# 🚀 QUICK SETUP: Ray-Ban Glasses + Laptop (1 Device)

---

## 📊 Your Setup

```
┌─────────────────────────────────────┐
│     YOUR LAPTOP                     │
│  ┌─────────────────────────────┐    │
│  │ Detection Server            │    │
│  │ (Python + YOLOv8)          │    │
│  │ Running on port 5000        │    │
│  │                             │    │
│  │ IP: 192.168.1.100          │    │
│  └──────────────┬──────────────┘    │
│                 │                    │
│           ┌─────▼─────┐              │
│           │  WiFi     │              │
│           └─────┬─────┘              │
│                 │                    │
└─────────────────┼────────────────────┘
                  │
                  │
                  │
┌─────────────────▼────────────────────┐
│     RAY-BAN META GLASSES             │
│                                       │
│  Browser + glasses_ui.html           │
│  Connects to: 192.168.1.100:5000     │
│                                       │
│  Shows: Alerts + Direction Arrows    │
│  Vibrates + Plays Sound              │
└───────────────────────────────────────┘
```

---

## ✅ BEFORE YOU START

- [ ] Python 3.11+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed:
  ```bash
  pip install opencv-python ultralytics flask flask-socketio
  ```
- [ ] Webcam working (test: python test_yolo.py)
- [ ] Ray-Ban glasses charged
- [ ] Glasses + laptop on SAME WiFi network

---

## 🎬 5-MINUTE SETUP

### **Step 1: Find Your Laptop IP (1 min)**

**Windows:**
```bash
ipconfig
```
Look for line: `IPv4 Address . . . . . . . . . . : 192.168.X.X`

**Mac/Linux:**
```bash
ifconfig
```
Look for: `inet 192.168.X.X`

**Example:** `192.168.1.100` ← **Remember this!**

---

### **Step 2: Copy Detection Server Code (2 min)**

Save this as `detector_server.py`:

```python
from ultralytics import YOLO
import cv2
from flask import Flask
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

model = YOLO('yolov8n.pt')
workers = {}
latest_detections = []

def get_simple_direction(x_center, frame_width):
    third = frame_width / 3
    if x_center < third:
        return "LEFT"
    elif x_center > 2 * third:
        return "RIGHT"
    else:
        return "FRONT"

def detection_loop():
    global latest_detections
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        h, w = frame.shape[:2]
        results = model(frame)
        people_count = 0
        detections = []
        danger_alert = None
        
        for detection in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = detection
            if confidence > 0.5:
                class_name = model.names[int(class_id)]
                if class_name == 'person':
                    people_count += 1
                    center_x = int((x1 + x2) / 2)
                    direction = get_simple_direction(center_x, w)
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'direction': direction,
                        'center_x': center_x
                    })
        
        latest_detections = detections
        
        if people_count >= 2 and len(detections) >= 2:
            for i, det1 in enumerate(detections):
                for det2 in detections[i+1:]:
                    distance = abs(det1['center_x'] - det2['center_x'])
                    if distance < 100:
                        danger_alert = {
                            'type': 'COLLISION_RISK',
                            'severity': 'HIGH',
                            'direction': det1['direction'],
                            'message': f'⚠️ Person approaching from {det1["direction"]}!',
                            'timestamp': time.time()
                        }
        
        for worker_id in workers:
            if danger_alert:
                socketio.emit('danger_alert', danger_alert, to=workers[worker_id]['sid'])
            else:
                socketio.emit('detection_update', {'detections': detections}, to=workers[worker_id]['sid'])
        
        time.sleep(0.05)
    cap.release()

@socketio.on('connect')
def handle_connect():
    print(f'✓ Connected: {request.sid}')

@socketio.on('register_worker')
def handle_register_worker(data):
    worker_id = data['worker_id']
    device_type = data.get('device_type', 'unknown')
    workers[worker_id] = {'sid': request.sid, 'device_type': device_type}
    print(f'✓ Registered: {worker_id} ({device_type})')
    emit('confirmed', {'status': 'registered', 'worker_id': worker_id})

@socketio.on('disconnect')
def handle_disconnect():
    for worker_id, info in list(workers.items()):
        if info['sid'] == request.sid:
            del workers[worker_id]
            print(f'✗ Disconnected: {worker_id}')

@app.route('/health')
def health():
    return {
        'status': 'ok',
        'connected_workers': len(workers),
        'workers': list(workers.keys())
    }

if __name__ == '__main__':
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    
    print('\n' + '='*60)
    print('🚀 Ray-Ban Smart Glasses Detection Server')
    print('='*60)
    print('\nServer running on:')
    print('  Local:    http://localhost:5000')
    print('  Network:  http://<YOUR-LAPTOP-IP>:5000')
    print('\n'+'='*60 + '\n')
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
```

---

### **Step 3: Copy Glasses UI (1 min)**

Save this as `glasses_ui.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ray-Ban Safety System</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #000;
            font-family: Arial, sans-serif;
            color: #fff;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .container {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            position: relative;
        }
        
        .screen {
            width: 100%;
            height: 100%;
            background: #111;
            border: 3px solid #00ff00;
            position: relative;
            overflow: hidden;
        }
        
        .feed-info {
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 24px;
            color: #00ff00;
        }
        
        .alert {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 0, 0, 0.95);
            padding: 50px 80px;
            border-radius: 15px;
            text-align: center;
            display: none;
            z-index: 1000;
            box-shadow: 0 0 50px rgba(255, 0, 0, 0.8);
        }
        
        .alert.active {
            display: block;
            animation: pulse 0.5s infinite alternate;
        }
        
        @keyframes pulse {
            from { box-shadow: 0 0 30px rgba(255, 0, 0, 0.5); }
            to { box-shadow: 0 0 60px rgba(255, 0, 0, 1); }
        }
        
        .alert h1 {
            font-size: 60px;
            margin-bottom: 20px;
        }
        
        .alert p {
            font-size: 32px;
            font-weight: bold;
        }
        
        .arrow {
            position: fixed;
            top: 100px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 100px;
            color: #00ff00;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .arrow.show {
            opacity: 1;
            animation: bounce 0.6s infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateX(-50%) translateY(0); }
            50% { transform: translateX(-50%) translateY(-20px); }
        }
        
        .status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 255, 0, 0.3);
            border: 2px solid #00ff00;
            padding: 15px 20px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
        }
        
        .status.disconnected {
            background: rgba(255, 0, 0, 0.3);
            border-color: #ff0000;
            color: #ff0000;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="screen">
            <div class="feed-info">
                Connected to Detection Server ✓
            </div>
            <div class="arrow" id="arrow">↑</div>
            <div class="alert" id="alert">
                <h1>⚠️ DANGER!</h1>
                <p id="msg">Person approaching from FRONT</p>
            </div>
            <div class="status" id="status">Connecting...</div>
        </div>
    </div>

    <script>
        const WORKER_ID = 'rayban_' + Math.random().toString(36).substr(2, 9);
        const socket = io(window.location.origin);
        
        const arrows = {
            'LEFT': '←',
            'RIGHT': '→',
            'FRONT': '↑',
            'BEHIND': '↓'
        };
        
        socket.on('connect', () => {
            console.log('Connected');
            socket.emit('register_worker', {
                worker_id: WORKER_ID,
                device_type: 'Ray-Ban Meta Glasses'
            });
            document.getElementById('status').textContent = 'Connected ✓';
            document.getElementById('status').classList.remove('disconnected');
        });
        
        socket.on('danger_alert', (data) => {
            console.log('🚨 Alert:', data);
            
            // Vibrate
            if (navigator.vibrate) {
                navigator.vibrate([100, 50, 100, 50, 200]);
            }
            
            // Show alert
            const alertEl = document.getElementById('alert');
            const arrowEl = document.getElementById('arrow');
            
            document.getElementById('msg').textContent = data.message;
            arrowEl.textContent = arrows[data.direction] || '?';
            
            arrowEl.classList.add('show');
            alertEl.classList.add('active');
            
            playSound();
            
            setTimeout(() => {
                alertEl.classList.remove('active');
                arrowEl.classList.remove('show');
            }, 4000);
        });
        
        socket.on('disconnect', () => {
            document.getElementById('status').textContent = 'Disconnected ✗';
            document.getElementById('status').classList.add('disconnected');
        });
        
        function playSound() {
            try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.frequency.value = 800;
                gain.gain.setValueAtTime(0.3, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.5);
            } catch(e) {}
        }
    </script>
</body>
</html>
```

---

### **Step 4: Run Server (1 min)**

```bash
cd C:\shared_perception
env\Scripts\activate
python detector_server.py
```

You should see:
```
============================================================
🚀 Ray-Ban Smart Glasses Detection Server
============================================================

Server running on:
  Local:    http://localhost:5000
  Network:  http://192.168.1.100:5000

============================================================
```

---

### **Step 5: Connect Glasses (1 min)**

1. On Ray-Ban glasses, open browser
2. Connect to same WiFi as laptop
3. Go to: `http://192.168.1.100:5000/glasses_ui.html` (replace IP with yours)
4. You should see green screen with "Connected to Detection Server ✓"

---

## 🧪 TEST IT

### **Test 1: People Detection**
```
1. Have 1 person in front of laptop camera
2. Check glasses screen - should say "Connected ✓"
3. Person count should show in detection logs
```

### **Test 2: Trigger Alert**
```
1. Get 2 people in front of camera
2. Move them close together (~1 meter)
3. Glasses should show RED ALERT
4. Direction arrow appears (↑ or ← or →)
5. Glasses vibrate
6. Alert sound plays
7. Alert disappears after 4 seconds
```

---

## 🎯 SUCCESS CHECKLIST

```
☐ Python server starts without errors
☐ Glasses connects to same WiFi
☐ Browser loads glasses_ui.html
☐ Green screen shows "Connected ✓"
☐ Detection logs print to terminal
☐ When 2 people close: RED alert appears
☐ Direction arrow shows correctly
☐ Glasses vibrate (if supported)
☐ Alert sound plays
☐ Alert auto-hides after 4 seconds
```

---

## 🔴 QUICK FIXES

| Problem | Fix |
|---------|-----|
| **Glasses can't find server** | Both on same WiFi? Check IP address is correct |
| **"Connection refused" error** | Server not running? Try: `python detector_server.py` |
| **No vibration** | Some glasses don't support haptic yet. Check settings. |
| **No alert sound** | Browser sound muted? Check browser volume settings. |
| **Slow detection** | Reduce video resolution or use YOLOv8n model |

---

## 📱 NEXT: MULTIPLE DEVICES

Once working with 1 device, you can add:
- Webcam on glasses (use glasses' built-in camera)
- Multiple static cameras (add more in detection loop)
- Multiple workers (just open URL on other devices)

---

## 🎬 DEMO TIME!

```
FLOW:
1. "Server is running on my laptop"
2. "Glasses connected via WiFi"
3. "Detection running in real-time"
4. [Person walks in front of camera]
5. "Glasses receive alert: DANGER from FRONT!"
6. "Red alert with direction arrow"
7. "Instant feedback to worker"
```

---

**You're ready! Start with Step 1! 🚀**
