# 🕶️ Ray-Ban Meta SDK Integration Guide
## With Single Device Testing Strategy

---

## ⚠️ IMPORTANT: Ray-Ban Meta SDK Reality Check

**Current Status (2024-2026):**
- Ray-Ban Meta glasses have **LIMITED official SDK**
- No full public API for third-party apps yet
- Meta mainly supports: video recording, voice commands, basic AR

**What we CAN do:**
1. ✅ Send data via their cloud API (if available)
2. ✅ Create companion mobile app that syncs with glasses
3. ✅ Use their WebXR API (experimental)
4. ✅ Build for Android (glasses run Android-based OS)

**What we CANNOT do easily:**
- ❌ Direct haptic feedback control
- ❌ Full AR overlay without their framework
- ❌ Real-time video processing on the glasses

---

## 🎯 RECOMMENDED APPROACH FOR 1 WEEK + 1 DEVICE

### **Option A: Companion Mobile App (EASIEST) ⭐ RECOMMENDED**
- Glasses → WiFi → Your Laptop (Detection)
- Your Laptop → WiFi → Android Phone
- Phone syncs with Glasses via Bluetooth/WiFi

### **Option B: Direct Glasses API (HARDER)**
- Glasses run custom Android app
- Connects directly to your detection server
- Requires Android development knowledge

### **Option C: Web-based (QUICKEST FOR DEMO)**
- Just use browser on glasses for now
- Perfect for 1-week prototype

---

## 📋 IMPLEMENTATION PLAN

### **STEP 1: Get Ray-Ban Meta Developer Access (Day 1)**

**Go to:**
- https://www.meta.com/en/developers/
- OR https://developers.meta.com/

**Sign up for:**
1. Meta Developer Account (free)
2. Ray-Ban Smart Glasses Developer Program
3. Request access to their documentation

**You'll get:**
- Developer console
- API keys
- Documentation access
- Sample code

**Note:** This can take 24-48 hours to approve

---

## 🔧 SETUP: 3 DIFFERENT APPROACHES

---

## **APPROACH 1: BROWSER-BASED (Fastest for Demo) ⭐**

### Why this is best for 1 week + 1 device:
- No app installation needed
- Works on any browser (including glasses browser)
- Can be tested on your laptop immediately
- Easy to show at demo

### How it works:
```
Glasses (Chrome Browser)
        ↓ (WiFi)
    Detection Server
        ↓
    Your Laptop
```

### Implementation:

**Step 1: Modify `detector_server.py` to add CORS headers**

```python
# detector_server.py
from ultralytics import YOLO
import cv2
import json
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# CORS headers for glasses browser
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO('yolov8n.pt')

workers = {}
latest_detections = []

def get_simple_direction(x_center, frame_width):
    """Simple direction based on x position in frame"""
    third = frame_width / 3
    
    if x_center < third:
        return "LEFT"
    elif x_center > 2 * third:
        return "RIGHT"
    else:
        return "FRONT"

def detection_loop():
    """Main detection loop"""
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
        
        # Danger detection logic
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
        
        # Send to all workers
        for worker_id in workers:
            if danger_alert:
                socketio.emit('danger_alert', danger_alert, 
                            to=workers[worker_id]['sid'])
            else:
                socketio.emit('detection_update', 
                            {'detections': detections}, 
                            to=workers[worker_id]['sid'])
        
        time.sleep(0.05)
    
    cap.release()

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    print(f'✓ Connected: {request.sid}')

@socketio.on('register_worker')
def handle_register_worker(data):
    worker_id = data['worker_id']
    device_type = data.get('device_type', 'unknown')
    workers[worker_id] = {
        'sid': request.sid,
        'device_type': device_type
    }
    print(f'✓ Registered: {worker_id} ({device_type})')
    emit('confirmed', {'status': 'registered', 'worker_id': worker_id})

@socketio.on('disconnect')
def handle_disconnect():
    for worker_id, info in list(workers.items()):
        if info['sid'] == request.sid:
            del workers[worker_id]
            print(f'✗ Disconnected: {worker_id}')

# HTTP Endpoints
@app.route('/health')
def health():
    return {
        'status': 'ok',
        'connected_workers': len(workers),
        'workers': list(workers.keys())
    }

if __name__ == '__main__':
    # Start detection
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    
    print('\n' + '='*60)
    print('🚀 Ray-Ban Smart Glasses Detection Server')
    print('='*60)
    print('\nServer running on:')
    print('  Local:    http://localhost:5000')
    print('  Network:  http://<YOUR-LAPTOP-IP>:5000')
    print('\nConnect your Ray-Ban glasses to the URL above\n')
    print('='*60 + '\n')
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
```

**Step 2: Create Glasses UI - `glasses_ui.html`**

Save this file and open it in browser on glasses:
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ray-Ban Safety System</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: #000;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #fff;
            height: 100vh;
            overflow: hidden;
        }
        
        .container {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        }
        
        /* Main view */
        .main-display {
            width: 100%;
            height: 100%;
            position: relative;
            overflow: hidden;
        }
        
        .camera-feed-placeholder {
            width: 100%;
            height: 100%;
            background: #111;
            border: 2px solid #00ff00;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 18px;
            color: #00ff00;
            font-weight: bold;
            text-align: center;
            padding: 40px;
        }
        
        /* Alert box */
        .alert-container {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 1000;
            pointer-events: none;
        }
        
        .alert-box {
            background: rgba(255, 0, 0, 0.95);
            border: 3px solid #ff0000;
            border-radius: 15px;
            padding: 40px 60px;
            text-align: center;
            opacity: 0;
            transform: scale(0.5);
            transition: all 0.3s ease;
            box-shadow: 0 0 50px rgba(255, 0, 0, 0.8);
        }
        
        .alert-box.active {
            opacity: 1;
            transform: scale(1);
            animation: pulse 0.5s infinite alternate;
        }
        
        @keyframes pulse {
            from { box-shadow: 0 0 30px rgba(255, 0, 0, 0.5); }
            to { box-shadow: 0 0 60px rgba(255, 0, 0, 1); }
        }
        
        .alert-box h1 {
            font-size: 48px;
            margin-bottom: 20px;
            text-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        }
        
        .alert-box p {
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
            text-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        }
        
        .severity {
            font-size: 20px;
            opacity: 0.9;
            margin-top: 15px;
        }
        
        /* Direction indicator */
        .direction-indicator {
            position: fixed;
            top: 80px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 120px;
            opacity: 0;
            transition: opacity 0.3s ease;
            filter: drop-shadow(0 0 10px #00ff00);
        }
        
        .direction-indicator.show {
            opacity: 1;
            animation: bounce 0.6s infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateX(-50%) translateY(0); }
            50% { transform: translateX(-50%) translateY(-20px); }
        }
        
        /* Status bar */
        .status-bar {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 255, 0, 0.2);
            border: 2px solid #00ff00;
            padding: 15px 25px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            max-width: 300px;
        }
        
        .status-bar.disconnected {
            background: rgba(255, 0, 0, 0.2);
            border-color: #ff0000;
            color: #ff0000;
        }
        
        .status-bar.connected {
            color: #00ff00;
        }
        
        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            animation: blink 1s infinite;
        }
        
        .status-dot.active {
            background: #00ff00;
        }
        
        .status-dot.inactive {
            background: #ff0000;
        }
        
        @keyframes blink {
            0%, 49%, 100% { opacity: 1; }
            50%, 99% { opacity: 0.3; }
        }
        
        /* Detection stats */
        .stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid #00ff00;
            padding: 15px;
            border-radius: 8px;
            font-size: 14px;
            font-family: monospace;
            max-width: 250px;
        }
        
        .stat-item {
            margin: 5px 0;
            color: #00ff00;
        }
        
        .stat-label {
            color: #00aa00;
            opacity: 0.7;
        }
        
        /* Mobile optimization */
        @media (max-width: 768px) {
            .alert-box {
                padding: 30px 40px;
            }
            
            .alert-box h1 {
                font-size: 36px;
            }
            
            .alert-box p {
                font-size: 24px;
            }
            
            .direction-indicator {
                font-size: 80px;
                top: 60px;
            }
            
            .stats {
                font-size: 12px;
                padding: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-display">
            <div class="camera-feed-placeholder">
                📹 Connected to Detection Server<br>
                <span style="font-size: 14px; margin-top: 20px;">Waiting for camera feed...</span>
            </div>
            
            <!-- Direction indicator -->
            <div class="direction-indicator" id="arrow">↑</div>
            
            <!-- Alert box -->
            <div class="alert-container">
                <div class="alert-box" id="alert">
                    <h1>⚠️ DANGER</h1>
                    <p id="alert-msg">Object approaching!</p>
                    <div class="severity">
                        Severity: <span id="alert-severity">HIGH</span>
                    </div>
                </div>
            </div>
            
            <!-- Status bar -->
            <div class="status-bar disconnected" id="status">
                <span class="status-dot inactive" id="status-dot"></span>
                <span id="status-text">Connecting...</span>
            </div>
            
            <!-- Stats -->
            <div class="stats">
                <div class="stat-item">
                    <span class="stat-label">Device:</span>
                    <span id="device-type">Loading...</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">People Detected:</span>
                    <span id="people-count">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Status:</span>
                    <span id="system-status">Initializing</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Configuration
        const WORKER_ID = 'rayban_' + Math.random().toString(36).substr(2, 9);
        const SERVER_URL = window.location.origin; // Connect to same server
        
        // Direction arrow mapping
        const directionArrows = {
            'LEFT': '←',
            'RIGHT': '→',
            'FRONT': '↑',
            'BEHIND': '↓',
            'UNKNOWN': '?'
        };
        
        // Get laptop IP for display
        function getLocalIP() {
            fetch('/health')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('system-status').textContent = 'Running';
                })
                .catch(() => {
                    document.getElementById('system-status').textContent = 'Offline';
                });
        }
        
        // Socket.io connection
        const socket = io(SERVER_URL);
        
        socket.on('connect', () => {
            console.log('✓ Connected to server');
            
            // Register this worker
            socket.emit('register_worker', {
                worker_id: WORKER_ID,
                device_type: 'Ray-Ban Meta Glasses'
            });
            
            // Update UI
            updateStatus('Connected', true);
            document.getElementById('device-type').textContent = 'Ray-Ban Meta';
        });
        
        socket.on('confirmed', (data) => {
            console.log('✓ Registered:', data);
        });
        
        socket.on('danger_alert', (data) => {
            console.log('🚨 ALERT RECEIVED:', data);
            
            // Vibrate if available (haptic feedback)
            if (navigator.vibrate) {
                navigator.vibrate([100, 50, 100, 50, 200]);
            }
            
            // Show alert
            const alertBox = document.getElementById('alert');
            const arrow = document.getElementById('arrow');
            
            document.getElementById('alert-msg').textContent = 
                data.message || `${data.type} from ${data.direction}!`;
            document.getElementById('alert-severity').textContent = data.severity;
            
            // Show direction arrow
            arrow.textContent = directionArrows[data.direction] || '?';
            arrow.classList.add('show');
            
            // Activate alert box
            alertBox.classList.add('active');
            
            // Play sound alert if available
            playAlertSound();
            
            // Auto-hide after 4 seconds
            setTimeout(() => {
                alertBox.classList.remove('active');
                arrow.classList.remove('show');
            }, 4000);
        });
        
        socket.on('detection_update', (data) => {
            // Update people count
            const peopleCount = data.detections ? data.detections.length : 0;
            document.getElementById('people-count').textContent = peopleCount;
        });
        
        socket.on('disconnect', () => {
            console.log('✗ Disconnected from server');
            updateStatus('Disconnected', false);
            document.getElementById('system-status').textContent = 'Offline';
        });
        
        function updateStatus(text, connected) {
            const statusEl = document.getElementById('status');
            const statusDot = document.getElementById('status-dot');
            const statusText = document.getElementById('status-text');
            
            statusText.textContent = text;
            
            if (connected) {
                statusEl.classList.remove('disconnected');
                statusEl.classList.add('connected');
                statusDot.classList.remove('inactive');
                statusDot.classList.add('active');
            } else {
                statusEl.classList.remove('connected');
                statusEl.classList.add('disconnected');
                statusDot.classList.remove('active');
                statusDot.classList.add('inactive');
            }
        }
        
        function playAlertSound() {
            // Create audio context for alert beep
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.value = 800;
                oscillator.type = 'sine';
                
                gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
                
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.5);
            } catch (e) {
                console.log('Audio context not available');
            }
        }
        
        // Initialize
        getLocalIP();
        updateStatus('Connecting...', false);
    </script>
</body>
</html>
```

---

## 🧪 TESTING WITH 1 DEVICE (Ray-Ban Meta Glasses)

### **Setup:**

**Step 1: Get Your Laptop IP**
```bash
# Windows
ipconfig
# Look for IPv4 Address (e.g., 192.168.1.100)

# Mac/Linux
ifconfig
# Look for inet (e.g., 192.168.1.100)
```

**Step 2: Start Detection Server on Laptop**
```bash
python detector_server.py
```

You'll see:
```
============================================================
🚀 Ray-Ban Smart Glasses Detection Server
============================================================

Server running on:
  Local:    http://localhost:5000
  Network:  http://192.168.1.100:5000

Connect your Ray-Ban glasses to the URL above
```

**Step 3: On Ray-Ban Glasses**
1. Connect glasses to same WiFi as laptop
2. Open browser (Chrome/Safari on glasses)
3. Go to: `http://192.168.1.100:5000/glasses_ui.html`
4. You should see the green bordered interface

**Step 4: Test Alert**
1. Position 2 people in front of laptop camera
2. Move them close together (~1 meter)
3. **Glasses should show RED ALERT** with direction arrow
4. Glasses should vibrate (if supported)
5. Glasses should play alert sound

---

### **Testing Checklist:**

```
□ Laptop and glasses on same WiFi
□ Server starts without errors
□ Glasses browser loads UI
□ Status shows "Connected"
□ People detected count updates in real-time
□ When 2 people close: RED ALERT appears
□ Direction arrow appears correctly
□ Glasses vibrate (optional)
□ Alert sound plays (optional)
□ Alert auto-dismisses after 4 seconds
```

---

## 🎮 DEMO SCRIPT (1 Device)

```
SETUP:
- Laptop with detection server running
- Ray-Ban glasses connected to same WiFi
- 1-2 people available for testing

FLOW:
1. "Server is running, monitoring camera feed"
2. "I open browser on glasses"
3. "Green interface loads - shows connection status"
4. "Now I'll trigger a danger alert"
5. "Two people walk close together in front of camera"
6. "Alert appears on glasses: RED with direction arrow"
7. "Glasses vibrate and play alert sound"
8. "Alert auto-dismisses after 4 seconds"
9. "System ready for next alert"

TALKING POINTS:
✓ "This detects collision risks in real-time"
✓ "Alert is sent instantly via WiFi"
✓ "Works with glasses browser - no app needed"
✓ "Can be extended with actual Ray-Ban SDK"
✓ "Scalable to multiple glasses + cameras"
```

---

## 📱 APPROACH 2: ACTUAL RAY-BAN APP (For Later)

After 1 week, if you want to build actual app:

### Step 1: Install Android Studio
- Download from: https://developer.android.com/studio

### Step 2: Create Android Project
```gradle
// Build.gradle
dependencies {
    implementation 'com.squareup.okhttp3:okhttp:4.9.1'
    implementation 'com.google.code.gson:gson:2.8.9'
    implementation 'io.socket:socket.io-client-java:2.1.0'
}
```

### Step 3: Basic Android Code
```kotlin
// MainActivity.kt
import io.socket.client.IO
import io.socket.client.Socket

class MainActivity : AppCompatActivity() {
    private lateinit var socket: Socket
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Connect to detection server
        val opts = IO.Options().apply {
            reconnection = true
        }
        
        socket = IO.socket("http://192.168.1.100:5000", opts)
        
        socket.on(Socket.EVENT_CONNECT) {
            socket.emit("register_worker", mapOf(
                "worker_id" to "rayban_phone",
                "device_type" to "Ray-Ban Meta Companion"
            ))
        }
        
        socket.on("danger_alert") { args ->
            val data = args[0] as JSONObject
            showAlert(data.optString("message"))
            vibratePhone()
        }
        
        socket.connect()
    }
    
    private fun showAlert(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        }
    }
    
    private fun vibratePhone() {
        val vibrator = getSystemService(VIBRATOR_SERVICE) as Vibrator
        vibrator.vibrate(longArrayOf(0, 100, 50, 100), -1)
    }
}
```

---

## 🚨 APPROACH 3: OFFICIAL META SDK (Future)

Meta has announced plans for:
- **MetaXR SDK** - For spatial computing apps
- **Glasses Dev Kit** - For enterprise
- **Cloud API** - For backend integration

**How to access:**
1. Go to: https://developers.meta.com/
2. Apply for Glasses Developer Program
3. Get access to official documentation
4. Download MetaXR SDK

---

## 📊 COMPARISON

| Aspect | Approach 1 (Browser) | Approach 2 (Android) | Approach 3 (SDK) |
|--------|-------------------|-----------------|--------|
| Time to setup | 30 min | 3-5 hours | 1-2 weeks |
| Requires coding | No | Yes (Kotlin) | Yes (C++) |
| Haptic feedback | Limited | Full | Full |
| Visual overlay | Limited | Better | Full AR |
| Vibration | Yes | Yes | Yes |
| Best for demo | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Production ready | No | Somewhat | Yes |

---

## 🎯 RECOMMENDATION FOR YOUR 1-WEEK SPRINT

**Use APPROACH 1 (Browser-based)**

Why:
- ✅ Works immediately
- ✅ No app installation
- ✅ Easy to test with 1 device
- ✅ Demonstrates full concept
- ✅ Can show in demo
- ✅ Builds foundation for Approach 2/3 later

**Then after Week 1:**
- Consider Approach 2 (Android app) for better integration
- Wait for official Meta SDK (Approach 3) for production

---

## 🔧 QUICK SETUP SUMMARY

```bash
# Terminal 1 - Run detection server
cd C:\shared_perception
env\Scripts\activate
python detector_server.py

# Then on Ray-Ban glasses:
# 1. Connect to same WiFi as laptop
# 2. Open browser
# 3. Go to: http://YOUR_LAPTOP_IP:5000/glasses_ui.html
# 4. Done! You're connected
```

---

## ⚡ REAL RAY-BAN META FEATURES YOU CAN USE

Once connected via browser, you can trigger:

✅ **Vibration**
```javascript
navigator.vibrate([100, 50, 100]);
```

✅ **Audio Alerts**
```javascript
playAlertSound();
```

✅ **Screen Changes**
```javascript
// Dynamic CSS updates
document.body.style.backgroundColor = 'red';
```

✅ **Notifications**
```javascript
new Notification("DANGER", { body: "Person approaching" });
```

❌ **NOT available yet** (need official SDK):
- Directional audio through glasses speakers
- AR overlay on glasses display
- Direct haptic motor control
- Eye tracking

---

## 🎬 NEXT STEPS

**For this week:**
1. Use browser approach (fastest)
2. Test with 1 device + laptop
3. Get demo working

**For next month:**
1. Explore Android app approach
2. Request official Meta SDK access
3. Build native app with full features

---

## 📞 TROUBLESHOOTING

**Problem: Glasses can't find server**
```
Solution: Make sure both on same WiFi network
Check: ping 192.168.1.100 (from glasses)
```

**Problem: No vibration on glasses**
```
Solution: Some glasses don't support haptic feedback yet
Alternative: Use audio alert (plays beep)
```

**Problem: Alert doesn't show**
```
Solution: Check browser console (F12) for errors
Check: Server is running (python detector_server.py)
Check: Worker registered (should show in terminal)
```

---

**You're all set! Ready to code? 🚀**
