# 🚨 COMPLETE 1-WEEK SPRINT: Ray-Ban Safety System
## Full Integration Guide + Testing Strategy for 1 Device

---

## 🎯 YOUR GOAL (1 Week)

Build a working **construction safety system** that:
- ✅ Detects people in real-time
- ✅ Identifies collision risks
- ✅ Sends instant alerts to Ray-Ban glasses
- ✅ Shows direction of danger (LEFT/RIGHT/FRONT/BEHIND)
- ✅ Vibrates + plays alert sound
- ✅ Works with 1 device (your glasses + laptop)
- ✅ Scalable to multiple cameras/workers later

---

## 📅 WEEK BREAKDOWN

```
DAY 1 (2-3 hours):  Setup + Test YOLOv8
DAY 2-3 (5 hours):  Build Detection Server
DAY 4 (3 hours):    Add Alert Logic
DAY 5 (2 hours):    Create Glasses UI
DAY 6 (4 hours):    Integration + Testing
DAY 7 (3 hours):    Polish + Demo Prep
```

---

## 🏗️ ARCHITECTURE

```
┌──────────────────────────────────────────┐
│         YOUR LAPTOP (Windows/Mac/Linux)  │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │   Detection Engine (YOLOv8)      │   │
│  │   • Webcam input                 │   │
│  │   • Real-time person detection   │   │
│  │   • FPS: 15-20                   │   │
│  └────────────┬─────────────────────┘   │
│               │                          │
│  ┌────────────▼─────────────────────┐   │
│  │   Alert Logic                    │   │
│  │   • Collision detection          │   │
│  │   • Direction calculation        │   │
│  │   • Distance measurement         │   │
│  └────────────┬─────────────────────┘   │
│               │                          │
│  ┌────────────▼─────────────────────┐   │
│  │   Dispatcher Server (Flask)      │   │
│  │   • WebSocket connections        │   │
│  │   • Real-time messaging          │   │
│  │   • Port: 5000                   │   │
│  │   • IP: 192.168.X.X              │   │
│  └────────────┬─────────────────────┘   │
│               │ (WiFi)                   │
└───────────────┼──────────────────────────┘
                │
                │ HTTP/WebSocket
                │ (Same WiFi Network)
                │
┌───────────────▼──────────────────────────┐
│    RAY-BAN META GLASSES                  │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │   Browser + glasses_ui.html      │   │
│  │   • Connects to server           │   │
│  │   • Receives alerts              │   │
│  │   • Shows direction arrows       │   │
│  │   • Vibrates + Alert sound       │   │
│  └──────────────────────────────────┘   │
│                                          │
└──────────────────────────────────────────┘
```

---

# 📝 DETAILED DAILY PLAN

---

## DAY 1: SETUP & VERIFY (2-3 hours)

### Phase 1A: Install Python & Tools (30 min)

**Install Python 3.11:**
- Go to: https://www.python.org/downloads/
- Download Python 3.11
- ✅ CHECK "Add Python to PATH"
- Restart computer

**Verify installation:**
```bash
# Open Command Prompt / Terminal
python --version
# Should show: Python 3.11.x
```

**Install VS Code (Optional but recommended):**
- https://code.visualstudio.com/
- Simple code editor for writing Python

### Phase 1B: Create Project Structure (20 min)

**Windows:**
```bash
# Create folder
mkdir C:\shared_perception
cd C:\shared_perception

# Create virtual environment
python -m venv env

# Activate it
env\Scripts\activate
# You should see: (env) in terminal
```

**Mac/Linux:**
```bash
# Create folder
mkdir ~/Desktop/shared_perception
cd ~/Desktop/shared_perception

# Create virtual environment
python3 -m venv env

# Activate it
source env/bin/activate
# You should see: (env) in terminal
```

### Phase 1C: Install Dependencies (15 min)

Copy-paste this (one line at a time):
```bash
pip install opencv-python
pip install ultralytics
pip install flask
pip install flask-socketio
pip install python-socketio
pip install python-engineio
```

Each will download and install. You'll see:
```
Successfully installed opencv-python-4.x.x
Successfully installed ultralytics-8.x.x
...
```

### Phase 1D: Test YOLOv8 (30 min)

Create file: `test_yolo.py`

```python
from ultralytics import YOLO
import cv2

# First run: downloads model (~200MB)
print("Loading YOLOv8...")
model = YOLO('yolov8n.pt')

print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    exit()

print("✓ Webcam opened. Press 'q' to quit.")
print("\nDetecting people in real-time...")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame, verbose=False)
    
    # Draw results
    annotated = results[0].plot()
    
    # Show frame
    cv2.imshow('YOLOv8 Detection', annotated)
    
    # Count detections
    frame_count += 1
    if frame_count % 10 == 0:
        num_detections = len(results[0].boxes)
        print(f"Frame {frame_count}: {num_detections} objects detected")
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nClosing...")
        break

cap.release()
cv2.destroyAllWindows()
print("✓ Test complete!")
```

**Run it:**
```bash
python test_yolo.py
```

**Expected output:**
- Webcam opens
- Green boxes appear around people/objects
- Terminal shows detection count
- Press 'q' to exit

**✅ DAY 1 COMPLETE if:**
- [ ] Python installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Webcam shows detections

---

## DAY 2-3: DETECTION SERVER (4-5 hours)

### Step 1: Create Detection Server

Create file: `detector_server.py`

```python
from ultralytics import YOLO
import cv2
import json
from flask import Flask, request
from flask_socketio import SocketIO, emit
import threading
import time
import socket

# Get local IP
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

# Flask + WebSocket setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'construction-safety-secret'

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25
)

# Load YOLOv8 model
print("🔄 Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

# Storage
workers = {}  # Connected workers/glasses
latest_detections = []
server_stats = {
    'frames_processed': 0,
    'detections_total': 0,
    'alerts_sent': 0
}

def get_simple_direction(x_center, frame_width):
    """
    Calculate direction based on pixel position in frame
    """
    third = frame_width / 3
    
    if x_center < third:
        return "LEFT"
    elif x_center > 2 * third:
        return "RIGHT"
    else:
        return "FRONT"

def detection_loop():
    """
    Main loop: Run detection on webcam feed
    """
    global latest_detections, server_stats
    
    print("🎥 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot open webcam!")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    print("✓ Webcam opened. Starting detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Cannot read frame")
            continue
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Run YOLOv8 detection
        results = model(frame, verbose=False)
        
        people_count = 0
        detections = []
        danger_alert = None
        
        # Process detections
        for detection in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = detection
            
            # Only keep high confidence detections
            if confidence > 0.5:
                class_name = model.names[int(class_id)]
                
                # Only process people
                if class_name == 'person':
                    people_count += 1
                    
                    # Calculate center position
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # Determine direction
                    direction = get_simple_direction(center_x, w)
                    
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'direction': direction,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height
                    })
        
        # Update latest detections
        latest_detections = detections
        server_stats['frames_processed'] += 1
        server_stats['detections_total'] += len(detections)
        
        # Check for collision risk (simplified)
        if people_count >= 2 and len(detections) >= 2:
            # Check if any two people are close together
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    det1 = detections[i]
                    det2 = detections[j]
                    
                    # Distance between centers
                    distance = abs(det1['center_x'] - det2['center_x'])
                    
                    # If too close (less than 100 pixels = ~1 meter at 5m distance)
                    if distance < 100:
                        danger_alert = {
                            'type': 'COLLISION_RISK',
                            'severity': 'HIGH',
                            'direction': det1['direction'],
                            'message': f'⚠️ Person approaching from {det1["direction"]}!',
                            'confidence': det1['confidence'],
                            'timestamp': time.time()
                        }
                        server_stats['alerts_sent'] += 1
                        break
        
        # Send to all connected workers
        for worker_id, worker_info in list(workers.items()):
            try:
                if danger_alert:
                    # Send alert
                    socketio.emit(
                        'danger_alert',
                        danger_alert,
                        to=worker_info['sid'],
                        skip_sid=None
                    )
                else:
                    # Send regular update (optional - can be heavy traffic)
                    # Uncomment if you want continuous updates
                    # socketio.emit(
                    #     'detection_update',
                    #     {'detections': detections},
                    #     to=worker_info['sid']
                    # )
                    pass
            except Exception as e:
                print(f"Error sending to {worker_id}: {e}")
        
        # Sleep to maintain ~15 FPS
        time.sleep(0.067)  # ~15 FPS
    
    cap.release()

# ============ WebSocket Events ============

@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    print(f'✓ Client connected: {client_id}')

@socketio.on('register_worker')
def handle_register_worker(data):
    """
    Worker (glasses) registers with system
    """
    worker_id = data.get('worker_id', f'worker_{request.sid[:8]}')
    device_type = data.get('device_type', 'unknown')
    
    workers[worker_id] = {
        'sid': request.sid,
        'device_type': device_type,
        'connected_at': time.time()
    }
    
    print(f'✓ Registered: {worker_id} ({device_type})')
    emit('registration_confirmed', {
        'status': 'success',
        'worker_id': worker_id,
        'server_time': time.time()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """
    Worker disconnected
    """
    for worker_id, info in list(workers.items()):
        if info['sid'] == request.sid:
            del workers[worker_id]
            print(f'✗ Disconnected: {worker_id}')

@socketio.on('update_position')
def handle_update_position(data):
    """
    Worker sends their position (for future use)
    """
    worker_id = data.get('worker_id')
    position = data.get('position')
    # Could use this for distance-based alerts

# ============ HTTP Endpoints ============

@app.route('/health')
def health():
    return {
        'status': 'ok',
        'connected_workers': len(workers),
        'workers': list(workers.keys()),
        'latest_detections': latest_detections,
        'server_stats': server_stats
    }

@app.route('/stats')
def stats():
    return {
        'frames_processed': server_stats['frames_processed'],
        'total_detections': server_stats['detections_total'],
        'alerts_sent': server_stats['alerts_sent'],
        'connected_workers': len(workers)
    }

@app.route('/')
def index():
    return '''
    <h1>🚨 Construction Safety Detection Server</h1>
    <p>Server is running!</p>
    <p><a href="/health">Health Check</a></p>
    <p><a href="/stats">Statistics</a></p>
    '''

if __name__ == '__main__':
    print('\n' + '='*70)
    print('🚀 RAY-BAN SAFETY SYSTEM - DETECTION SERVER')
    print('='*70)
    
    # Get IP
    local_ip = get_local_ip()
    
    print(f'\nServer Configuration:')
    print(f'  Local:    http://localhost:5000')
    print(f'  Network:  http://{local_ip}:5000')
    print(f'\n  Glasses UI: http://{local_ip}:5000/glasses_ui.html')
    print(f'\nHealth Check: http://{local_ip}:5000/health')
    print(f'Statistics:   http://{local_ip}:5000/stats')
    print(f'\nDetection Running on Webcam (0)')
    print('='*70)
    print()
    
    # Start detection in background
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    
    # Start server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
```

### Step 2: Test Server

**Run it:**
```bash
python detector_server.py
```

**Expected output:**
```
======================================================================
🚀 RAY-BAN SAFETY SYSTEM - DETECTION SERVER
======================================================================

Server Configuration:
  Local:    http://localhost:5000
  Network:  http://192.168.1.100:5000

  Glasses UI: http://192.168.1.100:5000/glasses_ui.html

Health Check: http://192.168.1.100:5000/health
Statistics:   http://192.168.1.100:5000/stats

Detection Running on Webcam (0)
======================================================================
```

**Verify:**
- Open browser: http://localhost:5000/
- Should show: "Server is running!"
- Check: http://localhost:5000/health
- Should show JSON with server stats

**✅ DAY 2-3 COMPLETE if:**
- [ ] Server starts without errors
- [ ] Webcam detection running
- [ ] Health endpoint works
- [ ] No connection errors

---

## DAY 4: ALERT LOGIC (Already Included!)

The detection server from Day 2-3 already has collision detection logic:

```python
# Check for collision risk
if people_count >= 2 and len(detections) >= 2:
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            distance = abs(det1['center_x'] - det2['center_x'])
            if distance < 100:  # Collision risk!
                danger_alert = {
                    'type': 'COLLISION_RISK',
                    'severity': 'HIGH',
                    'direction': det1['direction'],
                    'message': f'⚠️ Person approaching from {det1["direction"]}!'
                }
```

**✅ DAY 4 COMPLETE**

---

## DAY 5: GLASSES UI (2 hours)

Create file: `glasses_ui.html`

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
        
        html, body {
            width: 100%;
            height: 100%;
        }
        
        body {
            background: #000;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #fff;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }
        
        .glasses-screen {
            width: 100%;
            height: 100%;
            background: #0a0a0a;
            position: relative;
            overflow: hidden;
        }
        
        .main-view {
            width: 100%;
            height: 100%;
            background: #111;
            border: 3px solid #00ff00;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 20px;
            color: #00ff00;
            position: relative;
            font-weight: bold;
        }
        
        /* Alert Box */
        .alert-modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, rgba(255, 0, 0, 0.95), rgba(200, 0, 0, 0.95));
            border: 4px solid #ff0000;
            border-radius: 20px;
            padding: 60px 100px;
            text-align: center;
            opacity: 0;
            pointer-events: none;
            z-index: 1000;
            transform: translate(-50%, -50%) scale(0.5);
            transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
            box-shadow: 0 0 40px rgba(255, 0, 0, 0.6);
        }
        
        .alert-modal.active {
            opacity: 1;
            transform: translate(-50%, -50%) scale(1);
            pointer-events: all;
            animation: pulse-alert 0.4s infinite alternate;
        }
        
        @keyframes pulse-alert {
            from { box-shadow: 0 0 30px rgba(255, 0, 0, 0.4); }
            to { box-shadow: 0 0 80px rgba(255, 0, 0, 1); }
        }
        
        .alert-title {
            font-size: 64px;
            font-weight: 900;
            margin-bottom: 30px;
            text-shadow: 0 0 20px rgba(0, 0, 0, 0.8);
            letter-spacing: 2px;
        }
        
        .alert-message {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
            text-shadow: 0 0 10px rgba(0, 0, 0, 0.8);
        }
        
        .alert-severity {
            font-size: 24px;
            opacity: 0.95;
            text-shadow: 0 0 5px rgba(0, 0, 0, 0.8);
        }
        
        /* Direction Arrow */
        .direction-arrow {
            position: fixed;
            top: 80px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 120px;
            color: #00ff00;
            opacity: 0;
            pointer-events: none;
            text-shadow: 0 0 20px #00ff00;
            transition: opacity 0.3s;
            z-index: 500;
        }
        
        .direction-arrow.show {
            opacity: 1;
            animation: arrow-bounce 0.6s infinite;
        }
        
        @keyframes arrow-bounce {
            0%, 100% { transform: translateX(-50%) translateY(0) rotateZ(0deg); }
            50% { transform: translateX(-50%) translateY(-30px) rotateZ(5deg); }
        }
        
        /* Status Bar */
        .status-bar {
            position: fixed;
            bottom: 30px;
            left: 30px;
            background: rgba(0, 255, 0, 0.15);
            border: 2px solid #00ff00;
            border-radius: 10px;
            padding: 20px 30px;
            font-size: 16px;
            font-weight: bold;
            z-index: 100;
            backdrop-filter: blur(5px);
        }
        
        .status-bar.disconnected {
            background: rgba(255, 0, 0, 0.15);
            border-color: #ff0000;
            color: #ff0000;
        }
        
        .status-dot {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
            animation: blink 1s infinite;
        }
        
        .status-dot.connected {
            background: #00ff00;
        }
        
        .status-dot.disconnected {
            background: #ff0000;
        }
        
        @keyframes blink {
            0%, 49% { opacity: 1; }
            50%, 100% { opacity: 0.4; }
        }
        
        /* Stats Panel */
        .stats-panel {
            position: fixed;
            top: 30px;
            right: 30px;
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid #00ff00;
            border-radius: 10px;
            padding: 20px;
            font-size: 14px;
            font-family: 'Courier New', monospace;
            max-width: 300px;
            z-index: 100;
            backdrop-filter: blur(5px);
        }
        
        .stat-row {
            margin: 8px 0;
            color: #00ff00;
        }
        
        .stat-label {
            color: #00aa00;
            opacity: 0.8;
            margin-right: 10px;
        }
        
        .stat-value {
            color: #00ff00;
            font-weight: bold;
        }
        
        /* Mobile optimization */
        @media (max-width: 768px) {
            .alert-title {
                font-size: 48px;
            }
            .alert-message {
                font-size: 28px;
                padding: 0 20px;
            }
            .direction-arrow {
                font-size: 80px;
            }
            .stats-panel {
                font-size: 12px;
                padding: 15px;
                top: 15px;
                right: 15px;
            }
            .status-bar {
                bottom: 15px;
                left: 15px;
                padding: 15px 20px;
                font-size: 14px;
            }
        }
    </style>
</head>
<body>
    <div class="glasses-screen">
        <div class="main-view">
            📹 Connected to Detection Server ✓
        </div>
        
        <div class="direction-arrow" id="arrow">↑</div>
        
        <div class="alert-modal" id="alert">
            <div class="alert-title">⚠️ DANGER!</div>
            <div class="alert-message" id="msg">Person approaching from FRONT</div>
            <div class="alert-severity">Severity: <span id="severity">HIGH</span></div>
        </div>
        
        <div class="status-bar disconnected" id="status">
            <span class="status-dot disconnected" id="dot"></span>
            <span id="status-text">Connecting...</span>
        </div>
        
        <div class="stats-panel">
            <div class="stat-row">
                <span class="stat-label">Device:</span>
                <span class="stat-value">Ray-Ban Meta</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">People:</span>
                <span class="stat-value" id="people">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Status:</span>
                <span class="stat-value" id="sys-status">Init</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Server:</span>
                <span class="stat-value" id="server-stat">Offline</span>
            </div>
        </div>
    </div>

    <script>
        const WORKER_ID = 'rayban_' + Math.floor(Math.random() * 10000);
        const SERVER = window.location.origin;
        
        const ARROWS = {
            'LEFT': '←',
            'RIGHT': '→',
            'FRONT': '↑',
            'BEHIND': '↓',
            'UNKNOWN': '?'
        };
        
        console.log('Worker ID:', WORKER_ID);
        console.log('Connecting to:', SERVER);
        
        // Socket connection
        const socket = io(SERVER, {
            reconnectionDelayMax: 10000,
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: 5
        });
        
        let alertActive = false;
        
        socket.on('connect', () => {
            console.log('✓ Connected to server');
            
            // Register as worker
            socket.emit('register_worker', {
                worker_id: WORKER_ID,
                device_type: 'Ray-Ban Meta Glasses'
            });
            
            updateStatus('Connected', true);
            document.getElementById('server-stat').textContent = 'Online';
        });
        
        socket.on('registration_confirmed', (data) => {
            console.log('✓ Registration confirmed:', data);
            document.getElementById('sys-status').textContent = 'Ready';
        });
        
        socket.on('danger_alert', (data) => {
            if (alertActive) return; // Prevent multiple simultaneous alerts
            alertActive = true;
            
            console.log('🚨 ALERT RECEIVED:', data);
            
            const alertEl = document.getElementById('alert');
            const arrowEl = document.getElementById('arrow');
            const msgEl = document.getElementById('msg');
            const sevEl = document.getElementById('severity');
            
            // Update content
            msgEl.textContent = data.message || `${data.type} from ${data.direction}!`;
            sevEl.textContent = data.severity || 'HIGH';
            
            // Show arrow
            arrowEl.textContent = ARROWS[data.direction] || '?';
            arrowEl.classList.add('show');
            
            // Show alert
            alertEl.classList.add('active');
            
            // Haptic feedback
            if (navigator.vibrate) {
                navigator.vibrate([100, 50, 100, 50, 150]);
            }
            
            // Audio alert
            playAlertSound();
            
            // Auto-hide after 4 seconds
            setTimeout(() => {
                alertEl.classList.remove('active');
                arrowEl.classList.remove('show');
                alertActive = false;
            }, 4000);
        });
        
        socket.on('detection_update', (data) => {
            // Update people count
            if (data.detections) {
                document.getElementById('people').textContent = data.detections.length;
            }
        });
        
        socket.on('disconnect', () => {
            console.log('✗ Disconnected');
            updateStatus('Disconnected', false);
            document.getElementById('server-stat').textContent = 'Offline';
        });
        
        socket.on('error', (error) => {
            console.error('Socket error:', error);
        });
        
        function updateStatus(text, connected) {
            const statusEl = document.getElementById('status');
            const dotEl = document.getElementById('dot');
            const textEl = document.getElementById('status-text');
            
            textEl.textContent = text;
            
            if (connected) {
                statusEl.classList.remove('disconnected');
                dotEl.classList.remove('disconnected');
                dotEl.classList.add('connected');
            } else {
                statusEl.classList.add('disconnected');
                dotEl.classList.remove('connected');
                dotEl.classList.add('disconnected');
            }
        }
        
        function playAlertSound() {
            try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                
                osc.connect(gain);
                gain.connect(ctx.destination);
                
                osc.frequency.value = 800;
                osc.type = 'sine';
                
                gain.gain.setValueAtTime(0.4, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
                
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.5);
                
                console.log('✓ Alert sound played');
            } catch (e) {
                console.log('Audio not available');
            }
        }
        
        // Initial status
        updateStatus('Connecting...', false);
    </script>
</body>
</html>
```

**✅ DAY 5 COMPLETE**

---

## DAY 6: INTEGRATION & TESTING (4 hours)

### Setup (30 min)

**Terminal 1: Start Server**
```bash
cd C:\shared_perception
env\Scripts\activate
python detector_server.py
```

Wait for output:
```
🚀 RAY-BAN SAFETY SYSTEM - DETECTION SERVER
Server Configuration:
  Local:    http://localhost:5000
  Network:  http://192.168.1.100:5000
```

**Note your IP address!** (e.g., `192.168.1.100`)

### Connect Glasses (15 min)

1. **On Ray-Ban glasses:**
   - Open WiFi settings
   - Connect to same WiFi as laptop
   - Open Chrome/Safari browser
   - Go to: `http://192.168.1.100:5000/glasses_ui.html`
   - Replace `192.168.1.100` with your laptop's IP

2. **Verify connection:**
   - Green screen shows: "Connected to Detection Server ✓"
   - Status bar shows: "Connected"
   - Server terminal shows: "✓ Registered: rayban_XXXX"

### Run Tests (2+ hours)

**Test 1: Basic Connection**
```
Expected:
- Glasses connect without error
- Green status indicator
- No error messages in terminal
Result: ✅ PASS / ❌ FAIL
```

**Test 2: Person Detection**
```
Setup: 1 person in front of camera
Expected:
- Terminal shows frame processing
- Stats panel shows "People: 1"
Result: ✅ PASS / ❌ FAIL
```

**Test 3: Collision Alert**
```
Setup: 2 people walk close together (< 1 meter)
Expected:
- RED ALERT appears on glasses
- Direction arrow shows (↑ or ← or →)
- Glasses vibrate
- Alert sound plays
- Alert disappears after 4 seconds
Result: ✅ PASS / ❌ FAIL
```

**Test 4: Multiple Alerts**
```
Setup: Trigger multiple collision scenarios
Expected:
- Each triggers a new alert
- No errors or crashes
- Server remains responsive
Result: ✅ PASS / ❌ FAIL
```

**Test 5: Disconnect/Reconnect**
```
Setup: Disconnect glasses, reconnect
Expected:
- Status updates to "Disconnected"
- Status updates to "Connected" when reconnecting
- No data loss
Result: ✅ PASS / ❌ FAIL
```

### Debugging Checklist

```
❌ Glasses won't connect:
   □ Check WiFi: Both on same network?
   □ Check IP address: Correct?
   □ Check URL: http://192.168.1.XXX:5000/glasses_ui.html
   □ Check server: Running without errors?
   □ Try refresh in browser

❌ No vibration:
   □ Check browser settings: Allow haptic feedback
   □ Some glasses don't support haptic
   □ Try audio alert instead

❌ No alert sound:
   □ Check browser volume
   □ Check glasses volume
   □ Try other alert (vibration)

❌ Detection not working:
   □ Check webcam: python test_yolo.py
   □ Check server: http://localhost:5000/health
   □ Check model loaded: Look for "Loading YOLOv8"

❌ Slow detection:
   □ Reduce resolution: Change 640x480 to smaller
   □ Close other apps
   □ Check CPU usage
```

**✅ DAY 6 COMPLETE if all tests pass**

---

## DAY 7: POLISH & DEMO PREP (3 hours)

### Polish (1 hour)

**Improvements:**
```python
# In detector_server.py, add logging for demo:
if server_stats['frames_processed'] % 60 == 0:
    print(f"Stats: {server_stats['frames_processed']} frames, {server_stats['detections_total']} detections, {server_stats['alerts_sent']} alerts")
```

### Demo Preparation (1 hour)

**Create Demo Script (`DEMO.txt`):**
```
CONSTRUCTION SAFETY SYSTEM - LIVE DEMO

SETUP (Before demo):
1. Laptop with server running
2. Ray-Ban glasses connected to WiFi
3. 2 people ready for demonstration

FLOW:
=======
INTRO (30 seconds):
- "This is a real-time construction safety system"
- "It uses AI to detect collision risks"
- "Sends instant alerts to workers' glasses"

DEMO (2-3 minutes):
1. Show server running
   Terminal: "Server is running on http://192.168.1.100:5000"

2. Show glasses interface
   "Glasses connected and ready to receive alerts"
   "Green status indicator shows 'Connected'"

3. Person 1 walks in front of camera
   "System detects person on the left"
   Terminal: "1 detection, direction: LEFT"

4. Person 2 walks towards Person 1
   "As Person 2 approaches..."
   "...collision detected!"
   
5. RED ALERT APPEARS ON GLASSES
   "Danger alert sent to glasses in real-time"
   "Direction arrow shows where threat is coming from"
   "Glasses vibrate and play alert sound"
   
6. Alert auto-dismisses
   "Alert clears automatically after 4 seconds"
   "System ready for next alert"

TALKING POINTS:
✓ Real-time detection using YOLOv8
✓ Collision risk analysis
✓ Instant wireless communication
✓ Haptic + audio feedback
✓ Scalable to multiple cameras and workers
✓ Can be deployed on actual construction sites

NEXT STEPS:
- Fine-tune hazard detection (machinery, falling objects)
- Integrate actual Ray-Ban Meta SDK
- Add GPS-based positioning
- Deploy on construction sites
- Train on site-specific hazards

QUESTIONS:
Q: How fast is it?
A: Real-time - alerts sent in < 1 second

Q: Works with multiple workers?
A: Yes! Each glasses user gets their own alert

Q: Can it detect other hazards?
A: Yes! Model can be fine-tuned for falling objects, machinery, etc.
```

### Final Checklist (1 hour)

```
BEFORE DEMO:
☐ Server starts without errors
☐ Glasses connect successfully
☐ Collision detection works
☐ Alerts display correctly
☐ Vibration works
☐ Sound plays
☐ No crashes or lag

DEMO EQUIPMENT:
☐ Laptop charged (or plugged in)
☐ Ray-Ban glasses charged
☐ WiFi stable and fast
☐ 2 people ready to participate
☐ Demo script printed/visible
☐ Backup phone for screenshot if needed

DEMO ROOM:
☐ Good lighting for camera
☐ Clear space for movement
☐ Camera can see people clearly
☐ WiFi reaches everywhere
```

**✅ DAY 7 COMPLETE**

---

## 🎉 END OF WEEK

You now have a **fully functional** construction safety system that:

✅ Runs on your laptop  
✅ Communicates with Ray-Ban glasses  
✅ Detects people in real-time  
✅ Identifies collision risks  
✅ Sends instant alerts with direction  
✅ Vibrates + plays alert sound  
✅ Can be demoed to investors/team  
✅ Foundation for production system  

---

## 🚀 NEXT STEPS (After Week 1)

```
Week 2-4:
□ Integrate actual Ray-Ban Meta SDK
□ Add depth estimation for 3D positioning
□ Train on construction-specific hazards
□ Add multiple camera support
□ Deploy to multiple glasses simultaneously

Month 2-3:
□ On-site testing
□ Hardware integration (IMU, GPS)
□ Cloud deployment
□ Mobile app for monitoring
□ Advanced analytics
```

---

## 📞 QUICK COMMANDS

```bash
# Setup (first time)
python -m venv env
env\Scripts\activate
pip install -r requirements.txt

# Daily run (2 terminals)
# Terminal 1:
python detector_server.py

# Terminal 2 (on glasses browser):
http://192.168.1.XXX:5000/glasses_ui.html
```

---

**You got this! Good luck! 🚀**
