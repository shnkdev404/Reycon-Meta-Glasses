# ⚡ CHEAT SHEET: Ray-Ban Safety System
## Quick Reference (Print this!)

---

## 🚀 QUICK START (5 min)

```bash
# Step 1: Setup
mkdir C:\shared_perception
cd C:\shared_perception
python -m venv env
env\Scripts\activate

# Step 2: Install
pip install opencv-python ultralytics flask flask-socketio

# Step 3: Run
python detector_server.py

# Step 4: Connect
# On glasses: http://YOUR_IP:5000/glasses_ui.html
```

---

## 📋 FILES YOU NEED

| File | Purpose | Copy From |
|------|---------|-----------|
| `detector_server.py` | Detection + server | COMPLETE_1WEEK_RAYBAN.md |
| `glasses_ui.html` | Glasses interface | COMPLETE_1WEEK_RAYBAN.md |
| `test_yolo.py` | Verify setup | ONE_WEEK_SPRINT.md |

---

## 🧪 QUICK TESTS

### Test 1: Is Python working?
```bash
python --version
# Should show: Python 3.11.x
```

### Test 2: Is YOLOv8 working?
```bash
python test_yolo.py
# Should show webcam with green boxes
```

### Test 3: Is server running?
```bash
python detector_server.py
# Should show: "🚀 RAY-BAN SAFETY SYSTEM"
```

### Test 4: Is glasses connecting?
```
Browser: http://localhost:5000/glasses_ui.html
Should show: "Connected to Detection Server ✓"
```

---

## 🎯 ALERT TRIGGER

**To see RED ALERT on glasses:**
1. Get 2 people in front of camera
2. Move them close together (< 1 meter)
3. Wait for collision detection
4. RED ALERT appears on glasses!

---

## 🔧 KEY COMMANDS

```bash
# Find your IP
ipconfig  (Windows)
ifconfig  (Mac/Linux)

# Start server
python detector_server.py

# Check server health
http://localhost:5000/health

# View statistics
http://localhost:5000/stats

# Check detections
http://localhost:5000

# Stop server
Ctrl + C
```

---

## 📞 QUICK FIXES

| Issue | Fix |
|-------|-----|
| "Port already in use" | `python detector_server.py --port 5001` |
| "No module named 'ultralytics'" | `pip install ultralytics` |
| "Cannot open webcam" | Try `cv2.VideoCapture(1)` instead of 0 |
| "Glasses won't connect" | Check both on same WiFi + IP correct |
| "No detections" | Check lighting + run `test_yolo.py` |
| "Slow alerts" | Close other apps + check WiFi |
| "No vibration" | Check browser permissions + volume |

---

## 🎯 DETECTION PARAMETERS

```python
# In detector_server.py, modify these:

# Confidence threshold (0-1, higher = stricter)
if confidence > 0.5:  # Change to 0.3 for more detections

# Collision distance (pixels, lower = closer)
if distance < 100:  # Change to 50 for stricter collision detection

# FPS (higher = more responsive but slower)
time.sleep(0.067)  # 15 FPS (0.033 for 30 FPS)

# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Can reduce to 320
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Can reduce to 240
```

---

## 📊 PERFORMANCE TUNING

| Change | Effect | Use When |
|--------|--------|----------|
| Lower confidence | More detections | Missing people |
| Higher confidence | Fewer false positives | Too many false alerts |
| Lower resolution | Faster | Laptop too slow |
| Higher FPS | More responsive | Enough compute power |
| Reduce frame size | 10x faster | Emergency speed needed |

---

## 🧬 CODE SNIPPETS

### Add new alert type:
```python
if some_condition:
    danger_alert = {
        'type': 'NEW_HAZARD',
        'severity': 'HIGH',
        'direction': 'FRONT',
        'message': 'Custom alert message'
    }
```

### Change direction calculation:
```python
def get_simple_direction(x_center, frame_width):
    third = frame_width / 3
    if x_center < third:
        return "LEFT"
    elif x_center > 2 * third:
        return "RIGHT"
    else:
        return "FRONT"
```

### Add new camera:
```python
CAMERAS = [0, 1]  # Multiple cameras

for cap_idx, cap in enumerate(cameras):
    ret, frame = cap.read()
    # Process each camera
```

---

## 📱 GLASSES INTERFACE CUSTOMIZATION

### Change alert color:
```html
<div class="alert-modal" style="background: rgba(255, 100, 0, 0.95);">
```

### Change alert text:
```html
<div class="alert-message" id="msg">Your custom message</div>
```

### Change direction arrow size:
```css
.direction-arrow {
    font-size: 80px;  /* Change from 120px */
}
```

### Change alert duration:
```javascript
setTimeout(() => {
    alertEl.classList.remove('active');
    arrowEl.classList.remove('show');
}, 3000);  // Change from 4000ms
```

---

## 🎬 DEMO TIMING

| Part | Time | What to Do |
|------|------|-----------|
| Intro | 30s | Explain concept |
| Server demo | 30s | Show terminal running |
| Glasses connection | 30s | Show connected interface |
| Detection demo | 30s | Person walks in front |
| Alert trigger | 60s | 2 people collide |
| Explanation | 60s | Explain real-time response |
| Q&A | Open | Answer questions |

**Total: ~5 minutes**

---

## 📈 SCALING UP

### For 2 Cameras:
```python
CAMERAS = [0, 1]
for cap in CAMERAS:
    detect_on_camera(cap)
```

### For Multiple Workers:
```python
# Already supported!
# Just open URL on multiple glasses
# Each gets their own worker_id
```

### For Cloud Deployment:
```python
# Change:
socketio.run(app, host='0.0.0.0', port=5000)
# To cloud IP (AWS/Azure/GCP)
```

---

## 🚨 ALERT SEVERITY LEVELS

```python
if distance < 50:
    severity = "CRITICAL"
elif distance < 100:
    severity = "HIGH"
elif distance < 150:
    severity = "MEDIUM"
else:
    severity = "LOW"
```

---

## 📊 MONITORING

### Check server health:
```bash
curl http://localhost:5000/health
```

### Check statistics:
```bash
curl http://localhost:5000/stats
```

### Monitor in real-time:
```bash
watch -n 1 curl http://localhost:5000/stats
```

---

## 🔐 SECURITY NOTES

```python
# Currently: Anyone can connect (development)
# For production, add:

@app.before_request
def check_auth():
    token = request.headers.get('Authorization')
    if not validate_token(token):
        return {'error': 'Unauthorized'}, 401
```

---

## 📦 DEPLOYMENT CHECKLIST

Before putting in production:

- [ ] Test with multiple users
- [ ] Test various lighting conditions
- [ ] Test at various distances
- [ ] Test with different clothing colors
- [ ] Add error logging
- [ ] Add crash recovery
- [ ] Add database for alerts
- [ ] Add authentication
- [ ] Add rate limiting
- [ ] Add monitoring/alerting
- [ ] Document all parameters
- [ ] Train team on system
- [ ] Have backup plan

---

## 🎓 LEARNING PATHS

**To Understand Better:**

1. **YOLOv8** → https://docs.ultralytics.com/
2. **Flask** → https://flask.palletsprojects.com/
3. **WebSocket** → https://developer.mozilla.org/en-US/docs/Web/API/WebSocket
4. **Computer Vision** → https://docs.opencv.org/
5. **Ray-Ban SDK** → https://www.meta.com/developers/

---

## ✅ SUCCESS SIGNALS

When things are working, you should see:

```
Terminal:
✓ YOLOv8 model loaded
✓ Webcam opened
✓ Server running on http://192.168.1.100:5000
✓ Client connected: abc123
✓ Registered: rayban_1234 (Ray-Ban Meta Glasses)

Glasses:
✓ Green screen shows "Connected to Detection Server ✓"
✓ Status bar shows "Connected"

Detection:
✓ When person enters frame: "Person detected"
✓ When 2 people close: "RED ALERT" appears
✓ Glasses vibrate
✓ Alert sound plays
✓ Alert disappears after 4 seconds
```

---

## 🆘 HELP!

**If stuck:**

1. **Check terminal output** - Most errors shown there
2. **Run test_yolo.py** - Verify detection works
3. **Check http://localhost:5000/health** - Server status
4. **Check browser console** (F12) - JS errors
5. **Restart everything** - Server + browser
6. **Check WiFi** - Both devices connected?
7. **Ask for help** - Paste error message in search

---

## 💾 BACKUP IMPORTANT FILES

```bash
# Create backup
cp detector_server.py detector_server.py.backup
cp glasses_ui.html glasses_ui.html.backup

# Or use Git:
git init
git add .
git commit -m "Working version"
```

---

## 🎉 YOU'RE READY!

Print this sheet, grab your laptop, and start coding!

**Remember:** Start small, test often, debug quickly.

**Good luck! 🚀**

---

## 📞 CONTACT COMMANDS

```bash
# Get IP for glasses
ipconfig | grep IPv4  (Windows)
ifconfig | grep inet   (Mac/Linux)

# Test connectivity
ping 192.168.1.100

# Check if port open
netstat -an | grep 5000

# Kill process on port 5000
lsof -i :5000 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

---

**Last updated:** 2026  
**Status:** Ready to use  
**Next update:** After week 1 complete
