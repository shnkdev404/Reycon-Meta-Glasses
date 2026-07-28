# Shared Perception Server & WebSockets

A production-ready real-time FastAPI WebSocket and REST server application.

---

## 📁 Project Structure

```
kaya-hackathon/
├── app/
│   ├── __init__.py
│   ├── main.py                     # Main FastAPI app initialization
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py               # REST API endpoints (/, /health)
│   │   └── websocket.py            # WebSocket real-time endpoint (/ws)
│   └── services/
│       ├── __init__.py
│       └── connection_manager.py   # Thread-safe WebSocket connection manager
├── clients/
│   ├── python_client.py            # Async CLI WebSocket client
│   └── web_client.html             # Glassmorphic Web browser client
├── roadmap/                        # Hackathon guides & Ray-Ban SDK integration docs
├── run.py                          # Primary server launch script
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project overview & documentation
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
Launch the FastAPI Uvicorn server:
```bash
python run.py
```
- Server will run at: `http://127.0.0.1:8000`
- WebSocket endpoint: `ws://127.0.0.1:8000/ws`
- REST Health check: `http://127.0.0.1:8000/health`

### 3. Connect Clients

#### **Option A: Python CLI Client**
Open a new terminal and run:
```bash
python clients/python_client.py
```
*(You can open multiple terminal windows to run simultaneous clients)*

#### **Option B: Web Client**
Open `clients/web_client.html` (or `client.html`) in your browser to use the real-time glassmorphic chat interface.
