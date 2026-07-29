"""
Shared Perception Safety System - Central Server Main App.
"""
import sys
import os

# Ensure workspace root is in sys.path when executed directly as `python app/main.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Core application routers
from app.api.websocket import router as ws_router
from app.api.routes import router as api_router
from app.dashboard.visualizer import router as dashboard_router

# Shared Perception routers
try:
    from app.api.shared_websocket import router as shared_ws_router
except (ImportError, AttributeError):
    from fastapi import APIRouter
    shared_ws_router = APIRouter()

try:
    from app.api.shared_routes import router as shared_api_router
except (ImportError, AttributeError):
    from fastapi import APIRouter
    shared_api_router = APIRouter()

app = FastAPI(
    title="Shared Perception Safety System",
    description="Laptop-Centric Collaborative Perception & Threat Warning System",
    version="1.0.0"
)

# Enable CORS for web visualizers & external mobile web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(ws_router)
app.include_router(api_router)
app.include_router(dashboard_router)
app.include_router(shared_ws_router)
app.include_router(shared_api_router)

# Mount /public static files directory
os.makedirs("public", exist_ok=True)
app.mount("/public", StaticFiles(directory="public"), name="public")


@app.get("/")
async def root():
    return {
        "status": "online",
        "system": "Shared Perception Safety System",
        "websocket_endpoint": "/ws/mobile"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
