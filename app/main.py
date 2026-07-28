"""
Shared Perception Safety System - Central Laptop Server.
FastAPI + WebSockets + YOLOv8 + OpenCV Computer Vision.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.websocket import router as ws_router

app = FastAPI(
    title="Shared Perception Safety System",
    description="Laptop-Centric Collaborative Perception & Threat Warning System for Construction Sites",
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

# Mount WebSocket Router
app.include_router(ws_router)


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
