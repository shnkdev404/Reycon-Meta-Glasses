"""
Shared Perception Platform - Main Application Entrypoint.
Collaborative spatial perception & threat prediction backend for Ray-Ban Meta Smart Glasses.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health_router, ws_router, routes_router
from app.dashboard import dashboard_router
from app.utils.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Collaborative Spatial Perception & Threat Prediction Stack for Ray-Ban Meta Smart Glasses",
    version=settings.VERSION
)

# Enable CORS for web visualizers & external client apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST, WebSocket, and Dashboard Routers
app.include_router(routes_router)
app.include_router(health_router)
app.include_router(ws_router)
app.include_router(dashboard_router)
