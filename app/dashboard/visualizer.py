"""
Server Command Center Dashboard Router.

Serves the interactive high-tech Server Command Center Webpage
displaying the 3D spatial radar, metric counters, connected devices,
combined tracked objects, directed non-broadcast alert feed, and raw JSON state.
"""
import os
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

SHARED_DASHBOARD_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "public", "shared_dashboard.html"))
DASHBOARD_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clients", "server_dashboard.html"))
MOBILE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clients", "mobile_client.html"))


@router.get("/dashboard", response_class=HTMLResponse)
@router.get("/server", response_class=HTMLResponse)
@router.get("/shared", response_class=HTMLResponse)
async def render_dashboard():
    """Renders the REYCON Server Command Center & 3D Spatial Radar Webpage."""
    target_file = SHARED_DASHBOARD_FILE if os.path.exists(SHARED_DASHBOARD_FILE) else DASHBOARD_FILE
    try:
        if os.path.exists(target_file):
            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()
            return HTMLResponse(content=content)
    except Exception as e:
        pass
    
    return HTMLResponse(content="<h1>Server Command Center</h1><p>Loading dashboard...</p>")


@router.get("/mobile", response_class=HTMLResponse)
@router.get("/mobile_client.html", response_class=HTMLResponse)
async def render_mobile():
    """Renders the REYCON Tactical AR Mobile Client Webpage."""
    try:
        if os.path.exists(MOBILE_FILE):
            with open(MOBILE_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            return HTMLResponse(content=content)
    except Exception as e:
        pass
    
    return HTMLResponse(content="<h1>Mobile AR Client</h1><p>Loading HUD...</p>")

