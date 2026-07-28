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

DASHBOARD_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clients", "server_dashboard.html"))


@router.get("/dashboard", response_class=HTMLResponse)
@router.get("/server", response_class=HTMLResponse)
async def render_dashboard():
    """Renders the REYCON Server Command Center Webpage."""
    try:
        if os.path.exists(DASHBOARD_FILE):
            with open(DASHBOARD_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            return HTMLResponse(content=content)
    except Exception as e:
        pass
    
    return HTMLResponse(content="<h1>Server Command Center</h1><p>Loading dashboard...</p>")
