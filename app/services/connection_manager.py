"""
Phase 1: Networking Layer & Connection Manager.

Handles multiple simultaneous Meta Smart Glass WebSocket connections,
reconnection handling, heartbeats, and non-broadcast direct messaging.
"""
import asyncio
import time
from typing import Dict, Optional
from fastapi import WebSocket
from app.utils.logger import get_logger
from app.utils.config import settings

logger = get_logger("ConnectionManager")


class ConnectionManager:
    """Manages active smart glass WebSocket clients with auth and direct messaging."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.last_heartbeat: Dict[str, float] = {}

    async def connect(self, glass_id: str, websocket: WebSocket, auth_token: Optional[str] = None) -> bool:
        """
        Authenticate and accept a new smart glass connection.
        Placeholder auth check included for hackathon extension.
        """
        # Authentication Placeholder Check
        if auth_token and auth_token != settings.AUTH_SECRET_KEY:
            logger.warning(f"🔒 Auth failed for glass '{glass_id}'. Rejecting connection.")
            await websocket.close(code=4001, reason="Unauthorized connection request")
            return False

        await websocket.accept()
        self.active_connections[glass_id] = websocket
        self.last_heartbeat[glass_id] = time.time()
        logger.info(f"✅ Glass '{glass_id}' connected. ({len(self.active_connections)} active glasses)")
        return True

    def disconnect(self, glass_id: str):
        """Clean up connection on disconnect."""
        if glass_id in self.active_connections:
            del self.active_connections[glass_id]
        if glass_id in self.last_heartbeat:
            del self.last_heartbeat[glass_id]
        logger.info(f"❌ Glass '{glass_id}' disconnected. ({len(self.active_connections)} remaining)")

    def update_heartbeat(self, glass_id: str):
        """Update last active ping timestamp."""
        self.last_heartbeat[glass_id] = time.time()

    async def send_direct_message(self, glass_id: str, message: dict) -> bool:
        """
        Deliver message EXCLUSIVELY to a single specified smart glass (Non-broadcast).
        """
        websocket = self.active_connections.get(glass_id)
        if not websocket:
            logger.warning(f"⚠️ Failed to send direct message: Glass '{glass_id}' not connected.")
            return False

        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"⚠️ Error sending direct message to '{glass_id}': {e}")
            self.disconnect(glass_id)
            return False


connection_manager = ConnectionManager()
