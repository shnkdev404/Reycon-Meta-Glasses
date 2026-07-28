import logging
from typing import Dict
from fastapi import WebSocket

logger = logging.getLogger("ConnectionManager")
logging.basicConfig(level=logging.INFO)


class ConnectionManager:
    """Manages active WebSocket connections from client smartphones/glasses."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, glass_id: str, websocket: WebSocket):
        """Accept incoming WebSocket connection and register client."""
        await websocket.accept()
        self.active_connections[glass_id] = websocket
        logger.info(f"Client connected: '{glass_id}'. Total active connections: {len(self.active_connections)}")

    def disconnect(self, glass_id: str):
        """Remove client from active connections list on disconnect."""
        if glass_id in self.active_connections:
            del self.active_connections[glass_id]
            logger.info(f"Client disconnected: '{glass_id}'. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast JSON message to all currently connected clients."""
        disconnected_clients = []
        for glass_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client '{glass_id}': {e}")
                disconnected_clients.append(glass_id)

        # Cleanup failed connections
        for glass_id in disconnected_clients:
            self.disconnect(glass_id)

    async def send_personal_message(self, message: dict, glass_id: str) -> bool:
        """Send JSON message directly to a specific connected client."""
        connection = self.active_connections.get(glass_id)
        if connection:
            try:
                await connection.send_json(message)
                return True
            except Exception as e:
                logger.error(f"Error sending personal message to '{glass_id}': {e}")
                self.disconnect(glass_id)
        return False


connection_manager = ConnectionManager()
