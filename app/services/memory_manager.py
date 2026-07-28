"""
Phase 7: Persistent Map Memory & History Manager.
Saves and reloads global SLAM world maps, tracked object histories, and threat logs across sessions.
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("MemoryManager")


class PersistentMemoryManager:
    """Manages persistent memory storage for SLAM maps, tracked object trajectories, and threat logs."""

    def __init__(self, storage_dir: str = "data/memory"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_world_memory(self, memory_id: str, world_state: Dict[str, Any]) -> bool:
        """Persist complete world model state to JSON on disk."""
        try:
            file_path = os.path.join(self.storage_dir, f"{memory_id}.json")
            with open(file_path, "w") as f:
                json.dump(world_state, f, indent=2)
            logger.info(f"💾 Persistent memory saved successfully to '{file_path}'.")
            return True
        except Exception as e:
            logger.error(f"Error saving persistent memory: {e}")
            return False

    def load_world_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Load persistent world model state from JSON on disk."""
        try:
            file_path = os.path.join(self.storage_dir, f"{memory_id}.json")
            if not os.path.exists(file_path):
                return None
            with open(file_path, "r") as f:
                data = json.load(f)
            logger.info(f"📂 Persistent memory loaded from '{file_path}'.")
            return data
        except Exception as e:
            logger.error(f"Error loading persistent memory: {e}")
            return None

    def list_saved_memories(self) -> Dict[str, Any]:
        """List all saved world map memory sessions on disk."""
        try:
            files = [f.replace(".json", "") for f in os.listdir(self.storage_dir) if f.endswith(".json")]
            return {"memories": files, "count": len(files)}
        except Exception as e:
            logger.error(f"Error listing saved memories: {e}")
            return {"memories": [], "count": 0}


memory_manager = PersistentMemoryManager()
