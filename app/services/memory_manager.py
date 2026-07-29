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
    """Manages persistent memory storage for SLAM maps, tracked object trajectories, persistent 3D objects, and threat logs."""

    def __init__(self, storage_dir: str = "data/memory"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.objects_file = os.path.join(self.storage_dir, "persistent_objects.json")
        self.persistent_objects: Dict[str, Dict[str, Any]] = {}
        self.load_persistent_objects()

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

    # ============ PERSISTENT 3D OBJECT MEMORY STORE ============

    def load_persistent_objects(self):
        """Load persistent objects from disk."""
        try:
            if os.path.exists(self.objects_file):
                with open(self.objects_file, "r") as f:
                    self.persistent_objects = json.load(f)
                logger.info(f"📂 Persistent objects loaded: {len(self.persistent_objects)} stored objects.")
        except Exception as e:
            logger.error(f"Error loading persistent objects from disk: {e}")

    def save_persistent_objects(self):
        """Save persistent objects to disk."""
        try:
            with open(self.objects_file, "w") as f:
                json.dump(self.persistent_objects, f, indent=2)
            logger.info(f"💾 Persistent objects saved: {len(self.persistent_objects)} stored objects.")
        except Exception as e:
            logger.error(f"Error saving persistent objects to disk: {e}")

    def add_or_update_object(
        self,
        object_id: str,
        label: str,
        position: Dict[str, float],
        detected_by: str,
        confidence: float = 0.85
    ) -> Dict[str, Any]:
        """Add or update an object in persistent memory with spatial deduplication (<2.0m threshold)."""
        now = time.time()
        import math

        px = position.get("x", 0.0) if isinstance(position, dict) else 0.0
        py = position.get("y", 0.0) if isinstance(position, dict) else 0.0
        pz = position.get("z", 0.0) if isinstance(position, dict) else 0.0

        match_id = None
        if object_id in self.persistent_objects:
            match_id = object_id
        else:
            clean_label = label.lower().split(' #')[0]
            for oid, obj in self.persistent_objects.items():
                obj_lbl = str(obj.get("label", "")).lower().split(' #')[0]
                orig_lbl = str(obj.get("original_label", "")).lower().split(' #')[0]

                if obj_lbl == clean_label or orig_lbl == clean_label:
                    opos = obj.get("position", {})
                    if isinstance(opos, dict):
                        ox, oy, oz = opos.get("x", 0.0), opos.get("y", 0.0), opos.get("z", 0.0)
                        dist = math.sqrt((px - ox)**2 + (py - oy)**2 + (pz - oz)**2)
                        if dist <= 2.0:
                            match_id = oid
                            break

        target_key = match_id or object_id

        if target_key in self.persistent_objects:
            obj = self.persistent_objects[target_key]
            # Keep corrected label if object was manually overridden
            if not obj.get("is_corrected", False):
                obj["label"] = label
            obj["position"] = position
            obj["confidence"] = max(float(obj.get("confidence", 0.0)), float(confidence))
            obj["last_seen"] = now
            obj["detected_by"] = detected_by
        else:
            obj = {
                "object_id": target_key,
                "label": label,
                "original_label": label,
                "corrected_label": None,
                "is_corrected": False,
                "position": position,
                "detected_by": detected_by,
                "confidence": float(confidence),
                "first_seen": now,
                "last_seen": now
            }
            self.persistent_objects[target_key] = obj

        self.save_persistent_objects()
        return obj

    def correct_object_label(self, object_id: str, new_label: str) -> Optional[Dict[str, Any]]:
        """
        Manually correct/override an object's detected classification.
        Saves updated state permanently to disk.
        """
        if object_id in self.persistent_objects:
            obj = self.persistent_objects[object_id]
            obj["corrected_label"] = new_label
            obj["label"] = new_label
            obj["is_corrected"] = True
            obj["last_seen"] = time.time()
            self.save_persistent_objects()
            logger.info(f"✏️ Object '{object_id}' corrected: label updated to '{new_label}'.")
            return obj
        
        # If object_id not found directly, search by threat ID suffix or key match
        for oid, obj in self.persistent_objects.items():
            if object_id in oid or oid in object_id:
                obj["corrected_label"] = new_label
                obj["label"] = new_label
                obj["is_corrected"] = True
                obj["last_seen"] = time.time()
                self.save_persistent_objects()
                logger.info(f"✏️ Object '{oid}' corrected: label updated to '{new_label}'.")
                return obj

        return None

    def get_all_persistent_objects(self) -> List[Dict[str, Any]]:
        """Get all persistent memory objects."""
        return list(self.persistent_objects.values())

    def clear_persistent_objects(self):
        """Clear all stored persistent memory objects."""
        self.persistent_objects.clear()
        self.save_persistent_objects()
        logger.info("🗑️ Persistent memory store cleared.")


memory_manager = PersistentMemoryManager()


