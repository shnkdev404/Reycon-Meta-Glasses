"""
Phase 7 & Phase 15: Persistent Map Memory & SQLite History Manager.

Saves and reloads global SLAM world maps, tracked object histories, and threat logs across sessions.
Upgrades JSON persistence to indexed SQLite database storage with spatial and attribute queries.
"""
import os
import json
import time
import math
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("MemoryManager")


class PersistentMemoryManager:
    """
    Manages persistent SQLite memory storage for SLAM maps, tracked object trajectories,
    persistent 3D objects, and threat logs with spatial ($X, Y$) and attribute indexing.
    """

    def __init__(self, storage_dir: str = "data/memory", db_name: str = "persistent_objects.db"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.db_path = os.path.join(self.storage_dir, db_name)
        self.objects_file = os.path.join(self.storage_dir, "persistent_objects.json")
        self.persistent_objects: Dict[str, Dict[str, Any]] = {}
        
        self._init_sqlite_db()
        self.load_persistent_objects()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite_db(self):
        """Initialize SQLite table schema and create spatial & attribute indexes."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS persistent_objects (
                        object_id TEXT PRIMARY KEY,
                        object_class TEXT,
                        label TEXT,
                        original_label TEXT,
                        corrected_label TEXT,
                        is_corrected INTEGER DEFAULT 0,
                        pos_x REAL DEFAULT 0.0,
                        pos_y REAL DEFAULT 0.0,
                        pos_z REAL DEFAULT 0.0,
                        heading REAL DEFAULT 0.0,
                        threat_level TEXT DEFAULT 'LOW',
                        threat_score REAL DEFAULT 0.0,
                        confidence REAL DEFAULT 0.85,
                        detected_by TEXT,
                        first_seen REAL,
                        last_seen REAL,
                        metadata_json TEXT
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_object_class ON persistent_objects(object_class);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_threat_level ON persistent_objects(threat_level);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_spatial_pos ON persistent_objects(pos_x, pos_y);")
                conn.commit()
            logger.info(f"✅ SQLite Memory DB initialized at '{self.db_path}'.")
        except Exception as e:
            logger.error(f"Error initializing SQLite DB: {e}")

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

    # ============ PERSISTENT 3D OBJECT MEMORY STORE & SQLITE QUERY ENGINE ============

    def load_persistent_objects(self):
        """Load persistent objects from SQLite database (falling back to JSON import if fresh)."""
        self.persistent_objects.clear()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                rows = cursor.execute("SELECT * FROM persistent_objects").fetchall()
                for row in rows:
                    obj_dict = self._row_to_dict(row)
                    self.persistent_objects[obj_dict["object_id"]] = obj_dict

            # Migration fallback: If SQLite DB is empty but JSON exists, import JSON records
            if not self.persistent_objects and os.path.exists(self.objects_file):
                try:
                    with open(self.objects_file, "r") as f:
                        json_data = json.load(f)
                        if isinstance(json_data, dict):
                            for oid, obj in json_data.items():
                                self.add_or_update_object(
                                    object_id=oid,
                                    label=obj.get("label", "unknown"),
                                    position=obj.get("position", {"x": 0, "y": 0, "z": 0}),
                                    detected_by=obj.get("detected_by", "system"),
                                    confidence=obj.get("confidence", 0.85)
                                )
                except Exception as ex:
                    logger.debug(f"JSON migration warning: {ex}")

            logger.info(f"📂 Persistent objects loaded into memory & SQLite: {len(self.persistent_objects)} stored objects.")
        except Exception as e:
            logger.error(f"Error loading persistent objects from SQLite: {e}")

    def save_persistent_objects(self):
        """Sync current persistent objects state to disk (JSON backup & SQLite database)."""
        try:
            # Sync to JSON backup file
            with open(self.objects_file, "w") as f:
                json.dump(self.persistent_objects, f, indent=2)

            # Sync memory dict entries to SQLite database
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for oid, obj in self.persistent_objects.items():
                    self._upsert_sqlite_object(cursor, obj)
                conn.commit()
            logger.info(f"💾 Persistent objects saved to SQLite & JSON: {len(self.persistent_objects)} objects.")
        except Exception as e:
            logger.error(f"Error saving persistent objects to SQLite: {e}")

    def _upsert_sqlite_object(self, cursor: sqlite3.Cursor, obj: Dict[str, Any]):
        """Upsert a single object dictionary into SQLite database table."""
        oid = obj.get("object_id")
        label = obj.get("label", "unknown")
        clean_class = label.lower().split(" #")[0]
        pos = obj.get("position", {})
        px = pos.get("x", 0.0) if isinstance(pos, dict) else 0.0
        py = pos.get("y", 0.0) if isinstance(pos, dict) else 0.0
        pz = pos.get("z", 0.0) if isinstance(pos, dict) else 0.0

        cursor.execute("""
            INSERT INTO persistent_objects (
                object_id, object_class, label, original_label, corrected_label,
                is_corrected, pos_x, pos_y, pos_z, heading, threat_level, threat_score,
                confidence, detected_by, first_seen, last_seen, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(object_id) DO UPDATE SET
                object_class = excluded.object_class,
                label = excluded.label,
                corrected_label = excluded.corrected_label,
                is_corrected = excluded.is_corrected,
                pos_x = excluded.pos_x,
                pos_y = excluded.pos_y,
                pos_z = excluded.pos_z,
                heading = excluded.heading,
                threat_level = excluded.threat_level,
                threat_score = excluded.threat_score,
                confidence = excluded.confidence,
                detected_by = excluded.detected_by,
                last_seen = excluded.last_seen,
                metadata_json = excluded.metadata_json;
        """, (
            oid,
            clean_class,
            label,
            obj.get("original_label", label),
            obj.get("corrected_label"),
            1 if obj.get("is_corrected", False) else 0,
            px, py, pz,
            obj.get("heading", 0.0),
            obj.get("threat_level", "LOW"),
            float(obj.get("threat_score", 0.0)),
            float(obj.get("confidence", 0.85)),
            obj.get("detected_by", "system"),
            float(obj.get("first_seen", time.time())),
            float(obj.get("last_seen", time.time())),
            json.dumps(obj)
        ))

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row into standard python object dictionary representation."""
        metadata = {}
        if row["metadata_json"]:
            try:
                metadata = json.loads(row["metadata_json"])
            except Exception:
                pass

        pos = {
            "x": float(row["pos_x"]),
            "y": float(row["pos_y"]),
            "z": float(row["pos_z"])
        }

        res = {
            "object_id": row["object_id"],
            "object_class": row["object_class"],
            "label": row["label"],
            "original_label": row["original_label"],
            "corrected_label": row["corrected_label"],
            "is_corrected": bool(row["is_corrected"]),
            "position": pos,
            "heading": float(row["heading"]),
            "threat_level": row["threat_level"],
            "threat_score": float(row["threat_score"]),
            "confidence": float(row["confidence"]),
            "detected_by": row["detected_by"],
            "first_seen": float(row["first_seen"]),
            "last_seen": float(row["last_seen"])
        }
        if metadata:
            for k, v in metadata.items():
                if k not in res:
                    res[k] = v
        return res

    def add_or_update_object(
        self,
        object_id: str,
        label: str,
        position: Dict[str, float],
        detected_by: str,
        confidence: float = 0.85,
        threat_level: str = "LOW",
        threat_score: float = 0.0
    ) -> Dict[str, Any]:
        """Add or update an object in persistent memory & SQLite DB with spatial deduplication (<2.0m threshold)."""
        now = time.time()

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
            if not obj.get("is_corrected", False):
                obj["label"] = label
            obj["position"] = position
            obj["confidence"] = max(float(obj.get("confidence", 0.0)), float(confidence))
            obj["threat_level"] = threat_level
            obj["threat_score"] = threat_score
            obj["last_seen"] = now
            obj["detected_by"] = detected_by
        else:
            clean_class = label.lower().split(' #')[0]
            obj = {
                "object_id": target_key,
                "object_class": clean_class,
                "label": label,
                "original_label": label,
                "corrected_label": None,
                "is_corrected": False,
                "position": position,
                "detected_by": detected_by,
                "confidence": float(confidence),
                "threat_level": threat_level,
                "threat_score": threat_score,
                "first_seen": now,
                "last_seen": now
            }
            self.persistent_objects[target_key] = obj

        # Upsert directly to SQLite & backup
        with self._get_connection() as conn:
            cursor = conn.cursor()
            self._upsert_sqlite_object(cursor, obj)
            conn.commit()

        self.save_persistent_objects()
        return obj

    def correct_object_label(self, object_id: str, new_label: str) -> Optional[Dict[str, Any]]:
        """
        Manually correct/override an object's detected classification.
        Saves updated state permanently to SQLite database and disk.
        """
        target_obj = None
        target_id = None

        if object_id in self.persistent_objects:
            target_id = object_id
            target_obj = self.persistent_objects[object_id]
        else:
            for oid, obj in self.persistent_objects.items():
                if object_id in oid or oid in object_id:
                    target_id = oid
                    target_obj = obj
                    break

        if target_obj and target_id:
            target_obj["corrected_label"] = new_label
            target_obj["label"] = new_label
            target_obj["object_class"] = new_label.lower().split(' #')[0]
            target_obj["is_corrected"] = True
            target_obj["last_seen"] = time.time()

            with self._get_connection() as conn:
                cursor = conn.cursor()
                self._upsert_sqlite_object(cursor, target_obj)
                conn.commit()

            self.save_persistent_objects()
            logger.info(f"✏️ Object '{target_id}' corrected: label updated to '{new_label}'.")
            return target_obj

        return None

    # ============ SQLITE QUERY SYSTEM (TASK 15) ============

    def query_objects_near(
        self,
        x: float,
        y: float,
        radius: float = 5.0,
        object_class: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes spatial proximity query: "Find all people (or objects of class X) near location (x, y) within radius meters."
        Usage matching prompt specification:
          conn = sqlite3.connect("persistent_objects.db")
          # Query: "Find all people near location X"
        """
        try:
            sql = """
                SELECT *, 
                       ((pos_x - ?)*(pos_x - ?) + (pos_y - ?)*(pos_y - ?)) as dist_sq 
                FROM persistent_objects 
                WHERE ((pos_x - ?)*(pos_x - ?) + (pos_y - ?)*(pos_y - ?)) <= ? * ?
            """
            params = [x, x, y, y, x, x, y, y, radius, radius]

            if object_class:
                sql += " AND (object_class LIKE ? OR label LIKE ?)"
                clean_cls = f"%{object_class.lower()}%"
                params.extend([clean_cls, clean_cls])

            sql += " ORDER BY dist_sq ASC"

            with self._get_connection() as conn:
                cursor = conn.cursor()
                rows = cursor.execute(sql, tuple(params)).fetchall()
                results = [self._row_to_dict(row) for row in rows]
                logger.info(f"🔍 Spatial Query ({x}, {y}, r={radius}m, class='{object_class}'): Found {len(results)} objects.")
                return results
        except Exception as e:
            logger.error(f"Error executing spatial query: {e}")
            return []

    def query_by_class(self, object_class: str) -> List[Dict[str, Any]]:
        """Find all persistent objects matching a given class (e.g. 'person', 'vehicle', 'forklift')."""
        try:
            sql = "SELECT * FROM persistent_objects WHERE object_class LIKE ? OR label LIKE ?"
            param = f"%{object_class.lower()}%"
            with self._get_connection() as conn:
                cursor = conn.cursor()
                rows = cursor.execute(sql, (param, param)).fetchall()
                return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error querying by class '{object_class}': {e}")
            return []

    def query_by_threat_level(self, threat_level: str) -> List[Dict[str, Any]]:
        """Find all persistent objects with a specific threat level (CRITICAL, HIGH, MEDIUM, LOW)."""
        try:
            sql = "SELECT * FROM persistent_objects WHERE UPPER(threat_level) = ?"
            with self._get_connection() as conn:
                cursor = conn.cursor()
                rows = cursor.execute(sql, (threat_level.upper(),)).fetchall()
                return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error querying by threat level '{threat_level}': {e}")
            return []

    def query_custom(self, sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Execute arbitrary parameter-bound SQL query against persistent_objects table."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                rows = cursor.execute(sql, params).fetchall()
                return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error executing custom query '{sql}': {e}")
            return []

    def get_all_persistent_objects(self) -> List[Dict[str, Any]]:
        """Get all persistent memory objects."""
        return list(self.persistent_objects.values())

    def clear_persistent_objects(self):
        """Clear all stored persistent memory objects from SQLite DB and memory cache."""
        self.persistent_objects.clear()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM persistent_objects;")
                conn.commit()
        except Exception as e:
            logger.error(f"Error clearing SQLite table: {e}")

        if os.path.exists(self.objects_file):
            try:
                with open(self.objects_file, "w") as f:
                    json.dump({}, f)
            except Exception:
                pass
        logger.info("🗑️ Persistent memory store & SQLite DB cleared.")


memory_manager = PersistentMemoryManager()
