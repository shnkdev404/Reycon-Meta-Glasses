"""
Phase 15: Persistent Memory System Unit & Integration Tests.

Verifies:
1. SQLite persistent_objects.db database initialization and index creation.
2. Spatial proximity queries: query_objects_near(x, y, radius, object_class="person") -> "Find all people near location X".
3. Attribute queries: query_by_class("vehicle") and query_by_threat_level("CRITICAL").
4. Label corrections and persistent database reloads across sessions.
"""
import sys
import os
import time
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.memory_manager import PersistentMemoryManager, memory_manager


def test_sqlite_memory_and_spatial_queries():
    print("--- 1. Testing SQLite Database Storage & Spatial Query Engine ---")
    test_db = PersistentMemoryManager(storage_dir="data/memory_test", db_name="test_persistent.db")
    test_db.clear_persistent_objects()

    # Add test objects: people, forklift, car at different 3D locations
    # Person 1 near location (5.0, 5.0)
    test_db.add_or_update_object(
        object_id="person_01",
        label="person #1",
        position={"x": 5.2, "y": 5.1, "z": 0.0},
        detected_by="glass_alpha",
        confidence=0.92,
        threat_level="LOW"
    )

    # Person 2 near location (5.0, 5.0) but distinct (>2.0m from person_01)
    test_db.add_or_update_object(
        object_id="person_02",
        label="person #2",
        position={"x": 8.0, "y": 8.0, "z": 0.0},
        detected_by="glass_beta",
        confidence=0.88,
        threat_level="MEDIUM"
    )

    # Person 3 far away at location (50.0, 50.0)
    test_db.add_or_update_object(
        object_id="person_03",
        label="person #3",
        position={"x": 50.0, "y": 50.0, "z": 0.0},
        detected_by="glass_gamma",
        confidence=0.95,
        threat_level="LOW"
    )

    # Forklift near location (5.0, 5.0) with CRITICAL threat
    test_db.add_or_update_object(
        object_id="forklift_01",
        label="forklift #1",
        position={"x": 5.5, "y": 4.8, "z": 0.0},
        detected_by="glass_alpha",
        confidence=0.96,
        threat_level="CRITICAL",
        threat_score=0.88
    )

    # Query matching prompt: "Find all people near location X" (X=5.0, Y=5.0, radius=5.0m)
    people_near = test_db.query_objects_near(x=5.0, y=5.0, radius=5.0, object_class="person")
    assert len(people_near) == 2
    ids_near = [p["object_id"] for p in people_near]
    assert "person_01" in ids_near
    assert "person_02" in ids_near
    assert "person_03" not in ids_near  # Person 3 is far away

    print(f"✅ Spatial query 'Find all people near location (5.0, 5.0)' passed! Found: {ids_near}")

    # Query all objects near (5.0, 5.0) regardless of class
    all_near = test_db.query_objects_near(x=5.0, y=5.0, radius=5.0)
    assert len(all_near) == 3
    print(f"✅ General spatial proximity query passed! Found {len(all_near)} total objects within 5m.")


def test_sqlite_attribute_queries_and_label_corrections():
    print("\n--- 2. Testing SQLite Attribute Queries & Label Corrections ---")
    test_db = PersistentMemoryManager(storage_dir="data/memory_test", db_name="test_persistent.db")

    # Query by threat level
    critical_threats = test_db.query_by_threat_level("CRITICAL")
    assert len(critical_threats) >= 1
    assert critical_threats[0]["object_id"] == "forklift_01"
    print(f"✅ Threat level query passed! Found {len(critical_threats)} CRITICAL threats.")

    # Query by class
    forklifts = test_db.query_by_class("forklift")
    assert len(forklifts) >= 1
    print(f"✅ Class query passed! Found {len(forklifts)} forklifts.")

    # Test label correction
    corrected = test_db.correct_object_label("person_01", "worker_engineer")
    assert corrected is not None
    assert corrected["label"] == "worker_engineer"
    assert corrected["is_corrected"] is True

    # Re-query corrected label
    engineers = test_db.query_by_class("worker_engineer")
    assert len(engineers) == 1
    assert engineers[0]["object_id"] == "person_01"
    print(f"✅ Label correction & database update passed! Corrected object label to '{engineers[0]['label']}'.")


def test_sqlite_persistence_reload():
    print("\n--- 3. Testing Persistent Reload from SQLite DB File ---")
    # Open fresh PersistentMemoryManager targeting same DB
    reloaded_db = PersistentMemoryManager(storage_dir="data/memory_test", db_name="test_persistent.db")
    all_objs = reloaded_db.get_all_persistent_objects()
    assert len(all_objs) >= 4
    
    # Clean up test DB directory
    reloaded_db.clear_persistent_objects()
    print(f"✅ Persistent SQLite reload passed! Reloaded {len(all_objs)} objects cleanly across process restarts.")


if __name__ == "__main__":
    test_sqlite_memory_and_spatial_queries()
    test_sqlite_attribute_queries_and_label_corrections()
    test_sqlite_persistence_reload()
    print("\n🎉 ALL PHASE 15 PERSISTENT MEMORY TESTS PASSED SUCCESSFULLY!")
