"""
Phase 7 Unit & Integration Test Suite: Persistent Map Memory & Interactive Debug Dashboard.
Verifies saving/reloading SLAM map memory, object history retention, and dashboard visualizer rendering.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from app.services.memory_manager import memory_manager
from app.dashboard.visualizer import render_dashboard


def test_phase_7_memory_and_dashboard():
    print("\n==========================================================================")
    print("🚀 EXECUTING PHASE 7: PERSISTENT MAP MEMORY & DASHBOARD VISUALIZER TESTS")
    print("==========================================================================")

    # Step 1: Save World Model Memory Session to Disk
    memory_id = "test_phase7_memory_session"
    sample_world_state = {
        "memory_id": memory_id,
        "glasses": {
            "glass_01": {"x": 10.0, "y": 20.0, "heading": 90.0}
        },
        "world_objects": {
            "obj_truck_1": {
                "object_id": "obj_truck_1",
                "label": "truck #1",
                "position_x": 15.0,
                "position_y": 22.0,
                "velocity_x": -2.0,
                "velocity_y": 0.0,
                "history": [
                    {"time": 1785260000.0, "x": 17.0, "y": 22.0},
                    {"time": 1785260001.0, "x": 15.0, "y": 22.0}
                ]
            }
        },
        "saved_at": 1785260001.0
    }

    saved = memory_manager.save_world_memory(memory_id, sample_world_state)
    assert saved is True
    print(f"💾 Persistent Map Memory Saved: Session '{memory_id}' written to disk.")

    # Step 2: Reload Persistent Map Memory Session
    loaded_data = memory_manager.load_world_memory(memory_id)
    assert loaded_data is not None
    assert loaded_data["memory_id"] == memory_id
    assert "obj_truck_1" in loaded_data["world_objects"]
    assert len(loaded_data["world_objects"]["obj_truck_1"]["history"]) == 2
    print(f"📂 Persistent Map Memory Reloaded: Restored {len(loaded_data['world_objects'])} objects with historical trajectory paths.")

    # Step 3: Test Saved Memories Listing
    mem_list = memory_manager.list_saved_memories()
    assert memory_id in mem_list["memories"]
    print(f"📋 Saved Memory Session List verified ({mem_list['count']} sessions found).")

    # Step 4: Verify Interactive Debug Dashboard Visualizer Rendering
    dashboard_response = asyncio.run(render_dashboard())
    assert dashboard_response.status_code == 200
    html_str = dashboard_response.body.decode("utf-8")
    assert "radarCanvas" in html_str
    assert "Server Command Center" in html_str
    print("💻 Interactive Debug Dashboard Visualizer rendering verified (Canvas & HTML UI loaded).")

    # Clean up test file
    test_file_path = os.path.join(memory_manager.storage_dir, f"{memory_id}.json")
    if os.path.exists(test_file_path):
        os.remove(test_file_path)

    print("\n==========================================================================")
    print("🎉 ALL PHASE 7 PERSISTENT MAP MEMORY & DASHBOARD TESTS PASSED CLEANLY!")
    print("==========================================================================\n")


if __name__ == "__main__":
    test_phase_7_memory_and_dashboard()
