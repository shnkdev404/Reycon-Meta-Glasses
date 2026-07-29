"""
Integration Test Suite: FastAPI REST Endpoints for Glasses Telemetry, Persistent Memory & Object Correction.
"""
import sys
import os
import unittest
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api.shared_routes import get_glasses, get_persistent_memory, correct_object_label
from app.services.shared_world_manager import world_manager as shared_wm, Position3D
from app.services.memory_manager import memory_manager


class TestAPIPersistentMemory(unittest.TestCase):

    def setUp(self):
        shared_wm.reset()
        memory_manager.persistent_objects.clear()

    def test_glasses_location_telemetry_endpoint(self):
        """Test GET /api/shared/glasses endpoint returns exact 3D position formatting."""
        shared_wm.register_glass("phone_beta", Position3D(2.5, -1.8, 0.0), heading=120.0)

        data = asyncio.run(get_glasses())
        self.assertEqual(data["count"], 1)
        phone = data["glasses"][0]
        self.assertEqual(phone["id"], "phone_beta")
        self.assertIn("formatted_position", phone)
        self.assertIn("X: +2.50m", phone["formatted_position"])

    def test_persistent_memory_and_correction_api(self):
        """Test GET /api/shared/persistent_memory and POST /api/shared/correct_object."""
        # 1. Register threat/object
        shared_wm.add_threat(
            threat_id="obj_truck_505",
            object_type="truck",
            position=Position3D(10.0, 5.0, 0.0),
            detected_by_glass_id="phone_beta",
            confidence=0.88
        )

        # 2. Fetch persistent memory via API
        data = asyncio.run(get_persistent_memory())
        self.assertEqual(data["status"], "success")
        self.assertTrue(len(data["objects"]) > 0)

        # 3. Post correction: truck -> forklift
        corr_data = asyncio.run(correct_object_label({
            "object_id": "obj_truck_505",
            "new_label": "forklift"
        }))
        self.assertEqual(corr_data["status"], "success")
        self.assertEqual(corr_data["object"]["label"], "forklift")
        self.assertTrue(corr_data["object"]["is_corrected"])


if __name__ == "__main__":
    unittest.main()
