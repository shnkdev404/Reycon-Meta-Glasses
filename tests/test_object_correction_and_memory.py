"""
Unit & Integration Test Suite: Phone 3D Locations, Persistent Memory Store & Object Correction.
"""
import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.memory_manager import memory_manager
from app.services.shared_world_manager import world_manager as shared_wm, Position3D


class TestObjectCorrectionAndMemory(unittest.TestCase):

    def setUp(self):
        shared_wm.reset()
        memory_manager.persistent_objects.clear()

    def test_phone_registration_and_exact_position(self):
        """Test registering a phone device and retrieving exact formatted 3D positions."""
        shared_wm.register_glass(
            glass_id="phone_alpha",
            position=Position3D(1.5, 2.3, 0.0),
            heading=45.0
        )

        glass_pos = shared_wm.get_glass_position("phone_alpha")
        self.assertIsNotNone(glass_pos)
        self.assertEqual(glass_pos.x, 1.5)
        self.assertEqual(glass_pos.y, 2.3)
        self.assertEqual(glass_pos.z, 0.0)

    def test_persistent_memory_store_and_object_correction(self):
        """Test saving object detections into persistent memory and correcting misclassified labels."""
        # 1. Add threat/object detection from phone_alpha
        threat = shared_wm.add_threat(
            threat_id="obj_bottle_101",
            object_type="bottle",
            position=Position3D(3.0, 4.0, 0.5),
            detected_by_glass_id="phone_alpha",
            confidence=0.92
        )

        self.assertEqual(threat.object_type, "bottle")

        # Verify object persisted in memory_manager
        memory_objs = memory_manager.get_all_persistent_objects()
        self.assertTrue(any(o["object_id"] == "obj_bottle_101" for o in memory_objs))

        # 2. Correct the misinterpreted object label (e.g. bottle -> cup)
        corrected = shared_wm.correct_object_label("obj_bottle_101", "cup")
        self.assertIsNotNone(corrected)
        self.assertEqual(corrected["label"], "cup")
        self.assertTrue(corrected["is_corrected"])

        # Verify active threat in shared_wm updated to "cup"
        all_threats = shared_wm.get_all_threats()
        target_threat = next((t for t in all_threats if t["threat_id"] == "obj_bottle_101"), None)
        self.assertIsNotNone(target_threat)
        self.assertEqual(target_threat["type"], "cup")


if __name__ == "__main__":
    unittest.main()
