import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import time
import numpy as np
from app.services.shared_world_manager import SharedWorldManager, Position3D
from app.models import GPSLocation


class TestDevicePruningAndPositioning(unittest.TestCase):

    def setUp(self):
        self.wm = SharedWorldManager()
        self.wm.reset(clear_persistent_memory=True)

    def test_stale_device_pruning(self):
        """Verify inactive glass devices are automatically pruned."""
        pos = Position3D(1.0, 2.0, 0.0)
        self.wm.register_glass("priyam", pos, heading=45.0)
        self.wm.register_glass("susu", pos, heading=90.0)
        self.wm.register_glass("phone_01", pos, heading=0.0)

        # Set timestamp to 20 seconds ago for priyam and susu
        self.wm.glasses["priyam"]["timestamp"] = time.time() - 20.0
        self.wm.glasses["susu"]["timestamp"] = time.time() - 20.0

        # Prune stale devices (> 15s)
        self.wm.prune_stale_glasses(max_age_seconds=15.0)

        # Only phone_01 should remain
        self.assertNotIn("priyam", self.wm.glasses)
        self.assertNotIn("susu", self.wm.glasses)
        self.assertIn("phone_01", self.wm.glasses)

    def test_gps_position_resolution(self):
        """Verify GPS info converts to metric Cartesian offsets when position is (0,0)."""
        ref_gps = GPSLocation(latitude=28.6139, longitude=77.2090)
        offset_gps = GPSLocation(latitude=28.6140, longitude=77.2091)

        pos_origin = Position3D(0.0, 0.0, 0.0)

        # First device sets reference GPS
        self.wm.update_glass_pose("phone_01", np.eye(4), pos_origin, heading=0.0, gps_info=ref_gps)
        pos1 = self.wm.get_glass_position("phone_01")
        self.assertEqual(pos1.x, 0.0)
        self.assertEqual(pos1.y, 0.0)

        # Second device with offset GPS converts to relative Cartesian meters
        self.wm.update_glass_pose("priyam", np.eye(4), pos_origin, heading=0.0, gps_info=offset_gps)
        pos2 = self.wm.get_glass_position("priyam")
        self.assertNotEqual(pos2.x, 0.0)
        self.assertNotEqual(pos2.y, 0.0)

    def test_reset_clears_devices(self):
        """Verify reset clears all active devices cleanly."""
        pos = Position3D(2.0, 3.0, 0.0)
        self.wm.register_glass("susu", pos, heading=15.0)
        self.assertEqual(len(self.wm.glasses), 1)

        self.wm.reset()
        self.assertEqual(len(self.wm.glasses), 0)


if __name__ == "__main__":
    unittest.main()
