"""
Unit tests for Meta Glasses RGB-D Depth Integration:
1. DepthSensor.get_depth_map() 2D metric array buffer.
2. estimate_distance_with_depth(bbox, depth_map) direct depth distance sampling.
3. MetaDepthAdapter hardware wrapper.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.sensors import DepthSensor, estimate_distance_with_depth
from app.meta import MetaDepthAdapter


def test_depth_sensor_and_get_depth_map():
    print("--- 1. Testing DepthSensor.get_depth_map() ---")
    sensor = DepthSensor(glass_id="rayban_meta_01")
    depth_map = sensor.get_depth_map()

    assert depth_map is not None, "get_depth_map() returned None!"
    assert hasattr(depth_map, "shape"), "depth_map must be a NumPy array!"
    assert len(depth_map.shape) == 2, f"Expected 2D array, got shape {depth_map.shape}"
    assert depth_map.dtype == np.float32 or depth_map.dtype == np.float64
    print(f"✅ DepthSensor.get_depth_map() passed! Shape: {depth_map.shape}, dtype: {depth_map.dtype}")


def test_estimate_distance_with_depth():
    print("\n--- 2. Testing estimate_distance_with_depth(bbox, depth_map) ---")
    # Synthetic depth map (384x640) with 2.85m metric depth in region (50, 50) to (150, 150)
    depth_map = np.zeros((384, 640), dtype=np.float32)
    depth_map[50:150, 50:150] = 2.85

    bbox = [50, 50, 150, 150]
    med_dist = estimate_distance_with_depth(bbox, depth_map)

    assert med_dist is not None, "estimate_distance_with_depth returned None!"
    assert abs(med_dist - 2.85) < 0.01, f"Expected 2.85m depth, got {med_dist}m"

    # Test invalid / zero depth filtering
    depth_map_with_zeros = np.zeros((384, 640), dtype=np.float32)
    depth_map_with_zeros[50:150, 50:150] = 0.0  # Invalid pixels
    depth_map_with_zeros[80:120, 80:120] = 4.10  # Valid subset

    med_dist_filtered = estimate_distance_with_depth(bbox, depth_map_with_zeros)
    assert med_dist_filtered == 4.10, f"Expected 4.10m after zero-filtering, got {med_dist_filtered}m"
    print(f"✅ estimate_distance_with_depth() passed! Distance: {med_dist}m (Zero-filtered: {med_dist_filtered}m)")


def test_meta_depth_adapter():
    print("\n--- 3. Testing MetaDepthAdapter ---")
    adapter = MetaDepthAdapter(glass_id="meta_glass_alpha")
    depth_map = adapter.get_depth_map()

    assert depth_map is not None, "MetaDepthAdapter.get_depth_map() returned None!"
    assert depth_map.shape == (384, 640)
    pixel_dist = adapter.get_distance_at_pixel(100, 100)
    assert pixel_dist > 0.0
    print(f"✅ MetaDepthAdapter passed! Sampled pixel depth at (100,100): {pixel_dist}m")


if __name__ == "__main__":
    test_depth_sensor_and_get_depth_map()
    test_estimate_distance_with_depth()
    test_meta_depth_adapter()
    print("\n🎉 ALL DEPTH INTEGRATION TESTS PASSED SUCCESSFULLY!")
