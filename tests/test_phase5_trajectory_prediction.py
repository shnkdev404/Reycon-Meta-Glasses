"""
Phase 5 Unit & Integration Test Suite: Trajectory Prediction, Velocity Estimation, & Collision Probability.
Verifies future trajectory extrapolation, velocity vectors, Time-To-Collision (TTC), and collision probability calculation.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from app.models.glass import GlassPose, GlassState
from app.models.object import WorldObject
from app.services.prediction_engine import ThreatPredictionEngine


def test_phase_5_trajectory_prediction():
    print("\n==========================================================================")
    print("🚀 EXECUTING PHASE 5: TRAJECTORY PREDICTION & COLLISION PROBABILITY TESTS")
    print("==========================================================================")

    pengine = ThreatPredictionEngine()

    # Create moving vehicle object heading toward (0, 0)
    moving_truck = WorldObject(
        object_id="obj_truck_test",
        label="truck #1",
        confidence=0.95,
        position_x=10.0,  # 10m away along +X
        position_y=0.0,
        position_z=0.0,
        velocity_x=-2.5,  # Moving left toward origin at 2.5 m/s
        velocity_y=0.0,
        velocity_z=0.0,
        last_seen=datetime.utcnow()
    )

    # Stationary target worker at origin (0, 0)
    target_worker = GlassState(
        glass_id="worker_target",
        pose_obj=GlassPose(x=0.0, y=0.0, z=0.0, heading=0.0)
    )

    # Step 1: Predict 3D Trajectory Points into the Future
    traj_points = pengine.predict_trajectory(moving_truck, time_horizon_sec=5.0, step_sec=1.0)
    assert len(traj_points) == 6
    assert traj_points[0]["x"] == 10.0  # t=0s -> x=10m
    assert traj_points[4]["x"] == 0.0   # t=4s -> x=0m (Collision intersection!)
    print(f"✅ Trajectory Extrapolation verified: Predicted {len(traj_points)} trajectory points (x=10m at t=0s -> x=0m at t=4s).")

    # Step 2: Estimate Collision Probability & Time-To-Collision (TTC)
    prob, ttc = pengine.estimate_collision_probability(moving_truck, target_worker)
    assert prob >= 0.90  # High collision probability near 1.0!
    assert abs(ttc - 4.0) < 0.5  # TTC ~ 4.0 seconds
    print(f"✅ Collision Probability & TTC verified: Collision Probability={prob*100:.0f}%, Time-To-Collision={ttc:.1f}s.")

    print("\n==========================================================================")
    print("🎉 ALL PHASE 5 TRAJECTORY PREDICTION & COLLISION PROBABILITY TESTS PASSED CLEANLY!")
    print("==========================================================================\n")


if __name__ == "__main__":
    test_phase_5_trajectory_prediction()
