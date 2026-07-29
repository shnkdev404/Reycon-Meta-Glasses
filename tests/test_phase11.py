"""
Automated unit and integration tests for Phase 11: Real-time Debug Dashboard Visualizer & REST APIs.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from app.api.routes import home, health, get_world_state, get_glasses, get_threats
from app.dashboard.visualizer import render_dashboard


def test_root_endpoint():
    print("--- 1. Testing Root REST Endpoint ('/') ---")
    data = home()
    assert data["status"] == "active"
    print("✅ Root REST Endpoint passed!")


def test_health_endpoint():
    print("\n--- 2. Testing Health REST Endpoint ('/health') ---")
    data = health()
    assert data["status"] == "OK"
    print("✅ Health REST Endpoint passed!")


def test_world_state_endpoint():
    print("\n--- 3. Testing Synchronized World Model REST Endpoint ('/world') ---")
    data = asyncio.run(get_world_state())
    assert "active_glasses_count" in data
    assert "glasses" in data
    assert "world_objects" in data
    assert "active_threats" in data
    print(f"✅ Synchronized World Model Endpoint passed! Active Glasses Count: {data['active_glasses_count']}")


def test_dashboard_visualizer_endpoint():
    print("\n--- 4. Testing Dashboard Visualizer HTML Endpoint ('/dashboard') ---")
    response = asyncio.run(render_dashboard())
    html_text = response.body.decode("utf-8")
    assert "<canvas id=\"radarCanvas\"></canvas>" in html_text
    assert "Server Command Center" in html_text
    print("✅ Dashboard Visualizer HTML Endpoint passed!")


if __name__ == "__main__":
    test_root_endpoint()
    test_health_endpoint()
    test_world_state_endpoint()
    test_dashboard_visualizer_endpoint()
    print("\n🎉 ALL PHASE 11 TESTS PASSED SUCCESSFULLY!")
