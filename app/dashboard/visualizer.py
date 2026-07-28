"""
Phase 11: Real-time Debug Dashboard Visualizer.

Provides an interactive HTML/JS dashboard for monitoring smart glasses poses,
fused object positions, threat vectors, and active alerts.
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.services.world_manager import world_manager

router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse)
async def render_dashboard():
    """Renders the Shared Perception real-time debug visualization dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shared Perception Stack - Visualizer Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: #090d16; color: #f8fafc; margin: 0; padding: 24px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
        h1 { color: #818cf8; margin: 0 0 6px 0; font-size: 24px; }
        .subtitle { color: #94a3b8; font-size: 14px; }
        .badge { background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #34d399; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 13px; }
        .grid { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 24px; }
        .card { background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(12px); border-radius: 16px; padding: 20px; }
        .card-header { font-size: 16px; font-weight: 600; color: #e2e8f0; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
        canvas { background: #020617; border-radius: 12px; width: 100%; height: 420px; border: 1px solid rgba(255, 255, 255, 0.08); }
        .threat-alert { background: rgba(239, 68, 68, 0.15); border-left: 4px solid #ef4444; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; color: #fca5a5; font-size: 13px; }
        .threat-alert.critical { background: rgba(220, 38, 38, 0.25); border-left-color: #dc2626; color: #f87171; font-weight: 600; }
        pre { background: #020617; padding: 16px; border-radius: 12px; overflow-x: auto; color: #38bdf8; font-size: 12px; height: 360px; border: 1px solid rgba(255, 255, 255, 0.08); }
        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
        .stat-box { background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05); text-align: center; }
        .stat-val { font-size: 20px; font-weight: 700; color: #818cf8; }
        .stat-lbl { font-size: 11px; color: #64748b; text-transform: uppercase; margin-top: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Shared Perception Stack Dashboard</h1>
            <div class="subtitle">Ray-Ban Meta Smart Glasses Collaborative Spatial Perception Pipeline</div>
        </div>
        <div class="badge">● SERVER ACTIVE</div>
    </div>

    <div class="stats-grid">
        <div class="stat-box"><div class="stat-val" id="statGlasses">0</div><div class="stat-lbl">Active Glasses</div></div>
        <div class="stat-box"><div class="stat-val" id="statObjects">0</div><div class="stat-lbl">Fused 3D Objects</div></div>
        <div class="stat-box"><div class="stat-box"><div class="stat-val" id="statThreats" style="color:#ef4444;">0</div><div class="stat-lbl">Active Threat Alerts</div></div></div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-header">
                <span>Live Spatial Radar Map</span>
                <span style="font-size: 12px; color: #64748b;">Scale: 1m = 20px</span>
            </div>
            <canvas id="radarCanvas"></canvas>
            <div id="threatBanners" style="margin-top: 16px;"></div>
        </div>
        <div class="card">
            <div class="card-header">Synchronized World Model State</div>
            <pre id="jsonState">Loading synchronized world model...</pre>
        </div>
    </div>

    <script>
        async function fetchWorldState() {
            try {
                const res = await fetch('/world');
                const data = await res.json();
                
                document.getElementById('jsonState').innerText = JSON.stringify(data, null, 2);
                document.getElementById('statGlasses').innerText = data.active_glasses_count || 0;
                document.getElementById('statObjects').innerText = Object.keys(data.world_objects || {}).length;
                document.getElementById('statThreats').innerText = (data.active_threats || []).length;
                
                renderThreatBanners(data.active_threats || []);
                drawRadar(data);
            } catch (e) {
                console.error('Error fetching world state:', e);
            }
        }

        function renderThreatBanners(threats) {
            const container = document.getElementById('threatBanners');
            if (threats.length === 0) {
                container.innerHTML = '<div style="font-size: 13px; color: #64748b; text-align: center; padding: 8px;">No active spatial threats detected.</div>';
                return;
            }
            container.innerHTML = threats.map(t => `
                <div class="threat-alert ${t.threat_level.toLowerCase()}">
                    🚨 <strong>${t.threat_level} THREAT</strong> [Target: ${t.target_glass_id}] — ${t.warning_message} (TTC: ${t.time_to_collision}s)
                </div>
            `).join('');
        }

        function drawRadar(data) {
            const canvas = document.getElementById('radarCanvas');
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;
            const scale = 20; // 1 meter = 20 pixels

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Concentric Radar Rings
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
            ctx.lineWidth = 1;
            for (let r = 40; r < 400; r += 40) {
                ctx.beginPath();
                ctx.arc(cx, cy, r, 0, 2 * Math.PI);
                ctx.stroke();
            }

            // Crosshair Axes
            ctx.beginPath();
            ctx.moveTo(cx, 0); ctx.lineTo(cx, canvas.height);
            ctx.moveTo(0, cy); ctx.lineTo(canvas.width, cy);
            ctx.stroke();

            // Render Fused World Objects
            if (data.world_objects) {
                Object.values(data.world_objects).forEach(obj => {
                    const x = cx + obj.position_x * scale;
                    const y = cy - obj.position_y * scale;

                    // Draw Object Dot & Velocity Vector
                    const isVehicle = obj.label.toLowerCase().includes('vehicle') || obj.label.toLowerCase().includes('car') || obj.label.toLowerCase().includes('forklift');
                    ctx.fillStyle = isVehicle ? '#ef4444' : '#f59e0b';
                    ctx.beginPath();
                    ctx.arc(x, y, 6, 0, 2 * Math.PI);
                    ctx.fill();

                    // Velocity Vector Arrow
                    if (obj.velocity_x !== 0 || obj.velocity_y !== 0) {
                        ctx.strokeStyle = '#38bdf8';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(x, y);
                        ctx.lineTo(x + obj.velocity_x * scale * 0.5, y - obj.velocity_y * scale * 0.5);
                        ctx.stroke();
                    }

                    // Label Tag
                    ctx.fillStyle = '#cbd5e1';
                    ctx.font = '11px Inter';
                    ctx.fillText(`${obj.label} (${obj.position_x.toFixed(1)}m, ${obj.position_y.toFixed(1)}m)`, x + 10, y + 4);
                });
            }

            // Render Glasses Poses & Heading Cones
            if (data.glasses) {
                Object.values(data.glasses).forEach(g => {
                    const x = cx + g.pose.x * scale;
                    const y = cy - g.pose.y * scale;
                    const headingRad = (g.pose.heading - 90) * (Math.PI / 180);

                    // Heading Cone (FOV)
                    ctx.fillStyle = 'rgba(99, 102, 241, 0.2)';
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.arc(x, y, 50, headingRad - Math.PI / 6, headingRad + Math.PI / 6);
                    ctx.closePath();
                    ctx.fill();

                    // Glass Circle Marker
                    ctx.fillStyle = '#6366f1';
                    ctx.beginPath();
                    ctx.arc(x, y, 8, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 2;
                    ctx.stroke();

                    // Glass Label
                    ctx.fillStyle = '#ffffff';
                    ctx.font = '600 12px Inter';
                    ctx.fillText(`👓 ${g.glass_id}`, x + 12, y - 10);
                });
            }
        }

        setInterval(fetchWorldState, 1000);
        fetchWorldState();
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

