import sys
import os
import socket
import uvicorn

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    
    local_ip = get_local_ip()
    
    ssl_kwargs = {}
    protocol = "http"
    if os.path.exists("cert.pem") and os.path.exists("key.pem"):
        ssl_kwargs["ssl_keyfile"] = "key.pem"
        ssl_kwargs["ssl_certfile"] = "cert.pem"
        protocol = "https"

    print("\n=======================================================")
    print(f"🚀 REYCON Shared Perception Server is LIVE!")
    print(f"📍 Local access:   {protocol}://localhost:8000")
    print(f"📱 Mobile HUD:     {protocol}://{local_ip}:8000/mobile")
    print(f"🖥️  Command Center: {protocol}://{local_ip}:8000/server")
    print("=======================================================\n")

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, **ssl_kwargs)

