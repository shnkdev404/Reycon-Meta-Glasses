import sys
import os
import socket
import argparse
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

    parser = argparse.ArgumentParser(description="REYCON Server Runner")
    parser.add_argument("--ssl", action="store_true", help="Enable HTTPS mode with SSL certificates")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on (default: 8000)")
    args, unknown = parser.parse_known_args()
    
    local_ip = get_local_ip()
    port = args.port
    
    ssl_kwargs = {}
    protocol = "http"
    
    if args.ssl and os.path.exists("cert.pem") and os.path.exists("key.pem"):
        ssl_kwargs["ssl_keyfile"] = "key.pem"
        ssl_kwargs["ssl_certfile"] = "cert.pem"
        protocol = "https"

    print("\n=======================================================")
    print(f"🚀 REYCON Shared Perception Server is LIVE! ({protocol.upper()} Mode)")
    print(f"📍 Local access:      {protocol}://localhost:{port}")
    print(f"📱 Mobile HUD:        {protocol}://{local_ip}:{port}/mobile")
    print(f"🖥️  Command Center:    {protocol}://{local_ip}:{port}/server")
    print(f"🏥 Health Check:      {protocol}://{local_ip}:{port}/health")
    if protocol == "https":
        print("\n⚠️  NOTE FOR HTTPS/SSL:")
        print("   If your browser shows a security warning for self-signed certificates,")
        print("   click 'Advanced' -> 'Proceed to localhost (unsafe)'.")
    else:
        print("\n💡 Running in HTTP mode (Recommended for browser access without cert warnings).")
        print("   To enable SSL/HTTPS, run: python run.py --ssl")
    print("=======================================================\n")

    try:
        uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True, **ssl_kwargs)
    except Exception as e:
        print(f"\n❌ Error starting server on port {port}: {e}")
        print("   If port is in use, try running with another port: python run.py --port 8001\n")


