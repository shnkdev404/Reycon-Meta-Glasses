import http.server
import ssl
import os

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="clients", **kwargs)

server_address = ('0.0.0.0', 8443)
httpd = http.server.HTTPServer(server_address, Handler)

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print("HTTPS Web Client Server running at https://0.0.0.0:8443/mobile_client.html")
httpd.serve_forever()
