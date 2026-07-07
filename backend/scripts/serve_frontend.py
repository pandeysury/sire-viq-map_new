#!/usr/bin/env python3
"""
Simple HTTP server to serve the frontend
"""
import http.server
import webbrowser
from pathlib import Path

PORT = 3000
DIRECTORY = str(Path(__file__).parent.parent.parent / "frontend")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    with http.server.ThreadingHTTPServer(("", PORT), Handler) as httpd:
        print(f"🌐 Frontend server running at http://localhost:{PORT}")
        print(f"📁 Serving files from: {DIRECTORY}")
        print("🚀 Opening browser...")
        
        # Open browser
        webbrowser.open(f"http://localhost:{PORT}")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    main()