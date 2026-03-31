#!/usr/bin/env python3
"""HTTP server with API endpoints for the compare page."""

import http.server
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

PORT = 8787
BASE = Path(__file__).parent


class PipelineHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/progress":
            self.send_json_file(BASE / "progress.json")
        elif self.path == "/api/run_info":
            self.send_json_file(BASE / "run_info.json")
        elif self.path == "/api/files":
            # List image files in webp/
            import json as jmod
            webp = BASE / "webp"
            exts = {".webp", ".jpg", ".jpeg", ".png"}
            files = [
                {"name": f.name, "stem": f.stem}
                for f in sorted(webp.iterdir(), key=lambda x: x.name)
                if f.suffix.lower() in exts
            ]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(jmod.dumps(files).encode())
        elif self.path == "/api/ratings":
            self.send_json_file(BASE / "ratings.json")
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/rerun":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode()
            # Save ratings
            ratings_path = BASE / "ratings.json"
            ratings_path.write_text(body)
            # Clear progress
            (BASE / "progress.json").write_text('{"done":false,"pass":0,"pass_name":"Starting...","current":0,"total":0,"filename":""}')
            # Run pipeline in background
            subprocess.Popen(
                [sys.executable, str(BASE / "rainbow_convert.py")],
                cwd=str(BASE),
                stdout=open(BASE / "pipeline.log", "w"),
                stderr=subprocess.STDOUT,
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"status":"started"}')

        elif self.path == "/api/stop":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"stopping"}')
            # Kill any running pipeline
            os.system("pkill -f rainbow_convert.py 2>/dev/null")
            # Stop server after response
            def shutdown():
                self.server.shutdown()
            import threading
            threading.Thread(target=shutdown).start()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def send_json_file(self, path):
        try:
            data = path.read_text()
        except FileNotFoundError:
            data = '{"done":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data.encode())

    def log_message(self, format, *args):
        # Suppress request logs
        pass


def main():
    os.chdir(BASE)
    server = http.server.HTTPServer(("", PORT), PipelineHandler)
    print(f"Server running at http://localhost:{PORT}/")
    print(f"Compare page: http://localhost:{PORT}/compare.html")
    print("Press Ctrl+C to stop.")

    # Write PID
    (BASE / ".server.pid").write_text(str(os.getpid()))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        (BASE / ".server.pid").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
