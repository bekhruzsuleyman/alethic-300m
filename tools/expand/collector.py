"""Local receiver for DeepSeek replies: POST /<name> with the reply text in the body is saved to incoming/<name>.txt.
Listens on 127.0.0.1 only. Run: python collector.py [folder]"""
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "incoming"
os.makedirs(FOLDER, exist_ok=True)


class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", self.path.strip("/"))[:80] or "unnamed"
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with open(os.path.join(FOLDER, name + ".txt"), "wb") as f:
            f.write(body)
        self.send_response(200)
        self._cors()
        self.end_headers()
        self.wfile.write(b"saved %d" % len(body))

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", 8765), H).serve_forever()
