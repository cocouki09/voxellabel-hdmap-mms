"""Serve the blind labeling page for one sample folder and save labels to disk.

  python serve_labeler.py <sample_dir> [port]

Serves labeler.html, samples.json and points.bin; POST /save writes
<sample_dir>/labels.json (previous version kept as labels.prev.json).
The KEY_* file is never served, so the pipeline label stays hidden.
"""
from __future__ import annotations

import json
import shutil
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE_DIR = Path(sys.argv[1]).resolve()
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8790
ALLOWED = {"/": HERE / "labeler.html", "/labeler.html": HERE / "labeler.html",
           "/samples.json": SAMPLE_DIR / "samples.json",
           "/points.bin": SAMPLE_DIR / "points.bin",
           "/labels.json": SAMPLE_DIR / "labels.json"}


META = {s["sample_id"]: s for s in
        json.loads((SAMPLE_DIR / "samples.json").read_text(encoding="utf-8"))["samples"]}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/sample/"):
            # one sample's points: float32 xyz (n*12 bytes) + uint8 rgb (n*3 bytes)
            try:
                s = META[int(path.rsplit("/", 1)[1])]
            except (ValueError, KeyError):
                self.send_error(404)
                return
            with open(SAMPLE_DIR / "points.bin", "rb") as f:
                f.seek(s["offset"])
                body = f.read(s["n"] * 15)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/photos/"):
            name = path.rsplit("/", 1)[1]
            f = SAMPLE_DIR / "photos" / name
            if not name.endswith(".jpg") or "/" in name or "\\" in name or not f.exists():
                self.send_error(404)
                return
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        p = ALLOWED.get(path)
        if p is not None and p.name == "labels.json" and not p.exists():
            p.write_text("{}", encoding="utf-8")  # first start: empty label set
        if p is None or not p.exists():
            self.send_error(404)
            return
        body = p.read_bytes()
        ctype = {".html": "text/html; charset=utf-8", ".json": "application/json",
                 ".bin": "application/octet-stream"}[p.suffix]
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = SAMPLE_DIR / "labels.json"
        if out.exists():
            shutil.copyfile(out, SAMPLE_DIR / "labels.prev.json")
        out.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"labeler on http://localhost:{PORT}/  (samples: {SAMPLE_DIR})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
