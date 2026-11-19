"""Minimal Prometheus exposition (no extra dependency).

Counters live in memory (reset on restart, as usual for Prometheus counters); per-video
states are read from the cache DB at scrape time, so they survive restarts.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_lock = threading.Lock()
_counters = {}   # (name, labels-tuple) -> float
_gauges = {}

HELP = {
    "jimaku_llm_requests_total": ("counter", "LLM requests by model and outcome"),
    "jimaku_llm_request_seconds_total": ("counter", "Time spent waiting for the LLM"),
    "jimaku_llm_tokens_total": ("counter", "Tokens reported by the provider, by model and kind (input/output)"),
    "jimaku_cache_hits_total": ("counter", "Translated batches served from the cache"),
    "jimaku_cues_translated_total": ("counter", "Subtitle cues translated, by target language"),
    "jimaku_ocr_cues_total": ("counter", "Subtitle cues recognised by OCR, by language and engine"),
    "jimaku_files_written_total": ("counter", "Subtitle files written, by kind (srt language / ass)"),
    "jimaku_scan_last_success_timestamp_seconds": ("gauge", "End of the last completed scan"),
    "jimaku_scan_duration_seconds": ("gauge", "Duration of the last scan"),
    "jimaku_videos": ("gauge", "Videos by last outcome (state, reason)"),
}


def inc(name, value=1.0, **labels):
    key = (name, tuple(sorted(labels.items())))
    with _lock:
        _counters[key] = _counters.get(key, 0.0) + value


def set_gauge(name, value, **labels):
    with _lock:
        _gauges[(name, tuple(sorted(labels.items())))] = float(value)


def _fmt(name, labels, value):
    lab = ",".join(f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
                   for k, v in labels)
    return f"{name}{{{lab}}} {value:g}" if lab else f"{name} {value:g}"


def render(cache=None):
    with _lock:
        items = list(_counters.items()) + list(_gauges.items())
    if cache is not None:
        try:
            for state, reason, n in cache.db.execute(
                    "SELECT state, reason, COUNT(*) FROM status GROUP BY state, reason"):
                items.append((("jimaku_videos", (("reason", reason), ("state", state))), float(n)))
        except Exception:
            pass
    lines, seen = [], set()
    for (name, labels), value in sorted(items, key=lambda x: (x[0][0], x[0][1])):
        if name not in seen:
            seen.add(name)
            kind, doc = HELP.get(name, ("untyped", name))
            lines += [f"# HELP {name} {doc}", f"# TYPE {name} {kind}"]
        lines.append(_fmt(name, labels, value))
    return "\n".join(lines) + "\n"


def serve(port, cache_path):
    """Start /metrics on a daemon thread. Uses its own SQLite connection (thread safety)."""
    import sqlite3

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] not in ("/metrics", "/"):
                self.send_error(404)
                return

            class _C:  # tiny adapter: render() only needs .db
                db = sqlite3.connect(cache_path)
            try:
                body = render(_C).encode()
            finally:
                _C.db.close()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep container logs clean
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True, name="metrics").start()
    return srv


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
