"""
Orchestrator API Server

Lightweight Express-like HTTP server that the Next.js frontend calls
to start/stop/query bot engine workers.

Runs alongside the Next.js app on a separate port (default 5000).

Endpoints:
  POST /api/bots/start   {botId}       → Start engine worker
  POST /api/bots/stop    {botId}       → Stop engine worker
  GET  /api/bots/status                → All worker statuses
  GET  /api/bots/status/:botId         → Single worker status
  GET  /api/engine/state/:botId        → Bot state (HMM, coinStates)
  GET  /api/engine/trades/:botId       → Active trades for a bot
"""
import os
import sys
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(__file__))

from bot_orchestrator import get_orchestrator
from db_adapter import DBAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("OrchestratorAPI")

PORT = int(os.getenv("ORCHESTRATOR_PORT", "5000"))
db = DBAdapter()


class OrchestratorHandler(BaseHTTPRequestHandler):

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            return json.loads(self.rfile.read(length))
        return {}

    def do_OPTIONS(self):
        self._send_json({})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        orch = get_orchestrator()

        if path == "/api/bots/status":
            self._send_json(orch.status())

        elif path.startswith("/api/bots/status/"):
            bot_id = path.split("/")[-1]
            running = orch.is_running(bot_id)
            self._send_json({"botId": bot_id, "running": running})

        elif path.startswith("/api/engine/state/"):
            bot_id = path.split("/")[-1]
            state = db.get_bot_state(bot_id)
            if state:
                # Convert datetime objects to ISO strings for JSON
                for k, v in state.items():
                    if hasattr(v, 'isoformat'):
                        state[k] = v.isoformat()
                self._send_json(state)
            else:
                self._send_json({"error": "Bot state not found"}, 404)

        elif path.startswith("/api/engine/trades/"):
            bot_id = path.split("/")[-1]
            trades = db.get_active_trades(bot_id)
            # Convert datetimes
            for t in trades:
                for k, v in t.items():
                    if hasattr(v, 'isoformat'):
                        t[k] = v.isoformat()
            self._send_json({"trades": trades, "count": len(trades)})

        elif path == "/health":
            self._send_json({"status": "ok", "workers": len(orch.status())})

        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        orch = get_orchestrator()

        if path == "/api/bots/start":
            body = self._read_body()
            bot_id = body.get("botId")
            if not bot_id:
                self._send_json({"error": "botId required"}, 400)
                return
            success = orch.start_bot(bot_id)
            if success:
                db.set_bot_status(bot_id, "running", True)
            self._send_json({"success": success, "botId": bot_id})

        elif path == "/api/bots/stop":
            body = self._read_body()
            bot_id = body.get("botId")
            if not bot_id:
                self._send_json({"error": "botId required"}, 400)
                return
            success = orch.stop_bot(bot_id)
            if success:
                db.set_bot_status(bot_id, "stopped", False)
            self._send_json({"success": success, "botId": bot_id})

        else:
            self._send_json({"error": "Not found"}, 404)

    def log_message(self, format, *args):
        """Suppress default access logs, use our logger instead."""
        logger.debug("%s %s", args[0] if args else "", args[1] if len(args) > 1 else "")


def main():
    orch = get_orchestrator()
    orch.start_health_monitor(check_interval=30)

    server = HTTPServer(("0.0.0.0", PORT), OrchestratorHandler)
    logger.info("🎯 Orchestrator API running on port %d", PORT)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        orch.stop_all()
        server.shutdown()
        db.close()


if __name__ == "__main__":
    main()
