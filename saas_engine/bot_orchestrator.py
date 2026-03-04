"""
Bot Orchestrator — Multi-Tenant Engine Manager

Manages Python engine worker processes for each active bot.
Called by the Next.js API when bots are toggled on/off.

Architecture:
  - Each bot gets its own subprocess running engine_worker.py
  - Workers read config from PostgreSQL via DBAdapter
  - Workers write trade events back to PostgreSQL
  - Orchestrator monitors health and restarts crashed workers

Usage:
  from bot_orchestrator import BotOrchestrator
  orch = BotOrchestrator()
  orch.start_bot("bot_cuid_123")
  orch.stop_bot("bot_cuid_123")
  orch.status()
"""
import os
import sys
import signal
import logging
import subprocess
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger("Orchestrator")


class BotOrchestrator:
    """Manages per-bot engine worker processes."""

    def __init__(self):
        self._workers: Dict[str, subprocess.Popen] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Path to the engine worker script
        self._worker_script = os.path.join(
            os.path.dirname(__file__), "engine_worker.py"
        )
        self._python = sys.executable

        logger.info("🎯 Bot Orchestrator initialized")

    def start_bot(self, bot_id: str, database_url: Optional[str] = None) -> bool:
        """Start an engine worker for a bot."""
        with self._lock:
            if bot_id in self._workers:
                proc = self._workers[bot_id]
                if proc.poll() is None:  # still running
                    logger.warning("Bot %s already running (pid=%d)", bot_id, proc.pid)
                    return True

            env = os.environ.copy()
            if database_url:
                env["DATABASE_URL"] = database_url

            env["BOT_ID"] = bot_id

            try:
                proc = subprocess.Popen(
                    [self._python, self._worker_script],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self._workers[bot_id] = proc
                logger.info("🟢 Started bot %s (pid=%d)", bot_id, proc.pid)

                # Start output reader thread
                t = threading.Thread(
                    target=self._read_output, args=(bot_id, proc),
                    daemon=True
                )
                t.start()

                return True
            except Exception as e:
                logger.error("Failed to start bot %s: %s", bot_id, e)
                return False

    def stop_bot(self, bot_id: str) -> bool:
        """Stop an engine worker for a bot."""
        with self._lock:
            proc = self._workers.get(bot_id)
            if not proc:
                logger.info("Bot %s not running", bot_id)
                return True

            if proc.poll() is not None:  # already exited
                del self._workers[bot_id]
                return True

            try:
                # Send SIGTERM for graceful shutdown
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

                del self._workers[bot_id]
                logger.info("🔴 Stopped bot %s", bot_id)
                return True
            except Exception as e:
                logger.error("Failed to stop bot %s: %s", bot_id, e)
                return False

    def is_running(self, bot_id: str) -> bool:
        """Check if a bot worker is running."""
        with self._lock:
            proc = self._workers.get(bot_id)
            if not proc:
                return False
            return proc.poll() is None

    def status(self) -> Dict[str, dict]:
        """Get status of all workers."""
        result = {}
        with self._lock:
            for bot_id, proc in list(self._workers.items()):
                running = proc.poll() is None
                result[bot_id] = {
                    "pid": proc.pid,
                    "running": running,
                    "returncode": proc.returncode if not running else None,
                }
                # Cleanup dead workers
                if not running:
                    del self._workers[bot_id]
        return result

    def start_health_monitor(self, check_interval: int = 30):
        """Start background thread that monitors and restarts crashed workers."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._health_loop, args=(check_interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("❤️ Health monitor started (interval=%ds)", check_interval)

    def stop_all(self):
        """Stop all workers and the health monitor."""
        self._running = False
        with self._lock:
            for bot_id in list(self._workers.keys()):
                self.stop_bot(bot_id)
        logger.info("⛔ All bots stopped")

    def _health_loop(self, check_interval: int):
        """Background loop that checks worker health."""
        while self._running:
            time.sleep(check_interval)
            with self._lock:
                for bot_id, proc in list(self._workers.items()):
                    if proc.poll() is not None:
                        rc = proc.returncode
                        logger.warning(
                            "⚠️ Bot %s crashed (exit=%d) — restarting...",
                            bot_id, rc
                        )
                        del self._workers[bot_id]
                        # Restart
                        self.start_bot(bot_id)

    @staticmethod
    def _read_output(bot_id: str, proc: subprocess.Popen):
        """Read worker stdout and log it."""
        prefix = f"[Worker:{bot_id[:8]}]"
        try:
            for line in proc.stdout:
                logger.info("%s %s", prefix, line.rstrip())
        except Exception:
            pass


# ─── Singleton ───────────────────────────────────────────────────────────────

_orchestrator: Optional[BotOrchestrator] = None


def get_orchestrator() -> BotOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = BotOrchestrator()
    return _orchestrator
