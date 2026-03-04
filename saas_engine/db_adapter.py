"""
PostgreSQL Database Adapter for HMMBOT Engine

Replaces JSON file I/O with PostgreSQL reads/writes via psycopg2.
Used by the engine worker to persist trades, bot state, and engine health.

Reads connection string from DATABASE_URL env var.
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import psycopg2
import psycopg2.extras

logger = logging.getLogger("DBAdapter")


class DBAdapter:
    """PostgreSQL adapter for multi-tenant HMMBOT engine."""

    def __init__(self, database_url: Optional[str] = None):
        self._url = database_url or os.getenv("DATABASE_URL", "")
        if not self._url:
            raise ValueError("DATABASE_URL env var required")
        self._conn = None

    def _get_conn(self):
        """Get or create a database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._url)
            self._conn.autocommit = True
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ─── Bot Config ──────────────────────────────────────────────────────

    def get_bot_config(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Load BotConfig for a bot."""
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                '''SELECT * FROM "BotConfig" WHERE "botId" = %s''',
                (bot_id,)
            )
            row = cur.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                if isinstance(result.get("coinList"), str):
                    result["coinList"] = json.loads(result["coinList"])
                if isinstance(result.get("leverageTiers"), str):
                    result["leverageTiers"] = json.loads(result["leverageTiers"])
                return result
            return None

    # ─── Bot State ───────────────────────────────────────────────────────

    def get_bot_state(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Load BotState for a bot."""
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                '''SELECT * FROM "BotState" WHERE "botId" = %s''',
                (bot_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def update_bot_state(self, bot_id: str, **kwargs):
        """Update BotState fields (engineStatus, lastCycleAt, cycleCount, coinStates, etc.)"""
        if not kwargs:
            return
        conn = self._get_conn()

        set_parts = []
        values = []
        for key, val in kwargs.items():
            # Convert camelCase to match Prisma column names
            if key == "coin_states" and isinstance(val, dict):
                val = json.dumps(val)
                key = "coinStates"
            elif key == "engine_status":
                key = "engineStatus"
            elif key == "last_cycle_at":
                key = "lastCycleAt"
            elif key == "cycle_count":
                key = "cycleCount"
            elif key == "cycle_duration_ms":
                key = "cycleDurationMs"
            elif key == "error_message":
                key = "errorMessage"
            elif key == "error_at":
                key = "errorAt"

            set_parts.append(f'"{key}" = %s')
            values.append(val)

        set_parts.append('"updatedAt" = %s')
        values.append(datetime.now(timezone.utc))
        values.append(bot_id)

        sql = f'''UPDATE "BotState" SET {", ".join(set_parts)} WHERE "botId" = %s'''
        with conn.cursor() as cur:
            cur.execute(sql, values)

    # ─── Exchange API Keys ───────────────────────────────────────────────

    def get_exchange_keys(self, user_id: str, exchange: str) -> Optional[Dict[str, str]]:
        """Get encrypted exchange API keys for a user."""
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                '''SELECT "apiKey", "apiSecret", "encryptionIv"
                   FROM "ExchangeApiKey"
                   WHERE "userId" = %s AND "exchange" = %s AND "isActive" = true''',
                (user_id, exchange)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # ─── Trades ──────────────────────────────────────────────────────────

    def get_active_trades(self, bot_id: str) -> List[Dict[str, Any]]:
        """Get all active trades for a bot."""
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                '''SELECT * FROM "Trade"
                   WHERE "botId" = %s AND "status" = 'active'
                   ORDER BY "entryTime" DESC''',
                (bot_id,)
            )
            return [dict(r) for r in cur.fetchall()]

    def create_trade(self, bot_id: str, trade_data: Dict[str, Any]) -> str:
        """Insert a new trade and return its ID."""
        conn = self._get_conn()
        # Generate cuid-like ID
        import uuid
        trade_id = str(uuid.uuid4()).replace('-', '')[:25]

        fields = {
            "id": trade_id,
            "botId": bot_id,
            "coin": trade_data.get("symbol", ""),
            "position": trade_data.get("side", "long").lower(),
            "regime": trade_data.get("regime", "unknown"),
            "confidence": trade_data.get("confidence", 0),
            "mode": trade_data.get("mode", "paper"),
            "leverage": trade_data.get("leverage", 1),
            "capital": trade_data.get("capital", 0),
            "quantity": trade_data.get("quantity", 0),
            "entryPrice": trade_data.get("entry_price", 0),
            "stopLoss": trade_data.get("stop_loss", 0),
            "takeProfit": trade_data.get("take_profit", 0),
            "t1Price": trade_data.get("t1_price"),
            "t2Price": trade_data.get("t2_price"),
            "t3Price": trade_data.get("t3_price"),
            "status": "active",
            "entryTime": datetime.now(timezone.utc),
            "createdAt": datetime.now(timezone.utc),
            "updatedAt": datetime.now(timezone.utc),
        }

        # Optional exchange refs
        if trade_data.get("order_id"):
            fields["exchangeOrderId"] = str(trade_data["order_id"])

        cols = ', '.join(f'"{k}"' for k in fields.keys())
        placeholders = ', '.join(['%s'] * len(fields))
        sql = f'INSERT INTO "Trade" ({cols}) VALUES ({placeholders})'

        with conn.cursor() as cur:
            cur.execute(sql, list(fields.values()))

        logger.info("📝 Trade created in DB: %s %s %s",
                     trade_data.get("symbol"), trade_data.get("side"), trade_id)
        return trade_id

    def update_trade(self, trade_id: str, **kwargs):
        """Update specific fields on a trade."""
        if not kwargs:
            return
        conn = self._get_conn()

        # Map snake_case → camelCase
        field_map = {
            "current_price": "currentPrice",
            "exit_price": "exitPrice",
            "stop_loss": "stopLoss",
            "take_profit": "takeProfit",
            "trailing_sl": "trailingSl",
            "trailing_tp": "trailingTp",
            "trailing_active": "trailingActive",
            "trail_sl_count": "trailSlCount",
            "capital_protection_active": "capitalProtectionActive",
            "t1_hit": "t1Hit",
            "t2_hit": "t2Hit",
            "original_qty": "originalQty",
            "original_capital": "originalCapital",
            "quantity": "quantity",
            "capital": "capital",
            "active_pnl": "activePnl",
            "active_pnl_pct": "activePnlPercent",
            "total_pnl": "totalPnl",
            "total_pnl_pct": "totalPnlPercent",
            "exit_reason": "exitReason",
            "exit_time": "exitTime",
            "status": "status",
        }

        set_parts = []
        values = []
        for key, val in kwargs.items():
            db_key = field_map.get(key, key)
            set_parts.append(f'"{db_key}" = %s')
            values.append(val)

        set_parts.append('"updatedAt" = %s')
        values.append(datetime.now(timezone.utc))
        values.append(trade_id)

        sql = f'''UPDATE "Trade" SET {", ".join(set_parts)} WHERE "id" = %s'''
        with conn.cursor() as cur:
            cur.execute(sql, values)

    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str,
                    pnl: float, pnl_pct: float):
        """Close a trade with exit details."""
        self.update_trade(
            trade_id,
            status="closed",
            exit_price=exit_price,
            exit_reason=exit_reason,
            total_pnl=pnl,
            total_pnl_pct=pnl_pct,
            active_pnl=0,
            active_pnl_pct=0,
            exit_time=datetime.now(timezone.utc),
        )
        logger.info("🔴 Trade closed in DB: %s reason=%s pnl=%.2f", trade_id, exit_reason, pnl)

    def create_partial_booking(self, trade_id: str, target: str, book_pct: float,
                                quantity: float, exit_price: float, pnl: float, pnl_pct: float):
        """Record a partial profit booking (T1/T2)."""
        conn = self._get_conn()
        import uuid
        booking_id = str(uuid.uuid4()).replace('-', '')[:25]

        sql = '''INSERT INTO "PartialBooking"
                 ("id", "tradeId", "target", "bookPercent", "quantity",
                  "exitPrice", "pnl", "pnlPercent", "createdAt")
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'''

        with conn.cursor() as cur:
            cur.execute(sql, (
                booking_id, trade_id, target, book_pct, quantity,
                exit_price, pnl, pnl_pct, datetime.now(timezone.utc)
            ))

        logger.info("📊 Partial booking: %s %s %.2f%% qty=%.6f",
                     trade_id, target, book_pct * 100, quantity)

    # ─── Bot Status ──────────────────────────────────────────────────────

    def set_bot_status(self, bot_id: str, status: str, is_active: bool):
        """Update bot running status."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc)
        extra = ""
        if status == "running":
            extra = ', "startedAt" = %s'
        elif status == "stopped":
            extra = ', "stoppedAt" = %s'

        sql = f'''UPDATE "Bot"
                  SET "status" = %s, "isActive" = %s, "updatedAt" = %s{extra}
                  WHERE "id" = %s'''

        values = [status, is_active, now]
        if extra:
            values.append(now)
        values.append(bot_id)

        with conn.cursor() as cur:
            cur.execute(sql, values)

    def get_bot_user_id(self, bot_id: str) -> Optional[str]:
        """Get the userId for a bot."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute('''SELECT "userId" FROM "Bot" WHERE "id" = %s''', (bot_id,))
            row = cur.fetchone()
            return row[0] if row else None
