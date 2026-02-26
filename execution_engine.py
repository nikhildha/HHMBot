"""
Project Regime-Master — Execution Engine
Handles futures order placement with protective SL/TP orders.
  • Paper mode → Binance testnet (simulated)
  • Live mode  → CoinDCX Futures (real orders)
"""
import logging
import csv
import os
from datetime import datetime

import config
from risk_manager import RiskManager

logger = logging.getLogger("ExecutionEngine")


class ExecutionEngine:
    """
    Places and manages futures trades.

    Key safety features:
      • ISOLATED margin per position
      • ATR-based SL/TP bracket orders
      • Paper trade mode (logs but doesn't execute)

    Live mode routes to CoinDCX Futures.
    Paper mode routes to Binance (testnet, simulated).
    """

    def __init__(self, client=None):
        self._client = client  # Binance client (paper only)
        self.risk = RiskManager()

    def _get_binance_client(self):
        """Lazy-init Binance client (paper mode only)."""
        if self._client is None:
            from binance.client import Client
            self._client = Client(
                api_key=config.BINANCE_API_KEY,
                api_secret=config.BINANCE_API_SECRET,
                testnet=config.TESTNET,
            )
        return self._client

    # Backward-compatible alias
    _get_client = _get_binance_client

    # ─── Margin & Leverage Setup (Paper/Binance) ──────────────────────────────

    def setup_position(self, symbol, leverage):
        """Set isolated margin and leverage for a symbol (Binance paper)."""
        client = self._get_binance_client()
        try:
            client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
            logger.info("Margin set to ISOLATED for %s.", symbol)
        except Exception:
            pass  # Already ISOLATED

        try:
            client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info("Leverage set to %dx for %s.", leverage, symbol)
        except Exception as e:
            logger.error("Failed to set leverage for %s: %s", symbol, e)

    # ─── Trade Execution ─────────────────────────────────────────────────────

    def execute_trade(self, symbol, side, leverage, quantity, atr,
                      regime=None, confidence=None, reason=""):
        """
        Execute a futures trade with protective SL/TP.

        Routes:
          Paper mode → simulated log entry
          Live mode  → CoinDCX Futures market order with TP/SL

        Parameters
        ----------
        symbol : str — Binance-style (BTCUSDT)
        side : str ('BUY' or 'SELL')
        leverage : int
        quantity : float
        atr : float (for computing SL/TP)
        regime : int (optional, for logging)
        confidence : float (optional)
        reason : str

        Returns
        -------
        dict: trade result or paper log
        """
        if leverage <= 0:
            logger.info("Leverage is 0 — no trade for %s.", symbol)
            return None

        regime_name = config.REGIME_NAMES.get(regime, "UNKNOWN")

        # ── Paper Trade Mode (Binance) ────────────────────────────
        if config.PAPER_TRADE:
            from data_pipeline import get_current_price
            price = get_current_price(symbol) or 0
            sl, tp = self.risk.calculate_atr_stops(price, atr, side)

            log_entry = {
                "timestamp":  datetime.utcnow().isoformat(),
                "symbol":     symbol,
                "side":       side,
                "leverage":   leverage,
                "quantity":   quantity,
                "entry_price": price,
                "stop_loss":  sl,
                "take_profit": tp,
                "regime":     regime_name,
                "confidence": f"{confidence:.2f}" if confidence else "N/A",
                "reason":     reason,
                "mode":       "PAPER",
            }
            self._log_trade(log_entry)
            logger.info(
                "📝 PAPER %s %s @ %.2f | %dx | SL=%.2f TP=%.2f | %s",
                side, symbol, price, leverage, sl, tp, regime_name,
            )
            return log_entry

        # ── Live Trade (CoinDCX Futures) ──────────────────────────
        return self._execute_coindcx(symbol, side, leverage, quantity, atr,
                                     regime, regime_name, confidence, reason)

    # ─── CoinDCX helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _cdx_qty_step(price):
        """
        Infer CoinDCX quantity step from unit price.

        Conservative heuristic (smaller steps to avoid rejection):
          BTC          (price >= $10,000) → step 0.001
          ETH/mid-caps ($10 – $10,000)   → step 0.01
          Low-caps     ($0.10 – $10)     → step 0.1
          Micro-caps   (< $0.10)         → step 1.0

        If CoinDCX rejects, the retry handler parses the real step from the error.
        """
        if price >= 10_000:
            return 0.001
        elif price >= 10:
            return 0.01
        elif price >= 0.10:
            return 0.1
        else:
            return 1.0

    @staticmethod
    def _round_to_step(qty, step):
        """Round quantity UP to the nearest step size."""
        import math
        return math.ceil(qty / step) * step

    def _execute_coindcx(self, symbol, side, leverage, quantity, atr,
                         regime, regime_name, confidence, reason):
        """
        Execute a live trade on CoinDCX Futures.

        Handles CoinDCX-specific requirements:
          • $120 minimum notional order value
          • Quantity step-size rounding per instrument
          • Max leverage auto-clamping with retry
          • Wallet balance pre-check
        """
        import math
        import re as _re
        import coindcx_client as cdx

        pair = cdx.to_coindcx_pair(symbol)
        coindcx_side = side.lower()

        COINDCX_MIN_NOTIONAL = 120.0

        try:
            # 1. Get current price
            price = cdx.get_current_price(pair)
            if price is None:
                logger.error("Cannot get price for %s — aborting trade.", pair)
                return None

            # 2. Enforce $120 minimum order + round to step size
            step = self._cdx_qty_step(price)
            notional = quantity * price
            if notional < COINDCX_MIN_NOTIONAL:
                quantity = self._round_to_step(COINDCX_MIN_NOTIONAL / price, step)
                logger.info(
                    "📐 Boosted %s qty to %.6f to meet $120 CoinDCX min", symbol, quantity,
                )
            else:
                quantity = self._round_to_step(quantity, step)
            notional = quantity * price

            # 3. Check margin vs wallet balance
            margin_needed = notional / leverage
            wallet = cdx.get_usdt_balance()
            if margin_needed > wallet:
                logger.warning(
                    "💰 Insufficient balance for %s: need $%.2f margin, have $%.2f — skipping.",
                    symbol, margin_needed, wallet,
                )
                return None

            # 4. Set leverage (auto-clamp if exchange rejects)
            try:
                cdx.update_leverage(pair, leverage)
            except Exception as lev_err:
                err_msg = str(lev_err)
                # Parse "Max allowed leverage … = 30.0x" from error
                m = _re.search(r"Max allowed leverage.*?=\s*([\d.]+)", err_msg)
                if m:
                    max_lev = int(float(m.group(1)))
                    logger.warning(
                        "⚡ %s max leverage %dx (requested %dx) — clamping.", symbol, max_lev, leverage,
                    )
                    leverage = max_lev
                    cdx.update_leverage(pair, leverage)
                    margin_needed = notional / leverage
                    if margin_needed > wallet:
                        logger.warning("💰 Margin $%.2f > balance $%.2f after clamp — skip.", margin_needed, wallet)
                        return None
                elif "Instrument is not active" in err_msg:
                    logger.warning("🚫 %s not active on CoinDCX — skipping.", symbol)
                    return None
                else:
                    raise

            sl, tp = self.risk.calculate_atr_stops(price, atr, side)

            # 5a. Round TP/SL to CoinDCX price tick sizes
            #     CoinDCX rejects prices with excess decimals.
            def _price_round(p):
                if p >= 1000:      return round(p, 1)
                elif p >= 10:      return round(p, 2)
                elif p >= 1:       return round(p, 3)
                elif p >= 0.01:    return round(p, 4)
                else:              return round(p, 5)

            sl = _price_round(sl)
            tp = _price_round(tp)

            # 5b. Place market order with inline TP/SL (retry on qty step error)
            try:
                result = cdx.create_order(
                    pair=pair, side=coindcx_side, order_type="market_order",
                    quantity=quantity, leverage=leverage,
                    take_profit_price=tp, stop_loss_price=sl,
                )
            except Exception as ord_err:
                err_msg = str(ord_err)
                # Parse "Quantity should be divisible by X" and retry
                m = _re.search(r"divisible by ([\d.]+)", err_msg)
                if m:
                    real_step = float(m.group(1))
                    quantity = self._round_to_step(COINDCX_MIN_NOTIONAL / price, real_step)
                    notional = quantity * price
                    margin_needed = notional / leverage
                    if margin_needed > wallet:
                        logger.warning("💰 Margin $%.2f > balance after re-step — skip.", margin_needed)
                        return None
                    logger.info("📐 Retry %s with step=%.6f → qty=%.6f", symbol, real_step, quantity)
                    result = cdx.create_order(
                        pair=pair, side=coindcx_side, order_type="market_order",
                        quantity=quantity, leverage=leverage,
                        take_profit_price=tp, stop_loss_price=sl,
                    )
                else:
                    raise

            logger.info(
                "✅ CoinDCX %s %s @ %.4f | %dx | qty=%.6f | SL=%.4f TP=%.4f | %s",
                side, symbol, price, leverage, quantity, sl, tp, regime_name,
            )

            # 6. Read back CONFIRMED position from CoinDCX (source of truth)
            import time as _time
            _time.sleep(0.5)  # Allow exchange to settle
            confirmed = {}
            try:
                positions = cdx.list_positions()
                for pos in positions:
                    if pos.get("pair") == pair and float(pos.get("active_pos", 0)) != 0:
                        confirmed = {
                            "avg_price":      float(pos.get("avg_price", price)),
                            "active_pos":     abs(float(pos.get("active_pos", quantity))),
                            "leverage":       int(float(pos.get("leverage", leverage))),
                            "locked_margin":  float(pos.get("locked_margin", 0)),
                            "mark_price":     float(pos.get("mark_price", price)),
                            "position_id":    pos.get("id"),
                            "sl_trigger":     pos.get("stop_loss_trigger"),
                            "tp_trigger":     pos.get("take_profit_trigger"),
                        }
                        break
            except Exception as read_err:
                logger.warning("Could not read back position for %s: %s", symbol, read_err)

            # Use CoinDCX-confirmed values where available; fallback to local
            fill_price   = confirmed.get("avg_price", price)
            fill_qty     = confirmed.get("active_pos", quantity)
            fill_lev     = confirmed.get("leverage", leverage)
            fill_margin  = confirmed.get("locked_margin", 0) or (fill_qty * fill_price / fill_lev)
            fill_sl      = confirmed.get("sl_trigger", sl)
            fill_tp      = confirmed.get("tp_trigger", tp)

            log_entry = {
                "timestamp":    datetime.utcnow().isoformat(),
                "symbol":       symbol,
                "side":         side,
                "leverage":     fill_lev,
                "quantity":     fill_qty,
                "entry_price":  fill_price,
                "stop_loss":    float(fill_sl) if fill_sl else sl,
                "take_profit":  float(fill_tp) if fill_tp else tp,
                "capital":      round(fill_margin, 2),
                "regime":       regime_name,
                "confidence":   confidence if confidence else 0,
                "reason":       reason,
                "mode":         "LIVE-COINDCX",
                "exchange":     "coindcx",
                "pair":         pair,
                "position_id":  confirmed.get("position_id"),
                "order_result": str(result),
            }
            self._log_trade(log_entry)
            return log_entry

        except Exception as e:
            logger.error("❌ CoinDCX Execution Error for %s: %s", symbol, e)
            return None

    # ─── Emergency Close ─────────────────────────────────────────────────────

    def close_all_positions(self, symbol=None):
        """
        Close all open futures positions.
        If symbol is provided, close only that symbol.
        """
        if config.PAPER_TRADE:
            logger.info("📝 PAPER: close_all_positions(%s)", symbol or "ALL")
            return

        # ── CoinDCX Live ──
        self._close_coindcx(symbol)

    def _close_coindcx(self, symbol=None):
        """Close positions on CoinDCX Futures."""
        import coindcx_client as cdx

        try:
            positions = cdx.list_positions()
            if not positions:
                logger.info("No CoinDCX positions to close.")
                return

            for pos in positions:
                active = float(pos.get("active_pos", 0))
                if active == 0:
                    continue

                pair = pos.get("pair", "")
                pos_id = pos.get("id", "")

                # Filter by symbol if specified
                if symbol:
                    target_pair = cdx.to_coindcx_pair(symbol)
                    if pair != target_pair:
                        continue

                cdx.exit_position(pos_id)
                logger.info("🔴 CoinDCX: Closed position %s (%s)", pair, pos_id)

            # Cancel all open orders
            cdx.cancel_all_open_orders()
            logger.info("CoinDCX: All positions closed and orders cancelled.")

        except Exception as e:
            logger.error("Failed to close CoinDCX positions: %s", e)

    # ─── Get Wallet Balance ──────────────────────────────────────────────────

    def get_futures_balance(self):
        """Get USDT futures wallet balance."""
        if config.PAPER_TRADE:
            return 1000.0  # Simulated starting balance

        # ── CoinDCX Live ──
        import coindcx_client as cdx
        return cdx.get_usdt_balance()

    # ─── Trade Logger ────────────────────────────────────────────────────────

    @staticmethod
    def _log_trade(entry):
        """Append trade to CSV log."""
        file_exists = os.path.exists(config.TRADE_LOG_FILE)
        try:
            with open(config.TRADE_LOG_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=entry.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(entry)
        except Exception as e:
            logger.error("Failed to log trade: %s", e)

