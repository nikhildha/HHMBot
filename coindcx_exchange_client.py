"""
Project Regime-Master — CoinDCX Exchange Client
Implements ExchangeClient interface wrapping the existing coindcx_client module.
"""
import logging
import math
import re
import time
from typing import Optional, Dict, Any

import config
import coindcx_client as cdx
from exchange_base import ExchangeClient

logger = logging.getLogger("CoinDCXExchange")


class CoinDCXExchangeClient(ExchangeClient):
    """
    Live trading on CoinDCX Futures.

    Wraps the existing coindcx_client.py functions into the
    ExchangeClient interface for multi-target support.
    """

    COINDCX_MIN_NOTIONAL = 120.0

    def __init__(self):
        logger.info("CoinDCX Exchange client initialized")

    @property
    def exchange_name(self) -> str:
        return "coindcx"

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _qty_step(price: float) -> float:
        """Infer CoinDCX quantity step from unit price."""
        if price >= 10_000:
            return 0.001
        elif price >= 10:
            return 0.01
        elif price >= 0.10:
            return 0.1
        else:
            return 1.0

    @staticmethod
    def _round_to_step(qty: float, step: float) -> float:
        """Round quantity UP to the nearest step size."""
        return math.ceil(qty / step) * step

    @staticmethod
    def _price_round(p: float) -> float:
        """Round price to CoinDCX tick sizes."""
        if p >= 1000:
            return round(p, 1)
        elif p >= 10:
            return round(p, 2)
        elif p >= 1:
            return round(p, 3)
        elif p >= 0.01:
            return round(p, 4)
        else:
            return round(p, 5)

    def _get_position_id(self, symbol: str) -> Optional[str]:
        """Get the CoinDCX position ID for a symbol."""
        pair = cdx.to_coindcx_pair(symbol)
        try:
            positions = cdx.list_positions()
            for pos in positions:
                if pos.get("pair") == pair and float(pos.get("active_pos", 0)) != 0:
                    return pos.get("id")
        except Exception as e:
            logger.error("Failed to find position for %s: %s", symbol, e)
        return None

    # ─── Interface Implementation ────────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol on CoinDCX."""
        pair = cdx.to_coindcx_pair(symbol)
        try:
            cdx.update_leverage(pair, leverage)
            logger.info("Set %s leverage to %dx on CoinDCX", symbol, leverage)
            return True
        except Exception as e:
            err_msg = str(e)
            m = re.search(r"Max allowed leverage.*?=\s*([\d.]+)", err_msg)
            if m:
                max_lev = int(float(m.group(1)))
                logger.warning("⚡ %s max leverage %dx — clamping", symbol, max_lev)
                cdx.update_leverage(pair, max_lev)
                return True
            logger.error("Failed to set leverage for %s: %s", symbol, e)
            return False

    def get_balance(self) -> float:
        """Get available USDT futures balance."""
        return cdx.get_usdt_balance()

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol."""
        pair = cdx.to_coindcx_pair(symbol)
        try:
            positions = cdx.list_positions()
            for pos in positions:
                if pos.get("pair") == pair:
                    qty = float(pos.get("active_pos", 0))
                    if qty != 0:
                        return {
                            "symbol": symbol,
                            "side": "BUY" if qty > 0 else "SELL",
                            "quantity": abs(qty),
                            "entry_price": float(pos.get("avg_price", 0)),
                            "leverage": int(float(pos.get("leverage", 1))),
                            "unrealized_pnl": float(pos.get("unrealized_pnl", 0)),
                            "position_id": pos.get("id"),
                        }
            return None
        except Exception as e:
            logger.error("Failed to get position for %s: %s", symbol, e)
            return None

    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: int,
        sl_price: float,
        tp_price: float,
        t1_price: Optional[float] = None,
        t2_price: Optional[float] = None,
        t3_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Open a futures position on CoinDCX with SL/TP."""
        pair = cdx.to_coindcx_pair(symbol)
        coindcx_side = side.lower()

        try:
            price = cdx.get_current_price(pair)
            if price is None:
                return {"order_id": None, "status": "FAILED", "error": "no price"}

            # Enforce minimum notional
            step = self._qty_step(price)
            notional = quantity * price
            if notional < self.COINDCX_MIN_NOTIONAL:
                quantity = self._round_to_step(self.COINDCX_MIN_NOTIONAL / price, step)
            else:
                quantity = self._round_to_step(quantity, step)

            # Check balance
            wallet = self.get_balance()
            margin_needed = (quantity * price) / leverage
            if margin_needed > wallet:
                return {"order_id": None, "status": "FAILED", "error": "insufficient balance"}

            # Set leverage
            self.set_leverage(symbol, leverage)

            sl = self._price_round(sl_price)
            tp = self._price_round(tp_price)

            # Place order with retry on qty step error
            try:
                result = cdx.create_order(
                    pair=pair, side=coindcx_side, order_type="market_order",
                    quantity=quantity, leverage=leverage,
                    take_profit_price=tp, stop_loss_price=sl,
                )
            except Exception as ord_err:
                m = re.search(r"divisible by ([\d.]+)", str(ord_err))
                if m:
                    real_step = float(m.group(1))
                    quantity = self._round_to_step(self.COINDCX_MIN_NOTIONAL / price, real_step)
                    result = cdx.create_order(
                        pair=pair, side=coindcx_side, order_type="market_order",
                        quantity=quantity, leverage=leverage,
                        take_profit_price=tp, stop_loss_price=sl,
                    )
                else:
                    raise

            # Read back confirmed position
            time.sleep(0.5)
            confirmed = {}
            try:
                positions = cdx.list_positions()
                for pos in positions:
                    if pos.get("pair") == pair and float(pos.get("active_pos", 0)) != 0:
                        confirmed = {
                            "avg_price": float(pos.get("avg_price", price)),
                            "filled_qty": abs(float(pos.get("active_pos", quantity))),
                            "position_id": pos.get("id"),
                        }
                        break
            except Exception:
                pass

            logger.info(
                "✅ CoinDCX OPEN %s %s @ %.4f | %dx | qty=%.6f",
                side, symbol, confirmed.get("avg_price", price), leverage, quantity,
            )

            return {
                "order_id": str(result) if result else None,
                "position_id": confirmed.get("position_id"),
                "filled_qty": confirmed.get("filled_qty", quantity),
                "avg_price": confirmed.get("avg_price", price),
                "status": "FILLED",
            }

        except Exception as e:
            logger.error("❌ CoinDCX open_position failed for %s: %s", symbol, e)
            return {"order_id": None, "status": "FAILED", "error": str(e)}

    def close_position(self, symbol: str) -> bool:
        """Fully close a position on CoinDCX."""
        try:
            pos_id = self._get_position_id(symbol)
            if not pos_id:
                logger.info("No position to close for %s", symbol)
                return True

            cdx.exit_position(pos_id)
            logger.info("🔴 CoinDCX CLOSED %s", symbol)
            return True
        except Exception as e:
            logger.error("Failed to close %s: %s", symbol, e)
            return False

    def partial_close(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """Partially close a position on CoinDCX."""
        pair = cdx.to_coindcx_pair(symbol)
        try:
            price = cdx.get_current_price(pair) or 0
            step = self._qty_step(price)
            qty = self._round_to_step(quantity, step)

            if qty <= 0:
                return {"order_id": None, "status": "SKIP", "error": "qty <= 0"}

            result = cdx.partial_close_position(
                pair=pair,
                side=side.lower(),
                quantity=qty,
            )

            logger.info("📊 CoinDCX PARTIAL CLOSE %s: %.6f qty", symbol, qty)
            return {
                "order_id": str(result) if result else None,
                "filled_qty": qty,
                "avg_price": price,
                "status": "FILLED",
            }
        except Exception as e:
            logger.error("Failed partial close %s: %s", symbol, e)
            return {"order_id": None, "status": "FAILED", "error": str(e)}

    def modify_sl(self, symbol: str, new_sl_price: float) -> bool:
        """Modify SL on CoinDCX by updating position's TP/SL."""
        try:
            pos_id = self._get_position_id(symbol)
            if not pos_id:
                logger.warning("No position for %s — cannot modify SL", symbol)
                return False

            new_sl = self._price_round(new_sl_price)
            cdx.modify_stop_loss(pos_id, new_sl)
            logger.info("🛡️ CoinDCX SL modified for %s → %.4f", symbol, new_sl)
            return True
        except Exception as e:
            logger.error("Failed to modify SL for %s: %s", symbol, e)
            return False

    def modify_tp(self, symbol: str, new_tp_price: float) -> bool:
        """Modify TP on CoinDCX by updating position's TP/SL."""
        try:
            pos_id = self._get_position_id(symbol)
            if not pos_id:
                return False

            new_tp = self._price_round(new_tp_price)
            cdx.modify_take_profit(pos_id, new_tp)
            logger.info("📈 CoinDCX TP modified for %s → %.4f", symbol, new_tp)
            return True
        except Exception as e:
            logger.error("Failed to modify TP for %s: %s", symbol, e)
            return False
