"""
Project Regime-Master — Binance Futures Client
Implements ExchangeClient interface for Binance USDT-M Futures.
Supports full multi-target flow: open, partial close, SL/TP modification.
"""
import logging
import math
import time
from typing import Optional, Dict, Any, List

import config
from exchange_base import ExchangeClient

logger = logging.getLogger("BinanceFutures")


class BinanceFuturesClient(ExchangeClient):
    """
    Live trading on Binance USDT-M Futures.

    Uses python-binance library. Supports:
      • Market/limit orders with bracket SL/TP
      • Partial close via reduce-only opposite-side orders
      • SL/TP modification via cancel + replace
      • Testnet mode for safe testing
    """

    def __init__(self, testnet: bool = True):
        from binance.client import Client
        self._testnet = testnet
        self._client = Client(
            api_key=config.BINANCE_API_KEY,
            api_secret=config.BINANCE_API_SECRET,
            testnet=testnet,
        )
        mode = "TESTNET" if testnet else "PRODUCTION"
        logger.info("Binance Futures client initialized (%s)", mode)

        # Cache for open SL/TP order IDs per symbol
        self._sl_orders: Dict[str, int] = {}
        self._tp_orders: Dict[str, int] = {}

    @property
    def exchange_name(self) -> str:
        return "binance"

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _qty_precision(self, symbol: str) -> int:
        """Get quantity precision for a symbol from exchange info."""
        try:
            info = self._client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step = float(f["stepSize"])
                            return max(0, int(round(-math.log10(step))))
            return 3  # Default
        except Exception:
            return 3

    def _price_precision(self, symbol: str) -> int:
        """Get price precision for a symbol."""
        try:
            info = self._client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "PRICE_FILTER":
                            tick = float(f["tickSize"])
                            return max(0, int(round(-math.log10(tick))))
            return 2
        except Exception:
            return 2

    def _round_qty(self, symbol: str, qty: float) -> float:
        """Round quantity to exchange precision."""
        precision = self._qty_precision(symbol)
        return round(qty, precision)

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to exchange precision."""
        precision = self._price_precision(symbol)
        return round(price, precision)

    # ─── Interface Implementation ────────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage and isolated margin for a symbol."""
        try:
            try:
                self._client.futures_change_margin_type(
                    symbol=symbol, marginType="ISOLATED"
                )
            except Exception:
                pass  # Already ISOLATED

            self._client.futures_change_leverage(
                symbol=symbol, leverage=leverage
            )
            logger.info("Set %s leverage to %dx (ISOLATED)", symbol, leverage)
            return True
        except Exception as e:
            logger.error("Failed to set leverage for %s: %s", symbol, e)
            return False

    def get_balance(self) -> float:
        """Get available USDT futures balance."""
        try:
            balances = self._client.futures_account_balance()
            for b in balances:
                if b["asset"] == "USDT":
                    return float(b["availableBalance"])
            return 0.0
        except Exception as e:
            logger.error("Failed to get balance: %s", e)
            return 0.0

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol."""
        try:
            positions = self._client.futures_position_information(symbol=symbol)
            for pos in positions:
                qty = float(pos["positionAmt"])
                if qty != 0:
                    return {
                        "symbol": symbol,
                        "side": "BUY" if qty > 0 else "SELL",
                        "quantity": abs(qty),
                        "entry_price": float(pos["entryPrice"]),
                        "leverage": int(pos["leverage"]),
                        "unrealized_pnl": float(pos["unRealizedProfit"]),
                        "mark_price": float(pos.get("markPrice", 0)),
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
        """
        Open a futures position with SL/TP bracket orders.

        Flow:
        1. Set leverage + ISOLATED margin
        2. Place market order
        3. Place separate SL and TP stop orders
        """
        try:
            # 1. Setup
            self.set_leverage(symbol, leverage)
            qty = self._round_qty(symbol, quantity)
            sl = self._round_price(symbol, sl_price)
            tp = self._round_price(symbol, tp_price)

            # 2. Market order
            result = self._client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=qty,
            )
            order_id = result["orderId"]
            avg_price = float(result.get("avgPrice", 0))

            # Wait for fill
            time.sleep(0.3)

            # Read back actual fill
            if avg_price == 0:
                pos = self.get_position(symbol)
                if pos:
                    avg_price = pos["entry_price"]

            logger.info(
                "✅ Binance OPEN %s %s @ %.4f | %dx | qty=%.6f",
                side, symbol, avg_price, leverage, qty,
            )

            # 3. Place SL order
            close_side = "SELL" if side == "BUY" else "BUY"
            try:
                sl_result = self._client.futures_create_order(
                    symbol=symbol,
                    side=close_side,
                    type="STOP_MARKET",
                    stopPrice=str(sl),
                    closePosition="true",
                    timeInForce="GTC",
                )
                self._sl_orders[symbol] = sl_result["orderId"]
                logger.info("   SL order placed @ %.4f", sl)
            except Exception as e:
                logger.warning("   Failed to place SL for %s: %s", symbol, e)

            # 4. Place TP order
            try:
                tp_result = self._client.futures_create_order(
                    symbol=symbol,
                    side=close_side,
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=str(tp),
                    closePosition="true",
                    timeInForce="GTC",
                )
                self._tp_orders[symbol] = tp_result["orderId"]
                logger.info("   TP order placed @ %.4f", tp)
            except Exception as e:
                logger.warning("   Failed to place TP for %s: %s", symbol, e)

            return {
                "order_id": order_id,
                "position_id": None,
                "filled_qty": qty,
                "avg_price": avg_price,
                "status": "FILLED",
                "sl_order_id": self._sl_orders.get(symbol),
                "tp_order_id": self._tp_orders.get(symbol),
            }

        except Exception as e:
            logger.error("❌ Binance open_position failed for %s: %s", symbol, e)
            return {"order_id": None, "status": "FAILED", "error": str(e)}

    def close_position(self, symbol: str) -> bool:
        """Fully close position + cancel all open orders for symbol."""
        try:
            pos = self.get_position(symbol)
            if not pos:
                logger.info("No position to close for %s", symbol)
                return True

            close_side = "SELL" if pos["side"] == "BUY" else "BUY"
            qty = self._round_qty(symbol, pos["quantity"])

            # Close via market order
            self._client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="MARKET",
                quantity=qty,
                reduceOnly="true",
            )

            # Cancel all open orders for this symbol
            try:
                self._client.futures_cancel_all_open_orders(symbol=symbol)
            except Exception:
                pass

            self._sl_orders.pop(symbol, None)
            self._tp_orders.pop(symbol, None)

            logger.info("🔴 Binance CLOSED %s (qty=%.6f)", symbol, qty)
            return True

        except Exception as e:
            logger.error("Failed to close %s: %s", symbol, e)
            return False

    def partial_close(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """
        Partially close a position via reduce-only opposite-side market order.

        For multi-target: T1 closes 25%, T2 closes 50% of remaining.
        """
        try:
            close_side = "SELL" if side == "BUY" else "BUY"
            qty = self._round_qty(symbol, quantity)

            if qty <= 0:
                return {"order_id": None, "status": "SKIP", "error": "qty <= 0"}

            result = self._client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="MARKET",
                quantity=qty,
                reduceOnly="true",
            )

            avg_price = float(result.get("avgPrice", 0))
            logger.info(
                "📊 Binance PARTIAL CLOSE %s: %.6f qty @ %.4f",
                symbol, qty, avg_price,
            )

            return {
                "order_id": result["orderId"],
                "filled_qty": qty,
                "avg_price": avg_price,
                "status": "FILLED",
            }

        except Exception as e:
            logger.error("Failed partial close %s: %s", symbol, e)
            return {"order_id": None, "status": "FAILED", "error": str(e)}

    def modify_sl(self, symbol: str, new_sl_price: float) -> bool:
        """
        Modify SL by cancelling old SL order and placing new one.

        Used after T1 hit (SL → breakeven) and T2 hit (SL → T1 price).
        """
        try:
            new_sl = self._round_price(symbol, new_sl_price)

            # Cancel existing SL order
            old_sl_id = self._sl_orders.get(symbol)
            if old_sl_id:
                try:
                    self._client.futures_cancel_order(
                        symbol=symbol, orderId=old_sl_id
                    )
                except Exception:
                    pass  # May already be filled/cancelled

            # Determine close side from position
            pos = self.get_position(symbol)
            if not pos:
                logger.warning("No position for %s — cannot modify SL", symbol)
                return False

            close_side = "SELL" if pos["side"] == "BUY" else "BUY"

            # Place new SL order
            sl_result = self._client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=str(new_sl),
                closePosition="true",
                timeInForce="GTC",
            )
            self._sl_orders[symbol] = sl_result["orderId"]

            logger.info("🛡️ Binance SL modified for %s: → %.4f", symbol, new_sl)
            return True

        except Exception as e:
            logger.error("Failed to modify SL for %s: %s", symbol, e)
            return False

    def modify_tp(self, symbol: str, new_tp_price: float) -> bool:
        """Modify TP by cancel + replace."""
        try:
            new_tp = self._round_price(symbol, new_tp_price)

            # Cancel existing TP
            old_tp_id = self._tp_orders.get(symbol)
            if old_tp_id:
                try:
                    self._client.futures_cancel_order(
                        symbol=symbol, orderId=old_tp_id
                    )
                except Exception:
                    pass

            pos = self.get_position(symbol)
            if not pos:
                return False

            close_side = "SELL" if pos["side"] == "BUY" else "BUY"

            tp_result = self._client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=str(new_tp),
                closePosition="true",
                timeInForce="GTC",
            )
            self._tp_orders[symbol] = tp_result["orderId"]

            logger.info("📈 Binance TP modified for %s: → %.4f", symbol, new_tp)
            return True

        except Exception as e:
            logger.error("Failed to modify TP for %s: %s", symbol, e)
            return False
