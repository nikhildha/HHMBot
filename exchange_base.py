"""
Project Regime-Master — Exchange Abstraction Layer
Base class for exchange clients (CoinDCX, Binance Futures).
All live-trading exchange implementations must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("ExchangeBase")


class ExchangeClient(ABC):
    """
    Abstract interface for live futures trading on any exchange.

    All methods accept Binance-style symbols (BTCUSDT) and
    convert internally if needed (e.g., CoinDCX uses B-BTC_USDT).
    """

    @abstractmethod
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
        Open a futures position with SL/TP bracket.

        Parameters
        ----------
        symbol : str — e.g. 'BTCUSDT'
        side : str — 'BUY' or 'SELL'
        quantity : float — base asset quantity
        leverage : int
        sl_price : float — initial stop loss price
        tp_price : float — initial take profit price (T3 for multi-target)
        t1_price, t2_price, t3_price : optional target prices

        Returns
        -------
        dict with keys: order_id, position_id, filled_qty, avg_price, status
        """
        ...

    @abstractmethod
    def close_position(self, symbol: str) -> bool:
        """Fully close an open position. Returns True on success."""
        ...

    @abstractmethod
    def partial_close(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """
        Partially close a position by placing a reduce-only order.

        Parameters
        ----------
        symbol : str
        side : str — original position side ('BUY' for long, 'SELL' for short)
        quantity : float — quantity to close

        Returns
        -------
        dict with keys: order_id, filled_qty, avg_price
        """
        ...

    @abstractmethod
    def modify_sl(self, symbol: str, new_sl_price: float) -> bool:
        """
        Modify the stop-loss price for an existing position.
        Cancels old SL order and places a new one.
        Returns True on success.
        """
        ...

    @abstractmethod
    def modify_tp(self, symbol: str, new_tp_price: float) -> bool:
        """Modify the take-profit price. Returns True on success."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current position for a symbol.

        Returns
        -------
        dict with keys: symbol, side, quantity, entry_price, leverage, unrealized_pnl
        or None if no position
        """
        ...

    @abstractmethod
    def get_balance(self) -> float:
        """Get available USDT balance for futures trading."""
        ...

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol. Returns True on success."""
        ...

    @property
    @abstractmethod
    def exchange_name(self) -> str:
        """Return the exchange name (e.g., 'binance', 'coindcx')."""
        ...
