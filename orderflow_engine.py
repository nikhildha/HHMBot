"""
Project Regime-Master — Order Flow Engine
Reads publicly available market microstructure data to produce an order flow signal.

Sources (all free, no API key needed):
  1. L2 Order Book  — Binance REST depth snapshot → bid/ask imbalance + wall detection
  2. Taker Flow     — Buy/sell volume already embedded in OHLCV klines (free extra columns)
  3. L/S Ratio      — Binance futures global long/short account ratio
  4. Cumulative Delta — Net buy − sell pressure from recent aggTrades

Output: OrderFlowSignal with a single composite score in [-1, +1]
  • Positive → buy pressure dominates (supports LONG)
  • Negative → sell pressure dominates (supports SHORT)
  • Near-zero → market is balanced / uncertain

Integration:
  • compute_conviction_score() reads orderflow_score (8th factor, 10 pts)
  • main.py calls get_engine().get_signal(symbol, df_15m) each analysis cycle
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

import config

logger = logging.getLogger("OrderFlowEngine")

_BINANCE_SPOT    = "https://api.binance.com/api/v3"
_BINANCE_FUTURES = "https://fapi.binance.com/fapi/v1"


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class WallLevel:
    """A price level with an unusually large resting order."""
    price: float
    size_usd: float     # notional at this level
    side: str           # "bid" or "ask"
    multiple: float     # how many × the average level size


@dataclass
class OrderFlowSignal:
    """Composite order-flow signal for one coin at a point in time."""
    symbol: str
    score: float                        # -1 to +1 composite
    book_imbalance: float               # -1 to +1 (bid depth vs ask depth)
    taker_buy_ratio: float              # 0 to 1 (1 = all buys)
    cumulative_delta: float             # -1 to +1 (recent net buy flow)
    ls_ratio: float                     # > 1 = more longs; < 1 = more shorts
    bid_walls: List[WallLevel] = field(default_factory=list)
    ask_walls: List[WallLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    note: str = ""

    @property
    def nearest_bid_wall(self) -> Optional[float]:
        return max((w.price for w in self.bid_walls), default=None)

    @property
    def nearest_ask_wall(self) -> Optional[float]:
        return min((w.price for w in self.ask_walls), default=None)


# ─── Engine ──────────────────────────────────────────────────────────────────

class OrderFlowEngine:
    """
    Fetches and combines microstructure signals into a single conviction modifier.

    Cache behaviour:
      • Each (symbol) result is cached for ORDERFLOW_CACHE_SECONDS
      • Stale cache is returned on network failure rather than crashing
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[float, OrderFlowSignal]] = {}  # symbol → (epoch, signal)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_signal(self, symbol: str, df_15m=None) -> Optional[OrderFlowSignal]:
        """
        Return an OrderFlowSignal for *symbol*.

        Parameters
        ----------
        symbol  : str, e.g. "BTCUSDT"
        df_15m  : optional pandas DataFrame of 15-min OHLCV klines.
                  When provided, taker buy/sell volumes are extracted from it
                  (avoids an extra REST call).

        Returns None only if ALL sub-fetches fail.
        """
        if not config.ORDERFLOW_ENABLED:
            return None

        # Return cached result if still fresh
        cached = self._cache.get(symbol)
        if cached:
            ts, sig = cached
            if time.time() - ts < config.ORDERFLOW_CACHE_SECONDS:
                return sig

        try:
            sig = self._compute(symbol, df_15m)
        except Exception as e:
            logger.warning("[OrderFlow] compute failed for %s: %s", symbol, e)
            # Return stale cache rather than None
            if cached:
                return cached[1]
            return None

        self._cache[symbol] = (time.time(), sig)
        return sig

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute(self, symbol: str, df_15m) -> OrderFlowSignal:
        """Fetch all sub-signals and combine into one score."""

        # 1. Order book imbalance + walls
        book_imbalance, bid_walls, ask_walls = self._fetch_book(symbol)

        # 2. Taker buy ratio — from klines df if available, otherwise REST
        taker_buy_ratio = self._get_taker_ratio(symbol, df_15m)

        # 3. Cumulative delta from recent aggTrades
        cum_delta = self._fetch_cum_delta(symbol)

        # 4. Long/short ratio from Binance futures (proxy for positioning bias)
        ls_ratio = self._fetch_ls_ratio(symbol) if config.ORDERFLOW_LS_ENABLED else 1.0

        # ── Combine sub-scores ───────────────────────────────────────────────
        # Each sub-score is mapped to [-1, +1] then weighted.
        #   book_imbalance  → already -1..+1     weight 0.35
        #   taker_score     → 2*(ratio-0.5)      weight 0.30
        #   cum_delta       → already -1..+1     weight 0.25
        #   ls_score        → (ratio-1)/max_dev  weight 0.10

        taker_score = 2.0 * (taker_buy_ratio - 0.5)   # -1 to +1
        ls_score    = _clamp((ls_ratio - 1.0) / 0.5)  # normalise around 1.0

        composite = (
            0.35 * book_imbalance
            + 0.30 * taker_score
            + 0.25 * cum_delta
            + 0.10 * ls_score
        )
        composite = _clamp(composite)

        notes = []
        if bid_walls:
            notes.append(f"{len(bid_walls)} bid wall(s) near ${bid_walls[0].price:,.0f}")
        if ask_walls:
            notes.append(f"{len(ask_walls)} ask wall(s) near ${ask_walls[0].price:,.0f}")

        return OrderFlowSignal(
            symbol=symbol,
            score=composite,
            book_imbalance=book_imbalance,
            taker_buy_ratio=taker_buy_ratio,
            cumulative_delta=cum_delta,
            ls_ratio=ls_ratio,
            bid_walls=bid_walls,
            ask_walls=ask_walls,
            note="; ".join(notes) if notes else "",
        )

    # ── 1. L2 Order Book ──────────────────────────────────────────────────────

    def _fetch_book(self, symbol: str) -> Tuple[float, List[WallLevel], List[WallLevel]]:
        """
        Fetch Binance depth snapshot and compute:
          • bid/ask imbalance   (total USD on bids vs asks within top N levels)
          • wall levels         (any level ≥ ORDERFLOW_WALL_THRESHOLD × avg)

        Returns (imbalance, bid_walls, ask_walls).
        imbalance = (bid_usd - ask_usd) / (bid_usd + ask_usd), range -1..+1
        """
        try:
            resp = requests.get(
                f"{_BINANCE_SPOT}/depth",
                params={"symbol": symbol, "limit": config.ORDERFLOW_DEPTH_LEVELS * 5},
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.debug("[OrderFlow] book fetch failed for %s: %s", symbol, e)
            return 0.0, [], []

        bids = [(float(p), float(q)) for p, q in data.get("bids", [])[:config.ORDERFLOW_DEPTH_LEVELS]]
        asks = [(float(p), float(q)) for p, q in data.get("asks", [])[:config.ORDERFLOW_DEPTH_LEVELS]]

        if not bids or not asks:
            return 0.0, [], []

        mid_price = (bids[0][0] + asks[0][0]) / 2.0

        # Compute USD notional at each level
        bid_usd_levels = [p * q for p, q in bids]
        ask_usd_levels = [p * q for p, q in asks]

        total_bid = sum(bid_usd_levels)
        total_ask = sum(ask_usd_levels)
        total = total_bid + total_ask
        imbalance = (total_bid - total_ask) / total if total > 0 else 0.0

        # Wall detection — levels significantly larger than the average
        avg_bid = total_bid / len(bid_usd_levels) if bid_usd_levels else 1
        avg_ask = total_ask / len(ask_usd_levels) if ask_usd_levels else 1
        threshold = config.ORDERFLOW_WALL_THRESHOLD

        bid_walls: List[WallLevel] = []
        for (price, qty), usd in zip(bids, bid_usd_levels):
            mult = usd / avg_bid if avg_bid > 0 else 0
            if mult >= threshold and usd >= config.ORDERFLOW_LARGE_ORDER_USD:
                bid_walls.append(WallLevel(price=price, size_usd=usd, side="bid", multiple=mult))
        bid_walls.sort(key=lambda w: w.price, reverse=True)  # nearest to mid first

        ask_walls: List[WallLevel] = []
        for (price, qty), usd in zip(asks, ask_usd_levels):
            mult = usd / avg_ask if avg_ask > 0 else 0
            if mult >= threshold and usd >= config.ORDERFLOW_LARGE_ORDER_USD:
                ask_walls.append(WallLevel(price=price, size_usd=usd, side="ask", multiple=mult))
        ask_walls.sort(key=lambda w: w.price)  # nearest to mid first

        logger.debug(
            "[OrderFlow] %s book: imbalance=%.3f bid_usd=%.0f ask_usd=%.0f walls(b=%d a=%d)",
            symbol, imbalance, total_bid, total_ask, len(bid_walls), len(ask_walls),
        )
        return _clamp(imbalance), bid_walls, ask_walls

    # ── 2. Taker Buy Ratio ───────────────────────────────────────────────────

    def _get_taker_ratio(self, symbol: str, df_15m) -> float:
        """
        Taker buy ratio = taker_buy_volume / total_volume over recent bars.

        Binance klines include these columns (already in our df_15m):
          index 9  → taker_buy_base_volume
          index 5  → total volume
        If df_15m is None, fetch from REST.
        """
        try:
            if df_15m is not None and len(df_15m) >= config.ORDERFLOW_LOOKBACK_BARS:
                # Columns: timestamp, open, high, low, close, volume, ...
                # After fetch_klines we keep named columns — check for taker column
                if "taker_buy_vol" in df_15m.columns:
                    n = config.ORDERFLOW_LOOKBACK_BARS
                    buy_vol  = df_15m["taker_buy_vol"].iloc[-n:].sum()
                    tot_vol  = df_15m["volume"].iloc[-n:].sum()
                    if tot_vol > 0:
                        return float(buy_vol / tot_vol)
        except Exception:
            pass

        # Fallback: fetch klines from Binance REST
        try:
            resp = requests.get(
                f"{_BINANCE_SPOT}/klines",
                params={
                    "symbol":   symbol,
                    "interval": "15m",
                    "limit":    config.ORDERFLOW_LOOKBACK_BARS,
                },
                timeout=8,
            )
            resp.raise_for_status()
            rows = resp.json()
            # kline index 5 = volume, 9 = taker_buy_base_volume
            buy_vol = sum(float(r[9]) for r in rows)
            tot_vol = sum(float(r[5]) for r in rows)
            return float(buy_vol / tot_vol) if tot_vol > 0 else 0.5
        except Exception as e:
            logger.debug("[OrderFlow] taker ratio fetch failed for %s: %s", symbol, e)
            return 0.5  # neutral default

    # ── 3. Cumulative Delta ───────────────────────────────────────────────────

    def _fetch_cum_delta(self, symbol: str) -> float:
        """
        Cumulative delta = (buy_vol - sell_vol) / (buy_vol + sell_vol)
        from the last 200 aggTrades. Range -1..+1.

        aggTrade field 'm' = True → market SELL (maker on bid side),
                             False → market BUY (maker on ask side).
        """
        try:
            resp = requests.get(
                f"{_BINANCE_SPOT}/aggTrades",
                params={"symbol": symbol, "limit": 500},
                timeout=8,
            )
            resp.raise_for_status()
            trades = resp.json()
        except Exception as e:
            logger.debug("[OrderFlow] aggTrade fetch failed for %s: %s", symbol, e)
            return 0.0

        buy_vol = sell_vol = 0.0
        for t in trades:
            qty   = float(t.get("q", 0))
            price = float(t.get("p", 0))
            usd   = qty * price
            if t.get("m"):        # m = True → the buyer is the maker → taker is SELLING
                sell_vol += usd
            else:
                buy_vol  += usd

        total = buy_vol + sell_vol
        delta = (buy_vol - sell_vol) / total if total > 0 else 0.0
        logger.debug("[OrderFlow] %s cumDelta=%.3f buy=%.0f sell=%.0f", symbol, delta, buy_vol, sell_vol)
        return _clamp(delta)

    # ── 4. Long/Short Ratio ───────────────────────────────────────────────────

    def _fetch_ls_ratio(self, symbol: str) -> float:
        """
        Binance futures global long/short account ratio.
        Returns ratio float (1.0 = balanced). Never raises.
        """
        try:
            resp = requests.get(
                "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                params={"symbol": symbol, "period": "15m", "limit": 1},
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                return float(data[0].get("longShortRatio", 1.0))
        except Exception as e:
            logger.debug("[OrderFlow] L/S ratio fetch failed for %s: %s", symbol, e)
        return 1.0  # neutral


# ─── Singleton access ─────────────────────────────────────────────────────────

_engine_instance: Optional[OrderFlowEngine] = None


def get_engine() -> OrderFlowEngine:
    """Return the module-level singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = OrderFlowEngine()
    return _engine_instance


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s — %(message)s")
    coins = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    eng = OrderFlowEngine()
    print("\n" + "=" * 65)
    print("ORDER FLOW ENGINE — Live Snapshot")
    print("=" * 65)
    for sym in coins:
        sig = eng.get_signal(sym)
        if sig is None:
            print(f"{sym:<12} — no data")
            continue
        bar = "▓" * int(abs(sig.score) * 20)
        direction = "▲" if sig.score > 0 else ("▼" if sig.score < 0 else "─")
        print(
            f"{sym:<12} score={sig.score:+.3f} {direction}{bar}"
            f"  book={sig.book_imbalance:+.3f}"
            f"  taker={sig.taker_buy_ratio:.2f}"
            f"  delta={sig.cumulative_delta:+.3f}"
            f"  L/S={sig.ls_ratio:.2f}"
        )
        if sig.bid_walls:
            for w in sig.bid_walls[:2]:
                print(f"  BID WALL  ${w.price:>12,.2f}  ${w.size_usd:>10,.0f}  ({w.multiple:.1f}× avg)")
        if sig.ask_walls:
            for w in sig.ask_walls[:2]:
                print(f"  ASK WALL  ${w.price:>12,.2f}  ${w.size_usd:>10,.0f}  ({w.multiple:.1f}× avg)")
        if sig.note:
            print(f"  Note: {sig.note}")
    print("=" * 65)
