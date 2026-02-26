"""
Project Regime-Master — Sideways Strategy (Mean Reversion Shield)
When HMM detects "Chop" regime, the bot switches to this module.
Uses Bollinger Bands + RSI to capture range-bound moves with low leverage.
"""
import logging

import config
from feature_engine import compute_indicators

logger = logging.getLogger("SidewaysStrategy")


def evaluate_mean_reversion(df, symbol=None):
    """
    Evaluate mean-reversion entry conditions for a sideways market.
    
    Conditions:
      BUY  : price ≤ lower Bollinger Band AND RSI < RSI_OVERSOLD
      SELL : price ≥ upper Bollinger Band AND RSI > RSI_OVERBOUGHT
    
    Parameters
    ----------
    df : pd.DataFrame with OHLCV data
    symbol : str (for logging)
    
    Returns
    -------
    dict or None:
        {'side': 'BUY'|'SELL', 'leverage': int, 'reason': str}
        Returns None if no setup detected.
    """
    symbol = symbol or config.PRIMARY_SYMBOL

    # Ensure indicators are computed
    if "rsi" not in df.columns or "bb_upper" not in df.columns:
        df = compute_indicators(df)

    current_price = df["close"].iloc[-1]
    lower_band    = df["bb_lower"].iloc[-1]
    upper_band    = df["bb_upper"].iloc[-1]
    rsi_val       = df["rsi"].iloc[-1]

    # Check for NaN (insufficient data)
    if any(v != v for v in [current_price, lower_band, upper_band, rsi_val]):
        logger.warning("Sideways: indicators not ready (NaN). Skipping.")
        return None

    leverage = config.LEVERAGE_LOW

    # ── Buy at floor ──────────────────────────────────────────
    if current_price <= lower_band and rsi_val < config.RSI_OVERSOLD:
        logger.info(
            "💎 Sideways BUY signal: %s @ %.2f ≤ BB_lower=%.2f, RSI=%.1f",
            symbol, current_price, lower_band, rsi_val,
        )
        return {
            "side":     "BUY",
            "leverage": leverage,
            "reason":   f"Mean Reversion BUY: Price({current_price:.2f}) ≤ BB_lower({lower_band:.2f}), RSI={rsi_val:.1f}",
        }

    # ── Sell at ceiling ───────────────────────────────────────
    if current_price >= upper_band and rsi_val > config.RSI_OVERBOUGHT:
        logger.info(
            "💰 Sideways SELL signal: %s @ %.2f ≥ BB_upper=%.2f, RSI=%.1f",
            symbol, current_price, upper_band, rsi_val,
        )
        return {
            "side":     "SELL",
            "leverage": leverage,
            "reason":   f"Mean Reversion SELL: Price({current_price:.2f}) ≥ BB_upper({upper_band:.2f}), RSI={rsi_val:.1f}",
        }

    logger.debug(
        "↔️ Sideways: No setup. Price=%.2f, BB=[%.2f, %.2f], RSI=%.1f",
        current_price, lower_band, upper_band, rsi_val,
    )
    return None
