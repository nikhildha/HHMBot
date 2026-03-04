"""
Project Regime-Master — Feature Engine
Computes HMM input features and technical indicators (RSI, Bollinger, ATR).
"""
import numpy as np
import pandas as pd
import logging

import config

logger = logging.getLogger("FeatureEngine")


# ─── HMM Features ───────────────────────────────────────────────────────────────

def compute_hmm_features(df):
    """
    Add HMM-ready features to an OHLCV DataFrame.
    
    Adds:
      - log_return : log(close / prev_close)
      - volatility : intraday range = (high - low) / close
      - volume_change : log(volume / prev_volume), clamped [-3, 3]
      - rsi_norm : (RSI_14 - 50) / 50, range [-1, +1]
    
    Parameters
    ----------
    df : pd.DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    
    Returns
    -------
    pd.DataFrame with added feature columns (NaN rows NOT dropped)
    """
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = (df["high"] - df["low"]) / df["close"]
    # Volume momentum — captures volume spikes signaling regime shifts
    df["volume_change"] = np.log(df["volume"] / df["volume"].shift(1).replace(0, np.nan))
    df["volume_change"] = df["volume_change"].fillna(0).clip(-3, 3)
    # Normalized RSI — momentum context for state classification
    rsi = compute_rsi(df["close"], length=14)
    df["rsi_norm"] = (rsi - 50) / 50  # Range: -1 to +1
    df["rsi_norm"] = df["rsi_norm"].fillna(0)
    return df


# ─── Technical Indicators ───────────────────────────────────────────────────────

def compute_rsi(series, length=None):
    """
    Compute Relative Strength Index.
    
    Parameters
    ----------
    series : pd.Series of close prices
    length : int, default from config.RSI_LENGTH
    
    Returns
    -------
    pd.Series
    """
    length = length or config.RSI_LENGTH
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_bollinger_bands(series, length=None, std=None):
    """
    Compute Bollinger Bands.
    
    Returns
    -------
    (middle, upper, lower) — each a pd.Series
    """
    length = length or config.BB_LENGTH
    std = std or config.BB_STD

    middle = series.rolling(window=length).mean()
    rolling_std = series.rolling(window=length).std()
    upper = middle + (rolling_std * std)
    lower = middle - (rolling_std * std)

    return middle, upper, lower


def compute_atr(df, length=14):
    """
    Compute Average True Range.
    
    Parameters
    ----------
    df : pd.DataFrame with 'high', 'low', 'close'
    length : int
    
    Returns
    -------
    pd.Series
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    return atr


def compute_indicators(df):
    """
    Add all technical indicators to an OHLCV DataFrame.
    
    Adds: rsi, bb_upper, bb_middle, bb_lower, atr
    
    Returns
    -------
    pd.DataFrame (copy with new columns)
    """
    df = df.copy()

    df["rsi"] = compute_rsi(df["close"])
    df["bb_middle"], df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(df["close"])
    df["atr"] = compute_atr(df)

    return df



def compute_ema(series, length):
    """Compute Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()


def compute_trend(df):
    """
    Determine trend direction using EMA 20/50 crossover + price position.
    Returns: 'UP', 'DOWN', or 'FLAT'
    """
    close = df["close"]
    ema_20 = compute_ema(close, 20)
    ema_50 = compute_ema(close, 50)

    last_close = float(close.iloc[-1])
    last_ema20 = float(ema_20.iloc[-1])
    last_ema50 = float(ema_50.iloc[-1])

    if last_ema20 > last_ema50 and last_close > last_ema20:
        return "UP"
    elif last_ema20 < last_ema50 and last_close < last_ema20:
        return "DOWN"
    return "FLAT"


def compute_support_resistance(df, n_swing=3, max_levels=3):
    """
    Identify support & resistance levels using swing high/low detection.
    
    Parameters
    ----------
    df : pd.DataFrame with 'high', 'low', 'close' columns (must have indicators)
    n_swing : int — lookback window on each side for swing detection
    max_levels : int — max number of S/R levels to return
    
    Returns
    -------
    dict with keys: support (list[float]), resistance (list[float]),
                    pivot (float), bb_pos (float 0-1)
    """
    highs = df["high"].values
    lows = df["low"].values
    close = float(df["close"].iloc[-1])
    
    swing_highs = []
    swing_lows = []
    
    for i in range(n_swing, len(df) - n_swing):
        # Swing high: local maximum
        if highs[i] == max(highs[i - n_swing: i + n_swing + 1]):
            swing_highs.append(float(highs[i]))
        # Swing low: local minimum
        if lows[i] == min(lows[i - n_swing: i + n_swing + 1]):
            swing_lows.append(float(lows[i]))
    
    # Filter: support = swing lows below current price, sorted closest first
    support = sorted([s for s in swing_lows if s < close], reverse=True)[:max_levels]
    # Filter: resistance = swing highs above current price, sorted closest first
    resistance = sorted([r for r in swing_highs if r > close])[:max_levels]
    
    # Pivot (traditional: (H + L + C) / 3 over last candle)
    last = df.iloc[-1]
    pivot = round((float(last["high"]) + float(last["low"]) + close) / 3, 6)
    
    # Bollinger Band position: 0 = at lower band, 1 = at upper band
    bb_pos = 0.5
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        bb_u = float(df["bb_upper"].iloc[-1])
        bb_l = float(df["bb_lower"].iloc[-1])
        if bb_u != bb_l:
            bb_pos = max(0, min(1, (close - bb_l) / (bb_u - bb_l)))
    
    return {
        "support": [round(s, 6) for s in support],
        "resistance": [round(r, 6) for r in resistance],
        "pivot": pivot,
        "bb_pos": round(bb_pos, 4),
    }


def compute_vwap(df, window=20):
    """
    Compute rolling VWAP (Volume Weighted Average Price).
    
    Returns pd.Series of VWAP values.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).rolling(window).sum()
    cum_vol = df["volume"].rolling(window).sum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def compute_sr_position(df, lookback=50):
    """
    Compute SR position (0-1) and VWAP position for conviction scoring.

    sr_position: 0 = at support (ideal for BUY), 1 = at resistance (ideal for SELL)
    vwap_position: >0 = above VWAP (bullish), <0 = below (bearish)

    Returns (sr_position: float, vwap_position: float)
    """
    if len(df) < lookback:
        return None, None

    window = df.iloc[-lookback:]
    close = float(df["close"].iloc[-1])
    recent_high = float(window["high"].max())
    recent_low = float(window["low"].min())

    # SR position: where is price in the recent range?
    if recent_high == recent_low:
        sr_pos = 0.5
    else:
        sr_pos = (close - recent_low) / (recent_high - recent_low)
    sr_pos = max(0.0, min(1.0, sr_pos))

    # VWAP position
    vwap = compute_vwap(df, window=min(20, len(df)))
    last_vwap = vwap.iloc[-1]
    if pd.isna(last_vwap) or last_vwap == 0:
        vwap_pos = None
    else:
        vwap_pos = (close - last_vwap) / last_vwap  # % above/below VWAP

    return sr_pos, vwap_pos


def compute_all_features(df):
    """
    Convenience: computes BOTH HMM features AND technical indicators.
    """
    df = compute_hmm_features(df)
    df = compute_indicators(df)
    
    return df



# ─── Synthetic Data Generator (for testing) ─────────────────────────────────────

def generate_synthetic_data(n=500, seed=42):
    """
    Generate synthetic OHLCV data for smoke-testing HMM training.
    Simulates 3 regimes: uptrend, downtrend, and sideways.
    """
    rng = np.random.RandomState(seed)
    
    # Build a price series with embedded regimes
    prices = [100.0]
    for i in range(1, n):
        phase = i / n
        if phase < 0.33:
            # Uptrend
            drift = 0.002
            vol = 0.01
        elif phase < 0.66:
            # Downtrend
            drift = -0.003
            vol = 0.02
        else:
            # Sideways
            drift = 0.0
            vol = 0.005
        
        ret = drift + vol * rng.randn()
        prices.append(prices[-1] * np.exp(ret))
    
    prices = np.array(prices)
    
    # Build synthetic OHLCV
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open":   prices * (1 + rng.uniform(-0.003, 0.003, n)),
        "high":   prices * (1 + rng.uniform(0.001, 0.015, n)),
        "low":    prices * (1 - rng.uniform(0.001, 0.015, n)),
        "close":  prices,
        "volume": rng.uniform(100, 5000, n),
    })
    
    return df
