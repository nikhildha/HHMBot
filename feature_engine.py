"""
Project Regime-Master — Feature Engine
Computes HMM input features and technical indicators (RSI, Bollinger, ATR, VWAP, S/R).
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
      - vwap_position : (close - VWAP) / ATR — how far price is from fair value
      - sr_position : 0 (at support) to 1 (at resistance) — where price sits in S/R range
    
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
    
    # ─── VWAP Position — distance from volume-weighted fair value ─────────
    vwap = compute_vwap(df)
    atr = compute_atr(df)
    df["vwap_position"] = ((df["close"] - vwap) / atr).fillna(0).clip(-3, 3)
    
    # ─── S/R Position — where price sits between support and resistance ───
    support, resistance = compute_support_resistance(df)
    sr_range = resistance - support
    sr_range = sr_range.replace(0, np.nan)  # avoid /0
    df["sr_position"] = ((df["close"] - support) / sr_range).fillna(0.5).clip(0, 1)
    # Normalize to [-1, 1] for HMM: 0=support → -1, 1=resistance → +1
    df["sr_position"] = df["sr_position"] * 2 - 1
    
    # ─── Open Interest Change — rate of change of market leverage ─────
    # Uses log(OI / prev_OI) — captures SHIFTS in positioning, not absolute size
    # This avoids noise from market growth and focuses on short-term leverage changes
    if "open_interest" in df.columns:
        oi = df["open_interest"].replace(0, np.nan)
        df["oi_change"] = np.log(oi / oi.shift(1)).fillna(0).clip(-2, 2)
    else:
        df["oi_change"] = 0.0
    
    # ─── Funding Rate (z-scored) — crowding indicator ────────────────
    # z-score over rolling window normalizes across different market conditions
    # High → longs crowded (short opportunity), Low → shorts crowded (long opportunity)
    if "funding_rate" in df.columns:
        fr = df["funding_rate"]
        fr_mean = fr.rolling(window=20, min_periods=1).mean()
        fr_std = fr.rolling(window=20, min_periods=1).std().replace(0, np.nan)
        df["funding_norm"] = ((fr - fr_mean) / fr_std).fillna(0).clip(-2, 2)
    else:
        df["funding_norm"] = 0.0
    
    return df


# ─── VWAP ────────────────────────────────────────────────────────────────────────

def compute_vwap(df, length=20):
    """
    Compute rolling VWAP (Volume-Weighted Average Price).
    
    Uses a rolling window instead of session-based VWAP since crypto trades 24/7.
    
    Parameters
    ----------
    df : pd.DataFrame with 'high', 'low', 'close', 'volume'
    length : int, rolling window size (default 20 bars)
    
    Returns
    -------
    pd.Series
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]
    vwap = tp_vol.rolling(window=length).sum() / df["volume"].rolling(window=length).sum()
    return vwap


# ─── Support / Resistance ────────────────────────────────────────────────────────

def compute_support_resistance(df, lookback=20, pivot_window=5):
    """
    Compute dynamic support and resistance levels using swing highs/lows.
    
    Method: Find local minima (support) and maxima (resistance) within a
    rolling window, then use the nearest ones as current S/R levels.
    
    Parameters
    ----------
    df : pd.DataFrame with 'high', 'low', 'close'
    lookback : int, how many bars back to search for S/R levels
    pivot_window : int, number of bars on each side to confirm a pivot
    
    Returns
    -------
    (support: pd.Series, resistance: pd.Series)
    """
    n = len(df)
    support = pd.Series(np.nan, index=df.index)
    resistance = pd.Series(np.nan, index=df.index)
    
    highs = df["high"].values
    lows = df["low"].values
    close = df["close"].values
    
    for i in range(lookback, n):
        window_start = max(0, i - lookback)
        
        # Find swing lows (support) in lookback window
        swing_lows = []
        for j in range(window_start + pivot_window, i - pivot_window + 1):
            left = lows[j - pivot_window:j]
            right = lows[j + 1:j + pivot_window + 1]
            if len(left) > 0 and len(right) > 0:
                if lows[j] <= left.min() and lows[j] <= right.min():
                    swing_lows.append(lows[j])
        
        # Find swing highs (resistance) in lookback window
        swing_highs = []
        for j in range(window_start + pivot_window, i - pivot_window + 1):
            left = highs[j - pivot_window:j]
            right = highs[j + 1:j + pivot_window + 1]
            if len(left) > 0 and len(right) > 0:
                if highs[j] >= left.max() and highs[j] >= right.max():
                    swing_highs.append(highs[j])
        
        # Use nearest support below price, nearest resistance above price
        current_price = close[i]
        
        supports_below = [s for s in swing_lows if s < current_price]
        resistances_above = [r for r in swing_highs if r > current_price]
        
        if supports_below:
            support.iloc[i] = max(supports_below)  # nearest support below
        else:
            support.iloc[i] = lows[window_start:i + 1].min()  # fallback: window low
        
        if resistances_above:
            resistance.iloc[i] = min(resistances_above)  # nearest resistance above
        else:
            resistance.iloc[i] = highs[window_start:i + 1].max()  # fallback: window high
    
    # Forward-fill early bars
    support = support.ffill().bfill()
    resistance = resistance.ffill().bfill()
    
    return support, resistance


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
    
    Adds: rsi, bb_upper, bb_middle, bb_lower, atr, vwap, support, resistance
    
    Returns
    -------
    pd.DataFrame (copy with new columns)
    """
    df = df.copy()

    df["rsi"] = compute_rsi(df["close"])
    df["bb_middle"], df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(df["close"])
    df["atr"] = compute_atr(df)
    df["vwap"] = compute_vwap(df)
    df["support"], df["resistance"] = compute_support_resistance(df)

    return df


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
