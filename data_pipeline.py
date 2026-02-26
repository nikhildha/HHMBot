"""
Project Regime-Master — Data Pipeline
Fetches multi-timeframe OHLCV data.
  • Paper mode → Binance (REST, testnet-safe)
  • Live mode  → CoinDCX Futures (REST)
"""
import pandas as pd
import logging

import config

logger = logging.getLogger("DataPipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# BINANCE CLIENT (Paper Trading)
# ═══════════════════════════════════════════════════════════════════════════════

_binance_client = None


def _get_binance_client():
    """Lazy-init the Binance client (paper trading only)."""
    global _binance_client
    if _binance_client is None:
        from binance.client import Client
        _binance_client = Client(
            api_key=config.BINANCE_API_KEY,
            api_secret=config.BINANCE_API_SECRET,
            testnet=config.TESTNET,
        )
        mode = "TESTNET" if config.TESTNET else "PRODUCTION"
        logger.info("Binance client initialized (%s).", mode)
    return _binance_client


# Keep backward-compatible alias
_get_client = _get_binance_client

INTERVAL_MAP = {
    "1m":  "1m",
    "3m":  "3m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
    "1w":  "1w",
}


# ═══════════════════════════════════════════════════════════════════════════════
# BINANCE FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_klines_binance(symbol, interval, limit=500):
    """Fetch candlesticks from Binance (spot or testnet)."""
    from binance.client import Client as BClient
    client = _get_binance_client()

    binance_map = {
        "1m": BClient.KLINE_INTERVAL_1MINUTE,   "3m": BClient.KLINE_INTERVAL_3MINUTE,
        "5m": BClient.KLINE_INTERVAL_5MINUTE,    "15m": BClient.KLINE_INTERVAL_15MINUTE,
        "30m": BClient.KLINE_INTERVAL_30MINUTE,  "1h": BClient.KLINE_INTERVAL_1HOUR,
        "4h": BClient.KLINE_INTERVAL_4HOUR,      "1d": BClient.KLINE_INTERVAL_1DAY,
        "1w": BClient.KLINE_INTERVAL_1WEEK,
    }
    binance_interval = binance_map.get(interval, interval)

    try:
        klines = client.get_klines(symbol=symbol, interval=binance_interval, limit=limit)
    except Exception as e:
        logger.error("Binance fetch %s %s failed: %s", symbol, interval, e)
        return None

    if not klines:
        return None

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore",
    ])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df.reset_index(drop=True, inplace=True)
    logger.debug("Binance: %d candles for %s %s.", len(df), symbol, interval)
    return df


def _fetch_futures_klines_binance(symbol, interval, limit=500):
    """Fetch Binance Futures candlesticks."""
    from binance.client import Client as BClient
    client = _get_binance_client()

    binance_map = {
        "1m": BClient.KLINE_INTERVAL_1MINUTE,   "3m": BClient.KLINE_INTERVAL_3MINUTE,
        "5m": BClient.KLINE_INTERVAL_5MINUTE,    "15m": BClient.KLINE_INTERVAL_15MINUTE,
        "30m": BClient.KLINE_INTERVAL_30MINUTE,  "1h": BClient.KLINE_INTERVAL_1HOUR,
        "4h": BClient.KLINE_INTERVAL_4HOUR,      "1d": BClient.KLINE_INTERVAL_1DAY,
        "1w": BClient.KLINE_INTERVAL_1WEEK,
    }
    binance_interval = binance_map.get(interval, interval)

    try:
        klines = client.futures_klines(symbol=symbol, interval=binance_interval, limit=limit)
    except Exception as e:
        logger.error("Binance futures fetch %s %s failed: %s", symbol, interval, e)
        return None

    if not klines:
        return None

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore",
    ])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_open_interest_hist(symbol="BTCUSDT", period="1h", limit=500):
    """
    Fetch historical Open Interest from Binance Futures.
    
    Returns DataFrame with columns: ['timestamp', 'open_interest']
    OI represents total outstanding futures contracts.
    """
    import requests
    
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": min(limit, 500)}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("OI fetch failed for %s: %s", symbol, e)
        return None
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open_interest"] = df["sumOpenInterest"].astype(float)
    df = df[["timestamp", "open_interest"]].copy()
    df.reset_index(drop=True, inplace=True)
    logger.debug("OI: got %d data points for %s", len(df), symbol)
    return df


def fetch_funding_rate(symbol="BTCUSDT", limit=500):
    """
    Fetch historical Funding Rate from Binance Futures.
    
    Returns DataFrame with columns: ['timestamp', 'funding_rate']
    Funding rate is typically every 8 hours.
    """
    import requests
    
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": min(limit, 1000)}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Funding rate fetch failed for %s: %s", symbol, e)
        return None
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]].copy()
    df.reset_index(drop=True, inplace=True)
    logger.debug("Funding: got %d data points for %s", len(df), symbol)
    return df


def enrich_with_oi_funding(df_ohlcv, symbol="BTCUSDT"):
    """
    Merge OI and Funding Rate data into an OHLCV DataFrame.
    
    Uses merge_asof to align different-frequency data (OI=1h, funding=8h)
    with the OHLCV candle timestamps. Forward-fills gaps.
    
    Parameters
    ----------
    df_ohlcv : pd.DataFrame with 'timestamp' column
    symbol : str
    
    Returns
    -------
    pd.DataFrame with added 'open_interest' and 'funding_rate' columns
    """
    df = df_ohlcv.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Fetch OI
    oi_df = fetch_open_interest_hist(symbol, period="1h", limit=500)
    if oi_df is not None and len(oi_df) > 0:
        oi_df = oi_df.sort_values("timestamp")
        df = pd.merge_asof(df, oi_df, on="timestamp", direction="backward")
        df["open_interest"] = df["open_interest"].ffill()
        logger.info("Enriched with %d OI data points", len(oi_df))
    else:
        df["open_interest"] = 0.0
        logger.warning("No OI data available, using zeros")
    
    # Fetch Funding Rate
    fr_df = fetch_funding_rate(symbol, limit=500)
    if fr_df is not None and len(fr_df) > 0:
        fr_df = fr_df.sort_values("timestamp")
        df = pd.merge_asof(df, fr_df, on="timestamp", direction="backward")
        df["funding_rate"] = df["funding_rate"].ffill()
        logger.info("Enriched with %d funding rate data points", len(fr_df))
    else:
        df["funding_rate"] = 0.0
        logger.warning("No funding data available, using zeros")
    
    return df


def _get_current_price_binance(symbol):
    """Get current price from Binance."""
    client = _get_binance_client()
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])
    except Exception as e:
        logger.error("Binance price fetch for %s: %s", symbol, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# COINDCX FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_klines_coindcx(symbol, interval, limit=500):
    """
    Fetch candlesticks from CoinDCX Futures.
    Accepts Binance-style symbol (BTCUSDT), converts internally.
    """
    import coindcx_client as cdx
    pair = cdx.to_coindcx_pair(symbol)
    return cdx.get_candlesticks(pair, interval, limit=limit)


def _get_current_price_coindcx(symbol):
    """Get current price from CoinDCX. Accepts Binance-style symbol."""
    import coindcx_client as cdx
    pair = cdx.to_coindcx_pair(symbol)
    return cdx.get_current_price(pair)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED PUBLIC API (auto-routes by mode)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_klines(symbol, interval, limit=500):
    """
    Fetch historical candlestick data.

    Always uses CoinDCX for price consistency between paper and live modes.
    Falls back to Binance only if CoinDCX fails.
    """
    try:
        return _fetch_klines_coindcx(symbol, interval, limit)
    except Exception as e:
        logger.warning("CoinDCX klines failed for %s, falling back to Binance: %s", symbol, e)
        return _fetch_klines_binance(symbol, interval, limit)


def fetch_futures_klines(symbol, interval, limit=500):
    """
    Fetch futures candlestick data.

    Routes:
      Paper mode → Binance Futures
      Live mode  → CoinDCX Futures
    """
    if config.PAPER_TRADE:
        return _fetch_futures_klines_binance(symbol, interval, limit)
    else:
        return _fetch_klines_coindcx(symbol, interval, limit)


def get_multi_timeframe_data(symbol=None, limit=500):
    """
    Fetch 15m, 1h, and 4h candles for a symbol.

    Returns
    -------
    dict: {'15m': DataFrame, '1h': DataFrame, '4h': DataFrame}
    Any timeframe that fails returns None.
    """
    symbol = symbol or config.PRIMARY_SYMBOL

    data = {
        config.TIMEFRAME_EXECUTION:     fetch_klines(symbol, config.TIMEFRAME_EXECUTION, limit),
        config.TIMEFRAME_CONFIRMATION:  fetch_klines(symbol, config.TIMEFRAME_CONFIRMATION, limit),
        config.TIMEFRAME_MACRO:         fetch_klines(symbol, config.TIMEFRAME_MACRO, limit),
    }

    success = sum(1 for v in data.values() if v is not None)
    logger.info("Multi-TF fetch for %s: %d/%d timeframes OK.", symbol, success, len(data))
    return data


def get_current_price(symbol=None):
    """Get the latest price for a symbol. Always uses CoinDCX."""
    symbol = symbol or config.PRIMARY_SYMBOL
    try:
        return _get_current_price_coindcx(symbol)
    except Exception as e:
        logger.warning("CoinDCX price failed for %s, falling back to Binance: %s", symbol, e)
        return _get_current_price_binance(symbol)

