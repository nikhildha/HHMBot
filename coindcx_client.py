"""
Project Regime-Master — CoinDCX Futures Client
REST API wrapper for CoinDCX Futures endpoints.
Used for LIVE trading (Binance retained for paper trading only).

API Docs: https://docs.coindcx.com/#futures-end-points
"""
import hmac
import hashlib
import json
import time
import logging
import re

import requests
import pandas as pd

import config

logger = logging.getLogger("CoinDCX")

# ─── Constants ───────────────────────────────────────────────────────────────────

BASE_URL = config.COINDCX_BASE_URL        # https://api.coindcx.com
PUBLIC_URL = config.COINDCX_PUBLIC_URL     # https://public.coindcx.com
FUTURES_PREFIX = "/exchange/v1/derivatives/futures"


# ─── Authentication ──────────────────────────────────────────────────────────────

def _sign(body_dict):
    """Generate HMAC-SHA256 signature for CoinDCX API request."""
    secret_bytes = bytes(config.COINDCX_API_SECRET, encoding="utf-8")
    json_body = json.dumps(body_dict, separators=(",", ":"))
    signature = hmac.new(secret_bytes, json_body.encode(), hashlib.sha256).hexdigest()
    return json_body, signature


def _auth_headers(signature):
    """Build auth headers for private endpoints."""
    return {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": config.COINDCX_API_KEY,
        "X-AUTH-SIGNATURE": signature,
    }


def _private_post(endpoint, body_dict):
    """Make an authenticated POST request to CoinDCX API."""
    body_dict["timestamp"] = int(round(time.time() * 1000))
    json_body, signature = _sign(body_dict)
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.post(url, data=json_body, headers=_auth_headers(signature), timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        # Include response body in the exception so callers can parse error details
        body_text = resp.text if resp is not None else ""
        logger.error("CoinDCX API error %s: %s — %s", endpoint, e, body_text)
        raise RuntimeError(f"CoinDCX {endpoint}: {body_text}") from e
    except Exception as e:
        logger.error("CoinDCX request failed %s: %s", endpoint, e)
        raise


def _private_get(endpoint, body_dict):
    """Make an authenticated GET request to CoinDCX API."""
    body_dict["timestamp"] = int(round(time.time() * 1000))
    json_body, signature = _sign(body_dict)
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, data=json_body, headers=_auth_headers(signature), timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        logger.error("CoinDCX API error %s: %s — %s", endpoint, e, resp.text)
        raise
    except Exception as e:
        logger.error("CoinDCX request failed %s: %s", endpoint, e)
        raise


# ─── Symbol Conversion ──────────────────────────────────────────────────────────
# Binance: BTCUSDT  →  CoinDCX: B-BTC_USDT

def to_coindcx_pair(binance_symbol):
    """Convert Binance symbol (BTCUSDT) to CoinDCX pair (B-BTC_USDT)."""
    # Handle common quote currencies
    for quote in ("USDT", "INR", "BUSD", "USDC"):
        if binance_symbol.endswith(quote):
            base = binance_symbol[: -len(quote)]
            return f"B-{base}_{quote}"
    # Fallback: assume USDT
    return f"B-{binance_symbol}_USDT"


def from_coindcx_pair(coindcx_pair):
    """Convert CoinDCX pair (B-BTC_USDT) to Binance symbol (BTCUSDT)."""
    # Remove 'B-' prefix, replace '_' with ''
    pair = coindcx_pair
    if pair.startswith("B-"):
        pair = pair[2:]
    return pair.replace("_", "")


# ─── Resolution Mapping ─────────────────────────────────────────────────────────
# CoinDCX candlestick resolutions: '1', '5', '15', '60', '1D'

RESOLUTION_MAP = {
    "1m":  "1",
    "3m":  "3",
    "5m":  "5",
    "15m": "15",
    "30m": "30",
    "1h":  "60",
    "4h":  "240",
    "1d":  "1D",
    "1w":  "1W",
}

# Interval in seconds (for computing candle time ranges)
INTERVAL_SECONDS = {
    "1m":  60,
    "3m":  180,
    "5m":  300,
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
    "4h":  14400,
    "1d":  86400,
    "1w":  604800,
}


# ─── Public Endpoints ────────────────────────────────────────────────────────────

def get_active_instruments(margin_currency=None):
    """
    Fetch list of all active futures instruments.

    Returns
    -------
    list[str] — e.g. ['B-BTC_USDT', 'B-ETH_USDT', ...]
    """
    margin_currency = margin_currency or config.COINDCX_MARGIN_CURRENCY
    url = (
        f"{BASE_URL}/exchange/v1/derivatives/futures/data/active_instruments"
        f"?margin_currency_short_name[]={margin_currency}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        instruments = resp.json()
        logger.info("CoinDCX: %d active %s futures instruments.", len(instruments), margin_currency)
        return instruments
    except Exception as e:
        logger.error("Failed to fetch CoinDCX instruments: %s", e)
        return []


def get_candlesticks(pair, interval, limit=500):
    """
    Fetch candlestick data from CoinDCX Futures.

    Parameters
    ----------
    pair : str — CoinDCX pair, e.g. 'B-BTC_USDT'
    interval : str — e.g. '15m', '1h', '1d'
    limit : int — number of candles (computed via time range)

    Returns
    -------
    pd.DataFrame with columns: timestamp, open, high, low, close, volume
    """
    resolution = RESOLUTION_MAP.get(interval)
    if resolution is None:
        logger.error("Unsupported interval for CoinDCX: %s", interval)
        return None

    interval_secs = INTERVAL_SECONDS.get(interval, 900)
    now = int(time.time())
    from_ts = now - (limit * interval_secs)

    url = f"{PUBLIC_URL}/market_data/candlesticks"
    params = {
        "pair": pair,
        "from": from_ts,
        "to": now,
        "resolution": resolution,
        "pcode": "f",  # futures
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Failed to fetch CoinDCX candles for %s %s: %s", pair, interval, e)
        return None

    if data.get("s") != "ok" or not data.get("data"):
        logger.warning("Empty candles for %s %s: %s", pair, interval, data.get("s"))
        return None

    candles = data["data"]
    df = pd.DataFrame(candles)

    # Standardize column names to match Binance format
    df = df.rename(columns={"time": "timestamp"})
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df.reset_index(drop=True, inplace=True)

    logger.debug("Fetched %d CoinDCX candles for %s %s.", len(df), pair, interval)
    return df


def get_current_prices():
    """
    Fetch real-time futures prices for all instruments.

    Returns
    -------
    dict — {pair: {ls: last_price, v: volume_24h, pc: price_change_pct, ...}}
    """
    url = f"{PUBLIC_URL}/market_data/v3/current_prices/futures/rt"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("prices", {})
    except Exception as e:
        logger.error("Failed to fetch CoinDCX prices: %s", e)
        return {}


def get_current_price(pair):
    """
    Get live price for a single futures instrument.

    Parameters
    ----------
    pair : str — CoinDCX pair, e.g. 'B-BTC_USDT'

    Returns
    -------
    float or None
    """
    prices = get_current_prices()
    info = prices.get(pair)
    if info and "ls" in info:
        return float(info["ls"])
    logger.warning("No price found for %s on CoinDCX.", pair)
    return None


# ─── Private Endpoints (Futures) ─────────────────────────────────────────────────

def create_order(pair, side, order_type, quantity, leverage,
                 price=None, stop_price=None,
                 take_profit_price=None, stop_loss_price=None,
                 time_in_force="good_till_cancel"):
    """
    Create a futures order on CoinDCX.

    Parameters
    ----------
    pair : str — CoinDCX pair (B-BTC_USDT)
    side : str — 'buy' or 'sell'
    order_type : str — 'market_order' or 'limit_order'
    quantity : float
    leverage : int
    price : float (required for limit orders)
    stop_price : float (optional)
    take_profit_price : float (optional, sets TP on position)
    stop_loss_price : float (optional, sets SL on position)

    Returns
    -------
    dict — API response
    """
    order = {
        "side": side.lower(),
        "pair": pair,
        "order_type": order_type,
        "total_quantity": quantity,
        "leverage": leverage,
        "notification": "no_notification",
        "hidden": False,
        "post_only": False,
    }

    if price is not None:
        order["price"] = str(price)
    if stop_price is not None:
        order["stop_price"] = str(stop_price)
    if take_profit_price is not None:
        order["take_profit_price"] = float(take_profit_price)
    if stop_loss_price is not None:
        order["stop_loss_price"] = float(stop_loss_price)

    # time_in_force only for non-market orders
    if order_type != "market_order":
        order["time_in_force"] = time_in_force

    body = {"order": order}

    logger.info(
        "CoinDCX Order: %s %s %s qty=%.6f lev=%dx TP=%s SL=%s",
        side, pair, order_type, quantity, leverage,
        take_profit_price, stop_loss_price,
    )

    return _private_post(f"{FUTURES_PREFIX}/orders/create", body)


def cancel_order(order_id):
    """Cancel a single futures order by ID."""
    return _private_post(f"{FUTURES_PREFIX}/orders/cancel", {"id": order_id})


def cancel_all_open_orders(margin_currency=None):
    """Cancel all open futures orders."""
    margin_currency = margin_currency or config.COINDCX_MARGIN_CURRENCY
    try:
        return _private_post(
            f"{FUTURES_PREFIX}/positions/cancel_all_open_orders",
            {"margin_currency_short_name": [margin_currency]},
        )
    except Exception as e:
        # 422 = "No open orders found" — not a real error
        if "422" in str(e) or "No open orders" in str(e):
            logger.debug("No open orders to cancel.")
            return {"message": "no_open_orders"}
        raise


def list_positions(margin_currency=None, page=1, size=50):
    """
    List all futures positions.

    Returns
    -------
    list[dict] — position objects with id, pair, active_pos, avg_price, leverage, etc.
    """
    margin_currency = margin_currency or config.COINDCX_MARGIN_CURRENCY
    return _private_post(
        f"{FUTURES_PREFIX}/positions",
        {
            "page": str(page),
            "size": str(size),
            "margin_currency_short_name": [margin_currency],
        },
    )


def get_position_by_pair(pair):
    """Get position for a specific pair."""
    return _private_post(
        f"{FUTURES_PREFIX}/positions/by_pairs_or_positionid",
        {"pair": pair},
    )


def exit_position(position_id):
    """Exit (close) a position by its ID."""
    logger.info("CoinDCX: Exiting position %s", position_id)
    return _private_post(f"{FUTURES_PREFIX}/positions/exit", {"id": position_id})


def update_leverage(pair, leverage):
    """Update leverage for a futures position."""
    logger.info("CoinDCX: Setting leverage %dx for %s", leverage, pair)
    return _private_post(
        f"{FUTURES_PREFIX}/positions/update_leverage",
        {"leverage": str(leverage), "pair": pair},
    )


def create_tpsl(position_id, take_profit_price=None, stop_loss_price=None):
    """
    Create Take Profit and Stop Loss for an existing position.

    Uses stop_market for SL and take_profit_market for TP.
    """
    body = {"id": position_id}
    if take_profit_price is not None:
        body["take_profit"] = {
            "stop_price": str(take_profit_price),
            "order_type": "take_profit_market",
        }
    if stop_loss_price is not None:
        body["stop_loss"] = {
            "stop_price": str(stop_loss_price),
            "order_type": "stop_market",
        }
    logger.info(
        "CoinDCX: Setting TP/SL on position %s — TP=%s SL=%s",
        position_id, take_profit_price, stop_loss_price,
    )
    return _private_post(f"{FUTURES_PREFIX}/positions/create_tpsl", body)


def get_wallet_details():
    """
    Fetch futures wallet balance.

    Returns
    -------
    list[dict] — [{currency_short_name, balance, locked_balance, ...}]
    """
    return _private_get(f"{FUTURES_PREFIX}/wallets", {})


def get_usdt_balance():
    """Get USDT futures wallet balance."""
    try:
        wallets = get_wallet_details()
        for w in wallets:
            if w.get("currency_short_name") == config.COINDCX_MARGIN_CURRENCY:
                return float(w.get("balance", 0))
    except Exception as e:
        logger.error("Failed to get CoinDCX balance: %s", e)
    return 0.0


# ─── CLI Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    print("\n=== CoinDCX Futures API Test ===\n")

    # 1. Active instruments
    instruments = get_active_instruments()
    print(f"Active instruments: {len(instruments)}")
    if instruments:
        print(f"  First 5: {instruments[:5]}")

    # 2. Symbol conversion
    print(f"\n  BTCUSDT → {to_coindcx_pair('BTCUSDT')}")
    print(f"  B-BTC_USDT → {from_coindcx_pair('B-BTC_USDT')}")

    # 3. Current price
    btc_price = get_current_price("B-BTC_USDT")
    print(f"\n  BTC Price: ${btc_price}")

    # 4. Candlesticks
    df = get_candlesticks("B-BTC_USDT", "15m", limit=10)
    if df is not None:
        print(f"\n  BTC 15m candles: {len(df)} rows")
        print(df.tail(3).to_string())

    # 5. Wallet balance
    try:
        balance = get_usdt_balance()
        print(f"\n  Wallet Balance: {balance} {config.COINDCX_MARGIN_CURRENCY}")
    except Exception as e:
        print(f"\n  Wallet Balance Error: {e}")

    print("\n=== Test Complete ===\n")
