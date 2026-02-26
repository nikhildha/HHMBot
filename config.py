"""
Project Regime-Master — Central Configuration
All settings, thresholds, and constants live here.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Binance API (used for PAPER trading) ────────────────────────────────────────
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TESTNET = os.getenv("TESTNET", "true").lower() == "true"
PAPER_TRADE = os.getenv("PAPER_TRADE", "true").lower() == "true"
PAPER_MAX_CAPITAL = 1500       # Total portfolio: 15 slots × $100/trade

# ─── CoinDCX API (used for LIVE trading) ────────────────────────────────────────
COINDCX_API_KEY = os.getenv("COINDCX_API_KEY", "")
COINDCX_API_SECRET = os.getenv("COINDCX_API_SECRET", "")
COINDCX_BASE_URL = "https://api.coindcx.com"
COINDCX_PUBLIC_URL = "https://public.coindcx.com"
COINDCX_MARGIN_CURRENCY = os.getenv("COINDCX_MARGIN_CURRENCY", "USDT")
EXCHANGE_LIVE = os.getenv("EXCHANGE_LIVE", "coindcx")  # Live exchange: coindcx

# ─── CoinDCX Fees ───────────────────────────────────────────────────────────────
TAKER_FEE_COINDCX = 0.0005    # 0.05% per leg
MAKER_FEE_COINDCX = 0.0002    # 0.02% per leg

# ─── Trading Symbols ────────────────────────────────────────────────────────────
PRIMARY_SYMBOL = "BTCUSDT"
SECONDARY_SYMBOLS = ["ETHUSDT"]

# ─── Timeframes ─────────────────────────────────────────────────────────────────
TIMEFRAME_EXECUTION = "15m"   # Entry / exit timing (optimized from 5m)
TIMEFRAME_CONFIRMATION = "1h" # Trend confirmation
TIMEFRAME_MACRO = "4h"        # Macro regime

# ─── HMM Brain ──────────────────────────────────────────────────────────────────
HMM_N_STATES = 4              # Bull, Bear, Chop, Crash
HMM_COVARIANCE = "full"       # Optimized: captures cross-feature correlations
HMM_ITERATIONS = 100
HMM_LOOKBACK = 250            # Candles used for training (reduced for speed)
HMM_RETRAIN_HOURS = 24        # Retrain every N hours

# ─── Regime Labels (assigned post-training by sorting mean returns) ──────────
REGIME_BULL = 0
REGIME_BEAR = 1
REGIME_CHOP = 2
REGIME_CRASH = 3

REGIME_NAMES = {
    REGIME_BULL:  "BULLISH",
    REGIME_BEAR:  "BEARISH",
    REGIME_CHOP:  "SIDEWAYS/CHOP",
    REGIME_CRASH: "CRASH/PANIC",
}

# ─── Leverage Tiers (Legacy — conviction system uses continuous 10x-35x) ────────
LEVERAGE_HIGH = 35       # Confidence > 99% → max conviction
LEVERAGE_MODERATE = 20   # Confidence 96–99%
LEVERAGE_LOW = 10        # Confidence 92–96% — minimum deployment
LEVERAGE_NONE = 1        # Observation mode

# ─── Confidence Thresholds ──────────────────────────────────────────────────────
CONFIDENCE_HIGH = 0.99   # Above 99% → 15x
CONFIDENCE_MEDIUM = 0.96 # 96–99% → 10x
CONFIDENCE_LOW = 0.92    # 92–96% → 5x, below 92% = no deploy

# ─── Counter-Trend Penalty ──────────────────────────────────────────────────────
# Require higher confidence for trades against the prevailing regime
# e.g., LONGs in a bear market or SHORTs in a bull market
COUNTER_TREND_CONF_PENALTY = 0.04  # Subtract 4% from confidence for counter-trend trades
SKIP_CHOP_TRADES = True              # Skip all SIDEWAYS/CHOP regime trades

# ─── Risk Management ────────────────────────────────────────────────────────────
RISK_PER_TRADE = 0.02         # 2% risk per trade (reduced from 4% for multi-coin)
KILL_SWITCH_DRAWDOWN = 0.10   # Pause bot if 10% drawdown in 24h
MAX_LOSS_PER_TRADE_PCT = -30
MIN_HOLD_MINUTES = 30         # Minimum hold time before regime-change exits
DEFAULT_QUANTITY = 0.002      # BTC quantity (overridden by position sizer)
MARGIN_TYPE = "ISOLATED"      # Never use CROSS for high leverage

# ─── Stop Loss / Take Profit ────────────────────────────────────────────────────
ATR_SL_MULTIPLIER = 1.5       # SL = ATR * multiplier (DEFAULT, used as fallback)
ATR_TP_MULTIPLIER = 3.0       # TP = ATR * multiplier (DEFAULT, used as fallback)
SLIPPAGE_BUFFER = 0.0005      # 0.05% slippage estimate

# SL width multiplier — wider SL reduces whipsaw stop-outs
SL_WIDTH_FACTOR = 1.5  # Widen SL by 50% to avoid premature stop-outs

def get_atr_multipliers(leverage=1):
    """Return (sl_mult, tp_mult) adjusted for leverage.
    Higher leverage → tighter SL/TP to keep effective portfolio risk consistent.
    SL is widened by SL_WIDTH_FACTOR to reduce whipsaw stop-outs.
    Always maintains ~1:2 risk-reward ratio."""
    if leverage >= 50:
        sl, tp = 0.5, 1.0
    elif leverage >= 25:
        sl, tp = 0.7, 1.4
    elif leverage >= 10:
        sl, tp = 1.0, 2.0
    elif leverage >= 5:
        sl, tp = 1.2, 2.4
    else:  # 1-4x
        sl, tp = ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER
    return (sl * SL_WIDTH_FACTOR, tp)  # Wider SL, same TP

# ─── Trailing SL / TP ──────────────────────────────────────────────────────────
TRAILING_SL_ENABLED = True
TRAILING_SL_ACTIVATION_ATR = 1.0     # Start trailing after price moves 1×ATR in favor
TRAILING_SL_DISTANCE_ATR = 1.0       # Trail distance: SL stays 1×ATR behind peak price
TRAILING_TP_ENABLED = True
TRAILING_TP_ACTIVATION_PCT = 0.75    # Start extending TP once 75% of TP distance reached
TRAILING_TP_EXTENSION_ATR = 1.5      # Extend TP by 1.5×ATR each time threshold is hit
TRAILING_TP_MAX_EXTENSIONS = 3       # Max TP extensions (prevents runaway)

# ─── Volatility Filter ─────────────────────────────────────────────────────────
VOL_FILTER_ENABLED = True
VOL_MIN_ATR_PCT = 0.003
VOL_MAX_ATR_PCT = 0.06

# ─── Fees ────────────────────────────────────────────────────────────────────────
TAKER_FEE = 0.0005            # 0.05% Binance futures taker per leg (0.1% round trip)
MAKER_FEE = 0.0002            # 0.02% Binance futures maker

# ─── Sideways Strategy ──────────────────────────────────────────────────────────
BB_LENGTH = 20
BB_STD = 2.0
RSI_LENGTH = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
SIDEWAYS_POSITION_REDUCTION = 0.30  # 30% smaller positions in chop

# ─── Bot Loop ────────────────────────────────────────────────────────────────────
LOOP_INTERVAL_SECONDS = 60        # 1-minute heartbeat (checks commands, updates state)
ANALYSIS_INTERVAL_SECONDS = 900   # 15-minute full analysis cycle (HMM scan, trades)
ERROR_RETRY_SECONDS = 60          # Retry after error

# ─── Multi-Coin Trading ──────────────────────────────────────────────────────────
MAX_CONCURRENT_POSITIONS = 15   # Max symbols traded at once
TOP_COINS_LIMIT = 15            # How many coins to scan (reduced from 50 for speed)
CAPITAL_PER_COIN_PCT = 0.05     # 5% of balance per coin (max 15 = 75% deployed)
SCAN_INTERVAL_CYCLES = 4        # Re-scan top coins every N analysis cycles (4 × 15m = 1h)
MULTI_COIN_MODE = True          # Enable multi-coin scanning

# ─── Telegram Notifications ──────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
TELEGRAM_NOTIFY_TRADES = os.getenv("TELEGRAM_NOTIFY_TRADES", "true").lower() == "true"
TELEGRAM_NOTIFY_ALERTS = os.getenv("TELEGRAM_NOTIFY_ALERTS", "true").lower() == "true"
TELEGRAM_NOTIFY_SUMMARY = os.getenv("TELEGRAM_NOTIFY_SUMMARY", "true").lower() == "true"

# ─── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRADE_LOG_FILE = os.path.join(DATA_DIR, "trade_log.csv")
STATE_FILE = os.path.join(DATA_DIR, "bot_state.json")
MULTI_STATE_FILE = os.path.join(DATA_DIR, "multi_bot_state.json")
COMMANDS_FILE = os.path.join(DATA_DIR, "commands.json")

os.makedirs(DATA_DIR, exist_ok=True)

# ─── Sentiment Engine ─────────────────────────────────────────────────────────
SENTIMENT_ENABLED        = os.getenv("SENTIMENT_ENABLED", "true").lower() == "true"
SENTIMENT_CACHE_MINUTES  = 15      # Refresh every 15 min (aligns with analysis cycle)
SENTIMENT_WINDOW_HOURS   = 4       # Look back N hours of articles when scoring
SENTIMENT_MIN_ARTICLES   = 2       # Min articles needed for a reliable signal
SENTIMENT_VETO_THRESHOLD = -0.65   # Skip trade when coin sentiment is below this
SENTIMENT_STRONG_POS     = 0.45    # "Strongly positive" threshold for max bonus
SENTIMENT_USE_FINBERT    = True    # Use FinBERT for news text (slower but more accurate)

# Sentiment API keys — set in .env (all optional; RSS + Fear&Greed work without keys)
CRYPTOPANIC_API_KEY  = os.getenv("CRYPTOPANIC_API_KEY",  "")   # cryptopanic.com/api/v1/posts
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID",     "")   # reddit.com/prefs/apps
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "HMMBOT/1.0 by SentimentBot")

# RSS feeds — always active, no key required
SENTIMENT_RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://www.theblock.co/rss.xml",
    "https://bitcoinmagazine.com/.rss/full/",
]

# ─── Conviction Score Weights — 7 factors, positive max sums to 100 ──────────
# Reduced existing weights by ~15% total to make room for the new sentiment factor.
CONVICTION_WEIGHT_HMM        = 25   # HMM model confidence       (was 30)
CONVICTION_WEIGHT_BTC_MACRO   = 20   # BTC macro alignment         (was 25)
CONVICTION_WEIGHT_FUNDING     = 12   # Funding rate crowding       (was 15)
CONVICTION_WEIGHT_SR_VWAP     = 12   # S/R + VWAP entry position  (was 15)
CONVICTION_WEIGHT_OI          =  8   # Open Interest momentum      (was 10)
CONVICTION_WEIGHT_VOLATILITY   =  5   # Volatility regime           (unchanged)
CONVICTION_WEIGHT_SENTIMENT   = 18   # Social/news sentiment       (NEW)

# Paths for sentiment data log (used by backtester and live engine)
SENTIMENT_LOG_FILE = os.path.join(DATA_DIR, "sentiment_log.csv")
