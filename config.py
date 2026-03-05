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
PAPER_TRADE = True
ENGINE_USER_ID = "cmmbvbo2l0000j1xo3rqvkfhz"  # Default user for engine trades (admin)
PAPER_MAX_CAPITAL = 2500       # Total portfolio: 25 slots × $100/trade

# ─── CoinDCX API (used for LIVE trading) ────────────────────────────────────────
COINDCX_API_KEY = os.getenv("COINDCX_API_KEY", "")
COINDCX_API_SECRET = os.getenv("COINDCX_API_SECRET", "")
COINDCX_BASE_URL = "https://api.coindcx.com"
COINDCX_PUBLIC_URL = "https://public.coindcx.com"
COINDCX_MARGIN_CURRENCY = os.getenv("COINDCX_MARGIN_CURRENCY", "USDT")
EXCHANGE_LIVE = os.getenv("EXCHANGE_LIVE", "")  # "coindcx" or "binance"
BINANCE_FUTURES_TESTNET = os.getenv("BINANCE_FUTURES_TESTNET", "true").lower() == "true"

# ─── Exchange Fees ──────────────────────────────────────────────────────────────
TAKER_FEE = 0.0005            # 0.05% taker per leg (Binance & CoinDCX)
MAKER_FEE = 0.0002            # 0.02% maker per leg

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

# ─── Leverage Tiers ─────────────────────────────────────────────────────────────
LEVERAGE_HIGH = 35       # Confidence > 95%
LEVERAGE_MODERATE = 25   # Confidence 91–95%
LEVERAGE_LOW = 15        # Confidence 85–90%
LEVERAGE_NONE = 1        # Observation mode

# ─── Confidence Thresholds ──────────────────────────────────────────────────────
CONFIDENCE_HIGH = 0.99   # Above 99% → 35x  (optimized from 0.95)
CONFIDENCE_MEDIUM = 0.96 # 96–99% → 25x  (optimized from 0.91)
CONFIDENCE_LOW = 0.92    # 92–96% → 15x  (optimized from 0.85, below 92% = no deploy)

# ─── Strategy Profiles ──────────────────────────────────────────────────────────
# Each profile defines its own conviction thresholds, leverage mapping, risk params.
# The HMM analysis runs ONCE; each profile then applies its own lens to the raw scores.
STRATEGY_PROFILES = {
    "standard": {
        "label": "SM-Standard",
        "confidence_min": 0.92,
        "confidence_tiers": {0.99: 35, 0.96: 25, 0.92: 15},
        "max_positions": 15,
        "capital_per_trade": 100,
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 3.0,
        "trailing_sl": True,
        "multi_target": True,
        "mt_rr_ratio": 5,
    },
    "conservative": {
        "label": "SM-Conservative",
        "confidence_min": 0.97,
        "confidence_tiers": {0.99: 10, 0.97: 5},
        "max_positions": 5,
        "capital_per_trade": 100,
        "atr_sl_mult": 1.0,
        "atr_tp_mult": 2.0,
        "trailing_sl": True,
        "multi_target": False,
        "mt_rr_ratio": 3,
    },
}
ACTIVE_PROFILES = list(STRATEGY_PROFILES.keys())  # Which profiles to run

# ─── Risk Management ────────────────────────────────────────────────────────────
RISK_PER_TRADE = 0.04
KILL_SWITCH_DRAWDOWN = 0.10   # Pause bot if 10% drawdown in 24h
MAX_LOSS_PER_TRADE_PCT = -15     # Hard max-loss per trade – flat for all leverage
MIN_HOLD_MINUTES = 30         # Minimum hold time before regime-change exits
DEFAULT_QUANTITY = 0.002      # BTC quantity (overridden by position sizer)
MARGIN_TYPE = "ISOLATED"      # Never use CROSS for high leverage

# ─── Stop Loss / Take Profit ────────────────────────────────────────────────────
ATR_SL_MULTIPLIER = 1.5       # SL = ATR * multiplier (DEFAULT, used as fallback)
ATR_TP_MULTIPLIER = 3.0       # TP = ATR * multiplier (DEFAULT, used as fallback)
SLIPPAGE_BUFFER = 0.0005      # 0.05% slippage estimate

def get_atr_multipliers(leverage=1):
    """Return (sl_mult, tp_mult) adjusted for leverage.
    Higher leverage → tighter SL/TP to keep effective portfolio risk consistent.
    Always maintains 1:2 risk-reward ratio."""
    if leverage >= 50:
        return (0.5, 1.0)
    elif leverage >= 25:
        return (0.7, 1.4)
    elif leverage >= 10:
        return (1.0, 2.0)
    elif leverage >= 5:
        return (1.2, 2.4)
    else:  # 1-4x
        return (ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER)

# ─── Trailing SL / TP ──────────────────────────────────────────────────────────
TRAILING_SL_ENABLED = True
TRAILING_SL_ACTIVATION_ATR = 1.0     # Start trailing after price moves 1×ATR in favor
TRAILING_SL_DISTANCE_ATR = 1.0       # Trail distance: SL stays 1×ATR behind peak price
TRAILING_TP_ENABLED = False       # Disabled — replaced by multi-target T1/T2/T3
TRAILING_TP_ACTIVATION_PCT = 0.75    # (legacy, unused when MT enabled)
TRAILING_TP_EXTENSION_ATR = 1.5      # (legacy, unused when MT enabled)
TRAILING_TP_MAX_EXTENSIONS = 3       # (legacy, unused when MT enabled)

# ─── Multi-Target Partial Profit Booking (0304_v1) ─────────────────────────────
MULTI_TARGET_ENABLED = True
MT_RR_RATIO = 5                  # SL : T3 = 1:5
MT_T1_FRAC = 0.333               # T1 at 33.3% of T3 distance (Even spacing)
MT_T2_FRAC = 0.666               # T2 at 66.6% of T3 distance
MT_T1_BOOK_PCT = 0.25            # Book 25% of original qty at T1
MT_T2_BOOK_PCT = 0.50            # Book 50% of remaining qty at T2

# ─── Capital Protection (Profit Lock) ──────────────────────────────────────────
CAPITAL_PROTECT_ENABLED = False      # Disabled — Phase 3 proved it hurts multi-target perf
CAPITAL_PROTECT_TRIGGER_PCT = 10.0   # Activate when leveraged P&L ≥ 10%
CAPITAL_PROTECT_LOCK_PCT = 4.0       # Move SL to lock in +4% profit above/below entry

# ─── Volatility Filter ─────────────────────────────────────────────────────────
VOL_FILTER_ENABLED = True
VOL_MIN_ATR_PCT = 0.003
VOL_MAX_ATR_PCT = 0.06

# ─── Sideways Strategy ──────────────────────────────────────────────────────────
BB_LENGTH = 20
BB_STD = 2.0
RSI_LENGTH = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
SIDEWAYS_POSITION_REDUCTION = 0.30  # 30% smaller positions in chop

# ─── Bot Loop ────────────────────────────────────────────────────────────────────
LOOP_INTERVAL_SECONDS = 60        # 1-minute heartbeat (checks commands, updates state)
ANALYSIS_INTERVAL_SECONDS = 300   # 5-minute full analysis cycle (HMM scan, trades)
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
TELEGRAM_ENABLED = False
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
SENTIMENT_ENABLED           = True
SENTIMENT_CACHE_MINUTES     = 15       # Cache per-coin results for N minutes
SENTIMENT_WINDOW_HOURS      = 4        # Look back N hours of articles
SENTIMENT_MIN_ARTICLES      = 3        # Minimum articles to compute a score
SENTIMENT_VETO_THRESHOLD    = -0.65    # Hard veto gate (fast path before conviction)
SENTIMENT_STRONG_POS        = 0.45     # Threshold for "strongly positive" label
SENTIMENT_USE_FINBERT       = True     # Use FinBERT in addition to VADER (requires transformers)
SENTIMENT_VADER_WEIGHT      = 0.4      # VADER contribution when blending with FinBERT
SENTIMENT_FINBERT_WEIGHT    = 0.6      # FinBERT contribution when blending with VADER
SENTIMENT_DEDUPE_URL_LIMIT  = 5000     # Max tracked URLs before trimming seen-url set
SENTIMENT_DEDUPE_URL_TRIM   = 2000     # Keep last N URLs when trimming
SENTIMENT_CONFIDENCE_N_SCALE = 20      # N articles = 70% of confidence score
SENTIMENT_SOURCE_DIV_SCALE   = 3       # N unique sources = 30% of confidence score
SENTIMENT_RATE_LIMIT_SLEEP  = 1.2      # Seconds between paginated API requests (sources)
CRYPTOPANIC_API_KEY      = os.getenv("CRYPTOPANIC_API_KEY", "")
REDDIT_CLIENT_ID         = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET     = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT        = "RegimeMaster/1.0"
SENTIMENT_RSS_FEEDS      = [
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://theblock.co/rss.xml",
    "https://bitcoinmagazine.com/.rss/full/",
]
SENTIMENT_LOG_FILE       = os.path.join(DATA_DIR, "sentiment_log.csv")

# ─── Order Flow Engine ────────────────────────────────────────────────────────
ORDERFLOW_ENABLED          = True
ORDERFLOW_CACHE_SECONDS    = 60        # Cache orderflow snapshot per coin (60s)
ORDERFLOW_DEPTH_LEVELS     = 20        # Number of L2 order book levels to fetch
ORDERFLOW_WALL_THRESHOLD   = 3.0       # A level is a "wall" if it is N× the avg level size
ORDERFLOW_LOOKBACK_BARS    = 4         # Bars of 15m taker data to sum for cumulative delta
ORDERFLOW_LS_ENABLED       = True      # Include L/S ratio from Binance futures
ORDERFLOW_LARGE_ORDER_USD  = 50_000    # USD threshold to flag a single order as "large"

# ─── Conviction Score Weights (must sum to 100) ───────────────────────────────
CONVICTION_WEIGHT_HMM       = 22   # HMM regime confidence
CONVICTION_WEIGHT_BTC_MACRO = 18   # BTC macro regime alignment
CONVICTION_WEIGHT_FUNDING   = 12   # Funding rate
CONVICTION_WEIGHT_SR_VWAP   = 10   # Support/Resistance + VWAP position
CONVICTION_WEIGHT_OI        = 8    # Open Interest change
CONVICTION_WEIGHT_VOL       = 5    # Volatility quality
CONVICTION_WEIGHT_SENTIMENT = 15   # Social/news sentiment
CONVICTION_WEIGHT_ORDERFLOW = 10   # Order book flow

# ─── Conviction Score: Leverage Bands ────────────────────────────────────────
CONVICTION_MIN_TRADE   = 40   # Below this → no trade (leverage = 0)
CONVICTION_BAND_LOW    = 55   # 40–54  → 10x leverage
CONVICTION_BAND_MED    = 70   # 55–69  → 15x leverage
CONVICTION_BAND_HIGH   = 85   # 70–84  → 25x; 85+ → 35x leverage

# ─── Conviction Score: Penalties ─────────────────────────────────────────────
CONVICTION_CRASH_PENALTY           = 10   # Macro crash regime hard penalty
CONVICTION_MACRO_FIGHT_PENALTY     = 8    # Trading against macro direction
CONVICTION_FUNDING_PENALTY         = 4    # Crowded funding rate penalty
CONVICTION_OI_PENALTY              = 3    # Adverse OI move penalty
CONVICTION_SENTIMENT_STRONG_PENALTY = 12  # Strong negative news penalty
CONVICTION_SENTIMENT_MILD_PENALTY  = 4    # Mild negative sentiment penalty
CONVICTION_SENTIMENT_NEG_THRESHOLD = -0.20  # Score below this = mild negative
CONVICTION_FLOW_MILD_PENALTY       = 3    # Mild opposing order flow
CONVICTION_FLOW_STRONG_PENALTY     = 7    # Strong opposing order flow

# ─── Conviction Score: HMM Confidence Tiers ──────────────────────────────────
HMM_CONF_TIER_HIGH     = 0.97   # Full weight (100%)
HMM_CONF_TIER_MED_HIGH = 0.94   # 85% weight
HMM_CONF_TIER_MED      = 0.90   # 65% weight
HMM_CONF_TIER_LOW      = 0.85   # 40% weight (below = no contribution)

# ─── Conviction Score: Funding Rate Thresholds ───────────────────────────────
FUNDING_NEG_STRONG =  -0.0001  # Below: longs paid → BUY favorable (full score)
FUNDING_POS_MED    =   0.0003  # Above: crowded longs → BUY penalty
FUNDING_POS_STRONG =   0.0001  # Above: shorts paid → SELL favorable (full score)
FUNDING_NEG_MED    =  -0.0003  # Below: crowded shorts → SELL penalty

# ─── Conviction Score: OI Change Thresholds ──────────────────────────────────
OI_CHANGE_HIGH     =  0.03   # > 3%: strong fresh positioning
OI_CHANGE_MED      =  0.01   # > 1%: moderate positioning
OI_CHANGE_NEG_HIGH = -0.03   # < -3%: OI falling (short-covering risk for BUY)
OI_CHANGE_NEG_MED  = -0.01   # < -1%: mild OI contraction

# ─── CoinDCX Execution ───────────────────────────────────────────────────────
COINDCX_MIN_NOTIONAL      = 120.0   # Minimum order size in USD
COINDCX_ORDER_SETTLE_SLEEP = 0.5    # Seconds to wait after placing order

# ─── Coin Scanner ────────────────────────────────────────────────────────────
SCANNER_RATE_LIMIT_SLEEP = 1.0   # Seconds between API calls to avoid rate limiting

