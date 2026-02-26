"""
Project Regime-Master — Sentiment Engine
Multi-source news/social sentiment scorer using VADER + FinBERT.

Pipeline:
  1. Fetch articles from all enabled sources (CryptoPanic, RSS, Reddit, Fear&Greed)
  2. Deduplicate by URL; filter to SENTIMENT_WINDOW_HOURS
  3. Score each article with VADER (fast, always) + FinBERT (optional, for news)
  4. Detect ALERT events (hack / exploit / scam / regulatory keywords)
  5. Compute weighted composite signal per coin (-1 to +1)
  6. Cache results for SENTIMENT_CACHE_MINUTES to stay within API rate limits
  7. Log each refresh to data/sentiment_log.csv for backtesting

Usage (in main bot loop):
    from sentiment_engine import SentimentEngine
    engine = SentimentEngine()
    signal = engine.get_coin_sentiment("BTCUSDT")  # or "BTC"
    print(signal.score, signal.alert)

    # Pass directly into conviction scorer:
    conviction = RiskManager.compute_conviction_score(
        ..., sentiment_score=signal.score
    )
"""
from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import config
from sentiment_sources import (
    ArticleItem,
    CryptoPanicSource,
    RSSNewsSource,
    RedditSource,
    FearGreedSource,
    FearGreedReading,
    COIN_KEYWORDS,
)

logger = logging.getLogger("SentimentEngine")

# ─── Alert keywords — any match immediately flags an article ─────────────────
_ALERT_KEYWORDS = [
    "hack", "hacked", "exploit", "exploited", "breach", "breached",
    "stolen", "theft", "rug pull", "rug pulled", "exit scam", "scam",
    "insolvency", "insolvent", "bankrupt", "bankruptcy",
    "sec charges", "sec lawsuit", "sec settlement",
    "ban", "banned", "crackdown", "shutdown", "delisted", "delisting",
    "flash loan attack", "flash crash", "51% attack",
    "investigation", "indictment", "arrested",
]

# Source quality weights — higher = more weight in composite score
_SOURCE_WEIGHTS = {
    "CryptoPanic": 0.85,
    "RSS":          0.80,
    "Reddit":       0.55,  # noisier; lower weight
}

# ─── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class SentimentSignal:
    """Composite sentiment signal for one coin."""
    coin: str
    score: float                        # weighted average, -1 to +1
    confidence: float                   # 0-1; based on article count & source diversity
    buzz_velocity: int                  # articles per hour in the window
    momentum: float                     # change in score vs 1 hour ago (-1 to +1)
    alert: bool                         # True if hack/exploit/scam detected
    alert_reason: str                   # Description of the alert (if any)
    fear_greed: Optional[float]         # Overall market Fear & Greed (-1 to +1)
    sources_used: List[str] = field(default_factory=list)
    article_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def effective_score(self) -> float:
        """Score to use downstream; -1.0 if alert, else score."""
        return -1.0 if self.alert else self.score

    def __repr__(self) -> str:
        alert_str = f" ALERT:{self.alert_reason}" if self.alert else ""
        return (f"<SentimentSignal {self.coin} score={self.score:.2f} "
                f"conf={self.confidence:.2f} buzz={self.buzz_velocity}/h"
                f"{alert_str}>")


# ─── VADER loader ─────────────────────────────────────────────────────────────

def _load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        logger.warning("vaderSentiment not installed. Run: pip install vaderSentiment")
        return None


# ─── FinBERT loader (lazy, optional) ─────────────────────────────────────────

_finbert_pipeline = None
_finbert_loaded   = False

def _load_finbert():
    """Lazy-load FinBERT.  Returns pipeline or None if unavailable."""
    global _finbert_pipeline, _finbert_loaded
    if _finbert_loaded:
        return _finbert_pipeline
    _finbert_loaded = True
    if not config.SENTIMENT_USE_FINBERT:
        return None
    try:
        from transformers import pipeline
        logger.info("Loading FinBERT model (first run downloads ~400 MB)...")
        _finbert_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded.")
    except Exception as e:
        logger.warning("FinBERT unavailable: %s  (VADER-only mode)", e)
        _finbert_pipeline = None
    return _finbert_pipeline


# ─── Main engine ─────────────────────────────────────────────────────────────

class SentimentEngine:
    """
    Orchestrates all sentiment sources, NLP scoring, caching, and logging.

    Thread-safe for single-process use (the trading bot is single-threaded).
    """

    def __init__(self):
        self._vader      = _load_vader()
        self._sources    = self._build_sources()
        self._fg_source  = FearGreedSource()

        # Cache: coin → (SentimentSignal, fetched_at)
        self._cache: Dict[str, Tuple[SentimentSignal, datetime]] = {}
        # Rolling 1-hour history for momentum: coin → (score, recorded_at)
        self._prev_scores: Dict[str, Tuple[float, datetime]] = {}
        # Deduplicated URL set across recent fetch
        self._seen_urls: set = set()
        # Global Fear & Greed cache
        self._fg_cache: Optional[Tuple[FearGreedReading, datetime]] = None

    # ─── Public API ──────────────────────────────────────────────────────────

    def get_coin_sentiment(self, symbol: str) -> SentimentSignal:
        """
        Return sentiment for a trading symbol (e.g. 'BTCUSDT' or 'BTC').
        Results are cached for SENTIMENT_CACHE_MINUTES.
        """
        coin = self._normalise(symbol)
        cached = self._cache.get(coin)
        if cached:
            signal, fetched_at = cached
            age_m = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 60
            if age_m < config.SENTIMENT_CACHE_MINUTES:
                return signal

        signal = self._build_signal(coin)
        self._cache[coin] = (signal, datetime.now(timezone.utc))
        self._log_signal(signal)
        return signal

    def get_market_sentiment(self) -> dict:
        """
        High-level market overview: BTC, ETH, global Fear & Greed.
        Returns dict with keys 'btc', 'eth', 'fear_greed', 'market_bias'.
        """
        btc = self.get_coin_sentiment("BTC")
        eth = self.get_coin_sentiment("ETH")
        fg  = self._get_fear_greed()

        market_bias = (btc.effective_score * 0.5 + eth.effective_score * 0.3
                       + (fg.normalized if fg else 0) * 0.2)

        return {
            "btc":         btc,
            "eth":         eth,
            "fear_greed":  fg,
            "market_bias": round(market_bias, 3),
        }

    def prefetch(self, symbols: List[str]):
        """Warm the cache for multiple coins (call once per analysis cycle)."""
        for sym in symbols:
            self.get_coin_sentiment(sym)

    # ─── Internal: building signals ──────────────────────────────────────────

    def _build_signal(self, coin: str) -> SentimentSignal:
        since = datetime.now(timezone.utc) - timedelta(hours=config.SENTIMENT_WINDOW_HOURS)
        articles = self._fetch_all(coin, since)
        fg = self._get_fear_greed()
        return self._compute_signal(coin, articles, fg)

    def _fetch_all(self, coin: str, since: datetime) -> List[ArticleItem]:
        """Collect from all sources, deduplicate, filter to window."""
        coins_to_query = [coin] if coin != "CRYPTO" else list(COIN_KEYWORDS.keys())[:20]
        all_items: List[ArticleItem] = []

        for source in self._sources:
            try:
                items = source.fetch(coins_to_query, since)
                all_items.extend(items)
            except Exception as e:
                logger.debug("[%s] fetch error: %s", source.name, e)

        # Deduplicate by URL
        unique: List[ArticleItem] = []
        for item in all_items:
            if item.url and item.url in self._seen_urls:
                continue
            if item.url:
                self._seen_urls.add(item.url)
            unique.append(item)

        # Keep seen_urls from growing unboundedly
        if len(self._seen_urls) > 5000:
            self._seen_urls = set(list(self._seen_urls)[-2000:])

        # Only keep articles relevant to the requested coin
        relevant = [
            a for a in unique
            if coin in a.coins or "CRYPTO" in a.coins
        ]
        return relevant

    def _compute_signal(self, coin: str, articles: List[ArticleItem],
                        fg: Optional[FearGreedReading]) -> SentimentSignal:
        if not articles:
            # No data — return neutral with low confidence
            prev_score, _ = self._get_prev_score(coin)
            return SentimentSignal(
                coin=coin, score=0.0, confidence=0.1,
                buzz_velocity=0, momentum=0.0,
                alert=False, alert_reason="",
                fear_greed=fg.normalized if fg else None,
                sources_used=[], article_count=0,
            )

        # Score each article
        scored: List[Tuple[float, float, str]] = []  # (score, weight, source)
        alert_detected = False
        alert_reasons: List[str] = []

        for art in articles:
            # Alert check first — full-text scan
            alert_match = self._check_alert(art.text)
            if alert_match:
                alert_detected = True
                alert_reasons.append(alert_match)

            nlp_score = self._score_text(art.text, use_finbert="RSS" in art.source)
            source_key = art.source.split(":")[0]
            source_w = _SOURCE_WEIGHTS.get(source_key, 0.6)
            weight = art.importance * source_w
            scored.append((nlp_score, weight, art.source))

        # Weighted average
        total_w = sum(w for _, w, _ in scored)
        composite = sum(s * w for s, w, _ in scored) / total_w if total_w > 0 else 0.0
        composite = max(-1.0, min(1.0, composite))

        # Confidence: rises with article count and source diversity
        n = len(articles)
        source_diversity = len(set(s.split(":")[0] for _, _, s in scored))
        confidence = min(1.0, (min(n, 20) / 20) * 0.7 + (min(source_diversity, 3) / 3) * 0.3)
        if n < config.SENTIMENT_MIN_ARTICLES:
            confidence *= 0.5

        # Buzz velocity (articles per hour)
        buzz = int(n / config.SENTIMENT_WINDOW_HOURS)

        # Momentum vs previous reading
        prev_score, prev_time = self._get_prev_score(coin)
        momentum = composite - prev_score
        momentum = max(-1.0, min(1.0, momentum))

        # Store this score for next momentum calc
        self._prev_scores[coin] = (composite, datetime.now(timezone.utc))

        sources_used = list(set(s.split(":")[0] for _, _, s in scored))

        # Include Fear & Greed as a mild global adjustment (not per-coin)
        fg_score = fg.normalized if fg else 0.0

        return SentimentSignal(
            coin=coin,
            score=composite,
            confidence=confidence,
            buzz_velocity=buzz,
            momentum=momentum,
            alert=alert_detected,
            alert_reason="; ".join(set(alert_reasons)),
            fear_greed=fg_score,
            sources_used=sources_used,
            article_count=n,
        )

    # ─── NLP Scoring ─────────────────────────────────────────────────────────

    def _score_text(self, text: str, use_finbert: bool = False) -> float:
        """
        Score text sentiment → float in [-1, +1].
        VADER is always run (fast, handles social media slang).
        FinBERT is layered on top for news articles (higher accuracy for formal text).
        """
        if not text.strip():
            return 0.0

        vader_score = self._vader_score(text)

        if use_finbert:
            fb = _load_finbert()
            if fb is not None:
                try:
                    result = fb(text[:512])[0]
                    label = result["label"].lower()
                    conf  = float(result["score"])
                    fb_score = conf if label == "positive" else (-conf if label == "negative" else 0.0)
                    # Blend: 40% VADER (social cues) + 60% FinBERT (financial context)
                    return 0.4 * vader_score + 0.6 * fb_score
                except Exception:
                    pass

        return vader_score

    def _vader_score(self, text: str) -> float:
        """VADER compound score normalised to [-1, +1]."""
        if self._vader is None:
            return 0.0
        try:
            return float(self._vader.polarity_scores(text)["compound"])
        except Exception:
            return 0.0

    # ─── Alert detection ─────────────────────────────────────────────────────

    @staticmethod
    def _check_alert(text: str) -> Optional[str]:
        """Return first matched alert keyword, or None."""
        text_lower = text.lower()
        for kw in _ALERT_KEYWORDS:
            if kw in text_lower:
                return kw
        return None

    # ─── Fear & Greed ─────────────────────────────────────────────────────────

    def _get_fear_greed(self) -> Optional[FearGreedReading]:
        """Return cached Fear & Greed reading (refreshes every 6 hours)."""
        if self._fg_cache:
            reading, fetched = self._fg_cache
            age_h = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
            if age_h < 6:
                return reading
        reading = self._fg_source.fetch()
        if reading:
            self._fg_cache = (reading, datetime.now(timezone.utc))
        return reading

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(symbol: str) -> str:
        """'BTCUSDT' → 'BTC',  'btc' → 'BTC'."""
        return symbol.upper().replace("USDT", "").replace("BUSD", "").replace("PERP", "")

    def _get_prev_score(self, coin: str) -> Tuple[float, Optional[datetime]]:
        entry = self._prev_scores.get(coin)
        if entry is None:
            return 0.0, None
        score, ts = entry
        # Discard if older than 2 hours (stale momentum)
        if (datetime.now(timezone.utc) - ts).total_seconds() > 7200:
            return 0.0, None
        return score, ts

    def _build_sources(self) -> list:
        sources = [RSSNewsSource()]
        sources.append(CryptoPanicSource())   # works without a key (limited)
        sources.append(RedditSource())         # falls back to public JSON without key
        return sources

    # ─── CSV logging ─────────────────────────────────────────────────────────

    def _log_signal(self, sig: SentimentSignal):
        """Append signal to sentiment_log.csv for backtesting analysis."""
        try:
            path = config.SENTIMENT_LOG_FILE
            is_new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow([
                        "timestamp", "coin", "score", "confidence",
                        "buzz_velocity", "momentum", "alert", "alert_reason",
                        "fear_greed", "article_count", "sources",
                    ])
                writer.writerow([
                    sig.timestamp.isoformat(),
                    sig.coin,
                    round(sig.score, 4),
                    round(sig.confidence, 4),
                    sig.buzz_velocity,
                    round(sig.momentum, 4),
                    sig.alert,
                    sig.alert_reason,
                    round(sig.fear_greed, 4) if sig.fear_greed is not None else "",
                    sig.article_count,
                    "|".join(sig.sources_used),
                ])
        except Exception as e:
            logger.debug("Sentiment log write failed: %s", e)


# ─── Module-level singleton (shared across the bot) ─────────────────────────

_engine: Optional[SentimentEngine] = None


def get_engine() -> SentimentEngine:
    """Return module-level singleton SentimentEngine (lazy-init)."""
    global _engine
    if _engine is None:
        _engine = SentimentEngine()
    return _engine


# ─── Quick CLI test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s — %(message)s")
    coins = sys.argv[1:] if len(sys.argv) > 1 else ["BTC", "ETH", "SOL"]
    engine = SentimentEngine()
    print(f"\n{'='*60}")
    print("SENTIMENT ENGINE — Live Test")
    print(f"{'='*60}")

    market = engine.get_market_sentiment()
    fg = market["fear_greed"]
    if fg:
        print(f"\nMarket Fear & Greed: {fg.score}/100 — {fg.label} ({fg.normalized:+.2f})")
    print(f"Market Bias (BTC+ETH+F&G): {market['market_bias']:+.3f}\n")

    for coin in coins:
        sig = engine.get_coin_sentiment(coin)
        alert_str = f"  ⚠ ALERT: {sig.alert_reason}" if sig.alert else ""
        print(f"{coin:6s}  score={sig.score:+.3f}  conf={sig.confidence:.2f}"
              f"  buzz={sig.buzz_velocity}/h  mom={sig.momentum:+.3f}"
              f"  articles={sig.article_count}  sources={sig.sources_used}"
              f"{alert_str}")
    print()
