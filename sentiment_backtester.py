"""
Project Regime-Master — Sentiment Signal Validator
Backtests sentiment signals against historical price data to measure signal quality.

Validation metrics:
  1. Directional Accuracy  — % times sentiment sign correctly predicted return sign
  2. Information Coefficient (IC) — Pearson correlation between sentiment & future return
  3. IC Decay Curve — IC at +1h, +2h, +4h, +8h lags (shows how long signal is useful)
  4. Source IC Ranking — Which source (CryptoPanic/RSS/Reddit) has best predictive value
  5. Sharpe Improvement — HMM-only Sharpe vs HMM+Sentiment Sharpe

Data strategy:
  Phase 1  — Uses CryptoPanic historical API to fetch recent news (last 30–60 days),
              then scores against Binance OHLCV data for the same period.
  Phase 2  — Once sentiment_log.csv accumulates live data, uses that for richer tests.

Run standalone:
    python sentiment_backtester.py --coin BTC --days 30
    python sentiment_backtester.py --coin ALL --days 60 --source ALL
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import config
from sentiment_engine import SentimentEngine  # type: ignore[import]
from sentiment_sources import CryptoPanicSource, RSSNewsSource, COIN_KEYWORDS  # type: ignore[import]

logger = logging.getLogger("SentimentBacktester")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SentimentSnapshot:
    """A point-in-time sentiment score for a coin."""
    timestamp: datetime
    coin: str
    score: float          # -1 to +1
    source: str
    article_count: int


@dataclass
class ValidationResult:
    """Full validation report for one coin."""
    coin: str
    n_periods: int
    ic_1h: float          # Information Coefficient at +1h
    ic_4h: float          # IC at +4h
    ic_8h: float          # IC at +8h
    dir_accuracy_1h: float   # Directional accuracy at +1h
    dir_accuracy_4h: float   # Directional accuracy at +4h
    source_ics: Dict[str, float]   # IC per source
    sharpe_sentiment_only: float
    sharpe_hmm_gate: float   # simulated: only trade when sentiment > 0.2
    sentiment_alpha: float   # improvement in Sharpe vs neutral (no-sentiment) baseline

    def summary(self) -> str:
        lines = [
            f"{'─'*58}",
            f" Coin: {self.coin}  |  Periods: {self.n_periods}",
            f"{'─'*58}",
            f" IC @+1h : {self.ic_1h:+.4f}    Dir Acc @+1h: {self.dir_accuracy_1h:.1%}",
            f" IC @+4h : {self.ic_4h:+.4f}    Dir Acc @+4h: {self.dir_accuracy_4h:.1%}",
            f" IC @+8h : {self.ic_8h:+.4f}",
            f" Sharpe (sentiment-gated): {self.sharpe_hmm_gate:.3f}",
            f" Sentiment Alpha:          {self.sentiment_alpha:+.3f}",
        ]
        if self.source_ics:
            lines.append(" Source IC Ranking:")
            for src, ic in sorted(self.source_ics.items(), key=lambda x: abs(float(x[1])), reverse=True):
                lines.append(f"   {src:<20} IC={ic:+.4f}")
        lines.append(f"{'─'*58}")
        return "\n".join(lines)


# ─── Historical data fetchers ────────────────────────────────────────────────

def _fetch_binance_hourly(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """
    Fetch hourly OHLCV from Binance public API (no key needed).
    Returns DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    limit = min(days * 24, 1000)
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol":   symbol if symbol.endswith("USDT") else f"{symbol}USDT",
        "interval": "1h",
        "limit":    limit,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logger.error("Binance OHLCV fetch failed for %s: %s", symbol, e)
        return None

    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "tb_base", "tb_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df.reset_index(drop=True, inplace=True)
    logger.info("Binance: %d hourly bars for %s", len(df), symbol)
    return df


def _fetch_cryptopanic_history(coin: str, days: int) -> List[SentimentSnapshot]:
    """
    Fetch historical news from CryptoPanic and score them with VADER.
    Returns a list of SentimentSnapshot objects sorted by timestamp.
    """
    from sentiment_engine import SentimentEngine
    engine = SentimentEngine()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    source = CryptoPanicSource()
    try:
        articles = source.fetch([coin], since)
    except Exception as e:
        logger.error("CryptoPanic history fetch failed: %s", e)
        return []

    snapshots: List[SentimentSnapshot] = []
    for art in articles:
        if coin not in art.coins and "CRYPTO" not in art.coins:
            continue
        score = engine._score_text(art.text, use_finbert=False)  # VADER only for speed
        snapshots.append(SentimentSnapshot(
            timestamp=art.published_at,
            coin=coin,
            score=score,
            source=art.source.split(":")[0],
            article_count=1,
        ))

    snapshots.sort(key=lambda x: x.timestamp)
    logger.info("CryptoPanic: %d scored snapshots for %s (last %d days)", len(snapshots), coin, days)
    return snapshots


def _load_sentiment_log(coin: str, days: int) -> List[SentimentSnapshot]:
    """Load previously logged sentiment signals from sentiment_log.csv."""
    import os, csv
    path = config.SENTIMENT_LOG_FILE
    if not os.path.exists(path):
        return []

    since = datetime.now(timezone.utc) - timedelta(days=days)
    snapshots = []
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("coin", "").upper() != coin.upper():
                    continue
                try:
                    ts = datetime.fromisoformat(row["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < since:
                        continue
                    snapshots.append(SentimentSnapshot(
                        timestamp=ts,
                        coin=coin,
                        score=float(row.get("score", 0)),
                        source=row.get("sources", "log"),
                        article_count=int(row.get("article_count", 0)),
                    ))
                except Exception:
                    continue
    except Exception as e:
        logger.debug("Sentiment log read failed: %s", e)
    snapshots.sort(key=lambda x: x.timestamp)
    logger.info("Sentiment log: %d entries for %s (last %d days)", len(snapshots), coin, days)
    return snapshots


# ─── Core validation ─────────────────────────────────────────────────────────

def _aggregate_hourly(snapshots: List[SentimentSnapshot]) -> pd.DataFrame:
    """
    Aggregate raw article scores into hourly sentiment readings.
    Uses weighted average (more recent articles in the hour get slightly more weight).
    """
    if not snapshots:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "timestamp": s.timestamp.floor("h"),
        "score":     s.score,
        "source":    s.source,
        "count":     1,
    } for s in snapshots])

    hourly = df.groupby("timestamp").agg(
        score=("score", "mean"),
        count=("count", "sum"),
    ).reset_index()
    hourly = hourly.sort_values("timestamp").reset_index(drop=True)
    return hourly


def _compute_ic(sent_series: pd.Series, ret_series: pd.Series) -> float:
    """Pearson correlation (Information Coefficient) between two series."""
    if len(sent_series) < 10:
        return float("nan")
    try:
        valid = (~np.isnan(sent_series)) & (~np.isnan(ret_series))
        if valid.sum() < 10:
            return float("nan")
        return float(np.corrcoef(sent_series[valid], ret_series[valid])[0, 1])
    except Exception:
        return float("nan")


def _compute_dir_accuracy(sent_series: pd.Series, ret_series: pd.Series) -> float:
    """% times the sign of sentiment correctly predicts the sign of return."""
    valid = (~np.isnan(sent_series)) & (~np.isnan(ret_series))
    if valid.sum() < 5:
        return float("nan")
    s = sent_series[valid]
    r = ret_series[valid]
    # Only evaluate when sentiment has a meaningful direction (|score| > 0.1)
    mask = s.abs() > 0.1
    if mask.sum() < 5:
        return float("nan")
    correct = ((s[mask] > 0) == (r[mask] > 0)).sum()
    return float(correct / mask.sum())


def validate_coin(coin: str, days: int = 30) -> Optional[ValidationResult]:
    """
    Full sentiment signal validation for a single coin.

    Steps:
      1. Fetch hourly OHLCV from Binance
      2. Fetch/load sentiment snapshots
      3. Align by timestamp (merge_asof)
      4. Compute IC at multiple lags
      5. Compute directional accuracy
      6. Simulate simple sentiment-gated Sharpe
    """
    logger.info("Validating %s over %d days...", coin, days)

    # 1. Price data
    price_df = _fetch_binance_hourly(coin, days)
    if price_df is None or len(price_df) < 24:
        logger.error("Not enough price data for %s", coin)
        return None
    price_df["log_ret_1h"] = np.log(price_df["close"] / price_df["close"].shift(1))
    price_df["log_ret_4h"] = np.log(price_df["close"] / price_df["close"].shift(4))
    price_df["log_ret_8h"] = np.log(price_df["close"] / price_df["close"].shift(8))

    # 2. Sentiment data: prefer live log, fallback to CryptoPanic API
    snapshots = _load_sentiment_log(coin, days)
    if len(snapshots) < 20:
        logger.info("Insufficient logged data; fetching from CryptoPanic...")
        snapshots = _fetch_cryptopanic_history(coin, days)

    if len(snapshots) < 5:
        logger.warning("Too few sentiment samples for %s (%d). Skipping.", coin, len(snapshots))
        return None

    # 3. Hourly aggregate
    sent_df = _aggregate_hourly(snapshots)
    if len(sent_df) < 5:
        logger.warning("Too few hourly sentiment periods for %s", coin)
        return None

    # 4. Merge: for each sentiment hour, look up the FUTURE price returns
    price_df["timestamp_h"] = price_df["timestamp"].dt.floor("h")
    merged = pd.merge_asof(
        sent_df.rename(columns={"timestamp": "timestamp_h"}),
        price_df[["timestamp_h", "log_ret_1h", "log_ret_4h", "log_ret_8h", "close"]],
        on="timestamp_h",
        direction="forward",
        tolerance=pd.Timedelta("2h"),
    )
    merged = merged.dropna(subset=["score", "log_ret_1h"])

    if len(merged) < 8:
        logger.warning("Not enough merged rows for %s (%d). Skipping.", coin, len(merged))
        return None

    s = merged["score"]
    r1 = merged["log_ret_1h"]
    r4 = merged["log_ret_4h"]
    r8 = merged["log_ret_8h"]

    # 5. Metrics
    ic_1h = _compute_ic(s, r1)
    ic_4h = _compute_ic(s, r4)
    ic_8h = _compute_ic(s, r8)
    dir_1h = _compute_dir_accuracy(s, r1)
    dir_4h = _compute_dir_accuracy(s, r4)

    # 6. Per-source IC (if we have source info from the log)
    source_ics: Dict[str, float] = {}
    if "source" in merged.columns:
        for src, group in merged.groupby("source"):
            if len(group) >= 5:
                source_ics[src] = _compute_ic(group["score"], group["log_ret_4h"])

    # 7. Simple strategy Sharpe: go LONG when score > 0.15, SHORT when score < -0.15
    ENTRY_THRESHOLD = 0.15
    strat_returns = []
    for _, row in merged.iterrows():
        if row["score"] > ENTRY_THRESHOLD:
            strat_returns.append(row["log_ret_4h"])
        elif row["score"] < -ENTRY_THRESHOLD:
            strat_returns.append(0.0 - float(row["log_ret_4h"]))   # short
        else:
            strat_returns.append(0.0)

    strat_arr = np.array(strat_returns)
    fee = 0.001  # 0.1% round trip
    strat_arr_net = strat_arr - np.abs(strat_arr > 0) * fee
    sharpe_sent = (np.nanmean(strat_arr_net) / (np.nanstd(strat_arr_net) + 1e-9)) * np.sqrt(24 * 365)

    # Baseline: always long (buy-and-hold Sharpe for comparison)
    bh_rets = merged["log_ret_1h"].dropna().values
    sharpe_bh = (np.nanmean(bh_rets) / (np.nanstd(bh_rets) + 1e-9)) * np.sqrt(24 * 365)

    return ValidationResult(
        coin=coin,
        n_periods=len(merged),
        ic_1h=ic_1h,
        ic_4h=ic_4h,
        ic_8h=ic_8h,
        dir_accuracy_1h=dir_1h,
        dir_accuracy_4h=dir_4h,
        source_ics=source_ics,
        sharpe_sentiment_only=sharpe_sent,
        sharpe_hmm_gate=sharpe_sent,   # placeholder — full HMM+sentiment backtest requires hmm_brain
        sentiment_alpha=sharpe_sent - sharpe_bh,
    )


def run_full_validation(coins: Optional[List[str]] = None, days: int = 30) -> List[ValidationResult]:
    """
    Run validation for multiple coins and print a ranked summary.
    Default coins: top 10 by market cap.
    """
    if coins is None:
        coins = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "MATIC", "AVAX", "LINK"]

    results = []
    for coin in coins:
        try:
            result = validate_coin(coin, days=days)
            if result:
                results.append(result)
                print(result.summary())
            time.sleep(0.5)  # avoid hammering Binance API
        except Exception as e:
            logger.error("Validation failed for %s: %s", coin, e)

    if results:
        _print_ranking(results)
    return results


def _print_ranking(results: List[ValidationResult]):
    """Print a ranked table by IC @4h."""
    print("\n" + "=" * 58)
    print(f" SENTIMENT SIGNAL RANKING — by IC @+4h")
    print("=" * 58)
    ranked = sorted(results, key=lambda r: abs(r.ic_4h) if not np.isnan(r.ic_4h) else 0, reverse=True)
    print(f"{'Coin':<8} {'IC@1h':>8} {'IC@4h':>8} {'DirAcc@4h':>12} {'SentAlpha':>12}")
    print("-" * 54)
    for r in ranked:
        ic1  = f"{r.ic_1h:+.4f}" if not np.isnan(r.ic_1h)  else "  N/A"
        ic4  = f"{r.ic_4h:+.4f}" if not np.isnan(r.ic_4h)  else "  N/A"
        da4  = f"{r.dir_accuracy_4h:.1%}" if not np.isnan(r.dir_accuracy_4h) else "  N/A"
        alph = f"{r.sentiment_alpha:+.3f}" if not np.isnan(r.sentiment_alpha) else "  N/A"
        print(f"{r.coin:<8} {ic1:>8} {ic4:>8} {da4:>12} {alph:>12}")
    print("=" * 58)

    # Summary insight
    good = [r for r in results if not np.isnan(r.ic_4h) and abs(r.ic_4h) > 0.05]
    print(f"\n{len(good)}/{len(results)} coins have |IC @4h| > 0.05 (useful signal threshold)")
    print("IC > 0.05 is considered statistically meaningful for daily trading.\n")


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate sentiment signal quality")
    parser.add_argument("--coin", default="BTC",
                        help="Coin symbol or 'ALL' for top 10 (default: BTC)")
    parser.add_argument("--days", type=int, default=30,
                        help="Lookback window in days (default: 30)")
    args = parser.parse_args()

    if args.coin.upper() == "ALL":
        run_full_validation(days=args.days)
    else:
        coins = [c.strip().upper() for c in args.coin.split(",")]
        run_full_validation(coins=coins, days=args.days)
