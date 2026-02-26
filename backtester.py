"""
Project Regime-Master — Backtester (The Time Machine)
Simulates HMM-driven strategy on historical data.

Logic mirrors live trading:
  - Confidence-aware leverage (85% min → 15x/25x/35x)
  - No regime-change exits (trades stay until SL/TP)
  - Fees only on new entries
"""
import numpy as np
import pandas as pd
import logging

import config
from hmm_brain import HMMBrain
from feature_engine import compute_all_features, compute_hmm_features
from risk_manager import RiskManager

logger = logging.getLogger("Backtester")


def backtest_hmm_strategy(df, states=None, brain=None):
    """
    Confidence-aware backtest of the HMM regime strategy.

    - Per-bar leverage based on HMM confidence (85% min, 15x/25x/35x tiers)
    - NO regime-change exits: positions carry until SL/TP would fire
    - Fees only on genuine new entries, not regime flips

    Parameters
    ----------
    df : pd.DataFrame with OHLCV data (already feature-enriched preferred)
    states : np.ndarray of canonical regime states (optional, computed if brain given)
    brain : HMMBrain instance (optional, used if states is None)

    Returns
    -------
    dict with equity curve, metrics, and regime breakdown
    """
    df = df.copy()

    # Ensure features exist
    if "log_return" not in df.columns:
        df = compute_all_features(df)

    # Get states
    if states is None and brain is not None:
        df_hmm = compute_hmm_features(df)
        states = brain.predict_all(df_hmm)
    elif states is None:
        raise ValueError("Either `states` or `brain` must be provided.")

    # ─── Compute per-bar confidence from HMM posteriors ─────────────────────
    confidences = None
    if brain is not None and brain._is_trained:
        df_hmm = compute_hmm_features(df)
        from hmm_brain import HMM_FEATURES
        hmm_feats = df_hmm[HMM_FEATURES].dropna().values
        hmm_scaled = (hmm_feats - brain._feat_mean) / brain._feat_std
        posteriors = brain.model.predict_proba(hmm_scaled)
        confidences = np.max(posteriors, axis=1)

    # Align states/confidences with DataFrame
    n = len(df)
    if len(states) < n:
        pad = n - len(states)
        states = np.concatenate([np.full(pad, config.REGIME_CHOP), states])
        if confidences is not None:
            confidences = np.concatenate([np.full(pad, 0.5), confidences])

    if confidences is None:
        confidences = np.full(n, 0.9)  # fallback if no brain

    df["state"] = states
    df["confidence"] = confidences
    df["log_ret"] = df["log_return"]

    # ─── Per-bar strategy with confidence-aware leverage ─────────────────────
    strategy_rets = np.zeros(n)
    leverages = np.zeros(n)
    prev_in_market = False

    for i in range(n):
        state = int(states[i])
        conf = confidences[i]
        log_ret = df["log_ret"].iloc[i] if not pd.isna(df["log_ret"].iloc[i]) else 0

        # Get leverage from regime engine (respects 85% min, 15/25/35x tiers)
        lev = RiskManager.get_dynamic_leverage(conf, state)
        leverages[i] = lev

        if lev == 0:
            # Below 85% confidence or crash regime — flat
            prev_in_market = False
            continue

        # Entry fee only on fresh entries
        if not prev_in_market:
            strategy_rets[i] -= (config.TAKER_FEE + config.SLIPPAGE_BUFFER)

        prev_in_market = True

        # Apply leveraged return based on regime direction
        if state == config.REGIME_BULL:
            strategy_rets[i] += log_ret * lev
        elif state == config.REGIME_BEAR:
            strategy_rets[i] += log_ret * -lev
        elif state == config.REGIME_CHOP:
            strategy_rets[i] += log_ret * lev * 0.3

    df["strategy_ret"] = strategy_rets
    df["leverage"] = leverages

    # ─── Cumulative Returns ──────────────────────────────────────────────────
    df["cumulative_market"]   = np.exp(df["log_ret"].cumsum())
    df["cumulative_strategy"] = np.exp(df["strategy_ret"].cumsum())

    # ─── Metrics ─────────────────────────────────────────────────────────────
    equity = df["cumulative_strategy"]
    final_mult = equity.iloc[-1] if not equity.empty else 1.0
    total_return = (final_mult - 1) * 100

    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    periods_per_year = 365 * 24
    strat_rets = df["strategy_ret"].dropna()
    if strat_rets.std() > 0:
        sharpe = (strat_rets.mean() / strat_rets.std()) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    gains = strat_rets[strat_rets > 0].sum()
    losses = abs(strat_rets[strat_rets < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("inf")

    # Trade count (flat → in-market transitions)
    in_market = (leverages > 0).astype(int)
    n_trades = int((np.diff(in_market, prepend=0) == 1).sum())

    # Selectivity stats
    active_mask = leverages > 0
    bars_active = int(active_mask.sum())
    bars_skipped = n - bars_active
    avg_leverage = float(leverages[active_mask].mean()) if bars_active > 0 else 0
    avg_conf = float(confidences[active_mask].mean()) if bars_active > 0 else 0

    # Regime breakdown
    regime_breakdown = {}
    for state_val, state_name in config.REGIME_NAMES.items():
        mask = df["state"] == state_val
        count = mask.sum()
        avg_ret = df.loc[mask, "strategy_ret"].mean() if count > 0 else 0
        regime_breakdown[state_name] = {
            "candles": int(count),
            "pct_time": f"{count / n * 100:.1f}%",
            "avg_return_per_candle": f"{avg_ret * 100:.4f}%",
        }

    results = {
        "total_return":     round(total_return, 2),
        "final_multiplier": round(final_mult, 4),
        "max_drawdown":     round(max_dd * 100, 2),
        "sharpe_ratio":     round(sharpe, 3),
        "profit_factor":    round(profit_factor, 3),
        "n_trades":         n_trades,
        "bars_active":      bars_active,
        "bars_skipped":     bars_skipped,
        "time_in_market":   f"{bars_active / n * 100:.1f}%",
        "avg_leverage":     round(avg_leverage, 1),
        "avg_confidence":   round(avg_conf, 3),
        "regime_breakdown": regime_breakdown,
        "equity_curve":     equity,
        "df":               df,
    }

    logger.info(
        "📊 Backtest: Return=%.1f%% (%.2fx) | MaxDD=%.1f%% | Sharpe=%.2f | PF=%.2f | Trades=%d | InMarket=%s | AvgLev=%.0fx",
        total_return, final_mult, max_dd * 100, sharpe, profit_factor, n_trades,
        results["time_in_market"], avg_leverage,
    )

    return results


def run_full_backtest(df_raw, n_states=4):
    """
    Convenience wrapper: trains HMM + runs backtest in one call.
    """
    brain = HMMBrain(n_states=n_states)

    df_feat = compute_all_features(df_raw)
    df_hmm = compute_hmm_features(df_raw)
    brain.train(df_hmm)

    return backtest_hmm_strategy(df_feat, brain=brain)


def print_backtest_report(results):
    """Pretty-print backtest results to console."""
    print("\n" + "=" * 60)
    print("  📊 REGIME-MASTER BACKTEST REPORT")
    print("=" * 60)
    print(f"  Total Return:     {results['total_return']:>10.2f}%")
    print(f"  Final Multiplier: {results['final_multiplier']:>10.4f}x")
    print(f"  Max Drawdown:     {results['max_drawdown']:>10.2f}%")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:>10.3f}")
    print(f"  Profit Factor:    {results['profit_factor']:>10.3f}")
    print(f"  Total Trades:     {results['n_trades']:>10d}")
    print(f"  Time in Market:   {results.get('time_in_market', '?'):>10s}")
    print(f"  Avg Leverage:     {results.get('avg_leverage', 0):>10.1f}x")
    print(f"  Avg Confidence:   {results.get('avg_confidence', 0):>10.3f}")
    print(f"  Bars Skipped:     {results.get('bars_skipped', 0):>10d}")
    print("-" * 60)
    print("  Regime Breakdown:")
    for regime, info in results["regime_breakdown"].items():
        print(f"    {regime:<16s} → {info['candles']:>5d} candles ({info['pct_time']:>6s}) | Avg Ret: {info['avg_return_per_candle']}")
    print("=" * 60 + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_pipeline import fetch_futures_klines

    print("📡 Fetching BTCUSDT 1h data (30 days)...")
    df = fetch_futures_klines("BTCUSDT", "1h", limit=720)

    if df is None or len(df) < 100:
        from feature_engine import generate_synthetic_data
        print("⚠️  Using synthetic data")
        df = generate_synthetic_data(720)

    print(f"✅ Got {len(df)} candles")
    results = run_full_backtest(df)
    print_backtest_report(results)
