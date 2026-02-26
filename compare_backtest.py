"""
Compare Old vs New Leverage/Confidence Logic via Backtest.

Old: LEVERAGE 75/50/25, CONFIDENCE thresholds 0.75/0.55/0.40
New: LEVERAGE 35/25/15, CONFIDENCE thresholds 0.95/0.91/0.85

Runs both configurations on the same historical data and compares metrics.
"""
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.WARNING)

from hmm_brain import HMMBrain
from feature_engine import compute_all_features, compute_hmm_features
import config

# ─── Configuration Profiles ─────────────────────────────────────────────────────

OLD_CONFIG = {
    "name": "OLD (Aggressive)",
    "leverage_high": 75,
    "leverage_moderate": 50,
    "leverage_low": 25,
    "confidence_high": 0.75,
    "confidence_medium": 0.55,
    "confidence_low": 0.40,
    "min_deploy": 0.40,
}

NEW_CONFIG = {
    "name": "NEW (Conservative)",
    "leverage_high": 35,
    "leverage_moderate": 25,
    "leverage_low": 15,
    "confidence_high": 0.95,
    "confidence_medium": 0.91,
    "confidence_low": 0.85,
    "min_deploy": 0.85,
}


def get_dynamic_leverage(confidence, regime, cfg):
    """Same logic as risk_manager.get_dynamic_leverage but with configurable tiers."""
    if regime == config.REGIME_CRASH:
        return 0
    if regime == config.REGIME_CHOP:
        return cfg["leverage_low"] if confidence >= cfg["confidence_low"] else 0
    # Trend
    if confidence >= cfg["confidence_high"]:
        return cfg["leverage_high"]
    elif confidence >= cfg["confidence_medium"]:
        return cfg["leverage_moderate"]
    elif confidence >= cfg["confidence_low"]:
        return cfg["leverage_low"]
    else:
        return 0  # Below minimum → don't trade


def backtest_with_config(df, states, confidences, cfg, regime_exit=False):
    """
    Run backtest with per-bar confidence-aware leverage.
    
    Parameters
    ----------
    df : DataFrame with features
    states : array of regime states per bar
    confidences : array of model confidence per bar
    cfg : config dict with leverage/confidence thresholds
    regime_exit : if True, force-close position on regime change (take fees)
    """
    df = df.copy()
    n = len(df)
    
    # Align arrays
    if len(states) < n:
        pad = n - len(states)
        states = np.concatenate([np.full(pad, config.REGIME_CHOP), states])
        confidences = np.concatenate([np.full(pad, 0.5), confidences])
    
    df["state"] = states
    df["confidence"] = confidences
    df["log_ret"] = df["log_return"]
    
    # ─── Per-bar leverage based on confidence ────────────────────────────────
    strategy_rets = np.zeros(n)
    leverages_used = np.zeros(n)
    trades_taken = 0
    trades_skipped = 0
    bars_in_market = 0
    bars_out = 0
    
    prev_state = None
    in_position = False
    
    for i in range(n):
        state = int(states[i])
        conf = confidences[i]
        log_ret = df["log_ret"].iloc[i] if not pd.isna(df["log_ret"].iloc[i]) else 0
        
        # Determine leverage for this bar
        lev = get_dynamic_leverage(conf, state, cfg)
        leverages_used[i] = lev
        
        # Regime changed?
        regime_changed = (prev_state is not None and state != prev_state)
        
        if regime_changed:
            if regime_exit and in_position:
                # Pay exit fee
                strategy_rets[i] -= (config.TAKER_FEE + config.SLIPPAGE_BUFFER)
            trades_taken += 1
        
        if lev == 0:
            # Skip this bar (low confidence or crash regime)
            trades_skipped += 1
            bars_out += 1
            in_position = False
        else:
            bars_in_market += 1
            in_position = True
            
            # Entry fee on new position
            if regime_changed:
                strategy_rets[i] -= (config.TAKER_FEE + config.SLIPPAGE_BUFFER)
            
            # Apply leveraged return based on regime direction
            if state == config.REGIME_BULL:
                strategy_rets[i] += log_ret * lev
            elif state == config.REGIME_BEAR:
                strategy_rets[i] += log_ret * -lev  # short
            elif state == config.REGIME_CHOP:
                strategy_rets[i] += log_ret * lev * 0.3  # mean reversion
        
        prev_state = state
    
    df["strategy_ret"] = strategy_rets
    df["leverage_used"] = leverages_used
    
    # ─── Metrics ─────────────────────────────────────────────────────────────
    equity = np.exp(np.cumsum(strategy_rets))
    final_mult = equity[-1] if len(equity) > 0 else 1.0
    total_return = (final_mult - 1) * 100
    
    # Max Drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = np.min(drawdowns) * 100
    
    # Sharpe (annualized hourly)
    periods_per_year = 365 * 24
    strat_rets = strategy_rets[strategy_rets != 0]
    if len(strat_rets) > 1 and np.std(strat_rets) > 0:
        sharpe = (np.mean(strat_rets) / np.std(strat_rets)) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0
    
    # Profit Factor
    gains = strategy_rets[strategy_rets > 0].sum()
    losses_sum = abs(strategy_rets[strategy_rets < 0].sum())
    profit_factor = gains / losses_sum if losses_sum > 0 else float("inf")
    
    # Win rate (per regime-change trade)
    trade_returns = []
    current_trade_ret = 0
    for i in range(1, n):
        current_trade_ret += strategy_rets[i]
        if i + 1 < n and int(states[i + 1]) != int(states[i]):
            trade_returns.append(current_trade_ret)
            current_trade_ret = 0
    if current_trade_ret != 0:
        trade_returns.append(current_trade_ret)
    
    wins = sum(1 for r in trade_returns if r > 0)
    win_rate = (wins / len(trade_returns) * 100) if trade_returns else 0
    
    # Average leverage
    active_levs = leverages_used[leverages_used > 0]
    avg_leverage = np.mean(active_levs) if len(active_levs) > 0 else 0
    
    # Aggressiveness score (0-100)
    # Combines: leverage intensity, time in market, drawdown exposure
    max_possible_lev = 75  # old max was 75x
    lev_score = avg_leverage / max_possible_lev * 100
    time_in_market = bars_in_market / n * 100
    dd_score = min(abs(max_dd), 100)
    aggressiveness = (lev_score * 0.4 + time_in_market * 0.3 + dd_score * 0.3)
    
    # Confidence distribution
    active_confs = df["confidence"][leverages_used > 0]
    
    return {
        "name": cfg["name"],
        "total_return": round(total_return, 2),
        "final_mult": round(final_mult, 4),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "profit_factor": round(profit_factor, 3),
        "n_trades": trades_taken,
        "win_rate": round(win_rate, 1),
        "bars_in_market": bars_in_market,
        "bars_out": bars_out,
        "time_in_market_pct": round(bars_in_market / n * 100, 1),
        "avg_leverage": round(avg_leverage, 1),
        "max_leverage_used": int(np.max(leverages_used)) if len(leverages_used) > 0 else 0,
        "aggressiveness": round(aggressiveness, 1),
        "trades_skipped": trades_skipped,
        "avg_confidence_when_trading": round(float(active_confs.mean()), 3) if len(active_confs) > 0 else 0,
        "equity_curve": equity,
    }


def print_comparison(old_result, new_result, regime_exit_old=None, regime_exit_new=None):
    """Print side-by-side comparison."""
    print("\n" + "═" * 72)
    print("  📊 BACKTEST COMPARISON: OLD vs NEW LOGIC")
    print("═" * 72)
    
    metrics = [
        ("Total Return", "total_return", "%", True),
        ("Final Multiplier", "final_mult", "x", True),
        ("Max Drawdown", "max_drawdown", "%", False),  # lower is better
        ("Sharpe Ratio", "sharpe", "", True),
        ("Profit Factor", "profit_factor", "", True),
        ("Win Rate", "win_rate", "%", True),
        ("Total Trades", "n_trades", "", None),
        ("Time in Market", "time_in_market_pct", "%", None),
        ("Avg Leverage", "avg_leverage", "x", None),
        ("Max Leverage", "max_leverage_used", "x", None),
        ("Bars Skipped (Low Conf)", "trades_skipped", "", None),
        ("Aggressiveness Score", "aggressiveness", "/100", None),
        ("Avg Confidence (Active)", "avg_confidence_when_trading", "", True),
    ]
    
    print(f"\n  {'Metric':<28s}  {'OLD (Aggressive)':>18s}  {'NEW (Conservative)':>18s}  {'Winner':>8s}")
    print(f"  {'─' * 28}  {'─' * 18}  {'─' * 18}  {'─' * 8}")
    
    for label, key, unit, higher_better in metrics:
        old_val = old_result[key]
        new_val = new_result[key]
        
        if higher_better is True:
            winner = "NEW ✅" if new_val > old_val else ("OLD" if old_val > new_val else "TIE")
        elif higher_better is False:
            winner = "NEW ✅" if abs(new_val) < abs(old_val) else ("OLD" if abs(old_val) < abs(new_val) else "TIE")
        else:
            winner = "—"
        
        print(f"  {label:<28s}  {old_val:>15.2f}{unit:<3s}  {new_val:>15.2f}{unit:<3s}  {winner:>8s}")
    
    # Regime exit comparison if available
    if regime_exit_old and regime_exit_new:
        print(f"\n  {'─' * 72}")
        print(f"  WITH REGIME-CHANGE EXITS:")
        print(f"  {'Return (w/ regime exit)':<28s}  {regime_exit_old['total_return']:>15.2f}%    {regime_exit_new['total_return']:>15.2f}%")
        print(f"  {'Max DD (w/ regime exit)':<28s}  {regime_exit_old['max_drawdown']:>15.2f}%    {regime_exit_new['max_drawdown']:>15.2f}%")
        print(f"  {'Sharpe (w/ regime exit)':<28s}  {regime_exit_old['sharpe']:>15.3f}     {regime_exit_new['sharpe']:>15.3f}")
    
    # Analysis
    print("\n" + "═" * 72)
    print("  💡 ANALYSIS")
    print("═" * 72)
    
    ret_change = new_result["total_return"] - old_result["total_return"]
    dd_change = abs(new_result["max_drawdown"]) - abs(old_result["max_drawdown"])
    skip_pct = new_result["trades_skipped"] / (new_result["bars_in_market"] + new_result["trades_skipped"]) * 100 if (new_result["bars_in_market"] + new_result["trades_skipped"]) > 0 else 0
    
    print(f"\n  Return Change:        {ret_change:+.2f}% ({'improved' if ret_change > 0 else 'reduced'})")
    print(f"  Max DD Change:        {dd_change:+.2f}% ({'tighter' if dd_change < 0 else 'wider'})")
    print(f"  Leverage Reduction:   {old_result['avg_leverage']:.0f}x → {new_result['avg_leverage']:.0f}x ({(1 - new_result['avg_leverage']/old_result['avg_leverage'])*100:.0f}% less aggressive)" if old_result['avg_leverage'] > 0 else "")
    print(f"  Bars Filtered Out:    {skip_pct:.1f}% of bars skipped in NEW (higher selectivity)")
    print(f"  Aggressiveness:       {old_result['aggressiveness']:.0f}/100 → {new_result['aggressiveness']:.0f}/100")
    
    # Risk-adjusted return
    old_rar = old_result["total_return"] / abs(old_result["max_drawdown"]) if old_result["max_drawdown"] != 0 else 0
    new_rar = new_result["total_return"] / abs(new_result["max_drawdown"]) if new_result["max_drawdown"] != 0 else 0
    print(f"\n  Risk-Adjusted Return: {old_rar:.2f} → {new_rar:.2f} ({'BETTER' if new_rar > old_rar else 'WORSE'} risk/reward)")
    
    print("\n" + "═" * 72)


# ─── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📡 Fetching BTCUSDT 1h data (last 30 days)...")
    from data_pipeline import fetch_futures_klines
    df_raw = fetch_futures_klines("BTCUSDT", "1h", limit=720)  # 30 days
    
    if df_raw is None or len(df_raw) < 100:
        print("❌ Not enough data. Trying synthetic...")
        from feature_engine import generate_synthetic_data
        df_raw = generate_synthetic_data(720)
    
    print(f"✅ Got {len(df_raw)} candles")
    
    # Compute features & train HMM
    print("🧠 Training HMM brain...")
    df_feat = compute_all_features(df_raw)
    df_hmm = compute_hmm_features(df_raw)
    
    brain = HMMBrain(n_states=4)
    brain.train(df_hmm)
    
    states = brain.predict_all(df_hmm)
    
    # Get per-bar confidence (posterior probability of assigned state)
    # Must use same 2 features + scaling as HMM was trained on
    hmm_feats = df_hmm[["log_return", "volatility"]].dropna().values
    hmm_scaled = (hmm_feats - brain._feat_mean) / brain._feat_std
    posteriors = brain.model.predict_proba(hmm_scaled)
    confidences = np.max(posteriors, axis=1)
    
    # Pad to match df length
    if len(states) < len(df_feat):
        pad = len(df_feat) - len(states)
        states = np.concatenate([np.full(pad, config.REGIME_CHOP), states])
        confidences = np.concatenate([np.full(pad, 0.5), confidences])
    
    print(f"📊 Confidence distribution: min={confidences.min():.2f} mean={confidences.mean():.2f} max={confidences.max():.2f}")
    print(f"   Bars ≥85% confidence: {(confidences >= 0.85).sum()} / {len(confidences)} ({(confidences >= 0.85).mean()*100:.1f}%)")
    print(f"   Bars ≥95% confidence: {(confidences >= 0.95).sum()} / {len(confidences)} ({(confidences >= 0.95).mean()*100:.1f}%)")
    
    # ─── Run backtests ───────────────────────────────────────────────────────
    print("\n🔄 Running OLD config backtest...")
    old_result = backtest_with_config(df_feat, states, confidences, OLD_CONFIG, regime_exit=False)
    
    print("🔄 Running NEW config backtest...")
    new_result = backtest_with_config(df_feat, states, confidences, NEW_CONFIG, regime_exit=False)
    
    # Also test with regime-change exits
    print("🔄 Running OLD config with regime exits...")
    old_regime_exit = backtest_with_config(df_feat, states, confidences, OLD_CONFIG, regime_exit=True)
    
    print("🔄 Running NEW config with regime exits...")
    new_regime_exit = backtest_with_config(df_feat, states, confidences, NEW_CONFIG, regime_exit=True)
    
    # ─── Print comparison ────────────────────────────────────────────────────
    print_comparison(old_result, new_result, old_regime_exit, new_regime_exit)
    
    # ─── Regime exit analysis ────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  🔄 REGIME-CHANGE EXIT ANALYSIS")
    print("═" * 72)
    
    old_diff = old_regime_exit["total_return"] - old_result["total_return"]
    new_diff = new_regime_exit["total_return"] - new_result["total_return"]
    old_dd_diff = abs(old_regime_exit["max_drawdown"]) - abs(old_result["max_drawdown"])
    new_dd_diff = abs(new_regime_exit["max_drawdown"]) - abs(new_result["max_drawdown"])
    
    print(f"\n  Old Config: Regime exits {'HELP' if old_diff > 0 else 'HURT'} returns by {old_diff:+.2f}%")
    print(f"  New Config: Regime exits {'HELP' if new_diff > 0 else 'HURT'} returns by {new_diff:+.2f}%")
    print(f"\n  Old Config: Regime exits {'tighten' if old_dd_diff < 0 else 'widen'} max DD by {old_dd_diff:+.2f}%")
    print(f"  New Config: Regime exits {'tighten' if new_dd_diff < 0 else 'widen'} max DD by {new_dd_diff:+.2f}%")
    
    print(f"\n  📝 Verdict on regime-change exits:")
    if old_diff < 0 and new_diff < 0:
        print("  ⚠️  Regime exits REDUCE returns in both configs.")
        print("  This is because the extra exit fees eat into profits,")
        print("  and the HMM's regime transitions often anticipate the move.")
        print("  RECOMMENDATION: Keep current behavior (no forced exit on regime change).")
    elif old_diff > 0 and new_diff > 0:
        print("  ✅ Regime exits IMPROVE returns in both configs.")
        print("  RECOMMENDATION: Implement regime-change exits in live trading.")
    else:
        print("  ⚖️  Mixed results. Test more datasets before deciding.")
    
    print("\n" + "═" * 72)
    
    # Save results
    results_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "data": "BTCUSDT 1h (30 days)",
        "candles": len(df_feat),
        "old_config": {k: v for k, v in old_result.items() if k != "equity_curve"},
        "new_config": {k: v for k, v in new_result.items() if k != "equity_curve"},
        "old_regime_exit": {k: v for k, v in old_regime_exit.items() if k != "equity_curve"},
        "new_regime_exit": {k: v for k, v in new_regime_exit.items() if k != "equity_curve"},
    }
    with open("data/backtest_comparison.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print("\n💾 Results saved to data/backtest_comparison.json")
