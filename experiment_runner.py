"""
Regime Engine Optimization — Multi-Experiment Backtester
=========================================================
Runs systematic backtests across different configurations to find optimal:
  1. Timeframe for HMM training (5m, 15m, 1h, 4h)
  2. Number of HMM states (3, 4, 5)
  3. Covariance type (diag, full, spherical)
  4. HMM input features (2-feat vs 4-feat vs 6-feat)
  5. HMM lookback window (200, 500, 1000)

Outputs a JSON + printed report comparing all configs.

Usage:
  python experiment_runner.py
"""
import sys
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_pipeline import fetch_futures_klines
from feature_engine import compute_all_features, compute_hmm_features, compute_atr, compute_rsi
from hmm_brain import HMMBrain
from risk_manager import RiskManager

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Experiment")
logger.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════════════════════
#  ENHANCED FEATURE SETS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_features_2(df):
    """Original 2-feature set: log_return + volatility."""
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = (df["high"] - df["low"]) / df["close"]
    return df

def compute_features_4(df):
    """4-feature set: log_return + volatility + volume_change + rsi_norm."""
    df = compute_features_2(df)
    df["volume_change"] = np.log(df["volume"] / df["volume"].shift(1).replace(0, np.nan))
    df["volume_change"] = df["volume_change"].fillna(0).clip(-3, 3)
    # Normalized RSI (0-1 scale, centered at 0)
    rsi = compute_rsi(df["close"], length=14)
    df["rsi_norm"] = (rsi - 50) / 50  # -1 to +1 range
    df["rsi_norm"] = df["rsi_norm"].fillna(0)
    return df

def compute_features_6(df):
    """6-feature set: 4-feat + atr_pct + momentum."""
    df = compute_features_4(df)
    atr = compute_atr(df, length=14)
    df["atr_pct"] = atr / df["close"]
    df["atr_pct"] = df["atr_pct"].fillna(0)
    # 10-bar momentum
    df["momentum"] = df["close"].pct_change(10)
    df["momentum"] = df["momentum"].fillna(0).clip(-0.3, 0.3)
    return df

FEATURE_SETS = {
    "2feat": {"fn": compute_features_2, "cols": ["log_return", "volatility"]},
    "4feat": {"fn": compute_features_4, "cols": ["log_return", "volatility", "volume_change", "rsi_norm"]},
    "6feat": {"fn": compute_features_6, "cols": ["log_return", "volatility", "volume_change", "rsi_norm", "atr_pct", "momentum"]},
}

# ═══════════════════════════════════════════════════════════════════════════════
#  MODIFIED HMM BRAIN (supports variable feature columns)
# ═══════════════════════════════════════════════════════════════════════════════

from hmmlearn.hmm import GaussianHMM

class FlexHMMBrain:
    """HMM Brain that supports variable feature columns."""
    
    def __init__(self, n_states=4, covariance="diag", n_iter=100, feature_cols=None):
        self.n_states = n_states
        self.covariance = covariance
        self.n_iter = n_iter
        self.feature_cols = feature_cols or ["log_return", "volatility"]
        self.model = None
        self._state_map = None
        self._feat_mean = None
        self._feat_std = None
        self._is_trained = False
    
    def train(self, df):
        features = df[self.feature_cols].dropna().values
        if len(features) < 50:
            return self
        
        self._feat_mean = features.mean(axis=0)
        self._feat_std = features.std(axis=0)
        self._feat_std[self._feat_std < 1e-10] = 1e-10
        features_scaled = (features - self._feat_mean) / self._feat_std
        
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance,
            n_iter=self.n_iter,
            random_state=42,
        )
        self.model.fit(features_scaled)
        self._build_state_map()
        self._is_trained = True
        return self
    
    def _build_state_map(self):
        means = self.model.means_[:, 0]  # log-return is always first col
        vols = self.model.means_[:, 1] if self.model.means_.shape[1] > 1 else np.zeros(self.n_states)
        sorted_indices = np.argsort(means)[::-1]
        
        if self.n_states >= 4:
            ranked = list(sorted_indices)
            mid = ranked[1:3]
            if vols[mid[0]] <= vols[mid[1]]:
                chop_raw, bear_raw = mid[0], mid[1]
            else:
                chop_raw, bear_raw = mid[1], mid[0]
            self._state_map = {
                ranked[0]: config.REGIME_BULL,
                bear_raw: config.REGIME_BEAR,
                chop_raw: config.REGIME_CHOP,
                ranked[-1]: config.REGIME_CRASH,
            }
        elif self.n_states == 3:
            self._state_map = {
                sorted_indices[0]: config.REGIME_BULL,
                sorted_indices[1]: config.REGIME_CHOP,
                sorted_indices[2]: config.REGIME_BEAR,
            }
        elif self.n_states == 5:
            ranked = list(sorted_indices)
            self._state_map = {
                ranked[0]: config.REGIME_BULL,
                ranked[1]: config.REGIME_BULL,   # strong + moderate bull → same regime
                ranked[2]: config.REGIME_CHOP,
                ranked[3]: config.REGIME_BEAR,
                ranked[4]: config.REGIME_CRASH,
            }
        else:
            self._state_map = {
                sorted_indices[0]: config.REGIME_BULL,
                sorted_indices[1]: config.REGIME_BEAR,
            }
    
    def predict_all(self, df):
        if not self._is_trained:
            return np.full(len(df), config.REGIME_CHOP)
        features = df[self.feature_cols].dropna().values
        features_scaled = (features - self._feat_mean) / self._feat_std
        raw_states = self.model.predict(features_scaled)
        return np.array([self._state_map.get(s, config.REGIME_CHOP) for s in raw_states])
    
    def predict_proba_all(self, df):
        if not self._is_trained:
            return np.zeros((len(df), self.n_states))
        features = df[self.feature_cols].dropna().values
        features_scaled = (features - self._feat_mean) / self._feat_std
        return self.model.predict_proba(features_scaled)

# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTESTER (self-contained, uses FlexHMMBrain)
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(df, brain, feature_cols):
    """Run a single backtest with given brain and features."""
    df = df.copy()
    
    # Get states and confidences
    states = brain.predict_all(df)
    
    # Get confidences
    features = df[feature_cols].dropna().values
    features_scaled = (features - brain._feat_mean) / brain._feat_std
    posteriors = brain.model.predict_proba(features_scaled)
    confidences = np.max(posteriors, axis=1)
    
    n = len(df)
    if len(states) < n:
        pad = n - len(states)
        states = np.concatenate([np.full(pad, config.REGIME_CHOP), states])
        confidences = np.concatenate([np.full(pad, 0.5), confidences])
    
    # Ensure log_return exists
    if "log_return" not in df.columns or "log_ret" not in df.columns:
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    else:
        df["log_ret"] = df["log_return"]
    
    # Per-bar strategy
    strategy_rets = np.zeros(n)
    leverages = np.zeros(n)
    prev_in_market = False
    
    for i in range(n):
        state = int(states[i])
        conf = confidences[i]
        log_ret = df["log_ret"].iloc[i] if not pd.isna(df["log_ret"].iloc[i]) else 0
        
        lev = RiskManager.get_dynamic_leverage(conf, state)
        leverages[i] = lev
        
        if lev == 0:
            prev_in_market = False
            continue
        
        if not prev_in_market:
            strategy_rets[i] -= (config.TAKER_FEE + config.SLIPPAGE_BUFFER)
        prev_in_market = True
        
        if state == config.REGIME_BULL:
            strategy_rets[i] += log_ret * lev
        elif state == config.REGIME_BEAR:
            strategy_rets[i] += log_ret * -lev
        elif state == config.REGIME_CHOP:
            strategy_rets[i] += log_ret * lev * 0.3
    
    # Metrics
    equity = np.exp(np.cumsum(strategy_rets))
    final_mult = equity[-1] if len(equity) > 0 else 1.0
    total_return = (final_mult - 1) * 100
    
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd = np.min(drawdowns) * 100
    
    strat_rets = strategy_rets[~np.isnan(strategy_rets)]
    periods_per_year = 365 * 24
    if np.std(strat_rets) > 0:
        sharpe = (np.mean(strat_rets) / np.std(strat_rets)) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0
    
    gains = np.sum(strat_rets[strat_rets > 0])
    losses = abs(np.sum(strat_rets[strat_rets < 0]))
    profit_factor = gains / losses if losses > 0 else float("inf")
    
    in_market = (leverages > 0).astype(int)
    n_trades = int((np.diff(in_market, prepend=0) == 1).sum())
    
    active_mask = leverages > 0
    bars_active = int(active_mask.sum())
    avg_leverage = float(leverages[active_mask].mean()) if bars_active > 0 else 0
    avg_conf = float(confidences[active_mask].mean()) if bars_active > 0 else 0
    
    # Regime breakdown
    regime_stats = {}
    for state_val, state_name in config.REGIME_NAMES.items():
        mask = states[:n] == state_val
        count = int(mask.sum())
        avg_ret = float(np.mean(strategy_rets[mask])) * 100 if count > 0 else 0
        regime_stats[state_name] = {
            "candles": count,
            "pct": round(count / n * 100, 1),
            "avg_ret_pct": round(avg_ret, 4),
        }
    
    return {
        "total_return": round(total_return, 2),
        "final_multiplier": round(float(final_mult), 4),
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "profit_factor": round(profit_factor, 3),
        "n_trades": n_trades,
        "time_in_market": round(bars_active / n * 100, 1),
        "avg_leverage": round(avg_leverage, 1),
        "avg_confidence": round(avg_conf, 3),
        "bars_skipped_pct": round((n - bars_active) / n * 100, 1),
        "regime_breakdown": regime_stats,
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # ── Group A: Timeframe comparison (current features, 4 states, diag) ────
    {"name": "A1_5m_baseline",    "tf": "5m",  "n_states": 4, "cov": "diag", "features": "2feat", "lookback": 500},
    {"name": "A2_15m",            "tf": "15m", "n_states": 4, "cov": "diag", "features": "2feat", "lookback": 500},
    {"name": "A3_1h",             "tf": "1h",  "n_states": 4, "cov": "diag", "features": "2feat", "lookback": 500},
    {"name": "A4_4h",             "tf": "4h",  "n_states": 4, "cov": "diag", "features": "2feat", "lookback": 500},
    
    # ── Group B: HMM states comparison (1h timeframe, 2-feat, diag) ────────
    {"name": "B1_3states",        "tf": "1h",  "n_states": 3, "cov": "diag", "features": "2feat", "lookback": 500},
    {"name": "B2_4states",        "tf": "1h",  "n_states": 4, "cov": "diag", "features": "2feat", "lookback": 500},
    {"name": "B3_5states",        "tf": "1h",  "n_states": 5, "cov": "diag", "features": "2feat", "lookback": 500},
    
    # ── Group C: Feature enrichment (1h, 4 states, diag) ──────────────────
    {"name": "C1_2feat",          "tf": "1h",  "n_states": 4, "cov": "diag", "features": "2feat", "lookback": 500},
    {"name": "C2_4feat",          "tf": "1h",  "n_states": 4, "cov": "diag", "features": "4feat", "lookback": 500},
    {"name": "C3_6feat",          "tf": "1h",  "n_states": 4, "cov": "diag", "features": "6feat", "lookback": 500},
    
    # ── Group D: Covariance type (1h, 4 states, 4-feat) ────────────────────
    {"name": "D1_diag",           "tf": "1h",  "n_states": 4, "cov": "diag",      "features": "4feat", "lookback": 500},
    {"name": "D2_full",           "tf": "1h",  "n_states": 4, "cov": "full",      "features": "4feat", "lookback": 500},
    {"name": "D3_spherical",      "tf": "1h",  "n_states": 4, "cov": "spherical", "features": "4feat", "lookback": 500},
    
    # ── Group E: Lookback window (1h, 4 states, 4-feat, diag) ─────────────
    {"name": "E1_lookback_200",   "tf": "1h",  "n_states": 4, "cov": "diag", "features": "4feat", "lookback": 200},
    {"name": "E2_lookback_500",   "tf": "1h",  "n_states": 4, "cov": "diag", "features": "4feat", "lookback": 500},
    {"name": "E3_lookback_1000",  "tf": "1h",  "n_states": 4, "cov": "diag", "features": "4feat", "lookback": 1000},
    
    # ── Group F: Best-combo candidates ────────────────────────────────────
    {"name": "F1_15m_4feat_full", "tf": "15m", "n_states": 4, "cov": "full", "features": "4feat", "lookback": 500},
    {"name": "F2_1h_4feat_full",  "tf": "1h",  "n_states": 4, "cov": "full", "features": "4feat", "lookback": 500},
    {"name": "F3_1h_6feat_full",  "tf": "1h",  "n_states": 4, "cov": "full", "features": "6feat", "lookback": 500},
    {"name": "F4_15m_6feat_full", "tf": "15m", "n_states": 4, "cov": "full", "features": "6feat", "lookback": 500},
]

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_data_for_timeframe(tf, limit=1000):
    """Fetch BTCUSDT data for a given timeframe."""
    logger.info(f"📡 Fetching BTCUSDT {tf} data (limit={limit})...")
    df = fetch_futures_klines("BTCUSDT", tf, limit=limit)
    if df is None or len(df) < 100:
        logger.error(f"❌ Failed to fetch sufficient {tf} data")
        return None
    logger.info(f"✅ Got {len(df)} candles for {tf}")
    return df

def run_single_experiment(exp, data_cache):
    """Run a single experiment configuration."""
    tf = exp["tf"]
    
    # Get or fetch data
    lookback = exp["lookback"]
    cache_key = f"{tf}_{lookback}"
    if cache_key not in data_cache:
        # Fetch enough data: lookback for training + rest for testing
        fetch_limit = min(lookback + 500, 1500)
        data_cache[cache_key] = fetch_data_for_timeframe(tf, limit=fetch_limit)
    
    df_raw = data_cache[cache_key]
    if df_raw is None:
        return {"error": f"No data for {tf}"}
    
    # Compute features
    feat_info = FEATURE_SETS[exp["features"]]
    df = feat_info["fn"](df_raw)
    feature_cols = feat_info["cols"]
    
    # Train HMM
    brain = FlexHMMBrain(
        n_states=exp["n_states"],
        covariance=exp["cov"],
        n_iter=100,
        feature_cols=feature_cols,
    )
    
    # Use first `lookback` rows for training, full series for backtest
    train_df = df.iloc[:lookback]
    brain.train(train_df)
    
    if not brain._is_trained:
        return {"error": "HMM training failed"}
    
    # Run backtest on full data
    try:
        results = run_backtest(df, brain, feature_cols)
        return results
    except Exception as e:
        return {"error": str(e)}

def main():
    print("\n" + "=" * 80)
    print("  🔬 REGIME ENGINE OPTIMIZATION — MULTI-EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"  Experiments to run: {len(EXPERIMENTS)}")
    print(f"  Symbol: BTCUSDT (Binance Futures)")
    print("=" * 80 + "\n")
    
    data_cache = {}
    all_results = {}
    
    for i, exp in enumerate(EXPERIMENTS):
        name = exp["name"]
        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{len(EXPERIMENTS)}] {name}")
        print(f"  TF={exp['tf']}  States={exp['n_states']}  Cov={exp['cov']}  Features={exp['features']}  Lookback={exp['lookback']}")
        print(f"{'─' * 60}")
        
        t0 = time.time()
        try:
            result = run_single_experiment(exp, data_cache)
            elapsed = time.time() - t0
            result["elapsed_seconds"] = round(elapsed, 1)
            result["config"] = exp
            all_results[name] = result
            
            if "error" in result:
                print(f"  ❌ ERROR: {result['error']}")
            else:
                print(f"  Return:  {result['total_return']:>+10.2f}%  ({result['final_multiplier']:.4f}x)")
                print(f"  MaxDD:   {result['max_drawdown']:>10.2f}%")
                print(f"  Sharpe:  {result['sharpe_ratio']:>10.3f}")
                print(f"  PF:      {result['profit_factor']:>10.3f}")
                print(f"  Trades:  {result['n_trades']:>10d}")
                print(f"  InMkt:   {result['time_in_market']:>9.1f}%")
                print(f"  AvgLev:  {result['avg_leverage']:>9.1f}x")
                print(f"  AvgConf: {result['avg_confidence']:>9.3f}")
                print(f"  ⏱ {elapsed:.1f}s")
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            all_results[name] = {"error": str(e), "config": exp}
    
    # ── Save results ──────────────────────────────────────────────────────
    output_path = os.path.join(config.DATA_DIR, "experiment_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {output_path}")
    
    # ── Print comparison table ────────────────────────────────────────────
    print_comparison(all_results)
    
    return all_results

def print_comparison(results):
    """Print a sorted comparison table."""
    print("\n" + "=" * 120)
    print("  📊 COMPARISON TABLE — Sorted by Sharpe Ratio (descending)")
    print("=" * 120)
    
    # Header
    print(f"  {'Experiment':<25s} {'Return':>10s} {'MaxDD':>8s} {'Sharpe':>8s} {'PF':>7s} {'Trades':>7s} {'InMkt':>7s} {'AvgLev':>7s} {'AvgConf':>8s} {'Skip%':>7s}")
    print("  " + "─" * 116)
    
    # Sort by Sharpe
    valid = {k: v for k, v in results.items() if "error" not in v}
    failed = {k: v for k, v in results.items() if "error" in v}
    
    sorted_keys = sorted(valid.keys(), key=lambda k: valid[k].get("sharpe_ratio", -999), reverse=True)
    
    for name in sorted_keys:
        r = valid[name]
        ret_str = f"{r['total_return']:>+.1f}%"
        dd_str = f"{r['max_drawdown']:.1f}%"
        sharpe_str = f"{r['sharpe_ratio']:.3f}"
        pf_str = f"{r['profit_factor']:.2f}"
        trades_str = f"{r['n_trades']}"
        inmkt_str = f"{r['time_in_market']:.0f}%"
        avglev_str = f"{r['avg_leverage']:.0f}x"
        avgconf_str = f"{r['avg_confidence']:.3f}"
        skip_str = f"{r['bars_skipped_pct']:.0f}%"
        
        # Highlight best
        marker = " 🏆" if name == sorted_keys[0] else ""
        print(f"  {name:<25s} {ret_str:>10s} {dd_str:>8s} {sharpe_str:>8s} {pf_str:>7s} {trades_str:>7s} {inmkt_str:>7s} {avglev_str:>7s} {avgconf_str:>8s} {skip_str:>7s}{marker}")
    
    for name in failed:
        print(f"  {name:<25s} {'FAILED':>10s}  — {failed[name].get('error', '?')}")
    
    # Best picks
    if sorted_keys:
        best = sorted_keys[0]
        r = valid[best]
        print(f"\n  🥇 BEST: {best}")
        print(f"     Return={r['total_return']:+.1f}% | Sharpe={r['sharpe_ratio']:.3f} | PF={r['profit_factor']:.2f} | MaxDD={r['max_drawdown']:.1f}%")
        print(f"     Config: TF={r['config']['tf']}, States={r['config']['n_states']}, Cov={r['config']['cov']}, Features={r['config']['features']}, Lookback={r['config']['lookback']}")
    
    # Group bests
    print(f"\n  {'─' * 60}")
    print("  📈 GROUP WINNERS:")
    for group in ["A", "B", "C", "D", "E", "F"]:
        group_keys = [k for k in sorted_keys if k.startswith(group)]
        if group_keys:
            best_g = group_keys[0]
            r = valid[best_g]
            group_names = {"A": "Timeframe", "B": "HMM States", "C": "Features", "D": "Covariance", "E": "Lookback", "F": "Best Combos"}
            print(f"    {group_names.get(group, group)}: {best_g} → Sharpe={r['sharpe_ratio']:.3f}, Return={r['total_return']:+.1f}%, PF={r['profit_factor']:.2f}")
    
    print("=" * 120 + "\n")

if __name__ == "__main__":
    main()
