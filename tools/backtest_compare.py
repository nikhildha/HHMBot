"""
HMM Feature Comparison — Before vs After Bug Fix
================================================
Demonstrates the impact of adding log_return to HMM_FEATURES.

BUG:  HMM_FEATURES = ["volatility", "volume_change", "rsi_norm"]
      _build_state_map() assumes means_[:,0] = log_return  → WRONG
      Regime labels sorted by VOLATILITY not by RETURN DIRECTION

FIX:  HMM_FEATURES = ["log_return", "volatility", "volume_change", "rsi_norm"]
      means_[:,0] = log_return  → states correctly sorted by RETURN

Run:  python tools/backtest_compare.py
"""
import sys
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hmmlearn.hmm import GaussianHMM
from feature_engine import compute_hmm_features, compute_rsi

# ─── Synthetic market data generator ─────────────────────────────────────────

def _make_synthetic_ohlcv(n, mean_ret, vol, seed_offset=0):
    """Generate synthetic OHLCV candles with a fixed regime."""
    np.random.seed(42 + seed_offset)
    returns = np.random.normal(mean_ret, vol, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high  = close * (1 + np.abs(np.random.normal(0, vol * 0.5, n)))
    low   = close * (1 - np.abs(np.random.normal(0, vol * 0.5, n)))
    volume = np.random.lognormal(8, 0.4, n)
    df = pd.DataFrame({"open": close * 0.999, "high": high, "low": low,
                       "close": close, "volume": volume})
    return df

def make_full_dataset():
    """
    Stitch together 4 regime periods with known ground truth.
    Returns (df, labels) where labels = 0=BULL, 1=BEAR, 2=CHOP, 3=CRASH
    """
    periods = [
        ("BULL",  200, +0.005, 0.012, 0),   # positive drift, moderate vol
        ("CHOP",  200, +0.000, 0.005, 1),   # zero drift, low vol
        ("BEAR",  200, -0.003, 0.015, 2),   # negative drift, moderate-high vol
        ("CRASH", 100, -0.015, 0.035, 3),   # strong negative drift, very high vol
    ]
    dfs, true_labels = [], []
    label_map = {"BULL": 0, "CHOP": 2, "BEAR": 1, "CRASH": 3}
    for name, n, mean_ret, vol, offset in periods:
        df = _make_synthetic_ohlcv(n, mean_ret, vol, seed_offset=offset)
        dfs.append(df)
        true_labels.extend([label_map[name]] * n)

    combined = pd.concat(dfs, ignore_index=True)
    return combined, np.array(true_labels)


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_hmm(features_array, n_states=4):
    """Train a GaussianHMM and return the fitted model."""
    feat_mean = features_array.mean(axis=0)
    feat_std  = features_array.std(axis=0)
    feat_std[feat_std < 1e-10] = 1e-10
    scaled = (features_array - feat_mean) / feat_std

    model = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=100, random_state=42)
    model.fit(scaled)
    return model, feat_mean, feat_std, scaled


def build_state_map_by_col0(model):
    """Map raw states → canonical regimes using means[:,0] for sorting."""
    means = model.means_[:, 0]
    vols  = model.means_[:, 1]
    sorted_indices = np.argsort(means)[::-1]   # highest first

    ranked = list(sorted_indices)
    mid = ranked[1:3]
    if vols[mid[0]] <= vols[mid[1]]:
        chop_raw, bear_raw = mid[0], mid[1]
    else:
        chop_raw, bear_raw = mid[1], mid[0]

    regime_names = {0: "BULL", 1: "BEAR", 2: "CHOP", 3: "CRASH"}
    return {
        ranked[0]:  0,   # BULL
        bear_raw:   1,   # BEAR
        chop_raw:   2,   # CHOP
        ranked[-1]: 3,   # CRASH
    }, regime_names


def predict_regimes(model, scaled_features, state_map):
    raw_states = model.predict(scaled_features)
    return np.array([state_map.get(s, 2) for s in raw_states])


def accuracy(predictions, ground_truth, period_bounds):
    """Compute per-period accuracy (what % of candles in each period got the right label)."""
    results = {}
    for name, start, end in period_bounds:
        true = ground_truth[start:end]
        pred = predictions[start:end]
        acc = (pred == true).mean() * 100
        results[name] = acc
    return results


# ─── Main comparison ─────────────────────────────────────────────────────────

def run_comparison():
    print("=" * 65)
    print("  HMM FEATURE BUG FIX — BEFORE vs AFTER COMPARISON")
    print("=" * 65)

    # Build synthetic dataset
    df_raw, ground_truth = make_full_dataset()
    df_feat = compute_hmm_features(df_raw)
    df_feat = df_feat.dropna().reset_index(drop=True)
    gt = ground_truth[:len(df_feat)]  # align after dropna

    period_bounds = [
        ("BULL",  0,   200),
        ("CHOP",  200, 400),
        ("BEAR",  400, 600),
        ("CRASH", 600, 700),
    ]
    # Adjust for dropna trim (first few rows become NaN)
    trim = len(ground_truth) - len(gt)
    period_bounds = [(n, max(0, s - trim), min(len(gt), e - trim))
                     for n, s, e in period_bounds]

    # ── BEFORE: 3 features (volatility first → col 0 is volatility, NOT log_return)
    print("\n[BEFORE] HMM_FEATURES = ['volatility', 'volume_change', 'rsi_norm']")
    print("         _build_state_map uses means[:,0] thinking it is log_return")
    print("         ACTUAL means[:,0] = volatility  ← BUG\n")

    before_feats = ["volatility", "volume_change", "rsi_norm"]
    before_arr   = df_feat[before_feats].values
    before_model, bm, bs, before_scaled = train_hmm(before_arr)
    before_map, _ = build_state_map_by_col0(before_model)

    before_means_col0 = before_model.means_[:, 0]
    print(f"  State means for col 0 (volatility — NOT log_return):")
    for s in range(4):
        canonical = before_map.get(s, 2)
        names = ["BULL","BEAR","CHOP","CRASH"]
        print(f"    Raw state {s} → labeled {names[canonical]:5s} | col0 mean = {before_means_col0[s]:+.6f}")

    before_preds = predict_regimes(before_model, before_scaled, before_map)
    before_acc = accuracy(before_preds, gt, period_bounds)

    print("\n  Per-period labeling accuracy (BEFORE):")
    total_before = 0
    weights = {"BULL": 200, "CHOP": 200, "BEAR": 200, "CRASH": 100}
    for name, acc in before_acc.items():
        bar = "#" * int(acc / 5)
        print(f"    {name:5s}: {acc:5.1f}%  {bar}")
        total_before += acc * weights[name]
    wavg_before = total_before / sum(weights.values())
    print(f"  Weighted avg accuracy (BEFORE): {wavg_before:.1f}%")

    # ── AFTER: 4 features (log_return first → col 0 is log_return, as intended)
    print("\n" + "-" * 65)
    print("\n[AFTER]  HMM_FEATURES = ['log_return', 'volatility', 'volume_change', 'rsi_norm']")
    print("         _build_state_map uses means[:,0] = log_return  ← CORRECT\n")

    after_feats = ["log_return", "volatility", "volume_change", "rsi_norm"]
    after_arr   = df_feat[after_feats].values
    after_model, am, as_, after_scaled = train_hmm(after_arr)
    after_map, _ = build_state_map_by_col0(after_model)

    after_means_col0 = after_model.means_[:, 0]
    print(f"  State means for col 0 (log_return — correctly used):")
    for s in range(4):
        canonical = after_map.get(s, 2)
        names = ["BULL","BEAR","CHOP","CRASH"]
        print(f"    Raw state {s} → labeled {names[canonical]:5s} | log_return mean = {after_means_col0[s]:+.6f}")

    after_preds = predict_regimes(after_model, after_scaled, after_map)
    after_acc = accuracy(after_preds, gt, period_bounds)

    print("\n  Per-period labeling accuracy (AFTER):")
    total_after = 0
    for name, acc in after_acc.items():
        bar = "#" * int(acc / 5)
        print(f"    {name:5s}: {acc:5.1f}%  {bar}")
        total_after += acc * weights[name]
    wavg_after = total_after / sum(weights.values())
    print(f"  Weighted avg accuracy (AFTER):  {wavg_after:.1f}%")

    # ── Confidence comparison
    print("\n" + "-" * 65)
    print("\n  Confidence score comparison (higher = more decisive signals):")

    before_proba = before_model.predict_proba(before_scaled)
    after_proba  = after_model.predict_proba(after_scaled)
    before_conf = before_proba.max(axis=1).mean()
    after_conf  = after_proba.max(axis=1).mean()
    print(f"    Mean confidence  BEFORE: {before_conf:.4f}")
    print(f"    Mean confidence  AFTER:  {after_conf:.4f}")
    conf_delta = (after_conf - before_conf) / before_conf * 100
    print(f"    Change: {conf_delta:+.1f}%")

    # ── Summary
    print("\n" + "=" * 65)
    acc_delta = wavg_after - wavg_before
    print(f"\n  REGIME LABEL ACCURACY:  {wavg_before:.1f}%  →  {wavg_after:.1f}%  ({acc_delta:+.1f} pp)")
    print(f"  MEAN CONFIDENCE:        {before_conf:.4f}  →  {after_conf:.4f}  ({conf_delta:+.1f}%)")
    if wavg_after > wavg_before:
        print("\n  RESULT: Bug fix IMPROVES regime classification accuracy.")
    else:
        print("\n  RESULT: No accuracy change (model may have converged similarly).")
    print("=" * 65)
    return wavg_before, wavg_after, before_conf, after_conf


if __name__ == "__main__":
    run_comparison()
