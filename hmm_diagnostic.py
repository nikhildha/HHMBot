"""
SENTINEL — Comprehensive HMM Model Diagnostic
Tests whether the HMM regime detection makes practical sense.

Checks:
  1. Regime Stability — Do regimes persist or flip randomly?
  2. Feature Separation — Are BULL/BEAR/CHOP/CRASH statistically distinct?
  3. Look-Ahead Bias — In-sample vs Out-of-sample performance
  4. Regime-Return Alignment — Does BULL actually have positive returns?
  5. Transition Matrix Realism — Self-transition should dominate
  6. Confidence Distribution — Is confidence meaningful or always ~1.0?
  7. Random Baseline — Does HMM beat random regime assignment?
  8. Backtest Sanity — Are the astronomical returns real or artifacts?
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.WARNING)

import config
from data_pipeline import fetch_futures_klines
from feature_engine import compute_all_features, compute_hmm_features
from hmm_brain import HMMBrain, HMM_FEATURES
from backtester import backtest_hmm_strategy
from risk_manager import RiskManager

W = 100
DIV = "=" * W
SUBDIV = "-" * W

def header(title):
    print(f"\n{DIV}")
    print(f"  {title}")
    print(DIV)

def subheader(title):
    print(f"\n  {title}")
    print(f"  {SUBDIV[:len(title)+4]}")

# ═══════════════════════════════════════════════════════════════════════════════
#  FETCH DATA
# ═══════════════════════════════════════════════════════════════════════════════

header("📡 FETCHING DATA — BTCUSDT 1h (1000 candles ≈ 42 days)")
df_raw = fetch_futures_klines("BTCUSDT", "1h", limit=1000)
if df_raw is None or len(df_raw) < 200:
    print("  ❌ Could not fetch enough data. Aborting.")
    sys.exit(1)
print(f"  ✅ Got {len(df_raw)} candles | {df_raw['close'].iloc[0]:.0f} → {df_raw['close'].iloc[-1]:.0f}")

df = compute_all_features(df_raw)
df_hmm = compute_hmm_features(df_raw)

# Split: 70% train, 30% test (chronological, no shuffle)
split = int(len(df) * 0.7)
df_train, df_test = df.iloc[:split].copy(), df.iloc[split:].copy()
df_hmm_train, df_hmm_test = df_hmm.iloc[:split].copy(), df_hmm.iloc[split:].copy()

print(f"  📊 Train: {split} candles | Test: {len(df) - split} candles")

# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1: TRAIN HMM & BASIC SANITY
# ═══════════════════════════════════════════════════════════════════════════════

header("🧠 TEST 1: HMM Training & Basic Sanity")

brain = HMMBrain(n_states=4)
brain.train(df_hmm_train)

if not brain.is_trained:
    print("  ❌ HMM failed to train!")
    sys.exit(1)

print("  ✅ HMM trained successfully")
print(f"  State map (raw → canonical): {brain._state_map}")

# Show state means (unscaled)
means = brain.model.means_
stds = np.sqrt(np.array([np.diag(c) for c in brain.model.covars_]))
print(f"\n  {'State':<12} {'Mean LogRet':>12} {'Mean Vol':>12} {'Mean VolChg':>12} {'Mean RSI_n':>12}")
print(f"  {'-'*60}")
for raw_state in range(4):
    canonical = brain._state_map.get(raw_state, -1)
    name = config.REGIME_NAMES.get(canonical, "???")
    m = means[raw_state]
    print(f"  {name:<12} {m[0]:>12.6f} {m[1]:>12.6f} {m[2]:>12.6f} {m[3]:>12.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2: REGIME STABILITY (do regimes persist?)
# ═══════════════════════════════════════════════════════════════════════════════

header("📊 TEST 2: Regime Stability — Do Regimes Persist or Flip Randomly?")

states_all = brain.predict_all(df_hmm)
n = len(states_all)

# Count regime transitions
transitions = np.sum(np.diff(states_all) != 0)
persistence = 1 - (transitions / (n - 1))

# Average regime duration (in candles)
durations = []
current_dur = 1
for i in range(1, n):
    if states_all[i] == states_all[i-1]:
        current_dur += 1
    else:
        durations.append(current_dur)
        current_dur = 1
durations.append(current_dur)

avg_duration = np.mean(durations)
median_duration = np.median(durations)
max_duration = np.max(durations)

print(f"  Total transitions: {transitions} out of {n-1} bars")
print(f"  Persistence rate:  {persistence:.1%}")
print(f"  Avg regime duration:    {avg_duration:.1f} candles ({avg_duration:.1f}h)")
print(f"  Median regime duration: {median_duration:.0f} candles ({median_duration:.0f}h)")
print(f"  Max regime duration:    {max_duration} candles ({max_duration}h)")

if persistence < 0.7:
    print("  ⚠️  LOW PERSISTENCE — HMM is flipping states too often (noisy)")
elif persistence > 0.98:
    print("  ⚠️  OVERLY STICKY — HMM barely changes state (insensitive)")
else:
    print("  ✅ Good persistence rate — regimes are meaningful")

# Regime distribution
print(f"\n  {'Regime':<16} {'Count':>6} {'% Time':>8} {'Avg Duration':>14}")
print(f"  {'-'*50}")
for state_val, state_name in config.REGIME_NAMES.items():
    mask = states_all == state_val
    count = mask.sum()
    # Calculate durations for this regime
    regime_durations = []
    d = 0
    for s in states_all:
        if s == state_val:
            d += 1
        else:
            if d > 0:
                regime_durations.append(d)
            d = 0
    if d > 0:
        regime_durations.append(d)
    avg_d = np.mean(regime_durations) if regime_durations else 0
    print(f"  {state_name:<16} {count:>6} {count/n*100:>7.1f}% {avg_d:>12.1f}h")

# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3: FEATURE SEPARATION — Are regimes statistically distinct?
# ═══════════════════════════════════════════════════════════════════════════════

header("🔬 TEST 3: Feature Separation — Are Regimes Statistically Distinct?")

df_feat_check = df.copy()
# predict_all drops NaN rows so may be shorter — pad from front
if len(states_all) < len(df_feat_check):
    pad = len(df_feat_check) - len(states_all)
    states_all_padded = np.concatenate([np.full(pad, config.REGIME_CHOP), states_all])
else:
    states_all_padded = states_all
df_feat_check["regime"] = states_all_padded

features_to_check = ["log_return", "volatility", "volume_change"]
print(f"\n  {'Feature':<16} {'BULL Mean':>12} {'BEAR Mean':>12} {'CHOP Mean':>12} {'CRASH Mean':>12} {'F-stat':>10} {'Verdict':>10}")
print(f"  {'-'*85}")

from scipy import stats as sp_stats

for feat in features_to_check:
    groups = []
    means_by_regime = {}
    for state_val, state_name in config.REGIME_NAMES.items():
        mask = df_feat_check["regime"] == state_val
        vals = df_feat_check.loc[mask, feat].dropna().values
        short_name = state_name.split("/")[0][:5]
        if len(vals) > 0:
            groups.append(vals)
            means_by_regime[short_name] = np.mean(vals)
        else:
            means_by_regime[short_name] = 0.0
    
    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
        f_stat, p_val = sp_stats.f_oneway(*groups)
        verdict = "✅ GOOD" if p_val < 0.05 else "⚠️ WEAK"
    else:
        f_stat, p_val = 0, 1
        verdict = "❌ N/A"
    
    print(f"  {feat:<16} {means_by_regime.get('BULLI', 0):>12.6f} {means_by_regime.get('BEARI', 0):>12.6f} "
          f"{means_by_regime.get('SIDEW', 0):>12.6f} {means_by_regime.get('CRASH', 0):>12.6f} "
          f"{f_stat:>10.1f} {verdict:>10}")

# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4: LOOK-AHEAD BIAS — In-Sample vs Out-of-Sample
# ═══════════════════════════════════════════════════════════════════════════════

header("🔍 TEST 4: Look-Ahead Bias — In-Sample vs Out-of-Sample")

# In-sample: train on all, predict on all (what the current backtester does)
brain_insample = HMMBrain(n_states=4)
brain_insample.train(df_hmm)
results_insample = backtest_hmm_strategy(df, brain=brain_insample)

# Out-of-sample: train on first 70%, predict on last 30%
brain_oos = HMMBrain(n_states=4)
brain_oos.train(df_hmm_train)
results_oos = backtest_hmm_strategy(df_test, brain=brain_oos)

print(f"\n  {'Metric':<22} {'In-Sample (cheating)':>22} {'Out-of-Sample (true)':>22} {'Verdict':>10}")
print(f"  {'-'*80}")

metrics = [
    ("Total Return", f"{results_insample['total_return']:.1f}%", f"{results_oos['total_return']:.1f}%"),
    ("Max Drawdown", f"{results_insample['max_drawdown']:.1f}%", f"{results_oos['max_drawdown']:.1f}%"),
    ("Sharpe Ratio", f"{results_insample['sharpe_ratio']:.3f}", f"{results_oos['sharpe_ratio']:.3f}"),
    ("Profit Factor", f"{results_insample['profit_factor']:.3f}", f"{results_oos['profit_factor']:.3f}"),
    ("Trades", f"{results_insample['n_trades']}", f"{results_oos['n_trades']}"),
    ("Time in Market", f"{results_insample['time_in_market']}", f"{results_oos['time_in_market']}"),
    ("Avg Leverage", f"{results_insample['avg_leverage']:.1f}x", f"{results_oos['avg_leverage']:.1f}x"),
]

for name, insample, oos in metrics:
    print(f"  {name:<22} {insample:>22} {oos:>22}")

# Check if OOS significantly worse
is_ratio = results_insample['total_return']
oos_ratio = results_oos['total_return']
if is_ratio > 0 and oos_ratio < 0:
    print(f"\n  🚨 CRITICAL: In-sample positive but OOS negative → LOOK-AHEAD BIAS CONFIRMED")
    print(f"     The model is memorizing in-sample patterns, not learning generalizable regimes.")
elif is_ratio > 0 and oos_ratio > 0 and is_ratio / max(oos_ratio, 0.01) > 10:
    print(f"\n  ⚠️  WARNING: In-sample return is {is_ratio/max(oos_ratio,0.01):.0f}x better than OOS → likely overfitting")
else:
    print(f"\n  ✅ Reasonable in-sample vs OOS gap")

# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5: REGIME-RETURN ALIGNMENT — Does BULL = positive returns?
# ═══════════════════════════════════════════════════════════════════════════════

header("📈 TEST 5: Regime-Return Alignment — Does BULL = Positive Returns?")

# Use OUT-OF-SAMPLE predictions only
states_oos = brain_oos.predict_all(df_hmm_test)
df_test_check = df_test.copy()
# Pad if predict_all drops NaN rows
if len(states_oos) < len(df_test_check):
    pad = len(df_test_check) - len(states_oos)
    states_oos_padded = np.concatenate([np.full(pad, config.REGIME_CHOP), states_oos])
else:
    states_oos_padded = states_oos
df_test_check["regime"] = states_oos_padded

print(f"\n  {'Regime':<16} {'Avg Return':>12} {'Median Ret':>12} {'Std Dev':>10} {'Win Rate':>10} {'Correct?':>10}")
print(f"  {'-'*72}")

for state_val, state_name in config.REGIME_NAMES.items():
    mask = df_test_check["regime"] == state_val
    rets = df_test_check.loc[mask, "log_return"].dropna()
    if len(rets) > 0:
        avg_ret = rets.mean() * 100
        med_ret = rets.median() * 100
        std_ret = rets.std() * 100
        win_rate = (rets > 0).mean() * 100
        
        if state_val == config.REGIME_BULL:
            correct = "✅" if avg_ret > 0 else "❌"
        elif state_val == config.REGIME_BEAR:
            correct = "✅" if avg_ret < 0 else "❌"
        elif state_val == config.REGIME_CRASH:
            correct = "✅" if avg_ret < 0 and std_ret > 0.5 else "❌"
        else:
            correct = "✅" if abs(avg_ret) < std_ret else "⚠️"
        
        print(f"  {state_name:<16} {avg_ret:>11.4f}% {med_ret:>11.4f}% {std_ret:>9.4f}% {win_rate:>9.1f}% {correct:>10}")
    else:
        print(f"  {state_name:<16} {'— no data —':>50}")

# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 6: TRANSITION MATRIX REALISM
# ═══════════════════════════════════════════════════════════════════════════════

header("🔄 TEST 6: Transition Matrix — Is It Realistic?")

# Compute empirical transition matrix from OOS predictions
trans_matrix = np.zeros((4, 4))
for i in range(1, len(states_oos)):
    trans_matrix[states_oos[i-1], states_oos[i]] += 1

# Normalize rows
row_sums = trans_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
trans_matrix = trans_matrix / row_sums

print(f"\n  Empirical Transition Matrix (from OOS predictions):")
regime_short = ["BULL", "BEAR", "CHOP", "CRASH"]
print(f"  {'FROM \\ TO':<10}", end="")
for name in regime_short:
    print(f"{name:>8}", end="")
print()
print(f"  {'-'*42}")

for i, name in enumerate(regime_short):
    print(f"  {name:<10}", end="")
    for j in range(4):
        val = trans_matrix[i][j]
        marker = " ●" if val > 0.5 else ""
        print(f"{val:>7.2f}{marker}", end="")
    print()

# Check diagonal dominance (self-transitions should be highest)
diag_dominant = all(trans_matrix[i][i] == max(trans_matrix[i]) for i in range(4) if row_sums[i] > 0)
avg_self_trans = np.mean([trans_matrix[i][i] for i in range(4)])
print(f"\n  Average self-transition: {avg_self_trans:.2%}")
if diag_dominant and avg_self_trans > 0.5:
    print("  ✅ Diagonal dominant — regimes are persistent (realistic)")
elif avg_self_trans < 0.3:
    print("  ❌ Low self-transition — HMM is flipping states chaotically")
else:
    print("  ⚠️  Mixed — some regimes persist, others don't")

# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 7: CONFIDENCE DISTRIBUTION — Is It Meaningful?
# ═══════════════════════════════════════════════════════════════════════════════

header("📊 TEST 7: Confidence Distribution — Is Confidence Meaningful?")

# Get confidence for OOS data
feats_oos = df_hmm_test[HMM_FEATURES].dropna().values
feats_scaled = (feats_oos - brain_oos._feat_mean) / brain_oos._feat_std
posteriors = brain_oos.model.predict_proba(feats_scaled)
confidences = np.max(posteriors, axis=1)

print(f"\n  Total predictions: {len(confidences)}")
print(f"  Mean confidence:   {np.mean(confidences):.3f}")
print(f"  Median confidence: {np.median(confidences):.3f}")
print(f"  Min confidence:    {np.min(confidences):.3f}")
print(f"  Max confidence:    {np.max(confidences):.3f}")
print(f"  Std confidence:    {np.std(confidences):.3f}")

# Distribution buckets
buckets = [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.92), (0.92, 0.96), (0.96, 0.99), (0.99, 1.01)]
print(f"\n  {'Confidence Bucket':>20} {'Count':>8} {'% Total':>8}")
print(f"  {'-'*40}")
for lo, hi in buckets:
    count = np.sum((confidences >= lo) & (confidences < hi))
    pct = count / len(confidences) * 100
    bar = "█" * int(pct / 2)
    print(f"  {f'{lo:.0%}–{hi:.0%}':>20} {count:>8} {pct:>7.1f}% {bar}")

# Check if confidence is always >99% (meaningless)
high_conf_pct = np.mean(confidences > 0.99) * 100
if high_conf_pct > 90:
    print(f"\n  ⚠️  {high_conf_pct:.0f}% of predictions have >99% confidence")
    print(f"     Confidence is NOT discriminative — the model is overconfident.")
    print(f"     The leverage tiers (85/91/95%) are effectively bypassed.")
elif high_conf_pct > 50:
    print(f"\n  ⚠️  {high_conf_pct:.0f}% of predictions are >99% — somewhat overconfident")
else:
    print(f"\n  ✅ Confidence has meaningful spread — leverage tiers will differentiate")

# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 8: RANDOM BASELINE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

header("🎲 TEST 8: Does HMM Beat Random Regime Assignment?")

# Run 100 random simulations and compare
rng = np.random.RandomState(42)
random_sharpes = []
random_returns = []

for trial in range(100):
    random_states = rng.randint(0, 4, size=len(df_test))
    try:
        results_rand = backtest_hmm_strategy(df_test.copy(), states=random_states)
        random_sharpes.append(results_rand['sharpe_ratio'])
        random_returns.append(results_rand['total_return'])
    except:
        pass

hmm_sharpe = results_oos['sharpe_ratio']
hmm_return = results_oos['total_return']

print(f"\n  {'Metric':<22} {'HMM (OOS)':>15} {'Random Mean':>15} {'Random Median':>15} {'HMM Percentile':>15}")
print(f"  {'-'*85}")

if random_sharpes:
    rand_sharpe_mean = np.mean(random_sharpes)
    rand_sharpe_med = np.median(random_sharpes)
    hmm_percentile_sharpe = np.mean([s < hmm_sharpe for s in random_sharpes]) * 100
    
    rand_return_mean = np.mean(random_returns)
    rand_return_med = np.median(random_returns)
    hmm_percentile_return = np.mean([r < hmm_return for r in random_returns]) * 100
    
    print(f"  {'Sharpe Ratio':<22} {hmm_sharpe:>15.3f} {rand_sharpe_mean:>15.3f} {rand_sharpe_med:>15.3f} {hmm_percentile_sharpe:>14.0f}%")
    print(f"  {'Total Return':<22} {hmm_return:>14.1f}% {rand_return_mean:>14.1f}% {rand_return_med:>14.1f}% {hmm_percentile_return:>14.0f}%")
    
    if hmm_percentile_sharpe > 80:
        print(f"\n  ✅ HMM beats {hmm_percentile_sharpe:.0f}% of random baselines — model has genuine signal")
    elif hmm_percentile_sharpe > 50:
        print(f"\n  ⚠️  HMM beats {hmm_percentile_sharpe:.0f}% of random — marginal edge, possibly noise")
    else:
        print(f"\n  ❌ HMM does NOT beat random — model has NO predictive power")

# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 9: BACKTEST SANITY — Why are returns so high?
# ═══════════════════════════════════════════════════════════════════════════════

header("💡 TEST 9: Backtest Sanity — Why Are Returns So High?")

print("""
  The current backtester has these properties that inflate returns:

  1. ❌ IN-SAMPLE BIAS: Trains HMM on ALL data, then backtests on the SAME data.
     The model literally knows the future — it memorizes the exact regime sequence.

  2. ❌ NO SL/TP: The backtest applies raw leveraged returns per bar.
     With 25-35x leverage, even small correct predictions compound explosively.
     Real trading would hit stop losses and margin calls.

  3. ❌ NO POSITION SIZING: The backtest assumes 100% capital on every bar.
     Real trading uses 3-5% per position.

  4. ❌ UNLIMITED LEVERAGE: In the backtest, 35x leverage means returns multiply
     by 35 each bar. In reality, exchange limits + margin calls prevent this.

  5. ⚠️  BEAR = SHORT: The backtester goes SHORT during BEAR regimes with leverage.
     This doubles the alpha opportunity but doubles the risk.

  CONCLUSION: The +6,000,000% returns from Round 3 are MEANINGLESS.
  They are an artifact of in-sample fitting + unconstrained leverage compounding.
  The out-of-sample test above shows the REAL performance.
""")

# ═══════════════════════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

header("🏁 FINAL VERDICT")

print(f"""
  Summary of findings:

  HMM Model Architecture:
    ✅ Uses 4 reasonable features (log_return, volatility, volume_change, rsi_norm)
    ✅ State labeling by mean return is sound methodology
    ✅ Feature scaling prevents covariance issues
    
  Practical Issues:
    {'✅' if persistence > 0.7 and persistence < 0.98 else '⚠️ '} Regime Stability: {persistence:.1%} persistence
    {'✅' if avg_duration > 3 else '⚠️ '} Avg Duration: {avg_duration:.1f}h (need >3h to be tradeable)
    {'✅' if high_conf_pct < 50 else '⚠️ '} Confidence Spread: {high_conf_pct:.0f}% above 99%
    {'✅' if hmm_percentile_sharpe > 60 else '❌'} Beats Random: {hmm_percentile_sharpe:.0f}th percentile
    
  Backtest Issues:
    ❌ Current backtester trains + tests on SAME data (look-ahead bias)
    ❌ No stop-loss / take-profit in backtest (unrealistic returns)
    ❌ No position sizing (100% capital per bar)
    
  OOS Performance:
    Return: {results_oos['total_return']:.1f}%
    Sharpe: {results_oos['sharpe_ratio']:.3f}
    Max DD: {results_oos['max_drawdown']:.1f}%
""")

if results_oos['sharpe_ratio'] > 0.5 and hmm_percentile_sharpe > 60:
    print("  🟢 VERDICT: HMM has some genuine signal but astronomical backtest returns are fake.")
    print("     RECOMMENDATION: Fix backtester to use walk-forward testing with proper SL/TP.")
elif results_oos['sharpe_ratio'] > 0:
    print("  🟡 VERDICT: HMM has marginal edge. Keep the model but don't trust the backtest numbers.")
    print("     RECOMMENDATION: Use very conservative leverage (5-10x max) in live trading.")
else:
    print("  🔴 VERDICT: HMM shows NO out-of-sample edge. Exercise extreme caution in live trading.")
    print("     RECOMMENDATION: Reduce leverage drastically or disable live trading until model improves.")

print(f"\n{DIV}\n")
