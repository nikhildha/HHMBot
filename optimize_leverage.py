"""
SENTINEL — Leverage + Config Optimizer
Runs walk-forward backtest across multiple leverage configs to find optimal settings.
Uses backtester_v2 (realistic SL/TP, trailing stops, position sizing).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import config
from data_pipeline import fetch_futures_klines
from backtester_v2 import walk_forward_backtest

W = 100
DIV = "=" * W

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGS TO TEST
# ═══════════════════════════════════════════════════════════════════════════════

CONFIGS = [
    # (name, LEVERAGE_HIGH, LEVERAGE_MODERATE, LEVERAGE_LOW, CONF_HIGH, CONF_MED, CONF_LOW)
    ("Current_35_25_15",   35, 25, 15, 0.99, 0.96, 0.92),
    ("Moderate_15_10_5",   15, 10,  5, 0.99, 0.96, 0.92),
    ("Conservative_10_7_3", 10, 7,  3, 0.99, 0.96, 0.92),
    ("Safe_5_3_2",          5,  3,  2, 0.99, 0.96, 0.92),
    ("Ultra_Safe_3_2_1",    3,  2,  1, 0.99, 0.96, 0.92),
    ("No_Leverage_1_1_1",   1,  1,  1, 0.99, 0.96, 0.92),
]

print(f"\n{DIV}")
print(f"  🔬 LEVERAGE OPTIMIZER — Walk-Forward with Realistic SL/TP")
print(f"  Testing {len(CONFIGS)} configs × walk-forward folds")
print(DIV)

# ─── Fetch real data ──────────────────────────────────────────────────────
print("\n  📡 Fetching BTCUSDT 1h (1000 candles)...")
df_raw = fetch_futures_klines("BTCUSDT", "1h", limit=1000)
if df_raw is None or len(df_raw) < 600:
    print("  ❌ Could not fetch enough data.")
    sys.exit(1)
print(f"  ✅ Got {len(df_raw)} candles | {df_raw['close'].iloc[0]:.0f} → {df_raw['close'].iloc[-1]:.0f}")

# Market return for baseline
market_return = (df_raw['close'].iloc[-1] / df_raw['close'].iloc[0] - 1) * 100
print(f"  📈 Market (buy & hold): {market_return:+.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ALL CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════

results = []

for name, lev_h, lev_m, lev_l, conf_h, conf_m, conf_l in CONFIGS:
    print(f"\n  ⏳ Testing {name}...", end="", flush=True)
    
    # Temporarily override config
    orig = (config.LEVERAGE_HIGH, config.LEVERAGE_MODERATE, config.LEVERAGE_LOW,
            config.CONFIDENCE_HIGH, config.CONFIDENCE_MEDIUM, config.CONFIDENCE_LOW)
    
    config.LEVERAGE_HIGH = lev_h
    config.LEVERAGE_MODERATE = lev_m
    config.LEVERAGE_LOW = lev_l
    config.CONFIDENCE_HIGH = conf_h
    config.CONFIDENCE_MEDIUM = conf_m
    config.CONFIDENCE_LOW = conf_l
    
    try:
        r = walk_forward_backtest(
            df_raw,
            train_window=400,
            test_window=100,
            step=100,
            initial_capital=10000,
        )
        results.append((name, lev_h, lev_m, lev_l, r))
        print(f" Return: {r['total_return']:+.1f}% | DD: {r['max_drawdown']:.1f}% | "
              f"Sharpe: {r['sharpe_ratio']:.3f} | WR: {r['win_rate']:.0f}% | Trades: {r['total_trades']}")
    except Exception as e:
        print(f" ❌ Error: {e}")
    finally:
        # Restore config
        (config.LEVERAGE_HIGH, config.LEVERAGE_MODERATE, config.LEVERAGE_LOW,
         config.CONFIDENCE_HIGH, config.CONFIDENCE_MEDIUM, config.CONFIDENCE_LOW) = orig

# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{DIV}")
print(f"  📊 RESULTS — Walk-Forward with Realistic SL/TP/Trailing")
print(DIV)

print(f"\n  {'Config':<25} {'Lev':>8} {'Return':>10} {'MaxDD':>10} {'Sharpe':>8} {'PF':>8} {'WR':>8} {'Trades':>8} {'SL':>5} {'TP':>5} {'Trail':>6} {'Rating':>8}")
print(f"  {'─'*110}")

best = None
for name, lev_h, lev_m, lev_l, r in results:
    rating = "🟢" if r['total_return'] > 0 and r['sharpe_ratio'] > 0.3 else \
             "🟡" if r['total_return'] > -10 else "🔴"
    
    lev_str = f"{lev_h}/{lev_m}/{lev_l}"
    print(f"  {name:<25} {lev_str:>8} {r['total_return']:>+9.1f}% {r['max_drawdown']:>9.1f}% "
          f"{r['sharpe_ratio']:>8.3f} {r['profit_factor']:>8.3f} {r['win_rate']:>7.1f}% "
          f"{r['total_trades']:>8} {r['sl_hits']:>5} {r['tp_hits']:>5} {r.get('trail_activated',0):>6} {rating:>8}")
    
    if best is None or r['sharpe_ratio'] > best[1]['sharpe_ratio']:
        best = (name, r, lev_h, lev_m, lev_l)

print(f"\n  📈 Market Buy & Hold: {market_return:+.1f}%")

if best:
    print(f"\n  {'─'*80}")
    name, r, lev_h, lev_m, lev_l = best
    print(f"  🏆 BEST CONFIG: {name} ({lev_h}/{lev_m}/{lev_l}x)")
    print(f"     Return: {r['total_return']:+.1f}% | Sharpe: {r['sharpe_ratio']:.3f} | "
          f"MaxDD: {r['max_drawdown']:.1f}% | WinRate: {r['win_rate']:.1f}%")
    
    if r['total_return'] > market_return:
        print(f"     ✅ Beats buy & hold ({market_return:+.1f}%)")
    elif r['total_return'] > 0:
        print(f"     ⚠️  Positive but doesn't beat buy & hold ({market_return:+.1f}%)")
    else:
        print(f"     ❌ Negative return — model needs improvement")
    
    print(f"\n  📋 RECOMMENDED config.py UPDATE:")
    print(f"     LEVERAGE_HIGH = {lev_h}")
    print(f"     LEVERAGE_MODERATE = {lev_m}")
    print(f"     LEVERAGE_LOW = {lev_l}")

print(f"\n{DIV}\n")
