"""
Project Regime-Master — Experiment Script for S/R and VWAP features.

Runs a walk-forward backtest twice (with and without VWAP/SR filters) and compares the results.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import fetch_futures_klines
from backtester_v2 import walk_forward_backtest, print_report

def main():
    print("\n============================================================")
    print("  🔬 VWAP & S/R Strategy Impact Analysis")
    print("============================================================")
    
    print("\n  [1/3] Fetching BTCUSDT 1h data (800 candles)...")
    df_raw = fetch_futures_klines("BTCUSDT", "1h", limit=800)
    
    if df_raw is None or len(df_raw) < 500:
        print("  ❌ Could not fetch enough data for reliable backtest.")
        sys.exit(1)
        
    print(f"  ✅ Picked up {len(df_raw)} candles of historical data.")
    
    # Common parameters
    train_win = 300
    test_win = 100
    step = 50
    cap = 10000
    
    print("\n  [2/3] Running BASELINE Model (Without S/R and VWAP Bias)...")
    start_time = time.time()
    results_base = walk_forward_backtest(
        df_raw, train_window=train_win, test_window=test_win, step=step,
        initial_capital=cap, verbose=False, use_sr_vwap_filter=False
    )
    print(f"        Baseline run complete in {time.time() - start_time:.1f}s.")
    
    print("\n  [3/3] Running ENHANCED Model (With S/R and VWAP Bias)...")
    start_time = time.time()
    results_enhanced = walk_forward_backtest(
        df_raw, train_window=train_win, test_window=test_win, step=step,
        initial_capital=cap, verbose=False, use_sr_vwap_filter=True
    )
    print(f"        Enhanced run complete in {time.time() - start_time:.1f}s.")
    
    print("\n============================================================")
    print("  📊 EXPERIMENT CONCLUSION: Impact of VWAP + S/R Bias")
    print("============================================================")
    
    print(f"  {'Metric':<20} | {'Baseline':>12} | {'Enhanced':>12} | {'Diff':>10}")
    print("-" * 62)
    
    ret_b, ret_e = results_base['total_return'], results_enhanced['total_return']
    dd_b, dd_e = results_base['max_drawdown'], results_enhanced['max_drawdown']
    sh_b, sh_e = results_base['sharpe_ratio'], results_enhanced['sharpe_ratio']
    wr_b, wr_e = results_base['win_rate'], results_enhanced['win_rate']
    tr_b, tr_e = results_base['total_trades'], results_enhanced['total_trades']
    
    print(f"  {'Total Return':<20} | {ret_b:>11.2f}% | {ret_e:>11.2f}% | {(ret_e - ret_b):>9.2f}%")
    print(f"  {'Max Drawdown':<20} | {dd_b:>11.2f}% | {dd_e:>11.2f}% | {(dd_e - dd_b):>9.2f}%")
    print(f"  {'Sharpe Ratio':<20} | {sh_b:>12.3f} | {sh_e:>12.3f} | {(sh_e - sh_b):>10.3f}")
    print(f"  {'Win Rate':<20} | {wr_b:>11.1f}% | {wr_e:>11.1f}% | {(wr_e - wr_b):>9.1f}%")
    print(f"  {'Total Trades':<20} | {tr_b:>12} | {tr_e:>12} | {(tr_e - tr_b):>10}")
    print("-" * 62)
    
    # Simple recommendation engine based on results
    print("\n  🧠 AI VERDICT:")
    if sh_e > sh_b + 0.1 and ret_e > ret_b:
        print("  🟢 STRONGLY RECOMMENDED: VWAP & S/R logic substantially improves profitability and risk-adjusted returns.")
    elif sh_e > sh_b:
        print("  🟡 MINOR IMPROVEMENT: The filter reduces noise slightly and improves Sharpe.")
    elif sh_e < sh_b - 0.1:
        print("  🔴 NOT RECOMMENDED: The filter introduces too much noise or misses critical entries, degrading performance.")
    else:
        print("  ⚪ NEUTRAL / INCONCLUSIVE: The filter yields roughly the same performance.")

    print("\n============================================================\n")


if __name__ == "__main__":
    main()
