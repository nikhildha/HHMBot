"""
Round 3: Timeframe Comparison — 15m vs 30m vs 1h
==================================================
Runs the top 8 leverage/confidence configs on all three timeframes.
"""
import sys, os, json, time, warnings, numpy as np, pandas as pd, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_pipeline import fetch_futures_klines
from feature_engine import compute_atr, compute_rsi
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Exp3")
logger.setLevel(logging.INFO)

FEAT_COLS = ["log_return", "volatility", "volume_change", "rsi_norm"]

def compute_4feat(df):
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = (df["high"] - df["low"]) / df["close"]
    df["volume_change"] = np.log(df["volume"] / df["volume"].shift(1).replace(0, np.nan)).fillna(0).clip(-3, 3)
    rsi = compute_rsi(df["close"], length=14)
    df["rsi_norm"] = ((rsi - 50) / 50).fillna(0)
    return df

class FlexHMM:
    def __init__(self, n_states=4, cov="full"):
        self.n_states, self.cov = n_states, cov
        self.model = self._state_map = self._feat_mean = self._feat_std = None
        self._is_trained = False
    def train(self, df):
        f = df[FEAT_COLS].dropna().values
        if len(f) < 50: return self
        self._feat_mean, self._feat_std = f.mean(0), f.std(0)
        self._feat_std[self._feat_std < 1e-10] = 1e-10
        self.model = GaussianHMM(n_components=self.n_states, covariance_type=self.cov, n_iter=100, random_state=42)
        self.model.fit((f - self._feat_mean) / self._feat_std)
        m = self.model.means_[:, 0]; v = self.model.means_[:, 1]
        r = list(np.argsort(m)[::-1]); mid = r[1:3]
        ch, be = (mid[0], mid[1]) if v[mid[0]] <= v[mid[1]] else (mid[1], mid[0])
        self._state_map = {r[0]: 0, be: 1, ch: 2, r[-1]: 3}
        self._is_trained = True; return self
    def predict(self, df):
        f = df[FEAT_COLS].dropna().values
        s = (f - self._feat_mean) / self._feat_std
        raw = self.model.predict(s); probs = self.model.predict_proba(s)
        return np.array([self._state_map.get(x, 2) for x in raw]), np.max(probs, axis=1)

def get_lev(c, s, lt, ct):
    if s == 3: return 0
    if s == 2: return lt[2] if c >= ct[2] else 0
    if c >= ct[0]: return lt[0]
    elif c >= ct[1]: return lt[1]
    elif c >= ct[2]: return lt[2]
    return 0

def backtest(df, brain, lt, ct):
    df = df.copy()
    states, confs = brain.predict(df)
    n = len(df)
    if len(states) < n:
        p = n - len(states); states = np.concatenate([np.full(p, 2), states]); confs = np.concatenate([np.full(p, 0.5), confs])
    df["lr"] = np.log(df["close"] / df["close"].shift(1))
    rets, levs = np.zeros(n), np.zeros(n); prev = False
    for i in range(n):
        s, c = int(states[i]), confs[i]
        lr = df["lr"].iloc[i] if not pd.isna(df["lr"].iloc[i]) else 0
        lv = get_lev(c, s, lt, ct); levs[i] = lv
        if lv == 0: prev = False; continue
        if not prev: rets[i] -= 0.001
        prev = True
        if s == 0: rets[i] += lr * lv
        elif s == 1: rets[i] += lr * -lv
        elif s == 2: rets[i] += lr * lv * 0.3
    eq = np.exp(np.cumsum(rets)); rm = np.maximum.accumulate(eq)
    dd = np.min((eq - rm) / rm) * 100; sr = rets[~np.isnan(rets)]
    sharpe = (np.mean(sr) / np.std(sr) * np.sqrt(365*24)) if np.std(sr) > 0 else 0
    g = np.sum(sr[sr > 0]); l = abs(np.sum(sr[sr < 0]))
    pf = g / l if l > 0 else float("inf")
    im = (levs > 0).astype(int); trades = int((np.diff(im, prepend=0) == 1).sum())
    active = int((levs > 0).sum())
    return {
        "ret": round((eq[-1]-1)*100, 2), "dd": round(dd, 2), "sharpe": round(sharpe, 3),
        "pf": round(pf, 3), "trades": trades, "inmkt": round(active/n*100, 1),
        "avglev": round(float(levs[levs>0].mean()) if active > 0 else 0, 1),
    }

# Top configs from Round 2
CONFIGS = [
    {"name": "Balanced_35_25_15_@92",     "lev": (35, 25, 15), "conf": (0.99, 0.96, 0.92)},
    {"name": "Conservative_20_15_10_@95", "lev": (20, 15, 10), "conf": (0.99, 0.97, 0.95)},
    {"name": "Aggressive_50_35_25_@95",   "lev": (50, 35, 25), "conf": (0.99, 0.97, 0.95)},
    {"name": "Current_35_25_15_@85",      "lev": (35, 25, 15), "conf": (0.95, 0.91, 0.85)},
    {"name": "MidRange_40_30_20_@92",     "lev": (40, 30, 20), "conf": (0.98, 0.95, 0.92)},
    {"name": "Safe_10_8_5_@90",           "lev": (10, 8, 5),   "conf": (0.97, 0.93, 0.90)},
]

TIMEFRAMES = ["15m", "30m", "1h"]

def main():
    print("\n" + "=" * 120)
    print("  🔬 ROUND 3: TIMEFRAME COMPARISON — 15m vs 30m vs 1h")
    print("  Base: 4-feat / full cov / 500 lookback | 6 configs × 3 timeframes = 18 experiments")
    print("=" * 120)

    # Fetch and train per timeframe
    brains = {}
    dfs = {}
    for tf in TIMEFRAMES:
        logger.info(f"📡 Fetching {tf}...")
        raw = fetch_futures_klines("BTCUSDT", tf, limit=1000)
        if raw is None or len(raw) < 200:
            print(f"  ❌ {tf}: failed"); continue
        df = compute_4feat(raw)
        brain = FlexHMM(4, "full")
        brain.train(df.iloc[:500])
        if not brain._is_trained:
            print(f"  ❌ {tf}: HMM training failed"); continue
        brains[tf] = brain
        dfs[tf] = df
        print(f"  ✅ {tf}: {len(raw)} candles, HMM trained")

    # Run all combos
    results = {}
    for cfg in CONFIGS:
        for tf in TIMEFRAMES:
            if tf not in brains: continue
            key = f"{cfg['name']}_{tf}"
            r = backtest(dfs[tf], brains[tf], cfg["lev"], cfg["conf"])
            r["tf"] = tf; r["config_name"] = cfg["name"]
            results[key] = r

    # Print comparison grouped by config
    print("\n" + "=" * 120)
    print("  📊 SIDE-BY-SIDE COMPARISON: 15m vs 30m vs 1h")
    print("=" * 120)
    
    print(f"\n  {'Config':<32s} {'TF':>4s} {'Return':>11s} {'MaxDD':>8s} {'Sharpe':>8s} {'PF':>6s} {'Trades':>7s} {'InMkt':>6s} {'AvgLev':>7s} {'Rating':>8s}")
    print("  " + "─" * 114)

    for cfg in CONFIGS:
        for tf in TIMEFRAMES:
            key = f"{cfg['name']}_{tf}"
            if key not in results: continue
            r = results[key]
            dd_flag = "🟢" if r["dd"] > -30 else ("🟡" if r["dd"] > -50 else "🔴")
            best_marker = ""
            # Check if this is best TF for this config
            tf_results = {t: results.get(f"{cfg['name']}_{t}") for t in TIMEFRAMES if f"{cfg['name']}_{t}" in results}
            if tf_results:
                best_tf = max(tf_results, key=lambda t: tf_results[t]["sharpe"])
                if best_tf == tf: best_marker = " ⭐"
            print(f"  {cfg['name']:<32s} {tf:>4s} {r['ret']:>+10.1f}% {r['dd']:>7.1f}% {r['sharpe']:>8.3f} {r['pf']:>6.2f} {r['trades']:>7d} {r['inmkt']:>5.0f}% {r['avglev']:>6.1f}x {dd_flag}{best_marker}")
        print("  " + "·" * 114)

    # Overall winner per TF
    print(f"\n  {'─' * 60}")
    print("  🏆 BEST CONFIG PER TIMEFRAME (by Sharpe):")
    for tf in TIMEFRAMES:
        tf_res = {k: v for k, v in results.items() if v.get("tf") == tf}
        if tf_res:
            best = max(tf_res, key=lambda k: tf_res[k]["sharpe"])
            r = tf_res[best]
            print(f"    {tf}: {r['config_name']} → Sharpe={r['sharpe']:.3f}, Return={r['ret']:+.1f}%, DD={r['dd']:.1f}%")

    print(f"\n  🏆 BEST TIMEFRAME PER CONFIG (⭐ marked above):")
    for cfg in CONFIGS:
        tf_res = {tf: results[f"{cfg['name']}_{tf}"] for tf in TIMEFRAMES if f"{cfg['name']}_{tf}" in results}
        if tf_res:
            best_tf = max(tf_res, key=lambda t: tf_res[t]["sharpe"])
            r = tf_res[best_tf]
            print(f"    {cfg['name']}: {best_tf} → Sharpe={r['sharpe']:.3f}, Return={r['ret']:+.1f}%, DD={r['dd']:.1f}%")

    # Save
    out = os.path.join(config.DATA_DIR, "experiment_round3_results.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Saved: {out}")
    print("=" * 120 + "\n")

if __name__ == "__main__":
    main()
