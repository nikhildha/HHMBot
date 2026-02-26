"""
Regime Engine — Round 2 Experiments
====================================
Tests leverage tiers + confidence thresholds to find optimal
trade selectivity: fewer trades, controlled drawdown, max ROI.

Builds on best config from Round 1: 15m / 4feat / full covariance.
"""
import sys, os, json, time, warnings, numpy as np, pandas as pd, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_pipeline import fetch_futures_klines
from feature_engine import compute_atr, compute_rsi

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Exp2")
logger.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURES & HMM (reuse from round 1)
# ═══════════════════════════════════════════════════════════════════════════════
from hmmlearn.hmm import GaussianHMM

def compute_4feat(df):
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = (df["high"] - df["low"]) / df["close"]
    df["volume_change"] = np.log(df["volume"] / df["volume"].shift(1).replace(0, np.nan)).fillna(0).clip(-3, 3)
    rsi = compute_rsi(df["close"], length=14)
    df["rsi_norm"] = ((rsi - 50) / 50).fillna(0)
    return df

FEAT_COLS = ["log_return", "volatility", "volume_change", "rsi_norm"]

class FlexHMM:
    def __init__(self, n_states=4, cov="full"):
        self.n_states = n_states
        self.cov = cov
        self.model = None
        self._state_map = None
        self._feat_mean = None
        self._feat_std = None
        self._is_trained = False

    def train(self, df):
        features = df[FEAT_COLS].dropna().values
        if len(features) < 50: return self
        self._feat_mean = features.mean(axis=0)
        self._feat_std = features.std(axis=0)
        self._feat_std[self._feat_std < 1e-10] = 1e-10
        scaled = (features - self._feat_mean) / self._feat_std
        self.model = GaussianHMM(n_components=self.n_states, covariance_type=self.cov, n_iter=100, random_state=42)
        self.model.fit(scaled)
        self._build_map()
        self._is_trained = True
        return self

    def _build_map(self):
        means = self.model.means_[:, 0]
        vols = self.model.means_[:, 1]
        ranked = list(np.argsort(means)[::-1])
        if self.n_states >= 4:
            mid = ranked[1:3]
            chop, bear = (mid[0], mid[1]) if vols[mid[0]] <= vols[mid[1]] else (mid[1], mid[0])
            self._state_map = {ranked[0]: 0, bear: 1, chop: 2, ranked[-1]: 3}
        else:
            self._state_map = {ranked[0]: 0, ranked[1]: 2, ranked[2]: 1}

    def predict(self, df):
        feats = df[FEAT_COLS].dropna().values
        scaled = (feats - self._feat_mean) / self._feat_std
        raw = self.model.predict(scaled)
        probs = self.model.predict_proba(scaled)
        states = np.array([self._state_map.get(s, 2) for s in raw])
        confs = np.max(probs, axis=1)
        return states, confs

# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTESTER with configurable leverage/confidence
# ═══════════════════════════════════════════════════════════════════════════════

def get_leverage(conf, state, lev_tiers, conf_tiers):
    """Dynamic leverage with configurable tiers."""
    lev_high, lev_mod, lev_low = lev_tiers
    conf_high, conf_med, conf_low = conf_tiers

    if state == 3:  # CRASH
        return 0
    if state == 2:  # CHOP
        return lev_low if conf >= conf_low else 0
    # BULL / BEAR
    if conf >= conf_high:
        return lev_high
    elif conf >= conf_med:
        return lev_mod
    elif conf >= conf_low:
        return lev_low
    return 0

def backtest(df, brain, lev_tiers, conf_tiers):
    df = df.copy()
    states, confs = brain.predict(df)
    n = len(df)
    if len(states) < n:
        pad = n - len(states)
        states = np.concatenate([np.full(pad, 2), states])
        confs = np.concatenate([np.full(pad, 0.5), confs])

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    rets = np.zeros(n)
    levs = np.zeros(n)
    prev_in = False

    for i in range(n):
        s, c = int(states[i]), confs[i]
        lr = df["log_ret"].iloc[i] if not pd.isna(df["log_ret"].iloc[i]) else 0
        lev = get_leverage(c, s, lev_tiers, conf_tiers)
        levs[i] = lev
        if lev == 0:
            prev_in = False
            continue
        if not prev_in:
            rets[i] -= 0.001  # fee + slippage
        prev_in = True
        if s == 0: rets[i] += lr * lev       # BULL
        elif s == 1: rets[i] += lr * -lev     # BEAR
        elif s == 2: rets[i] += lr * lev * 0.3  # CHOP

    equity = np.exp(np.cumsum(rets))
    final = equity[-1]
    roll_max = np.maximum.accumulate(equity)
    dd = np.min((equity - roll_max) / roll_max) * 100
    sr = rets[~np.isnan(rets)]
    sharpe = (np.mean(sr) / np.std(sr) * np.sqrt(365*24)) if np.std(sr) > 0 else 0
    g = np.sum(sr[sr > 0])
    l = abs(np.sum(sr[sr < 0]))
    pf = g / l if l > 0 else float("inf")
    imk = (levs > 0).astype(int)
    trades = int((np.diff(imk, prepend=0) == 1).sum())
    active = int((levs > 0).sum())
    avg_lev = float(levs[levs > 0].mean()) if active > 0 else 0
    avg_conf = float(confs[levs > 0].mean()) if active > 0 else 0

    return {
        "return_pct": round((final - 1) * 100, 2),
        "multiplier": round(float(final), 4),
        "max_dd": round(dd, 2),
        "sharpe": round(sharpe, 3),
        "pf": round(pf, 3),
        "trades": trades,
        "in_market_pct": round(active / n * 100, 1),
        "avg_leverage": round(avg_lev, 1),
        "avg_confidence": round(avg_conf, 3),
        "skip_pct": round((n - active) / n * 100, 1),
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT MATRIX — Leverage × Confidence
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # ── Group G: Leverage levels (confidence thresholds fixed at 95/91/85) ──
    {"name": "G1_lev_10_15_20",  "lev": (20, 15, 10), "conf": (0.95, 0.91, 0.85)},
    {"name": "G2_lev_15_25_35",  "lev": (35, 25, 15), "conf": (0.95, 0.91, 0.85)},  # CURRENT
    {"name": "G3_lev_20_30_40",  "lev": (40, 30, 20), "conf": (0.95, 0.91, 0.85)},
    {"name": "G4_lev_25_35_50",  "lev": (50, 35, 25), "conf": (0.95, 0.91, 0.85)},
    {"name": "G5_lev_5_10_15",   "lev": (15, 10, 5),  "conf": (0.95, 0.91, 0.85)},
    {"name": "G6_lev_10_20_30",  "lev": (30, 20, 10), "conf": (0.95, 0.91, 0.85)},

    # ── Group H: Higher confidence thresholds (fewer trades) with current leverage ──
    {"name": "H1_conf_85_91_95", "lev": (35, 25, 15), "conf": (0.95, 0.91, 0.85)},  # CURRENT
    {"name": "H2_conf_88_93_97", "lev": (35, 25, 15), "conf": (0.97, 0.93, 0.88)},
    {"name": "H3_conf_90_95_98", "lev": (35, 25, 15), "conf": (0.98, 0.95, 0.90)},
    {"name": "H4_conf_92_96_99", "lev": (35, 25, 15), "conf": (0.99, 0.96, 0.92)},
    {"name": "H5_conf_90_93_95", "lev": (35, 25, 15), "conf": (0.95, 0.93, 0.90)},

    # ── Group I: Low leverage + ultra-selective (min drawdown, max ROI/trade) ──
    {"name": "I1_safe_10x_90",   "lev": (10, 8, 5),   "conf": (0.97, 0.93, 0.90)},
    {"name": "I2_safe_15x_92",   "lev": (15, 12, 8),  "conf": (0.98, 0.95, 0.92)},
    {"name": "I3_safe_20x_90",   "lev": (20, 15, 10), "conf": (0.97, 0.93, 0.90)},
    {"name": "I4_safe_20x_95",   "lev": (20, 15, 10), "conf": (0.99, 0.97, 0.95)},

    # ── Group J: Aggressive leverage + high confidence (max ROI, fewer trades) ──
    {"name": "J1_aggr_50x_90",   "lev": (50, 35, 25), "conf": (0.97, 0.93, 0.90)},
    {"name": "J2_aggr_50x_95",   "lev": (50, 35, 25), "conf": (0.99, 0.97, 0.95)},
    {"name": "J3_aggr_40x_92",   "lev": (40, 30, 20), "conf": (0.98, 0.95, 0.92)},
    {"name": "J4_aggr_35x_93",   "lev": (35, 25, 20), "conf": (0.98, 0.95, 0.93)},

    # ── Group K: Fixed leverage (no tiers) ──
    {"name": "K1_flat_15x",      "lev": (15, 15, 15), "conf": (0.95, 0.91, 0.85)},
    {"name": "K2_flat_25x",      "lev": (25, 25, 25), "conf": (0.95, 0.91, 0.85)},
    {"name": "K3_flat_20x_sel",  "lev": (20, 20, 20), "conf": (0.97, 0.93, 0.90)},
]

def main():
    print("\n" + "=" * 100)
    print("  🔬 ROUND 2: LEVERAGE & SELECTIVITY OPTIMIZATION")
    print("  Base config: 15m / 4-feat / full covariance / 500 lookback")
    print("=" * 100)

    # Fetch data and train brain once
    logger.info("📡 Fetching BTCUSDT 15m data...")
    df_raw = fetch_futures_klines("BTCUSDT", "15m", limit=1000)
    if df_raw is None or len(df_raw) < 200:
        print("❌ Failed to fetch data"); return
    print(f"  ✅ {len(df_raw)} candles loaded")

    df = compute_4feat(df_raw)
    brain = FlexHMM(n_states=4, cov="full")
    brain.train(df.iloc[:500])
    if not brain._is_trained:
        print("❌ HMM training failed"); return
    print(f"  ✅ HMM trained on 500 candles\n")

    results = {}
    for i, exp in enumerate(EXPERIMENTS):
        name = exp["name"]
        lev = exp["lev"]
        conf = exp["conf"]
        print(f"  [{i+1}/{len(EXPERIMENTS)}] {name}  Lev={lev}  Conf={conf}")

        try:
            r = backtest(df, brain, lev, conf)
            r["config"] = exp
            results[name] = r
            dd_flag = "🟢" if r["max_dd"] > -30 else ("🟡" if r["max_dd"] > -50 else "🔴")
            print(f"       Return={r['return_pct']:>+10.1f}%  DD={r['max_dd']:>7.1f}% {dd_flag}  Sharpe={r['sharpe']:>6.3f}  PF={r['pf']:>5.2f}  Trades={r['trades']:>4d}  AvgLev={r['avg_leverage']:>5.1f}x")
        except Exception as e:
            print(f"       ❌ {e}")
            results[name] = {"error": str(e), "config": exp}

    # Save
    out = os.path.join(config.DATA_DIR, "experiment_round2_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Saved: {out}")

    # ── Comparison Tables ──────────────────────────────────────────────────
    valid = {k: v for k, v in results.items() if "error" not in v}

    # Sort by ROI/trade (efficiency)
    print("\n" + "=" * 110)
    print("  📊 RANKED BY SHARPE RATIO")
    print("=" * 110)
    hdr = f"  {'Experiment':<22s} {'Return':>10s} {'MaxDD':>8s} {'Sharpe':>8s} {'PF':>6s} {'Trds':>5s} {'InMkt':>6s} {'AvgLev':>7s} {'AvgConf':>8s} {'$/Trade':>10s} {'DD Flag':>8s}"
    print(hdr)
    print("  " + "─" * 106)

    by_sharpe = sorted(valid.keys(), key=lambda k: valid[k]["sharpe"], reverse=True)
    for rank, name in enumerate(by_sharpe):
        r = valid[name]
        roi_per_trade = r["return_pct"] / r["trades"] if r["trades"] > 0 else 0
        dd_flag = "🟢 SAFE" if r["max_dd"] > -30 else ("🟡 MED" if r["max_dd"] > -50 else "🔴 HIGH")
        marker = " 🏆" if rank == 0 else ""
        print(f"  {name:<22s} {r['return_pct']:>+9.1f}% {r['max_dd']:>7.1f}% {r['sharpe']:>8.3f} {r['pf']:>6.2f} {r['trades']:>5d} {r['in_market_pct']:>5.0f}% {r['avg_leverage']:>6.1f}x {r['avg_confidence']:>8.3f} {roi_per_trade:>+9.1f}% {dd_flag}{marker}")

    # Best for each objective
    print(f"\n  {'─' * 60}")
    print("  🎯 BEST FOR EACH OBJECTIVE:")

    # Controlled DD (< -30%)
    safe = {k: v for k, v in valid.items() if v["max_dd"] > -30}
    if safe:
        best_safe = max(safe, key=lambda k: safe[k]["return_pct"])
        r = safe[best_safe]
        print(f"    🟢 Best controlled DD (<-30%): {best_safe} → Return={r['return_pct']:+.1f}%, DD={r['max_dd']:.1f}%, Sharpe={r['sharpe']:.3f}, Trades={r['trades']}")

    # Fewest trades with positive return
    pos = {k: v for k, v in valid.items() if v["return_pct"] > 0}
    if pos:
        fewest = min(pos, key=lambda k: pos[k]["trades"])
        r = pos[fewest]
        print(f"    📉 Fewest trades (profitable): {fewest} → Trades={r['trades']}, Return={r['return_pct']:+.1f}%, DD={r['max_dd']:.1f}%")

    # Max ROI per trade
    best_eff = max(valid, key=lambda k: valid[k]["return_pct"] / max(valid[k]["trades"], 1))
    r = valid[best_eff]
    eff = r["return_pct"] / r["trades"]
    print(f"    💰 Max ROI/trade: {best_eff} → {eff:+.1f}% per trade, Return={r['return_pct']:+.1f}%, Trades={r['trades']}")

    # Best overall (Sharpe)
    best_sharpe = by_sharpe[0]
    r = valid[best_sharpe]
    print(f"    📈 Best Sharpe: {best_sharpe} → Sharpe={r['sharpe']:.3f}, Return={r['return_pct']:+.1f}%, DD={r['max_dd']:.1f}%")

    print("=" * 110 + "\n")

if __name__ == "__main__":
    main()
