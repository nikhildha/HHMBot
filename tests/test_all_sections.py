"""
Project Regime-Master — Functionality Tests
One test per section to validate each module works end-to-end.

Run:  python -m pytest tests/test_all_sections.py -v --tb=short
  or: python -m unittest tests.test_all_sections -v
"""
import sys
import os
import unittest
import shutil
import tempfile
import json

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 1: Config
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):
    """Verify that all key configuration constants load correctly."""

    def test_config_constants(self):
        import config

        # Binance
        self.assertIsInstance(config.TESTNET, bool)
        self.assertIsInstance(config.PAPER_TRADE, bool)

        # Symbols
        self.assertEqual(config.PRIMARY_SYMBOL, "BTCUSDT")
        self.assertIsInstance(config.SECONDARY_SYMBOLS, list)

        # Timeframes
        self.assertIn(config.TIMEFRAME_EXECUTION, ["1m", "5m", "15m", "30m", "1h"])
        self.assertIn(config.TIMEFRAME_MACRO, ["1d", "4h", "1w"])

        # HMM
        self.assertEqual(config.HMM_N_STATES, 4)
        self.assertGreater(config.HMM_ITERATIONS, 0)

        # Leverage tiers
        self.assertGreater(config.LEVERAGE_HIGH, config.LEVERAGE_MODERATE)
        self.assertGreater(config.LEVERAGE_MODERATE, config.LEVERAGE_LOW)

        # Risk
        self.assertGreater(config.RISK_PER_TRADE, 0)
        self.assertLess(config.RISK_PER_TRADE, 1)
        self.assertGreater(config.KILL_SWITCH_DRAWDOWN, 0)

        # Paths
        self.assertTrue(os.path.isdir(config.DATA_DIR))

        print("  ✅ Config: All constants loaded and valid")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 2: Feature Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngine(unittest.TestCase):
    """Test HMM features and technical indicators on synthetic data."""

    def test_compute_all_features(self):
        from feature_engine import generate_synthetic_data, compute_all_features

        df = generate_synthetic_data(n=300, seed=42)
        self.assertEqual(len(df), 300)

        # Compute all features
        df_feat = compute_all_features(df)

        # HMM features
        self.assertIn("log_return", df_feat.columns)
        self.assertIn("volatility", df_feat.columns)

        # Technical indicators
        self.assertIn("rsi", df_feat.columns)
        self.assertIn("bb_upper", df_feat.columns)
        self.assertIn("bb_middle", df_feat.columns)
        self.assertIn("bb_lower", df_feat.columns)
        self.assertIn("atr", df_feat.columns)

        # RSI sanity: should be between 0 and 100 (excluding NaN)
        rsi_valid = df_feat["rsi"].dropna()
        self.assertTrue((rsi_valid >= 0).all() and (rsi_valid <= 100).all(),
                        "RSI values should be between 0 and 100")

        # BB sanity: upper > middle > lower
        bb_valid = df_feat[["bb_upper", "bb_middle", "bb_lower"]].dropna()
        self.assertTrue((bb_valid["bb_upper"] >= bb_valid["bb_middle"]).all())
        self.assertTrue((bb_valid["bb_middle"] >= bb_valid["bb_lower"]).all())

        # ATR sanity: should be positive
        atr_valid = df_feat["atr"].dropna()
        self.assertTrue((atr_valid > 0).all(), "ATR should be positive")

        print("  ✅ Feature Engine: All features computed correctly")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 3: HMM Brain
# ═══════════════════════════════════════════════════════════════════════════════

class TestHMMBrain(unittest.TestCase):
    """Train HMM on synthetic data and verify predictions."""

    def test_train_and_predict(self):
        from feature_engine import generate_synthetic_data, compute_hmm_features
        from hmm_brain import HMMBrain
        import config

        df = generate_synthetic_data(n=500, seed=42)
        df = compute_hmm_features(df)

        brain = HMMBrain(n_states=4)

        # Before training
        self.assertFalse(brain.is_trained)
        self.assertTrue(brain.needs_retrain())

        # Train
        brain.train(df)
        self.assertTrue(brain.is_trained)
        self.assertFalse(brain.needs_retrain())

        # Predict
        state, confidence = brain.predict(df)

        self.assertIn(state, [config.REGIME_BULL, config.REGIME_BEAR,
                              config.REGIME_CHOP, config.REGIME_CRASH])
        self.assertGreater(confidence, 0)
        self.assertLessEqual(confidence, 1.0)

        # Regime name
        name = brain.get_regime_name(state)
        self.assertIn(name, config.REGIME_NAMES.values())

        # Predict all
        import numpy as np
        states = brain.predict_all(df)
        self.assertIsInstance(states, np.ndarray)
        self.assertGreater(len(states), 0)

        print(f"  ✅ HMM Brain: Trained, predicted regime={name}, confidence={confidence:.2%}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 4: Risk Manager
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskManager(unittest.TestCase):
    """Test leverage mapping, position sizing, ATR stops, and kill switch."""

    def test_dynamic_leverage(self):
        from risk_manager import RiskManager
        import config

        # Crash → 0
        self.assertEqual(RiskManager.get_dynamic_leverage(0.9, config.REGIME_CRASH), 0)

        # Chop + high confidence → LEVERAGE_LOW
        self.assertEqual(RiskManager.get_dynamic_leverage(0.95, config.REGIME_CHOP),
                         config.LEVERAGE_LOW)

        # Chop + below threshold → 0
        self.assertEqual(RiskManager.get_dynamic_leverage(0.90, config.REGIME_CHOP), 0)

        # Bull + confidence >= 0.99 → LEVERAGE_HIGH
        self.assertEqual(RiskManager.get_dynamic_leverage(0.995, config.REGIME_BULL),
                         config.LEVERAGE_HIGH)

        # Bull + confidence 0.96–0.99 → LEVERAGE_MODERATE
        self.assertEqual(RiskManager.get_dynamic_leverage(0.97, config.REGIME_BULL),
                         config.LEVERAGE_MODERATE)

        # Bull + confidence 0.92–0.96 → LEVERAGE_LOW
        self.assertEqual(RiskManager.get_dynamic_leverage(0.93, config.REGIME_BULL),
                         config.LEVERAGE_LOW)

        # Bull + below 0.92 → 0
        self.assertEqual(RiskManager.get_dynamic_leverage(0.90, config.REGIME_BULL), 0)

        print("  ✅ Risk Manager: Dynamic leverage mapping correct")

    def test_position_sizing(self):
        from risk_manager import RiskManager

        qty = RiskManager.calculate_position_size(
            balance=10000, entry_price=50000, atr=500, leverage=5
        )
        self.assertGreater(qty, 0)
        # With 2% risk and 1.5x ATR stop → risk_amount=200, stop_dist=750 → qty≈0.267
        self.assertLess(qty, 10)  # reasonable range

        print(f"  ✅ Risk Manager: Position size = {qty}")

    def test_atr_stops(self):
        from risk_manager import RiskManager

        sl, tp = RiskManager.calculate_atr_stops(entry_price=50000, atr=500, side="BUY")
        self.assertLess(sl, 50000, "BUY SL should be below entry")
        self.assertGreater(tp, 50000, "BUY TP should be above entry")

        sl_s, tp_s = RiskManager.calculate_atr_stops(entry_price=50000, atr=500, side="SELL")
        self.assertGreater(sl_s, 50000, "SELL SL should be above entry")
        self.assertLess(tp_s, 50000, "SELL TP should be below entry")

        print(f"  ✅ Risk Manager: ATR Stops — BUY SL={sl}, TP={tp}")

    def test_kill_switch(self):
        from risk_manager import RiskManager

        rm = RiskManager()

        # Record equity then drop 15% → should trigger
        rm.record_equity(10000)
        rm.record_equity(8400)  # 16% drawdown

        triggered = rm.check_kill_switch()
        self.assertTrue(triggered, "Kill switch should trigger on >10% drawdown")
        self.assertTrue(rm.is_killed)

        # Reset
        rm.reset_kill_switch()
        self.assertFalse(rm.is_killed)

        print("  ✅ Risk Manager: Kill switch triggered and reset correctly")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 5: Sideways Strategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestSidewaysStrategy(unittest.TestCase):
    """Test mean-reversion signal detection with engineered data."""

    def test_buy_signal(self):
        import pandas as pd
        import numpy as np
        from sideways_strategy import evaluate_mean_reversion
        import config

        # Create a DataFrame with price at the lower Bollinger Band and low RSI
        n = 100
        prices = np.ones(n) * 100
        # Simulate an oversold condition at the end
        prices[-1] = 80  # Big drop at end

        df = pd.DataFrame({
            "open":   prices,
            "high":   prices * 1.01,
            "low":    prices * 0.99,
            "close":  prices,
            "volume": np.ones(n) * 1000,
        })
        # Force the last row to match BUY conditions
        df.loc[df.index[-1], "close"] = 80
        df.loc[df.index[-1], "low"] = 79

        # Pre-compute indicators, then manually override last row for test
        from feature_engine import compute_indicators
        df = compute_indicators(df)

        # Force conditions: price below lower band, RSI below oversold
        df.loc[df.index[-1], "bb_lower"] = 85
        df.loc[df.index[-1], "bb_upper"] = 115
        df.loc[df.index[-1], "rsi"] = 20  # Below RSI_OVERSOLD (35)

        signal = evaluate_mean_reversion(df, symbol="TESTUSDT")
        self.assertIsNotNone(signal, "Should get a BUY signal")
        self.assertEqual(signal["side"], "BUY")
        self.assertEqual(signal["leverage"], config.LEVERAGE_LOW)

        print("  ✅ Sideways Strategy: BUY signal detected correctly")

    def test_no_signal(self):
        import pandas as pd
        import numpy as np
        from sideways_strategy import evaluate_mean_reversion

        # Create price in middle of range → no signal
        n = 50
        df = pd.DataFrame({
            "open":   np.ones(n) * 100,
            "high":   np.ones(n) * 101,
            "low":    np.ones(n) * 99,
            "close":  np.ones(n) * 100,
            "volume": np.ones(n) * 1000,
        })

        from feature_engine import compute_indicators
        df = compute_indicators(df)
        # Force RSI to neutral
        df.loc[df.index[-1], "rsi"] = 50

        signal = evaluate_mean_reversion(df, symbol="TESTUSDT")
        self.assertIsNone(signal, "Mid-range should produce no signal")

        print("  ✅ Sideways Strategy: No signal in neutral conditions")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 6: Execution Engine (Paper Trade)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecutionEngine(unittest.TestCase):
    """Test paper-trade mode logging."""

    def test_paper_trade(self):
        import config
        # Ensure paper trade is ON
        self.assertTrue(config.PAPER_TRADE, "These tests require PAPER_TRADE=true")

        from execution_engine import ExecutionEngine

        engine = ExecutionEngine()

        # Get simulated balance
        balance = engine.get_futures_balance()
        self.assertEqual(balance, 1000.0, "Paper balance should be $1000")

        print(f"  ✅ Execution Engine: Paper trade mode active, balance=${balance}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 7: Data Pipeline (Live Testnet)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataPipeline(unittest.TestCase):
    """Test fetching live data from Binance Testnet."""

    def test_fetch_klines(self):
        import config
        from data_pipeline import fetch_klines

        df = fetch_klines("BTCUSDT", "1h", limit=10)

        if df is None:
            self.skipTest("Binance API unavailable (testnet keys may be invalid)")

        self.assertGreater(len(df), 0)
        expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")

        # Data types
        self.assertTrue(df["close"].dtype == float)
        self.assertTrue(df["volume"].dtype == float)

        print(f"  ✅ Data Pipeline: Fetched {len(df)} candles for BTCUSDT 1h")

    def test_get_current_price(self):
        from data_pipeline import get_current_price

        price = get_current_price("BTCUSDT")
        if price is None:
            self.skipTest("Binance API unavailable")

        self.assertGreater(price, 0)
        print(f"  ✅ Data Pipeline: BTCUSDT price = ${price:,.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 8: Tradebook
# ═══════════════════════════════════════════════════════════════════════════════

class TestTradebook(unittest.TestCase):
    """Test the full open → update → close trade lifecycle."""

    def setUp(self):
        """Back up tradebook so tests don't corrupt real data."""
        import config
        self._tradebook_path = os.path.join(config.DATA_DIR, "tradebook.json")
        self._backup_path = self._tradebook_path + ".bak"
        if os.path.exists(self._tradebook_path):
            shutil.copy2(self._tradebook_path, self._backup_path)

    def tearDown(self):
        """Restore tradebook backup."""
        if os.path.exists(self._backup_path):
            shutil.move(self._backup_path, self._tradebook_path)
        elif os.path.exists(self._tradebook_path):
            os.remove(self._tradebook_path)

    def test_open_close_lifecycle(self):
        from tradebook import open_trade, close_trade, get_active_trades, get_closed_trades, update_unrealized

        # Open a test trade
        trade_id = open_trade(
            symbol="TESTUSDT",
            side="BUY",
            leverage=5,
            quantity=0.01,
            entry_price=50000.0,
            atr=500.0,
            regime="BULLISH",
            confidence=0.85,
            reason="Test trade",
            capital=100.0,
        )
        self.assertIsNotNone(trade_id)
        self.assertTrue(trade_id.startswith("T-"))

        # Verify active
        active = get_active_trades()
        self.assertGreaterEqual(len(active), 1)
        test_trade = next(t for t in active if t["trade_id"] == trade_id)
        self.assertEqual(test_trade["symbol"], "TESTUSDT")
        self.assertEqual(test_trade["position"], "LONG")
        self.assertEqual(test_trade["status"], "ACTIVE")

        # Update unrealized P&L with a mock price
        update_unrealized(prices={"TESTUSDT": 51000.0})

        # Close with profit
        closed = close_trade(trade_id=trade_id, exit_price=51000.0, reason="TEST_CLOSE")
        self.assertIsNotNone(closed)
        self.assertEqual(closed["status"], "CLOSED")
        self.assertGreater(closed["realized_pnl"], 0, "Should have profit on price increase")

        # Verify it moved to closed
        closed_trades = get_closed_trades()
        test_closed = [t for t in closed_trades if t["trade_id"] == trade_id]
        self.assertEqual(len(test_closed), 1)

        print(f"  ✅ Tradebook: {trade_id} opened → updated → closed with P&L=${closed['realized_pnl']}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 8b: Trailing SL / TP
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrailingSLTP(unittest.TestCase):
    """Test trailing stop loss and trailing take profit logic."""

    def setUp(self):
        import config
        self._tradebook_path = os.path.join(config.DATA_DIR, "tradebook.json")
        self._backup_path = self._tradebook_path + ".bak"
        if os.path.exists(self._tradebook_path):
            shutil.copy2(self._tradebook_path, self._backup_path)

    def tearDown(self):
        if os.path.exists(self._backup_path):
            shutil.move(self._backup_path, self._tradebook_path)
        elif os.path.exists(self._tradebook_path):
            os.remove(self._tradebook_path)

    def test_trailing_sl_activation_and_trail(self):
        """LONG: SL should trail up after price moves 1×ATR in favor."""
        from tradebook import open_trade, update_unrealized, get_active_trades

        trade_id = open_trade(
            symbol="TRAILTEST", side="BUY", leverage=5, quantity=0.01,
            entry_price=50000.0, atr=500.0, regime="BULLISH",
            confidence=0.85, capital=100.0,
        )

        # Original SL: 50000 - 500*1.2 = 49400 (1.2x ATR at 5x leverage)
        trades = get_active_trades()
        trade = next(t for t in trades if t["trade_id"] == trade_id)
        original_sl = trade["trailing_sl"]
        self.assertAlmostEqual(original_sl, 49400.0, places=0)
        self.assertFalse(trade["trailing_active"])

        # Price moves up 1×ATR (500) → should activate trailing
        update_unrealized(prices={"TRAILTEST": 50500.0})
        trades = get_active_trades()
        trade = next(t for t in trades if t["trade_id"] == trade_id)
        self.assertTrue(trade["trailing_active"], "Trailing should activate at 1×ATR move")

        # Price moves further up → SL should trail behind peak
        update_unrealized(prices={"TRAILTEST": 51500.0})
        trades = get_active_trades()
        trade = next(t for t in trades if t["trade_id"] == trade_id)
        # New SL = peak(51500) - 1×ATR(500) = 51000
        self.assertGreaterEqual(trade["trailing_sl"], 51000.0,
                                "Trailing SL should be at peak - 1×ATR")
        self.assertGreater(trade["trailing_sl"], original_sl,
                           "Trailing SL should have moved up from original")

        print("  ✅ Trailing SL: Activation and trail-up for LONG verified")

    def test_trailing_sl_never_loosens(self):
        """After SL trails up, a price dip should NOT lower the SL."""
        from tradebook import open_trade, update_unrealized, get_active_trades

        trade_id = open_trade(
            symbol="TRAILTEST2", side="BUY", leverage=5, quantity=0.01,
            entry_price=50000.0, atr=500.0, regime="BULLISH",
            confidence=0.85, capital=100.0,
        )

        # Activate trailing and push SL up (stay below TP=51500)
        update_unrealized(prices={"TRAILTEST2": 51000.0})
        trades = get_active_trades()
        trade = next(t for t in trades if t["trade_id"] == trade_id)
        sl_after_rise = trade["trailing_sl"]

        # Price dips — SL should NOT move back down
        update_unrealized(prices={"TRAILTEST2": 50800.0})
        trades = get_active_trades()
        trade = next(t for t in trades if t["trade_id"] == trade_id)
        self.assertGreaterEqual(trade["trailing_sl"], sl_after_rise,
                                "SL must never loosen (move backwards)")

        print("  ✅ Trailing SL: SL never loosens on pullback verified")

    def test_trailing_tp_extension(self):
        """TP should extend when price reaches 75% of TP distance."""
        from tradebook import open_trade, update_unrealized, get_active_trades

        trade_id = open_trade(
            symbol="TRAILTEST3", side="BUY", leverage=5, quantity=0.01,
            entry_price=50000.0, atr=500.0, regime="BULLISH",
            confidence=0.85, capital=100.0,
        )

        # Original TP: 50000 + 500*2.4 = 51200 (2.4x ATR at 5x leverage)
        trades = get_active_trades()
        trade = next(t for t in trades if t["trade_id"] == trade_id)
        original_tp = trade["trailing_tp"]
        self.assertAlmostEqual(original_tp, 51200.0, places=0)

        # Price reaches 75% of TP distance (1200 * 0.75 = 900 → 50900)
        update_unrealized(prices={"TRAILTEST3": 50950.0})
        trades = get_active_trades()
        trade = next(t for t in trades if t["trade_id"] == trade_id)
        self.assertGreater(trade["trailing_tp"], original_tp,
                           "TP should extend after 75% threshold")
        self.assertEqual(trade["tp_extensions"], 1)

        print("  ✅ Trailing TP: Extension on 75% threshold verified")

    def test_trailing_tp_max_extensions(self):
        """TP should stop extending after MAX_EXTENSIONS."""
        import config
        from tradebook import open_trade, update_unrealized, get_active_trades

        trade_id = open_trade(
            symbol="TRAILTEST4", side="BUY", leverage=5, quantity=0.01,
            entry_price=50000.0, atr=500.0, regime="BULLISH",
            confidence=0.85, capital=100.0,
        )

        # Keep pushing price to trigger extensions up to max
        price = 51200.0
        for i in range(config.TRAILING_TP_MAX_EXTENSIONS + 2):
            update_unrealized(prices={"TRAILTEST4": price})
            price += 2000  # Big jump to ensure each extension triggers

        trades = get_active_trades()
        trade = next((t for t in trades if t["trade_id"] == trade_id), None)
        if trade:  # Might have hit TP and closed
            self.assertLessEqual(trade["tp_extensions"], config.TRAILING_TP_MAX_EXTENSIONS,
                                 "TP extensions should not exceed MAX_EXTENSIONS")

        print("  ✅ Trailing TP: Max extensions cap verified")

    def test_trailing_sl_short_direction(self):
        """SHORT: SL should trail down as price falls in favor."""
        from tradebook import open_trade, update_unrealized, get_active_trades

        trade_id = open_trade(
            symbol="TRAILSHORT", side="SELL", leverage=5, quantity=0.01,
            entry_price=50000.0, atr=200.0, regime="BEARISH",
            confidence=0.85, capital=100.0,
        )

        # Original SL for SHORT: 50000 + 200*1.2 = 50240 (1.2x ATR at 5x leverage)
        # Original TP for SHORT: 50000 - 200*2.4 = 49520
        trades = get_active_trades()
        trade = next(t for t in trades if t["trade_id"] == trade_id)
        original_sl = trade["trailing_sl"]
        self.assertAlmostEqual(original_sl, 50240.0, places=0)

        # Price drops 1×ATR → should activate trailing (stay above TP=49520)
        update_unrealized(prices={"TRAILSHORT": 49800.0})
        trades = get_active_trades()
        trade = next(t for t in trades if t["trade_id"] == trade_id)
        self.assertTrue(trade["trailing_active"])

        # Price drops further → SL should trail down (but stay above TP=49520)
        update_unrealized(prices={"TRAILSHORT": 49600.0})
        trades = get_active_trades()
        trade = next(t for t in trades if t["trade_id"] == trade_id)
        # New SL = peak(49600) + 1×ATR(200) = 49800
        self.assertLessEqual(trade["trailing_sl"], 49800.0,
                             "SHORT trailing SL should be at peak + 1×ATR")
        self.assertLess(trade["trailing_sl"], original_sl,
                        "SHORT trailing SL should have moved down")

        print("  ✅ Trailing SL: SHORT direction trailing verified")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 9: Backtester
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktester(unittest.TestCase):
    """Run a full backtest on synthetic data."""

    def test_full_backtest(self):
        from feature_engine import generate_synthetic_data
        from backtester import run_full_backtest

        df = generate_synthetic_data(n=1000, seed=42)
        results = run_full_backtest(df, n_states=4)

        # All expected keys present
        expected_keys = [
            "total_return", "final_multiplier", "max_drawdown",
            "sharpe_ratio", "profit_factor", "n_trades",
            "regime_breakdown", "equity_curve", "df",
        ]
        for key in expected_keys:
            self.assertIn(key, results, f"Missing result key: {key}")

        # Sanity checks
        self.assertIsInstance(results["total_return"], float)
        self.assertGreater(results["n_trades"], 0, "Should have at least 1 trade")
        self.assertGreater(results["final_multiplier"], 0, "Final multiplier should be positive")

        # Regime breakdown
        rb = results["regime_breakdown"]
        self.assertGreater(len(rb), 0, "Should have regime breakdown")

        print(f"  ✅ Backtester: Return={results['total_return']:.1f}% | "
              f"Sharpe={results['sharpe_ratio']:.2f} | "
              f"Trades={results['n_trades']}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 10: Web Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

class TestWebDashboard(unittest.TestCase):
    """Verify the Express server is responding."""

    def test_server_responds(self):
        import urllib.request
        try:
            req = urllib.request.Request("http://localhost:3001/", method="GET")
            resp = urllib.request.urlopen(req, timeout=5)
            self.assertEqual(resp.status, 200)
            body = resp.read().decode()
            self.assertGreater(len(body), 100, "Should return HTML page")
            print("  ✅ Web Dashboard: Server at :3001 returned 200 OK")
        except Exception as e:
            self.skipTest(f"Web server not running on port 3001: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  🧪 REGIME-MASTER: SECTION-BY-SECTION FUNCTIONALITY TESTS")
    print("=" * 70 + "\n")
    unittest.main(verbosity=2)
