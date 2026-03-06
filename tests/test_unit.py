"""
Project Regime-Master — Unit Tests (No Real API Calls)
Pure unit tests using unittest.mock for all external dependencies.
All tests pass without API keys, running bot, or internet connection.

Run:  python -m pytest tests/test_unit.py -v --tb=short
  or: python -m unittest tests.test_unit -v
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase G: Config — New Constants Coverage
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigNewConstants(unittest.TestCase):
    """Verify all constants added during cleanup are present and correct."""

    def test_coindcx_min_notional(self):
        self.assertEqual(config.COINDCX_MIN_NOTIONAL, 120.0)

    def test_conviction_leverage_bands(self):
        self.assertEqual(config.CONVICTION_MIN_TRADE, 40)
        self.assertEqual(config.CONVICTION_BAND_LOW, 55)
        self.assertEqual(config.CONVICTION_BAND_MED, 70)
        self.assertEqual(config.CONVICTION_BAND_HIGH, 85)

    def test_hmm_confidence_tiers(self):
        self.assertIn("HMM_CONF_TIER_HIGH", dir(config))
        self.assertIn("HMM_CONF_TIER_MED_HIGH", dir(config))
        self.assertIn("HMM_CONF_TIER_MED", dir(config))
        self.assertIn("HMM_CONF_TIER_LOW", dir(config))
        # Tiers must be descending
        self.assertGreater(config.HMM_CONF_TIER_HIGH, config.HMM_CONF_TIER_MED_HIGH)
        self.assertGreater(config.HMM_CONF_TIER_MED_HIGH, config.HMM_CONF_TIER_MED)
        self.assertGreater(config.HMM_CONF_TIER_MED, config.HMM_CONF_TIER_LOW)

    def test_funding_thresholds_exist(self):
        self.assertIn("FUNDING_NEG_STRONG", dir(config))
        self.assertIn("FUNDING_POS_MED", dir(config))
        self.assertIn("FUNDING_POS_STRONG", dir(config))
        self.assertIn("FUNDING_NEG_MED", dir(config))

    def test_oi_thresholds_exist(self):
        self.assertIn("OI_CHANGE_HIGH", dir(config))
        self.assertIn("OI_CHANGE_MED", dir(config))
        self.assertIn("OI_CHANGE_NEG_HIGH", dir(config))
        self.assertIn("OI_CHANGE_NEG_MED", dir(config))
        self.assertGreater(config.OI_CHANGE_HIGH, config.OI_CHANGE_MED)
        self.assertLess(config.OI_CHANGE_NEG_HIGH, config.OI_CHANGE_NEG_MED)

    def test_sentiment_weights_sum_to_one(self):
        total = config.SENTIMENT_VADER_WEIGHT + config.SENTIMENT_FINBERT_WEIGHT
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_conviction_weights_sum_to_100(self):
        total = (
            config.CONVICTION_WEIGHT_HMM
            + config.CONVICTION_WEIGHT_BTC_MACRO
            + config.CONVICTION_WEIGHT_FUNDING
            + config.CONVICTION_WEIGHT_SR_VWAP
            + config.CONVICTION_WEIGHT_OI
            + config.CONVICTION_WEIGHT_VOL
            + config.CONVICTION_WEIGHT_SENTIMENT
            + config.CONVICTION_WEIGHT_ORDERFLOW
        )
        self.assertEqual(total, 100)

    def test_scanner_rate_limit_sleep(self):
        self.assertGreater(config.SCANNER_RATE_LIMIT_SLEEP, 0)

    def test_sentiment_rate_limit_sleep(self):
        self.assertGreater(config.SENTIMENT_RATE_LIMIT_SLEEP, 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase A: RiskManager — 8 Factor Scoring Methods
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskManagerScoring(unittest.TestCase):
    """Unit tests for all 8 _score_* static methods and conviction score pipeline."""

    def setUp(self):
        from risk_manager import RiskManager
        self.RM = RiskManager

    # ── A1: _score_hmm ───────────────────────────────────────────────────────

    def test_score_hmm_full_weight_at_high_confidence(self):
        score = self.RM._score_hmm(config.HMM_CONF_TIER_HIGH)
        self.assertAlmostEqual(score, config.CONVICTION_WEIGHT_HMM, places=5)

    def test_score_hmm_above_high_tier(self):
        score = self.RM._score_hmm(0.99)
        self.assertEqual(score, config.CONVICTION_WEIGHT_HMM)

    def test_score_hmm_med_high_tier(self):
        score = self.RM._score_hmm(config.HMM_CONF_TIER_MED_HIGH)
        expected = config.CONVICTION_WEIGHT_HMM * 0.85
        self.assertAlmostEqual(score, expected, places=5)

    def test_score_hmm_med_tier(self):
        score = self.RM._score_hmm(config.HMM_CONF_TIER_MED)
        expected = config.CONVICTION_WEIGHT_HMM * 0.65
        self.assertAlmostEqual(score, expected, places=5)

    def test_score_hmm_low_tier(self):
        score = self.RM._score_hmm(config.HMM_CONF_TIER_LOW)
        expected = config.CONVICTION_WEIGHT_HMM * 0.40
        self.assertAlmostEqual(score, expected, places=5)

    def test_score_hmm_below_minimum_returns_zero(self):
        # HMM_CONF_TIER_LOW = 0.10 (margin scale); anything below returns 0
        score = self.RM._score_hmm(0.05)
        self.assertEqual(score, 0.0)

    def test_score_hmm_zero_confidence(self):
        score = self.RM._score_hmm(0.0)
        self.assertEqual(score, 0.0)

    def test_score_hmm_always_nonnegative(self):
        for conf in [0.0, 0.5, 0.85, 0.90, 0.94, 0.97, 1.0]:
            self.assertGreaterEqual(self.RM._score_hmm(conf), 0.0)

    def test_score_hmm_none_confidence_returns_zero(self):
        # Validates the production None guard — prevents TypeError when HMM predict fails
        score = self.RM._score_hmm(None)
        self.assertEqual(score, 0.0)

    # ── A2: _score_btc_macro ─────────────────────────────────────────────────

    def test_score_btc_macro_buy_aligned_bull(self):
        score = self.RM._score_btc_macro(config.REGIME_BULL, config.REGIME_BULL, "BUY")
        self.assertEqual(score, config.CONVICTION_WEIGHT_BTC_MACRO)

    def test_score_btc_macro_sell_aligned_bear(self):
        score = self.RM._score_btc_macro(config.REGIME_BEAR, config.REGIME_BEAR, "SELL")
        self.assertEqual(score, config.CONVICTION_WEIGHT_BTC_MACRO)

    def test_score_btc_macro_buy_against_bear(self):
        score = self.RM._score_btc_macro(config.REGIME_BEAR, config.REGIME_BULL, "BUY")
        self.assertEqual(score, -config.CONVICTION_MACRO_FIGHT_PENALTY)

    def test_score_btc_macro_sell_against_bull(self):
        score = self.RM._score_btc_macro(config.REGIME_BULL, config.REGIME_BEAR, "SELL")
        self.assertEqual(score, -config.CONVICTION_MACRO_FIGHT_PENALTY)

    def test_score_btc_macro_crash_penalty(self):
        score = self.RM._score_btc_macro(config.REGIME_CRASH, config.REGIME_BULL, "BUY")
        self.assertEqual(score, -config.CONVICTION_CRASH_PENALTY)

    def test_score_btc_macro_none_btc_returns_neutral(self):
        score = self.RM._score_btc_macro(None, config.REGIME_BULL, "BUY")
        expected = config.CONVICTION_WEIGHT_BTC_MACRO * 0.50
        self.assertAlmostEqual(score, expected, places=5)

    def test_score_btc_macro_chop_returns_small_boost(self):
        # CHOP btc_regime → small boost (neither aligned nor fighting)
        score = self.RM._score_btc_macro(config.REGIME_CHOP, config.REGIME_BULL, "BUY")
        self.assertGreater(score, 0)
        self.assertLess(score, config.CONVICTION_WEIGHT_BTC_MACRO)

    # ── A3: _score_funding ───────────────────────────────────────────────────

    def test_score_funding_negative_funding_favors_buy(self):
        # Very negative funding (below FUNDING_NEG_STRONG=-0.0001) → full score for BUY
        score = self.RM._score_funding(-0.0002, "BUY")
        self.assertEqual(score, config.CONVICTION_WEIGHT_FUNDING)

    def test_score_funding_high_positive_favors_sell(self):
        # High positive funding (above FUNDING_POS_STRONG=0.0001) → full score for SELL
        score = self.RM._score_funding(0.0005, "SELL")
        self.assertEqual(score, config.CONVICTION_WEIGHT_FUNDING)

    def test_score_funding_crowded_longs_penalty_for_buy(self):
        # High positive funding (above FUNDING_POS_MED=0.0003) → penalty for BUY
        score = self.RM._score_funding(0.0005, "BUY")
        self.assertEqual(score, -config.CONVICTION_FUNDING_PENALTY)

    def test_score_funding_crowded_shorts_penalty_for_sell(self):
        # Very negative funding (below FUNDING_NEG_MED=-0.0003) → penalty for SELL
        score = self.RM._score_funding(-0.0005, "SELL")
        self.assertEqual(score, -config.CONVICTION_FUNDING_PENALTY)

    def test_score_funding_neutral_funding(self):
        # Funding near 0 → partial score
        score_buy = self.RM._score_funding(0.00005, "BUY")
        self.assertGreater(score_buy, 0)
        self.assertLess(score_buy, config.CONVICTION_WEIGHT_FUNDING)

    def test_score_funding_none_returns_positive(self):
        # No funding data → mild positive
        score = self.RM._score_funding(None, "BUY")
        self.assertGreater(score, 0)

    # ── A4: _score_sr_vwap ───────────────────────────────────────────────────

    def test_score_sr_vwap_buy_at_support_above_vwap(self):
        # sr_position=0 (at support), vwap_position>0 (above VWAP), side=BUY → max score
        score = self.RM._score_sr_vwap(0.0, 0.5, "BUY")
        self.assertAlmostEqual(score, config.CONVICTION_WEIGHT_SR_VWAP, places=5)

    def test_score_sr_vwap_sell_at_resistance_below_vwap(self):
        # sr_position=1 (at resistance), vwap_position<0 (below VWAP), side=SELL → max score
        score = self.RM._score_sr_vwap(1.0, -0.5, "SELL")
        self.assertAlmostEqual(score, config.CONVICTION_WEIGHT_SR_VWAP, places=5)

    def test_score_sr_vwap_buy_at_resistance_no_vwap(self):
        # sr_position=1 (unfavorable for BUY), vwap neutral → lower score
        score_at_resistance = self.RM._score_sr_vwap(1.0, None, "BUY")
        score_at_support = self.RM._score_sr_vwap(0.0, None, "BUY")
        self.assertGreater(score_at_support, score_at_resistance)

    def test_score_sr_vwap_all_none_returns_neutral(self):
        score = self.RM._score_sr_vwap(None, None, "BUY")
        expected = config.CONVICTION_WEIGHT_SR_VWAP * 0.45
        self.assertAlmostEqual(score, expected, places=5)

    def test_score_sr_vwap_always_nonnegative_in_normal_cases(self):
        # sr_position in [0,1] → result should always be >= 0
        for sr in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for vwap in [-1.0, 0.0, 1.0]:
                score = self.RM._score_sr_vwap(sr, vwap, "BUY")
                self.assertGreaterEqual(score, 0.0)

    # ── A5: _score_oi ────────────────────────────────────────────────────────

    def test_score_oi_strong_oi_growth_for_buy(self):
        # OI > OI_CHANGE_HIGH=0.03 → full score for BUY
        score = self.RM._score_oi(0.05, "BUY")
        self.assertEqual(score, config.CONVICTION_WEIGHT_OI)

    def test_score_oi_strong_oi_decline_for_sell(self):
        # OI < OI_CHANGE_NEG_HIGH=-0.03 → full score for SELL
        score = self.RM._score_oi(-0.05, "SELL")
        self.assertEqual(score, config.CONVICTION_WEIGHT_OI)

    def test_score_oi_adverse_oi_for_buy_is_penalty(self):
        # OI < OI_CHANGE_NEG_HIGH → penalty for BUY (OI falling means short-covering risk)
        score = self.RM._score_oi(-0.05, "BUY")
        self.assertEqual(score, -config.CONVICTION_OI_PENALTY)

    def test_score_oi_adverse_oi_for_sell_is_penalty(self):
        # OI > OI_CHANGE_HIGH → penalty for SELL
        score = self.RM._score_oi(0.05, "SELL")
        self.assertEqual(score, -config.CONVICTION_OI_PENALTY)

    def test_score_oi_moderate_growth_for_buy(self):
        # OI_CHANGE_MED < oi < OI_CHANGE_HIGH → partial score
        score = self.RM._score_oi(0.02, "BUY")
        self.assertGreater(score, 0)
        self.assertLess(score, config.CONVICTION_WEIGHT_OI)

    def test_score_oi_none_returns_neutral(self):
        score = self.RM._score_oi(None, "BUY")
        expected = config.CONVICTION_WEIGHT_OI * 0.50
        self.assertAlmostEqual(score, expected, places=5)

    # ── A6: _score_volatility ────────────────────────────────────────────────

    def test_score_volatility_ideal_range_returns_full(self):
        # VOL_MIN_ATR_PCT=0.003, VOL_MAX_ATR_PCT=0.06, ideal is <= 50% of VOL_MAX
        ideal_vol = config.VOL_MIN_ATR_PCT + 0.001  # Just above minimum, within ideal
        score = self.RM._score_volatility(ideal_vol)
        self.assertEqual(score, config.CONVICTION_WEIGHT_VOL)

    def test_score_volatility_too_high_gets_reduced(self):
        # Way above VOL_MAX_ATR_PCT → very low score
        score = self.RM._score_volatility(config.VOL_MAX_ATR_PCT * 2)
        self.assertLess(score, config.CONVICTION_WEIGHT_VOL)

    def test_score_volatility_none_returns_partial(self):
        score = self.RM._score_volatility(None)
        self.assertGreater(score, 0)

    def test_score_volatility_always_positive(self):
        for vol in [0.001, 0.003, 0.01, 0.06, 0.15]:
            self.assertGreaterEqual(self.RM._score_volatility(vol), 0.0)

    # ── A7: _score_sentiment ─────────────────────────────────────────────────

    def test_score_sentiment_strongly_positive(self):
        # score > SENTIMENT_STRONG_POS=0.45 → full weight
        score = self.RM._score_sentiment(0.80)
        self.assertEqual(score, config.CONVICTION_WEIGHT_SENTIMENT)

    def test_score_sentiment_moderately_positive(self):
        # Between 0.20 (neg threshold) and 0.45 (strong pos) → partial
        score = self.RM._score_sentiment(0.30)
        self.assertGreater(score, 0)
        self.assertLess(score, config.CONVICTION_WEIGHT_SENTIMENT)

    def test_score_sentiment_neutral_band(self):
        # Between CONVICTION_SENTIMENT_NEG_THRESHOLD and its negation → partial
        score = self.RM._score_sentiment(0.0)
        self.assertGreaterEqual(score, 0)

    def test_score_sentiment_mild_negative(self):
        # Between SENTIMENT_VETO_THRESHOLD=-0.65 and CONVICTION_SENTIMENT_NEG_THRESHOLD=-0.20
        score = self.RM._score_sentiment(-0.40)
        self.assertEqual(score, -config.CONVICTION_SENTIMENT_MILD_PENALTY)

    def test_score_sentiment_strongly_negative(self):
        # Below SENTIMENT_VETO_THRESHOLD=-0.65 → strong penalty
        score = self.RM._score_sentiment(-0.70)
        self.assertEqual(score, -config.CONVICTION_SENTIMENT_STRONG_PENALTY)

    def test_score_sentiment_none_returns_mild(self):
        score = self.RM._score_sentiment(None)
        self.assertGreater(score, 0)

    # ── A8: _score_orderflow ─────────────────────────────────────────────────

    def test_score_orderflow_strong_buy_flow(self):
        # orderflow=0.8 > 0.5, side=BUY → full weight
        score = self.RM._score_orderflow(0.8, "BUY")
        self.assertEqual(score, config.CONVICTION_WEIGHT_ORDERFLOW)

    def test_score_orderflow_strong_sell_flow(self):
        # orderflow=-0.8, side=SELL → aligned = 0.8 > 0.5 → full weight
        score = self.RM._score_orderflow(-0.8, "SELL")
        self.assertEqual(score, config.CONVICTION_WEIGHT_ORDERFLOW)

    def test_score_orderflow_opposing_strong_flow_for_buy(self):
        # orderflow=-0.8 (strong sell flow), side=BUY → strong penalty
        score = self.RM._score_orderflow(-0.8, "BUY")
        self.assertEqual(score, -config.CONVICTION_FLOW_STRONG_PENALTY)

    def test_score_orderflow_opposing_mild_flow_for_buy(self):
        # orderflow=-0.35, side=BUY → mild penalty
        score = self.RM._score_orderflow(-0.35, "BUY")
        self.assertEqual(score, -config.CONVICTION_FLOW_MILD_PENALTY)

    def test_score_orderflow_neutral_flow(self):
        # orderflow=0.1, side=BUY → neutral, small positive
        score = self.RM._score_orderflow(0.1, "BUY")
        self.assertGreater(score, 0)
        self.assertLess(score, config.CONVICTION_WEIGHT_ORDERFLOW)

    def test_score_orderflow_none_returns_zero(self):
        score = self.RM._score_orderflow(None, "BUY")
        self.assertEqual(score, 0.0)

    # ── A9: compute_conviction_score ─────────────────────────────────────────

    def test_conviction_score_range_always_0_to_100(self):
        from risk_manager import RiskManager
        # All favorable inputs
        score_high = RiskManager.compute_conviction_score(
            confidence=0.99,
            regime=config.REGIME_BULL,
            side="BUY",
            btc_regime=config.REGIME_BULL,
            funding_rate=-0.0002,
            sr_position=0.0,
            vwap_position=0.5,
            oi_change=0.05,
            volatility=0.01,
            sentiment_score=0.80,
            orderflow_score=0.8,
        )
        self.assertGreaterEqual(score_high, 0.0)
        self.assertLessEqual(score_high, 100.0)
        # All unfavorable inputs
        score_low = RiskManager.compute_conviction_score(
            confidence=0.80,
            regime=config.REGIME_CRASH,
            side="BUY",
            btc_regime=config.REGIME_BEAR,
            funding_rate=0.0005,
            sr_position=1.0,
            vwap_position=-0.5,
            oi_change=-0.05,
            volatility=0.15,
            sentiment_score=-0.80,
            orderflow_score=-0.8,
        )
        self.assertGreaterEqual(score_low, 0.0)
        self.assertLessEqual(score_low, 100.0)

    def test_conviction_score_favorable_inputs_beat_unfavorable(self):
        from risk_manager import RiskManager
        score_high = RiskManager.compute_conviction_score(
            confidence=0.99,
            regime=config.REGIME_BULL,
            side="BUY",
            btc_regime=config.REGIME_BULL,
            funding_rate=-0.0002,
            sr_position=0.0,
            vwap_position=0.5,
            oi_change=0.05,
            volatility=0.01,
            sentiment_score=0.80,
            orderflow_score=0.8,
        )
        score_low = RiskManager.compute_conviction_score(
            confidence=0.80,
            regime=config.REGIME_CRASH,
            side="BUY",
        )
        self.assertGreater(score_high, score_low)

    def test_conviction_score_sentiment_alert_forces_zero(self):
        from risk_manager import RiskManager
        # sentiment_score <= -1.0 is the hard veto
        score = RiskManager.compute_conviction_score(
            confidence=0.99,
            regime=config.REGIME_BULL,
            side="BUY",
            sentiment_score=-1.0,
        )
        self.assertEqual(score, 0.0)

    def test_conviction_score_all_none_optional_fields(self):
        from risk_manager import RiskManager
        # Should not raise even with all optional fields None
        score = RiskManager.compute_conviction_score(
            confidence=0.95,
            regime=config.REGIME_BULL,
            side="BUY",
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    # ── A10: get_conviction_leverage ─────────────────────────────────────────

    def test_leverage_below_min_trade_is_zero(self):
        from risk_manager import RiskManager
        self.assertEqual(RiskManager.get_conviction_leverage(35), 0)
        self.assertEqual(RiskManager.get_conviction_leverage(0), 0)
        self.assertEqual(RiskManager.get_conviction_leverage(39.9), 0)

    def test_leverage_at_min_trade_threshold(self):
        from risk_manager import RiskManager
        lev = RiskManager.get_conviction_leverage(40)
        self.assertGreater(lev, 0)
        self.assertEqual(lev, 10)

    def test_leverage_band_low(self):
        from risk_manager import RiskManager
        # 40–54 → 10x
        self.assertEqual(RiskManager.get_conviction_leverage(40), 10)
        self.assertEqual(RiskManager.get_conviction_leverage(50), 10)
        self.assertEqual(RiskManager.get_conviction_leverage(54), 10)

    def test_leverage_band_med(self):
        from risk_manager import RiskManager
        # 55–69 → 15x
        self.assertEqual(RiskManager.get_conviction_leverage(55), 15)
        self.assertEqual(RiskManager.get_conviction_leverage(65), 15)
        self.assertEqual(RiskManager.get_conviction_leverage(69), 15)

    def test_leverage_band_high(self):
        from risk_manager import RiskManager
        # 70–84 → 25x
        self.assertEqual(RiskManager.get_conviction_leverage(70), 25)
        self.assertEqual(RiskManager.get_conviction_leverage(80), 25)
        self.assertEqual(RiskManager.get_conviction_leverage(84), 25)

    def test_leverage_band_max(self):
        from risk_manager import RiskManager
        # 85–100 → 35x
        self.assertEqual(RiskManager.get_conviction_leverage(85), 35)
        self.assertEqual(RiskManager.get_conviction_leverage(100), 35)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase B: ExecutionEngine — Pure Method Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecutionEngineUnit(unittest.TestCase):
    """Unit tests for ExecutionEngine helper methods — no real API calls."""

    def setUp(self):
        from execution_engine import ExecutionEngine
        self.engine = ExecutionEngine()
        self.EE = ExecutionEngine

    # ── B1: _cdx_price_round ─────────────────────────────────────────────────

    def test_price_round_large_price(self):
        # >= 1000 → 1 decimal
        result = self.EE._cdx_price_round(45000.123)
        self.assertEqual(result, 45000.1)

    def test_price_round_mid_price(self):
        # >= 10 → 2 decimals
        result = self.EE._cdx_price_round(500.1234)
        self.assertEqual(result, 500.12)

    def test_price_round_single_digit(self):
        # >= 1 → 3 decimals
        result = self.EE._cdx_price_round(5.12345)
        self.assertEqual(result, 5.123)

    def test_price_round_cent_price(self):
        # >= 0.01 → 4 decimals
        result = self.EE._cdx_price_round(0.05123)
        self.assertEqual(result, 0.0512)

    def test_price_round_micro_price(self):
        # < 0.01 → 5 decimals
        result = self.EE._cdx_price_round(0.001234)
        self.assertEqual(result, 0.00123)

    def test_price_round_exactly_at_boundary_1000(self):
        result = self.EE._cdx_price_round(1000.0)
        self.assertEqual(result, 1000.0)

    def test_price_round_exactly_at_boundary_10(self):
        result = self.EE._cdx_price_round(10.0)
        self.assertEqual(result, 10.0)

    def test_price_round_exactly_at_boundary_1(self):
        result = self.EE._cdx_price_round(1.0)
        self.assertEqual(result, 1.0)

    # ── B2: _validate_and_size_cdx_order ─────────────────────────────────────

    def test_validate_returns_none_when_price_unavailable(self):
        cdx_mock = MagicMock()
        cdx_mock.get_current_price.return_value = None
        result = self.engine._validate_and_size_cdx_order("BTCUSDT", "B-BTCUSDT", 0.001, 10, cdx_mock)
        self.assertIsNone(result)

    def test_validate_boosts_qty_when_below_min_notional(self):
        cdx_mock = MagicMock()
        cdx_mock.get_current_price.return_value = 50000.0
        cdx_mock.get_usdt_balance.return_value = 5000.0
        cdx_mock.update_leverage.return_value = None
        # quantity=0.001 → notional=50 < COINDCX_MIN_NOTIONAL=120
        result = self.engine._validate_and_size_cdx_order("BTCUSDT", "B-BTCUSDT", 0.001, 10, cdx_mock)
        self.assertIsNotNone(result)
        price, qty, lev, wallet = result
        self.assertGreaterEqual(qty * price, config.COINDCX_MIN_NOTIONAL)

    def test_validate_returns_none_when_insufficient_balance(self):
        cdx_mock = MagicMock()
        cdx_mock.get_current_price.return_value = 50000.0
        cdx_mock.get_usdt_balance.return_value = 0.50  # Almost no balance
        cdx_mock.update_leverage.return_value = None
        result = self.engine._validate_and_size_cdx_order("BTCUSDT", "B-BTCUSDT", 0.01, 1, cdx_mock)
        self.assertIsNone(result)

    def test_validate_returns_tuple_on_success(self):
        cdx_mock = MagicMock()
        cdx_mock.get_current_price.return_value = 50000.0
        cdx_mock.get_usdt_balance.return_value = 10000.0
        cdx_mock.update_leverage.return_value = None
        result = self.engine._validate_and_size_cdx_order("BTCUSDT", "B-BTCUSDT", 0.01, 10, cdx_mock)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        price, qty, lev, wallet = result
        self.assertEqual(price, 50000.0)
        self.assertEqual(wallet, 10000.0)

    def test_validate_clamps_leverage_on_rejection(self):
        cdx_mock = MagicMock()
        cdx_mock.get_current_price.return_value = 50000.0
        cdx_mock.get_usdt_balance.return_value = 10000.0
        # First leverage update fails with max=25 message
        cdx_mock.update_leverage.side_effect = [
            Exception("Max allowed leverage = 25"),
            None,  # Second call succeeds after clamping
        ]
        result = self.engine._validate_and_size_cdx_order("BTCUSDT", "B-BTCUSDT", 0.01, 35, cdx_mock)
        self.assertIsNotNone(result)
        _, _, lev, _ = result
        self.assertEqual(lev, 25)  # Clamped to max

    def test_validate_returns_none_for_inactive_instrument(self):
        cdx_mock = MagicMock()
        cdx_mock.get_current_price.return_value = 50000.0
        cdx_mock.get_usdt_balance.return_value = 10000.0
        cdx_mock.update_leverage.side_effect = Exception("Instrument is not active")
        result = self.engine._validate_and_size_cdx_order("BTCUSDT", "B-BTCUSDT", 0.01, 10, cdx_mock)
        self.assertIsNone(result)

    # ── B3: _place_cdx_order_with_retry ──────────────────────────────────────

    def test_place_order_success_first_try(self):
        cdx_mock = MagicMock()
        cdx_mock.create_order.return_value = {"status": "success", "order_id": "123"}
        result = self.engine._place_cdx_order_with_retry(
            "BTCUSDT", "B-BTCUSDT", "BUY", 0.01, 10, 50000.0, 49000.0, 52000.0, 5000.0, cdx_mock
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["order_id"], "123")

    def test_place_order_retries_on_step_error(self):
        cdx_mock = MagicMock()
        # First call fails with step error, second succeeds
        cdx_mock.create_order.side_effect = [
            Exception("quantity must be divisible by 0.01"),
            {"status": "success", "order_id": "456"},
        ]
        result = self.engine._place_cdx_order_with_retry(
            "BTCUSDT", "B-BTCUSDT", "BUY", 0.001, 10, 50000.0, 49000.0, 52000.0, 5000.0, cdx_mock
        )
        self.assertIsNotNone(result)
        self.assertEqual(cdx_mock.create_order.call_count, 2)

    def test_place_order_reraises_non_step_errors(self):
        cdx_mock = MagicMock()
        cdx_mock.create_order.side_effect = Exception("API rate limit exceeded")
        with self.assertRaises(Exception) as ctx:
            self.engine._place_cdx_order_with_retry(
                "BTCUSDT", "B-BTCUSDT", "BUY", 0.01, 10, 50000.0, 49000.0, 52000.0, 5000.0, cdx_mock
            )
        self.assertIn("rate limit", str(ctx.exception))

    def test_place_order_retry_returns_none_on_margin_breach(self):
        cdx_mock = MagicMock()
        # Step error triggers retry; retry qty results in margin > wallet
        cdx_mock.create_order.side_effect = [
            Exception("quantity must be divisible by 1.0"),
            None,  # Won't be called if margin check fails
        ]
        # wallet=0.50 → margin for retried qty will exceed wallet
        result = self.engine._place_cdx_order_with_retry(
            "BTCUSDT", "B-BTCUSDT", "BUY", 0.001, 1, 50000.0, 49000.0, 52000.0, 0.50, cdx_mock
        )
        self.assertIsNone(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase C: DataPipeline — Pure Function Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataPipelineUnit(unittest.TestCase):
    """Unit tests for data_pipeline helper functions — no real API calls."""

    def _make_sample_klines(self, n=5):
        """Generate a list of raw Binance kline rows."""
        now_ms = 1_609_459_200_000  # Fixed: 2021-01-01 00:00:00 UTC (deterministic)
        rows = []
        for i in range(n):
            ts = now_ms - (n - i) * 60000
            rows.append([
                ts,          # timestamp
                "50000.00",  # open
                "50500.00",  # high
                "49900.00",  # low
                "50200.00",  # close
                "10.5",      # volume
                ts + 59999,  # close_time
                "527100.0",  # quote_av
                100,         # trades
                "5.2",       # tb_base_av
                "261450.0",  # tb_quote_av
                "0",         # ignore
            ])
        return rows

    def test_parse_klines_df_column_names(self):
        from data_pipeline import _parse_klines_df
        klines = self._make_sample_klines(5)
        df = _parse_klines_df(klines)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            self.assertIn(col, df.columns)

    def test_parse_klines_df_length(self):
        from data_pipeline import _parse_klines_df
        klines = self._make_sample_klines(7)
        df = _parse_klines_df(klines)
        self.assertEqual(len(df), 7)

    def test_parse_klines_df_dtype_timestamp(self):
        from data_pipeline import _parse_klines_df
        klines = self._make_sample_klines(3)
        df = _parse_klines_df(klines)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))

    def test_parse_klines_df_dtypes_ohlcv_are_float(self):
        from data_pipeline import _parse_klines_df
        klines = self._make_sample_klines(3)
        df = _parse_klines_df(klines)
        for col in ["open", "high", "low", "close", "volume"]:
            self.assertTrue(pd.api.types.is_float_dtype(df[col]),
                            f"Expected {col} to be float, got {df[col].dtype}")

    def test_parse_klines_df_ohlcv_values_correct(self):
        from data_pipeline import _parse_klines_df
        klines = self._make_sample_klines(2)
        df = _parse_klines_df(klines)
        self.assertAlmostEqual(df.iloc[0]["open"], 50000.0)
        self.assertAlmostEqual(df.iloc[0]["high"], 50500.0)
        self.assertAlmostEqual(df.iloc[0]["low"], 49900.0)
        self.assertAlmostEqual(df.iloc[0]["close"], 50200.0)

    @patch("data_pipeline.fetch_klines")
    def test_get_multi_timeframe_data_returns_three_timeframes(self, mock_fetch):
        from data_pipeline import get_multi_timeframe_data
        sample_df = pd.DataFrame({"close": [1, 2, 3]})
        mock_fetch.return_value = sample_df
        result = get_multi_timeframe_data("BTCUSDT", limit=10)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        # Should have execution, confirmation, and macro timeframe keys
        self.assertIn(config.TIMEFRAME_EXECUTION, result)
        self.assertIn(config.TIMEFRAME_CONFIRMATION, result)
        self.assertIn(config.TIMEFRAME_MACRO, result)

    @patch("data_pipeline.fetch_klines")
    def test_get_multi_timeframe_partial_failure_handled(self, mock_fetch):
        from data_pipeline import get_multi_timeframe_data
        sample_df = pd.DataFrame({"close": [1, 2, 3]})
        mock_fetch.side_effect = [sample_df, None, sample_df]
        result = get_multi_timeframe_data("BTCUSDT", limit=10)
        # Partial failure is handled — result still has 3 keys, one is None
        self.assertEqual(len(result), 3)
        none_count = sum(1 for v in result.values() if v is None)
        self.assertEqual(none_count, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase E: SentimentSources — _reddit_importance_score
# ═══════════════════════════════════════════════════════════════════════════════

class TestSentimentSourcesUnit(unittest.TestCase):
    """Unit tests for sentiment_sources helper functions."""

    def setUp(self):
        from sentiment_sources import _reddit_importance_score
        self.score_fn = _reddit_importance_score

    def test_zero_reddit_score_returns_minimum(self):
        result = self.score_fn(0)
        self.assertAlmostEqual(result, 0.4, places=5)

    def test_high_reddit_score_approaches_one(self):
        result = self.score_fn(5000)
        self.assertAlmostEqual(result, 0.9, places=5)

    def test_very_high_reddit_score_at_max(self):
        # Formula: 0.4 + min(score, 5000) / 10000 — saturates at 0.9 when score >= 5000
        result = self.score_fn(10000)
        self.assertAlmostEqual(result, 0.9, places=5)

    def test_extremely_high_score_also_saturates(self):
        # min(999999, 5000) = 5000 → 0.4 + 0.5 = 0.9
        result = self.score_fn(999999)
        self.assertAlmostEqual(result, 0.9, places=5)

    def test_always_in_range_0_4_to_1(self):
        for s in [0, 100, 500, 1000, 5000, 10000, 50000]:
            result = self.score_fn(s)
            self.assertGreaterEqual(result, 0.4)
            self.assertLessEqual(result, 1.0)

    def test_monotonically_increasing(self):
        prev = self.score_fn(0)
        for s in [100, 500, 1000, 3000]:
            curr = self.score_fn(s)
            self.assertGreaterEqual(curr, prev)
            prev = curr


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase F: FeatureEngine — Individual Indicator Functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineUnit(unittest.TestCase):
    """Unit tests for individual technical indicator functions."""

    def _rising_prices(self, n=50, start=100.0, step=1.0):
        return pd.Series([start + i * step for i in range(n)])

    def _falling_prices(self, n=50, start=100.0, step=1.0):
        return pd.Series([start - i * step for i in range(n)])

    def _flat_prices(self, n=50, value=100.0):
        return pd.Series([value] * n)

    def _make_ohlcv(self, close_prices):
        n = len(close_prices)
        df = pd.DataFrame({
            "open":  close_prices * 0.99,
            "high":  close_prices * 1.01,
            "low":   close_prices * 0.98,
            "close": close_prices,
            "volume": pd.Series([1000.0] * n),
        })
        return df

    # ── F1: compute_rsi ──────────────────────────────────────────────────────

    def test_rsi_rising_prices_above_50(self):
        from feature_engine import compute_rsi
        # Use noisy rising prices so there are some losses — RSI > 50 but not NaN
        np.random.seed(42)
        prices = pd.Series(100.0 + np.cumsum(np.random.normal(0.5, 0.2, 100)))
        rsi = compute_rsi(prices)
        last_valid = rsi.dropna().iloc[-1]
        self.assertGreater(last_valid, 50)

    def test_rsi_falling_prices_below_50(self):
        from feature_engine import compute_rsi
        # Use noisy falling prices so there are some gains — RSI < 50 but not NaN
        np.random.seed(42)
        prices = pd.Series(100.0 + np.cumsum(np.random.normal(-0.5, 0.2, 100)))
        rsi = compute_rsi(prices)
        last_valid = rsi.dropna().iloc[-1]
        self.assertLess(last_valid, 50)

    def test_rsi_always_in_0_to_100(self):
        from feature_engine import compute_rsi
        prices = self._rising_prices(50)
        rsi = compute_rsi(prices)
        valid = rsi.dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 100).all())

    def test_rsi_with_custom_length(self):
        from feature_engine import compute_rsi
        prices = self._rising_prices(50)
        rsi_7  = compute_rsi(prices, length=7)
        rsi_14 = compute_rsi(prices, length=14)
        # Both should be Series of same length
        self.assertEqual(len(rsi_7), len(prices))
        self.assertEqual(len(rsi_14), len(prices))

    # ── F2: compute_atr ──────────────────────────────────────────────────────

    def test_atr_always_positive(self):
        from feature_engine import compute_atr
        close = self._rising_prices(50)
        df = self._make_ohlcv(close)
        atr = compute_atr(df)
        valid = atr.dropna()
        self.assertTrue((valid > 0).all())

    def test_atr_higher_vol_data_gives_higher_atr(self):
        from feature_engine import compute_atr
        # Low-volatility OHLCV
        close = pd.Series([100.0] * 50)
        df_low = pd.DataFrame({
            "open": close, "high": close + 0.01, "low": close - 0.01, "close": close, "volume": close
        })
        # High-volatility OHLCV
        df_high = pd.DataFrame({
            "open": close, "high": close + 5.0, "low": close - 5.0, "close": close, "volume": close
        })
        atr_low  = compute_atr(df_low).dropna().iloc[-1]
        atr_high = compute_atr(df_high).dropna().iloc[-1]
        self.assertGreater(atr_high, atr_low)

    # ── F3: compute_bollinger_bands ───────────────────────────────────────────

    def test_bollinger_bands_upper_above_lower(self):
        from feature_engine import compute_bollinger_bands
        prices = self._rising_prices(50)
        mid, upper, lower = compute_bollinger_bands(prices)
        valid_idx = upper.dropna().index
        self.assertTrue((upper[valid_idx] >= lower[valid_idx]).all())

    def test_bollinger_bands_price_inside_bands_for_stable_data(self):
        from feature_engine import compute_bollinger_bands
        # Stable prices centered → price should be near middle band
        prices = self._flat_prices(50)
        mid, upper, lower = compute_bollinger_bands(prices)
        valid_idx = mid.dropna().index
        last_mid = mid[valid_idx].iloc[-1]
        self.assertAlmostEqual(last_mid, 100.0, places=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase D: SentimentEngine — Pure Helper Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSentimentEngineUnit(unittest.TestCase):
    """Unit tests for sentiment_engine helper functions."""

    # ── D1: _source_category ─────────────────────────────────────────────────

    def test_source_category_rss_format(self):
        from sentiment_engine import _source_category
        self.assertEqual(_source_category("RSS:CoinTelegraph"), "RSS")

    def test_source_category_reddit_format(self):
        from sentiment_engine import _source_category
        self.assertEqual(_source_category("Reddit:r/bitcoin"), "Reddit")

    def test_source_category_cryptopanic(self):
        from sentiment_engine import _source_category
        self.assertEqual(_source_category("CryptoPanic"), "CryptoPanic")

    def test_source_category_feargreed(self):
        from sentiment_engine import _source_category
        self.assertEqual(_source_category("FearGreed"), "FearGreed")

    def test_source_category_no_separator(self):
        from sentiment_engine import _source_category
        self.assertEqual(_source_category("SomeSource"), "SomeSource")

    def test_source_category_empty_string(self):
        from sentiment_engine import _source_category
        self.assertEqual(_source_category(""), "")

    # ── D2: SentimentSignal.effective_score ──────────────────────────────────

    def test_effective_score_with_no_alert(self):
        from sentiment_engine import SentimentSignal
        signal = SentimentSignal(
            coin="BTC", score=0.5, confidence=0.8, buzz_velocity=5,
            momentum=0.1, alert=False, alert_reason="", fear_greed=None
        )
        self.assertEqual(signal.effective_score, 0.5)

    def test_effective_score_with_alert_returns_minus_one(self):
        from sentiment_engine import SentimentSignal
        signal = SentimentSignal(
            coin="BTC", score=0.3, confidence=0.8, buzz_velocity=5,
            momentum=0.1, alert=True, alert_reason="hack detected", fear_greed=None
        )
        self.assertEqual(signal.effective_score, -1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase H: HMMBrain — Training, Prediction, State Map
# ═══════════════════════════════════════════════════════════════════════════════

class TestHMMBrainUnit(unittest.TestCase):
    """Unit tests for HMMBrain using synthetic feature DataFrames (no API calls)."""

    @staticmethod
    def _make_hmm_df(n=200, seed=42):
        """Synthetic DataFrame with the 4 HMM_FEATURES columns."""
        np.random.seed(seed)
        # Two regime halves: bull-like returns then bear-like returns
        log_returns = np.concatenate([
            np.random.normal(+0.006, 0.010, n // 2),   # bull half
            np.random.normal(-0.006, 0.018, n // 2),   # bear half
        ])
        volatility = np.abs(log_returns) * 1.2 + np.abs(np.random.normal(0, 0.001, n))
        volume_change = np.random.normal(0, 0.08, n)
        rsi_norm = np.clip(np.random.normal(0.5, 0.15, n), 0.0, 1.0)
        return pd.DataFrame({
            "log_return":    log_returns,
            "volatility":    volatility,
            "volume_change": volume_change,
            "rsi_norm":      rsi_norm,
        })

    def setUp(self):
        from hmm_brain import HMMBrain
        self.HMMBrain = HMMBrain

    # ── H1: untrained state ──────────────────────────────────────────────────

    def test_untrained_predict_returns_chop_zero(self):
        brain = self.HMMBrain()
        df = self._make_hmm_df(20)
        state, conf = brain.predict(df)
        self.assertEqual(state, config.REGIME_CHOP)
        self.assertEqual(conf, 0.0)

    def test_untrained_is_trained_false(self):
        brain = self.HMMBrain()
        self.assertFalse(brain.is_trained)

    def test_untrained_needs_retrain_true(self):
        brain = self.HMMBrain()
        self.assertTrue(brain.needs_retrain())

    def test_untrained_predict_all_returns_chop_array(self):
        brain = self.HMMBrain()
        df = self._make_hmm_df(10)
        states = brain.predict_all(df)
        self.assertTrue(all(s == config.REGIME_CHOP for s in states))

    # ── H2: after training ───────────────────────────────────────────────────

    def test_train_sets_is_trained(self):
        brain = self.HMMBrain()
        brain.train(self._make_hmm_df())
        self.assertTrue(brain.is_trained)

    def test_not_needs_retrain_right_after_training(self):
        brain = self.HMMBrain()
        brain.train(self._make_hmm_df())
        self.assertFalse(brain.needs_retrain())

    def test_predict_confidence_in_0_1(self):
        brain = self.HMMBrain()
        brain.train(self._make_hmm_df())
        _, conf = brain.predict(self._make_hmm_df())
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_predict_returns_valid_regime(self):
        brain = self.HMMBrain()
        brain.train(self._make_hmm_df())
        state, _ = brain.predict(self._make_hmm_df())
        self.assertIn(state, [config.REGIME_BULL, config.REGIME_BEAR,
                               config.REGIME_CHOP, config.REGIME_CRASH])

    def test_predict_all_length_matches_input(self):
        brain = self.HMMBrain()
        df = self._make_hmm_df(100)
        brain.train(df)
        states = brain.predict_all(df)
        self.assertEqual(len(states), len(df))

    def test_predict_all_values_are_valid_regimes(self):
        brain = self.HMMBrain()
        df = self._make_hmm_df(100)
        brain.train(df)
        states = brain.predict_all(df)
        valid = {config.REGIME_BULL, config.REGIME_BEAR,
                 config.REGIME_CHOP, config.REGIME_CRASH}
        self.assertTrue(set(states).issubset(valid))

    # ── H3: state_map correctness (validates the HMM_FEATURES fix) ──────────

    def test_state_map_has_all_3_regimes(self):
        # 3-state model: BULL, CHOP, BEAR (CRASH merged into BEAR — removed)
        brain = self.HMMBrain()
        brain.train(self._make_hmm_df())
        self.assertEqual(len(brain._state_map), 3)
        self.assertEqual(
            set(brain._state_map.values()),
            {config.REGIME_BULL, config.REGIME_BEAR, config.REGIME_CHOP},
        )

    def test_bull_state_has_higher_log_return_than_bear(self):
        """Core correctness check: log_return is col 0, so BULL > BEAR by mean."""
        brain = self.HMMBrain()
        brain.train(self._make_hmm_df())
        bull_raw = [k for k, v in brain._state_map.items() if v == config.REGIME_BULL][0]
        bear_raw = [k for k, v in brain._state_map.items() if v == config.REGIME_BEAR][0]
        self.assertGreater(brain.model.means_[bull_raw][0],
                           brain.model.means_[bear_raw][0])

    # ── H4: regime name mapping ───────────────────────────────────────────────

    def test_get_regime_name_bull(self):
        brain = self.HMMBrain()
        self.assertEqual(brain.get_regime_name(config.REGIME_BULL), "BULLISH")

    def test_get_regime_name_bear(self):
        brain = self.HMMBrain()
        self.assertEqual(brain.get_regime_name(config.REGIME_BEAR), "BEARISH")

    def test_get_regime_name_chop(self):
        brain = self.HMMBrain()
        self.assertEqual(brain.get_regime_name(config.REGIME_CHOP), "SIDEWAYS/CHOP")

    def test_get_regime_name_crash(self):
        brain = self.HMMBrain()
        self.assertEqual(brain.get_regime_name(config.REGIME_CRASH), "CRASH/PANIC")

    # ── H5: edge cases ────────────────────────────────────────────────────────

    def test_insufficient_data_does_not_train(self):
        brain = self.HMMBrain()
        brain.train(self._make_hmm_df(20))  # well below min threshold
        self.assertFalse(brain.is_trained)

    def test_train_with_nan_rows_does_not_crash(self):
        brain = self.HMMBrain()
        df = self._make_hmm_df(200)
        df.iloc[10:15] = np.nan  # inject NaNs
        brain.train(df)  # should drop NaNs and train normally
        self.assertTrue(brain.is_trained)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase I: CoinScanner — Filtering, Sorting, Routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoinScannerUnit(unittest.TestCase):
    """Unit tests for coin_scanner functions — no real API calls."""

    def _make_ticker(self, symbol, volume):
        return {"symbol": symbol, "quoteVolume": str(volume)}

    # ── I1: COIN_EXCLUDE filtering ───────────────────────────────────────────

    @patch("coin_scanner._get_binance_client")
    def test_excluded_coins_not_in_results(self, mock_get_client):
        from coin_scanner import _get_top_coins_binance
        client = MagicMock()
        mock_get_client.return_value = client
        client.get_ticker.return_value = [
            self._make_ticker("BTCUSDT",  1_000_000),
            self._make_ticker("ETHUSDT",    800_000),
            self._make_ticker("EURUSDT",    500_000),   # excluded
            self._make_ticker("WBTCUSDT",   900_000),   # excluded
            self._make_ticker("USDCUSDT",   700_000),   # excluded
        ]
        result = _get_top_coins_binance(limit=10)
        self.assertIn("BTCUSDT", result)
        self.assertIn("ETHUSDT", result)
        self.assertNotIn("EURUSDT", result)
        self.assertNotIn("WBTCUSDT", result)
        self.assertNotIn("USDCUSDT", result)

    # ── I2: leverage token filtering ────────────────────────────────────────

    @patch("coin_scanner._get_binance_client")
    def test_leverage_tokens_filtered_out(self, mock_get_client):
        from coin_scanner import _get_top_coins_binance
        client = MagicMock()
        mock_get_client.return_value = client
        client.get_ticker.return_value = [
            self._make_ticker("BTCUSDT",     1_000_000),
            self._make_ticker("BTCUPUSDT",     900_000),   # UP token
            self._make_ticker("BTCDOWNUSDT",   800_000),   # DOWN token
            self._make_ticker("ETHBULLUSDT",   700_000),   # BULL token
            self._make_ticker("ETHBEARUSDT",   600_000),   # BEAR token
        ]
        result = _get_top_coins_binance(limit=10)
        self.assertIn("BTCUSDT", result)
        self.assertNotIn("BTCUPUSDT", result)
        self.assertNotIn("BTCDOWNUSDT", result)
        self.assertNotIn("ETHBULLUSDT", result)
        self.assertNotIn("ETHBEARUSDT", result)

    # ── I3: volume sort order ────────────────────────────────────────────────

    @patch("coin_scanner._get_binance_client")
    def test_results_sorted_by_volume_descending(self, mock_get_client):
        from coin_scanner import _get_top_coins_binance
        client = MagicMock()
        mock_get_client.return_value = client
        client.get_ticker.return_value = [
            self._make_ticker("LTCUSDT",  100_000),
            self._make_ticker("BTCUSDT", 1_000_000),
            self._make_ticker("ETHUSDT",  500_000),
        ]
        result = _get_top_coins_binance(limit=10)
        self.assertEqual(result[0], "BTCUSDT")
        self.assertEqual(result[1], "ETHUSDT")
        self.assertEqual(result[2], "LTCUSDT")

    # ── I4: limit respected ──────────────────────────────────────────────────

    @patch("coin_scanner._get_binance_client")
    def test_limit_caps_result_length(self, mock_get_client):
        from coin_scanner import _get_top_coins_binance
        client = MagicMock()
        mock_get_client.return_value = client
        client.get_ticker.return_value = [
            self._make_ticker(f"COIN{i:02d}USDT", 10_000 - i * 100)
            for i in range(20)
        ]
        result = _get_top_coins_binance(limit=5)
        self.assertEqual(len(result), 5)

    # ── I5: only USDT pairs ──────────────────────────────────────────────────

    @patch("coin_scanner._get_binance_client")
    def test_only_usdt_pairs_returned(self, mock_get_client):
        from coin_scanner import _get_top_coins_binance
        client = MagicMock()
        mock_get_client.return_value = client
        client.get_ticker.return_value = [
            self._make_ticker("BTCUSDT", 1_000_000),
            self._make_ticker("BTCBTC",    900_000),   # non-USDT
            self._make_ticker("ETHBNB",    800_000),   # non-USDT
            self._make_ticker("ETHUSDT",   700_000),
        ]
        result = _get_top_coins_binance(limit=10)
        for sym in result:
            self.assertTrue(sym.endswith("USDT"), f"{sym} should end with USDT")

    # ── I6: API failure fallback ─────────────────────────────────────────────

    @patch("coin_scanner._get_binance_client")
    def test_api_error_returns_primary_symbol(self, mock_get_client):
        from coin_scanner import _get_top_coins_binance
        client = MagicMock()
        mock_get_client.return_value = client
        client.get_ticker.side_effect = Exception("Connection timeout")
        result = _get_top_coins_binance(limit=10)
        self.assertEqual(result, [config.PRIMARY_SYMBOL])

    # ── I7: routing ──────────────────────────────────────────────────────────

    @patch("coin_scanner._get_binance_client")
    def test_paper_mode_routes_to_binance(self, mock_get_client):
        from coin_scanner import get_top_coins_by_volume
        client = MagicMock()
        mock_get_client.return_value = client
        client.get_ticker.return_value = [
            self._make_ticker("BTCUSDT", 1_000_000)
        ]
        with patch.object(config, "PAPER_TRADE", True):
            result = get_top_coins_by_volume(limit=5)
        self.assertIsInstance(result, list)
        client.get_ticker.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
