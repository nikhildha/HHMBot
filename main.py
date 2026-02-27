"""
Project Regime-Master — Main Bot Loop (Multi-Coin)
Scans top 50 coins by volume, runs HMM regime analysis on each,
and deploys paper/live trades on all eligible symbols simultaneously.
"""
import json
import os
import time
import logging
from datetime import datetime

import config
from hmm_brain import HMMBrain
from data_pipeline import fetch_klines, get_multi_timeframe_data
from feature_engine import compute_all_features, compute_hmm_features
from execution_engine import ExecutionEngine
from risk_manager import RiskManager
from sideways_strategy import evaluate_mean_reversion
from coin_scanner import get_top_coins_by_volume
import tradebook
import telegram as tg
import sentiment_engine as _sent_mod
import orderflow_engine as _of_mod

# ─── Logging Setup ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.DATA_DIR, "bot.log"), encoding="utf-8"),
    ]
)
logger = logging.getLogger("RegimeMaster")


class RegimeMasterBot:
    """
    Multi-coin orchestrator for Project Regime-Master.

    Heartbeat: every 1 minute (LOOP_INTERVAL_SECONDS)
      - Process commands (kill switch, reset)
      - Sync positions (detect SL/TP auto-closes)
      - Update unrealized P&L

    Full analysis: every 15 minutes (ANALYSIS_INTERVAL_SECONDS)
      1. Periodically refresh top-50 coin list (every SCAN_INTERVAL_CYCLES)
      2. For each coin: fetch data → HMM regime → check eligibility → trade
      3. Track active positions to respect MAX_CONCURRENT_POSITIONS
      4. Check global risk (kill switch, drawdown)
    """

    def __init__(self):
        self.executor = ExecutionEngine()
        self.risk = RiskManager()
        self._trade_count = 0
        self._cycle_count = 0
        self._last_analysis_time = 0.0  # epoch — triggers immediate first run

        # Multi-coin state
        self._coin_list = []
        self._active_positions = {}  # symbol → {regime, confidence, side, entry_time}
        self._coin_brains = {}       # symbol → HMMBrain (cached per coin)
        self._coin_states = {}       # symbol → latest state dict (for dashboard)

        # ── Startup: sync _active_positions from tradebook ──────────
        self._load_positions_from_tradebook()

        # ── Sentiment Engine (lazy singleton) ─────────────────────────
        self._sentiment = None
        if config.SENTIMENT_ENABLED:
            try:
                self._sentiment = _sent_mod.get_engine()
                logger.info("📰 Sentiment Engine ready (VADER%s)",
                            " + FinBERT" if config.SENTIMENT_USE_FINBERT else " only")
            except Exception as e:
                logger.warning("⚠️  Sentiment Engine failed to load: %s", e)

        # ── Order Flow Engine (lazy singleton) ────────────────────────
        self._orderflow = None
        if config.ORDERFLOW_ENABLED:
            try:
                self._orderflow = _of_mod.get_engine()
                logger.info("📊 Order Flow Engine ready (L2 depth + taker flow + cumDelta)")
            except Exception as e:
                logger.warning("⚠️  Order Flow Engine failed to load: %s", e)

    # ─── Main Loop ───────────────────────────────────────────────────────────

    def run(self):
        mode = "PAPER" if config.PAPER_TRADE else "LIVE"
        net = "TESTNET" if config.TESTNET else "PRODUCTION"
        coin_mode = "MULTI-COIN" if config.MULTI_COIN_MODE else "SINGLE"
        logger.info(
            "🚀 Regime-Master Bot Started | %s mode | %s | %s | Max Positions: %d",
            mode, net, coin_mode, config.MAX_CONCURRENT_POSITIONS,
        )
        logger.info(
            "⏱ Heartbeat: %ds | Full analysis: every %ds",
            config.LOOP_INTERVAL_SECONDS, config.ANALYSIS_INTERVAL_SECONDS,
        )

        while True:
            try:
                self._heartbeat()
                time.sleep(config.LOOP_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                logger.info("⏹ Bot stopped by user.")
                break
            except Exception as e:
                logger.error("⚠️ Loop error: %s", e, exc_info=True)
                time.sleep(config.ERROR_RETRY_SECONDS)

    def _heartbeat(self):
        """1-minute heartbeat: lightweight checks + trigger full analysis on schedule."""
        # ── Check engine pause state ──────────────────────────────────
        try:
            import json
            state_path = os.path.join(os.path.dirname(__file__), "data", "engine_state.json")
            if os.path.exists(state_path):
                with open(state_path) as f:
                    state = json.load(f)
                if state.get("status") == "paused":
                    # Check if timed halt has expired
                    halt_until = state.get("halt_until")
                    if halt_until:
                        try:
                            halt_dt = datetime.fromisoformat(halt_until.replace("Z", "+00:00")).replace(tzinfo=None)
                            if datetime.utcnow() >= halt_dt:
                                # Auto-resume: halt period expired
                                resume_state = {"status": "running", "resumed_at": datetime.utcnow().isoformat() + "Z", "paused_by": None}
                                with open(state_path, "w") as fw:
                                    json.dump(resume_state, fw, indent=2)
                                logger.info("✅ Auto-halt expired — engine RESUMED automatically")
                                self._pause_logged = False
                            else:
                                remaining = (halt_dt - datetime.utcnow()).total_seconds() / 60
                                if not getattr(self, '_pause_logged', False):
                                    reason = state.get("reason", "Auto-halted")
                                    logger.warning("⏸️  Engine HALTED: %s (%.0f min remaining)", reason, remaining)
                                    self._pause_logged = True
                                return  # Still halted
                        except Exception:
                            pass
                    else:
                        # Manual pause (no expiry)
                        if not getattr(self, '_pause_logged', False):
                            logger.info("⏸️  Engine PAUSED via dashboard — skipping all analysis")
                            self._pause_logged = True
                        return  # Skip entire heartbeat
            self._pause_logged = False
        except Exception:
            pass

        # Always: process commands (kill switch / reset)
        self._process_commands()

        if self.risk.is_killed:
            return

        # Always: sync positions (detect SL/TP auto-closes)
        self._sync_positions()

        # Always: update unrealized P&L + trailing SL/TP
        try:
            tradebook.update_unrealized()
        except Exception as e:
            logger.debug("Tradebook unrealized update error: %s", e)

        # Live mode: sync CoinDCX positions → tradebook → trailing SL/TP
        if not config.PAPER_TRADE:
            try:
                self._sync_coindcx_positions()
            except Exception as e:
                logger.debug("CoinDCX position sync error: %s", e)
            try:
                tradebook.sync_live_tpsl()
            except Exception as e:
                logger.debug("Live TPSL sync error: %s", e)

        # Check for manual trigger from dashboard
        trigger_file = os.path.join(config.DATA_DIR, "force_cycle.trigger")
        force = os.path.exists(trigger_file)
        if force:
            try:
                os.remove(trigger_file)
            except Exception:
                pass
            logger.info("⚡ Manual cycle trigger received from dashboard!")

        # Check if it's time for a full analysis cycle
        now = time.time()
        elapsed = now - self._last_analysis_time
        if force or elapsed >= config.ANALYSIS_INTERVAL_SECONDS:
            logger.info("🧠 Running full analysis cycle (%.0fs since last)...", elapsed)
            self._tick()
            self._last_analysis_time = time.time()
            self._save_timing()  # Update timing for dashboard
        else:
            remaining = config.ANALYSIS_INTERVAL_SECONDS - elapsed
            logger.debug("💤 Next analysis in %.0fs...", remaining)

    def _save_timing(self):
        """Persist last/next analysis timestamps for the dashboard."""
        try:
            multi = {}
            if os.path.exists(config.MULTI_STATE_FILE):
                with open(config.MULTI_STATE_FILE, "r") as f:
                    multi = json.load(f)
            multi["last_analysis_time"] = datetime.utcnow().isoformat() + "Z"
            nxt = self._last_analysis_time + config.ANALYSIS_INTERVAL_SECONDS
            multi["next_analysis_time"] = datetime.utcfromtimestamp(nxt).isoformat() + "Z"
            multi["analysis_interval_seconds"] = config.ANALYSIS_INTERVAL_SECONDS
            with open(config.MULTI_STATE_FILE, "w") as f:
                json.dump(multi, f, indent=2)
        except Exception:
            pass

    def _tick(self):
        """Full analysis cycle — runs every ANALYSIS_INTERVAL_SECONDS."""
        cycle_start = time.time()
        self._cycle_count += 1

        # ── 1. Refresh coin list periodically ────────────────────
        if config.MULTI_COIN_MODE:
            if not self._coin_list or self._cycle_count % config.SCAN_INTERVAL_CYCLES == 1:
                logger.info("🔍 Refreshing top %d coins by volume...", config.TOP_COINS_LIMIT)
                self._coin_list = get_top_coins_by_volume(limit=config.TOP_COINS_LIMIT)
                logger.info("📋 Tracking %d coins: %s ...", len(self._coin_list),
                            ", ".join(self._coin_list[:5]))
            symbols = self._coin_list
        else:
            symbols = [config.PRIMARY_SYMBOL]

        # ── 2. Global equity + kill switch check ─────────────────
        balance = self.executor.get_futures_balance()
        self.risk.record_equity(balance)
        if self.risk.check_kill_switch():
            logger.warning("🚨 Kill switch triggered! Closing all positions.")
            # Telegram kill switch alert
            try:
                peak = max(b for _, b in self.risk.equity_history) if self.risk.equity_history else 0
                current = self.risk.equity_history[-1][1] if self.risk.equity_history else 0
                dd = (peak - current) / peak * 100 if peak > 0 else 0
                tg.notify_kill_switch(dd, peak, current)
            except Exception:
                pass
            for sym in list(self._active_positions.keys()):
                tradebook.close_trade(symbol=sym, reason="KILL_SWITCH")
                self.executor.close_all_positions(sym)
            self._active_positions.clear()
            return

        # ── 3. Check exits for active positions ──────────────────
        self._check_exits(symbols)

        # ── 4. Scan each coin ────────────────────────────────────
        # SOLE SOURCE OF TRUTH: tradebook active count
        tradebook_active = tradebook.get_active_trades()
        tradebook_active_count = len(tradebook_active)
        tradebook_active_symbols = {t["symbol"] for t in tradebook_active}
        eligible_trades = []

        for symbol in symbols:
            try:
                result = self._analyze_coin(symbol, balance)
                if result:
                    eligible_trades.append(result)
            except Exception as e:
                logger.debug("Error analyzing %s: %s", symbol, e)
                continue

        # ── 5. Deploy eligible trades (respect position limit) ───
        slots_available = max(0, config.MAX_CONCURRENT_POSITIONS - tradebook_active_count)

        if slots_available == 0:
            logger.info(
                "📊 Position limit reached (%d/%d). No new deployments this cycle.",
                tradebook_active_count, config.MAX_CONCURRENT_POSITIONS,
            )
            # AUTO-HALT: 25 positions reached → pause engine for 1 hour
            if tradebook_active_count >= config.MAX_CONCURRENT_POSITIONS:
                try:
                    import json
                    from datetime import timedelta
                    halt_until = (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z"
                    state_path = os.path.join(os.path.dirname(__file__), "data", "engine_state.json")
                    halt_state = {
                        "status": "paused",
                        "paused_at": datetime.utcnow().isoformat() + "Z",
                        "paused_by": "auto_25_cap",
                        "halt_until": halt_until,
                        "reason": f"25 active positions reached — auto-halted for 1 hour until {halt_until}",
                    }
                    with open(state_path, "w") as f:
                        json.dump(halt_state, f, indent=2)
                    logger.warning(
                        "🛑 AUTO-HALT: %d positions active (max %d). Engine paused for 1 hour until %s",
                        tradebook_active_count, config.MAX_CONCURRENT_POSITIONS, halt_until,
                    )
                except Exception as e:
                    logger.error("Failed to write auto-halt state: %s", e)

        # ── Loss streak cooldown: pause 30 min after 5 consecutive losses ──
        LOSS_STREAK_LIMIT = 5
        COOLDOWN_MINUTES = 30
        streak, last_loss_ts = tradebook.get_current_loss_streak()
        if streak >= LOSS_STREAK_LIMIT and last_loss_ts:
            from datetime import datetime as _dt
            try:
                last_loss_time = _dt.fromisoformat(last_loss_ts.replace("Z", "+00:00"))
                elapsed = (datetime.utcnow() - last_loss_time.replace(tzinfo=None)).total_seconds() / 60
                if elapsed < COOLDOWN_MINUTES:
                    remaining = COOLDOWN_MINUTES - elapsed
                    logger.warning(
                        "⏸️  COOLDOWN: %d consecutive losses — pausing new deployments for %.0f more min",
                        streak, remaining,
                    )
                    return
                else:
                    logger.info("✅ Cooldown expired (%.0f min elapsed). Resuming deployments.", elapsed)
            except Exception:
                pass  # If timestamp parse fails, don't block

        # Sort by confidence (highest first)
        eligible_trades.sort(key=lambda x: x["confidence"], reverse=True)

        deployed = 0
        deployed_trades = []  # Collect for batch Telegram alert
        for trade in eligible_trades:
            if deployed >= slots_available:
                logger.info("📊 Position limit reached (%d/%d). Queuing rest.",
                            tradebook_active_count + deployed, config.MAX_CONCURRENT_POSITIONS)
                break

            # Re-check hard limit from tradebook EVERY iteration (bulletproof)
            current_total = len(tradebook.get_active_trades())
            if current_total >= config.MAX_CONCURRENT_POSITIONS:
                logger.warning(
                    "🛑 Hard limit reached: %d active trades in tradebook (max %d). Halting deployment.",
                    current_total, config.MAX_CONCURRENT_POSITIONS,
                )
                break

            sym = trade["symbol"]
            # Check tradebook (not dict!) for duplicate active positions on this symbol
            if sym in tradebook_active_symbols:
                continue  # Already have an active trade on this coin

            # Execute the trade
            logger.info(
                "🔥 DEPLOYING: %s %s @ %dx | Regime: %s (%.0f%%) | Qty: %.6f",
                trade["side"], sym, trade["leverage"],
                trade["regime_name"], trade["confidence"] * 100, trade["quantity"],
            )
            result = self.executor.execute_trade(
                symbol=sym,
                side=trade["side"],
                leverage=trade["leverage"],
                quantity=trade["quantity"],
                atr=trade["atr"],
                regime=trade["regime"],
                confidence=trade["confidence"],
                reason=trade["reason"],
            )

            # Record in tradebook — use CoinDCX-confirmed values for live mode
            entry_price = result.get("entry_price", 0) if result else 0
            fill_qty    = result.get("quantity", trade["quantity"]) if result else trade["quantity"]
            fill_lev    = result.get("leverage", trade["leverage"]) if result else trade["leverage"]
            fill_capital = result.get("capital", 100.0) if result else 100.0
            fill_sl     = result.get("stop_loss", 0) if result else 0
            fill_tp     = result.get("take_profit", 0) if result else 0

            tradebook.open_trade(
                symbol=sym,
                side=trade["side"],
                leverage=fill_lev,
                quantity=fill_qty,
                entry_price=entry_price,
                atr=trade["atr"],
                regime=trade["regime_name"],
                confidence=trade["confidence"],
                reason=trade["reason"],
                capital=fill_capital,
            )

            self._active_positions[sym] = {
                "regime": trade["regime_name"],
                "confidence": trade["confidence"],
                "side": trade["side"],
                "entry_time": datetime.utcnow().isoformat(),
                "leverage": fill_lev,
                "entry_price": entry_price,
                "quantity": fill_qty,
                "exchange": result.get("exchange", "binance") if result else "binance",
                "position_id": result.get("position_id") if result else None,
            }
            self._trade_count += 1
            deployed += 1

            # Collect trade info for batch alert
            deployed_trades.append({
                "symbol": sym,
                "position": "LONG" if trade["side"] == "BUY" else "SHORT",
                "regime": trade["regime_name"],
                "confidence": trade["confidence"],
                "leverage": fill_lev,
                "entry_price": entry_price,
                "stop_loss": fill_sl,
                "take_profit": fill_tp,
            })

        # ── Batch Telegram notification for all deployed trades ──
        if deployed_trades:
            try:
                # Re-read full trade records from tradebook for SL/TP info
                active = tradebook.get_active_trades()
                deployed_syms = {t["symbol"] for t in deployed_trades}
                full_records = [t for t in active if t["symbol"] in deployed_syms]
                # Use full records if available (has SL/TP), else use collected data
                tg.notify_batch_entries(full_records if full_records else deployed_trades)
            except Exception:
                pass

        # ── 6. Save state for dashboard ──────────────────────────
        cycle_duration = time.time() - cycle_start
        self._last_cycle_duration = cycle_duration
        self._save_multi_state(symbols, eligible_trades, deployed)

        logger.info(
            "📊 Cycle #%d complete | Scanned: %d | Eligible: %d | Deployed: %d | Active: %d/%d",
            self._cycle_count, len(symbols), len(eligible_trades), deployed,
            len(tradebook.get_active_trades()), config.MAX_CONCURRENT_POSITIONS,
        )

    # ─── Per-Coin Analysis ───────────────────────────────────────────────────

    def _analyze_coin(self, symbol, balance):
        """
        Analyze a single coin. Returns a trade dict if eligible, else None.
        Uses multi-timeframe analysis: 1h (primary) + 4h (macro confirmation).
        """
        # Fetch 1h data
        df_1h = fetch_klines(symbol, config.TIMEFRAME_CONFIRMATION, limit=config.HMM_LOOKBACK)
        if df_1h is None or len(df_1h) < 60:
            return None

        # Get or create brain for this coin (1h)
        brain = self._coin_brains.get(symbol)
        if brain is None:
            brain = HMMBrain()
            self._coin_brains[symbol] = brain

        # Compute features
        df_1h_feat = compute_all_features(df_1h)
        df_1h_hmm = compute_hmm_features(df_1h)

        # Train if needed
        if brain.needs_retrain():
            brain.train(df_1h_hmm)

        if not brain.is_trained:
            return None

        # Predict regime (1h)
        regime, conf = brain.predict(df_1h_feat)
        regime_name = brain.get_regime_name(regime)

        # ── 4h Macro Regime Confirmation ──
        macro_key = f"{symbol}_4h"
        macro_brain = self._coin_brains.get(macro_key)
        if macro_brain is None:
            macro_brain = HMMBrain()
            self._coin_brains[macro_key] = macro_brain

        macro_regime_name = None
        try:
            df_4h = fetch_klines(symbol, config.TIMEFRAME_MACRO, limit=config.HMM_LOOKBACK)
            if df_4h is not None and len(df_4h) >= 60:
                df_4h_feat = compute_all_features(df_4h)
                df_4h_hmm = compute_hmm_features(df_4h)
                if macro_brain.needs_retrain():
                    macro_brain.train(df_4h_hmm)
                if macro_brain.is_trained:
                    macro_regime, macro_conf = macro_brain.predict(df_4h_feat)
                    macro_regime_name = macro_brain.get_regime_name(macro_regime)
        except Exception as e:
            logger.debug("4h macro analysis failed for %s: %s", symbol, e)

        # Update coin state for dashboard
        current_price = float(df_1h_feat["close"].iloc[-1])

        # Extract latest HMM feature values for the feature heatmap
        _features = {}
        try:
            last = df_1h_feat.iloc[-1]
            _features = {
                "log_return":    round(float(last.get("log_return", 0)), 6),
                "volatility":    round(float(last.get("volatility", 0)), 6),
                "volume_change": round(float(last.get("volume_change", 0)), 6),
                "rsi_norm":      round(float(last.get("rsi_norm", 0)), 6),
            }
        except Exception:
            pass

        self._coin_states[symbol] = {
            "symbol": symbol,
            "regime": regime_name,
            "confidence": round(conf, 4),
            "price": current_price,
            "action": "ANALYZING",
            "macro_regime": macro_regime_name,
            "features": _features,
        }

        # ── CRASH on either timeframe → skip ──
        if regime == config.REGIME_CRASH:
            self._coin_states[symbol]["action"] = "CRASH_SKIP"
            return None
        if macro_regime_name == "CRASH":
            self._coin_states[symbol]["action"] = "MACRO_CRASH_SKIP"
            return None

        # ── Multi-TF conflict filter (1h vs 4h must agree on direction) ──
        if macro_regime_name:
            # BULL on 1h but BEAR on 4h → skip (and vice versa)
            if regime_name == "BULLISH" and macro_regime_name == "BEARISH":
                self._coin_states[symbol]["action"] = "MTF_CONFLICT"
                return None
            if regime_name == "BEARISH" and macro_regime_name == "BULLISH":
                self._coin_states[symbol]["action"] = "MTF_CONFLICT"
                return None

        # ── CHOP → sideways strategy ──
        if regime == config.REGIME_CHOP:
            signal = evaluate_mean_reversion(df_1h_feat, symbol)
            if signal:
                current_atr = df_1h_feat["atr"].iloc[-1] if "atr" in df_1h_feat.columns else 0
                coin_budget = balance * config.CAPITAL_PER_COIN_PCT
                quantity = self.risk.calculate_position_size(
                    coin_budget, current_price, current_atr, signal["leverage"],
                )
                quantity *= (1 - config.SIDEWAYS_POSITION_REDUCTION)
                quantity = round(quantity, 6)
                self._coin_states[symbol]["action"] = f"MEAN_REV_{signal['side']}"
                return {
                    "symbol": symbol,
                    "side": signal["side"],
                    "leverage": signal["leverage"],
                    "quantity": quantity,
                    "atr": current_atr,
                    "regime": regime,
                    "regime_name": regime_name,
                    "confidence": conf,
                    "reason": f"MeanRev {regime_name} | {signal['reason']}",
                }
            self._coin_states[symbol]["action"] = "CHOP_NO_SIGNAL"
            return None

        # ── TREND (BULL / BEAR) — 8-factor conviction flow ──────────────────────

        # 1. Determine side first (needed for sentiment gate + conviction)
        if regime == config.REGIME_BULL:
            side = "BUY"
        elif regime == config.REGIME_BEAR:
            side = "SELL"
        else:
            return None

        current_atr   = df_1h_feat["atr"].iloc[-1]   if "atr"   in df_1h_feat.columns else 0.0
        current_price = float(df_1h_feat["close"].iloc[-1])

        # 2. Volatility filter
        if config.VOL_FILTER_ENABLED and current_atr > 0:
            vol_ratio = current_atr / current_price
            if vol_ratio < config.VOL_MIN_ATR_PCT:
                self._coin_states[symbol]["action"] = "VOL_TOO_LOW"
                return None
            if vol_ratio > config.VOL_MAX_ATR_PCT:
                self._coin_states[symbol]["action"] = "VOL_TOO_HIGH"
                return None

        # 3. Sentiment (fast veto before conviction compute)
        sentiment_score = None
        coin_sym = symbol.replace("USDT", "").replace("BUSD", "")
        if self._sentiment:
            try:
                s_sig = self._sentiment.get_signal(coin_sym)
                if s_sig is not None:
                    if s_sig.alert:
                        self._coin_states[symbol]["action"] = f"SENTIMENT_ALERT:{s_sig.alert_reason}"
                        return None
                    sentiment_score = s_sig.effective_score
                    if sentiment_score <= config.SENTIMENT_VETO_THRESHOLD:
                        self._coin_states[symbol]["action"] = "SENTIMENT_VETO"
                        return None
            except Exception as _se:
                logger.debug("Sentiment fetch failed for %s: %s", symbol, _se)

        # 4. 15m momentum filter + order flow (fetch df_15m once for both)
        df_15m = None
        orderflow_score = None
        try:
            df_15m = fetch_klines(symbol, config.TIMEFRAME_EXECUTION, limit=50)
            if df_15m is not None and len(df_15m) >= 5:
                df_15m_feat = compute_all_features(df_15m)
                price_now   = float(df_15m_feat["close"].iloc[-1])
                price_5_ago = float(df_15m_feat["close"].iloc[-5])
                if side == "BUY"  and price_now <= price_5_ago:
                    self._coin_states[symbol]["action"] = "15M_FILTER_SKIP"
                    return None
                if side == "SELL" and price_now >= price_5_ago:
                    self._coin_states[symbol]["action"] = "15M_FILTER_SKIP"
                    return None
        except Exception:
            pass

        if self._orderflow:
            try:
                of_sig = self._orderflow.get_signal(symbol, df_15m)
                if of_sig is not None:
                    orderflow_score = of_sig.score
                    if of_sig.bid_walls or of_sig.ask_walls:
                        logger.info("🧱 %s order walls: %s", symbol, of_sig.note)
            except Exception as _oe:
                logger.debug("OrderFlow fetch failed for %s: %s", symbol, _oe)

        # 5. Full 8-factor conviction score
        _regime_name_to_int = {v: k for k, v in config.REGIME_NAMES.items()}
        btc_proxy   = _regime_name_to_int.get(macro_regime_name) if macro_regime_name else None
        funding     = df_1h_feat["funding_rate"].iloc[-1] if "funding_rate" in df_1h_feat.columns else None
        sr_pos      = df_1h_feat["sr_position"].iloc[-1]  if "sr_position"  in df_1h_feat.columns else None
        vwap_pos    = df_1h_feat["vwap_position"].iloc[-1] if "vwap_position" in df_1h_feat.columns else None
        oi_chg      = df_1h_feat["oi_change"].iloc[-1]    if "oi_change"    in df_1h_feat.columns else None
        volatility  = (current_atr / current_price)       if current_atr > 0 else None

        conviction = self.risk.compute_conviction_score(
            confidence=conf,
            regime=regime,
            side=side,
            btc_regime=btc_proxy,
            funding_rate=funding,
            sr_position=sr_pos,
            vwap_position=vwap_pos,
            oi_change=oi_chg,
            volatility=volatility,
            sentiment_score=sentiment_score,
            orderflow_score=orderflow_score,
        )
        leverage = self.risk.get_conviction_leverage(conviction)
        if leverage == 0:
            self._coin_states[symbol]["action"] = f"LOW_CONVICTION:{conviction:.1f}"
            return None

        # 6. Position sizing (per-coin budget)
        coin_budget = balance * config.CAPITAL_PER_COIN_PCT
        quantity    = self.risk.calculate_position_size(coin_budget, current_price, current_atr, leverage)
        quantity    = round(quantity, 6)

        of_note = f" | OF={orderflow_score:+.2f}" if orderflow_score is not None else ""
        sn_note = f" | sent={sentiment_score:+.2f}" if sentiment_score is not None else ""
        self._coin_states[symbol]["action"] = f"ELIGIBLE_{side}"
        self._coin_states[symbol].update({
            "conviction": round(conviction, 1),
            "orderflow":  round(orderflow_score, 3) if orderflow_score is not None else None,
            "sentiment":  round(sentiment_score, 3) if sentiment_score is not None else None,
        })
        return {
            "symbol": symbol,
            "side": side,
            "leverage": leverage,
            "quantity": quantity,
            "atr": current_atr,
            "regime": regime,
            "regime_name": regime_name,
            "confidence": conf,
            "conviction": conviction,
            "reason": f"Trend {regime_name} | conf={conf:.0%} | conv={conviction:.1f} | lev={leverage}x{sn_note}{of_note}",
        }

    # ─── Exit & Sync Logic ────────────────────────────────────────────────────

    def _check_exits(self, current_symbols):
        """
        DISABLED — Regime changes no longer trigger exits.
        
        Backtest confirmed: regime-change exits HURT returns because
        the HMM anticipates moves, and exit fees eat into profits.
        
        Trades now exit ONLY via:
          • ATR-based Stop Loss
          • ATR-based Take Profit
          • Trailing SL / Trailing TP
          • Max-loss guard (server.js)
        """
        # Sync _active_positions dict (remove entries closed by SL engine)
        active_syms = {t["symbol"] for t in tradebook.get_active_trades()}
        for sym in list(self._active_positions.keys()):
            if sym not in active_syms:
                del self._active_positions[sym]

    def _load_positions_from_tradebook(self):
        """Load active tradebook entries into _active_positions on startup."""
        try:
            active_trades = tradebook.get_active_trades()
            for t in active_trades:
                sym = t["symbol"]
                if sym not in self._active_positions:
                    self._active_positions[sym] = {
                        "regime": t.get("regime", "UNKNOWN"),
                        "confidence": t.get("confidence", 0),
                        "side": t.get("side", "BUY"),
                        "leverage": t.get("leverage", 1),
                        "entry_time": t.get("entry_timestamp", ""),
                    }
            if active_trades:
                logger.info(
                    "📂 Loaded %d active positions from tradebook: %s",
                    len(self._active_positions),
                    ", ".join(self._active_positions.keys()),
                )
        except Exception as e:
            logger.warning("Could not load tradebook positions on startup: %s", e)

    def _sync_positions(self):
        """
        Remove entries from _active_positions that were auto-closed
        by the tradebook (e.g., SL/TP hit during paper-mode simulation).
        """
        active_symbols = {t["symbol"] for t in tradebook.get_active_trades()}
        closed_out = [sym for sym in self._active_positions if sym not in active_symbols]
        for sym in closed_out:
            logger.info("📗 Position %s auto-closed by tradebook (SL/TP hit). Removing.", sym)
            del self._active_positions[sym]

    def _sync_coindcx_positions(self):
        """
        Sync CoinDCX positions → tradebook + dashboard (source of truth).

        Every heartbeat (1 min) this:
          1. Fetches all CoinDCX positions
          2. Auto-registers positions not in tradebook (manual opens)
          3. Detects exchange-side closures → close in tradebook
          4. Updates mark prices for P&L calculation
        """
        import coindcx_client as cdx

        try:
            cdx_positions = cdx.list_positions()
        except Exception as e:
            logger.debug("Failed to fetch CoinDCX positions: %s", e)
            return

        # Build map of active CoinDCX positions: symbol → position data
        cdx_active = {}
        for p in cdx_positions:
            active_pos = float(p.get("active_pos", 0))
            if active_pos == 0:
                continue
            pair = p.get("pair", "")
            try:
                symbol = cdx.from_coindcx_pair(pair)
            except Exception:
                continue
            cdx_active[symbol] = {
                "pair":          pair,
                "position_id":   p.get("id"),
                "active_pos":    active_pos,
                "avg_price":     float(p.get("avg_price", 0)),
                "mark_price":    float(p.get("mark_price", 0)),
                "leverage":      int(float(p.get("leverage", 1))),
                "locked_margin": float(p.get("locked_margin", 0)),
                "sl_trigger":    p.get("stop_loss_trigger"),
                "tp_trigger":    p.get("take_profit_trigger"),
                "side":          "BUY" if active_pos > 0 else "SELL",
            }

        # Get current tradebook active symbols
        tb_active = tradebook.get_active_trades()
        tb_symbols = {t["symbol"] for t in tb_active}

        # ── 1. Detect exchange-side closures ────────────────────────
        # If tradebook has an ACTIVE LIVE trade but CoinDCX doesn't → closed on exchange
        for trade in tb_active:
            sym = trade["symbol"]
            if trade.get("mode") != "LIVE":
                continue
            if sym not in cdx_active:
                logger.info(
                    "📕 %s closed on CoinDCX (SL/TP or manual). Closing in tradebook.", sym
                )
                tradebook.close_trade(symbol=sym, reason="EXCHANGE_CLOSED")
                if sym in self._active_positions:
                    del self._active_positions[sym]

        # ── 2. Auto-register external positions ─────────────────────
        # If CoinDCX has active position but tradebook doesn't → register it
        for sym, pos in cdx_active.items():
            if sym in tb_symbols:
                continue

            logger.info(
                "📘 Discovered untracked CoinDCX position: %s %s %dx @ $%.6f — registering.",
                pos["side"], sym, pos["leverage"], pos["avg_price"],
            )

            # Compute ATR (best-effort) for trailing
            try:
                from data_pipeline import fetch_klines
                from feature_engine import compute_all_features
                df = fetch_klines(sym, "1h", limit=200)
                df_feat = compute_all_features(df)
                atr = float(df_feat["atr"].iloc[-1])
            except Exception:
                atr = pos["avg_price"] * 0.015  # fallback 1.5%

            capital = pos["locked_margin"] if pos["locked_margin"] > 0 else 100.0

            trade_id = tradebook.open_trade(
                symbol=sym,
                side=pos["side"],
                leverage=pos["leverage"],
                quantity=abs(pos["active_pos"]),
                entry_price=pos["avg_price"],
                atr=atr,
                regime="BEARISH" if pos["side"] == "SELL" else "BULLISH",
                confidence=0.99,
                reason="Auto-synced from CoinDCX",
                capital=capital,
                mode="LIVE",
            )

            self._active_positions[sym] = {
                "regime": "BEARISH" if pos["side"] == "SELL" else "BULLISH",
                "confidence": 0.99,
                "side": pos["side"],
                "entry_time": datetime.utcnow().isoformat(),
                "leverage": pos["leverage"],
                "entry_price": pos["avg_price"],
                "quantity": abs(pos["active_pos"]),
                "exchange": "coindcx",
                "position_id": pos["position_id"],
            }
            logger.info("  → Registered as %s", trade_id)

        # ── 3. Push CoinDCX mark prices to tradebook ────────────────
        # This ensures unrealized P&L uses the exchange price, not Binance
        if cdx_active:
            cdx_prices = {sym: pos["mark_price"] for sym, pos in cdx_active.items()}
            tradebook.update_unrealized(prices=cdx_prices)

        # ── 4. Save multi_bot_state for dashboard ──────────────────
        # Keeps the dashboard positions card in sync with CoinDCX
        try:
            active_trades = tradebook.get_active_trades()
            positions_dict = {}
            coin_states_dict = {}
            for t in active_trades:
                sym = t["symbol"]
                positions_dict[sym] = {
                    "side": t.get("side", "SELL"),
                    "leverage": t.get("leverage", 1),
                    "entry_price": t.get("entry_price", 0),
                    "quantity": t.get("quantity", 0),
                    "atr": t.get("atr_at_entry", 0),
                    "status": "active",
                    "trade_id": t.get("trade_id"),
                    "exchange": "coindcx",
                    "unrealized_pnl": t.get("unrealized_pnl", 0),
                    "unrealized_pnl_pct": t.get("unrealized_pnl_pct", 0),
                    "current_price": t.get("current_price", 0),
                }
                coin_states_dict[sym] = {
                    "regime": t.get("regime", "UNKNOWN"),
                    "confidence": t.get("confidence", 0),
                    "action": f'{"LONG" if t.get("position") == "LONG" else "SHORT"} ACTIVE',
                    "side": t.get("side", "SELL"),
                    "leverage": t.get("leverage", 1),
                }
            multi_state = {
                "timestamp": datetime.utcnow().isoformat(),
                "cycle": getattr(self, "_cycle_count", 0),
                "coins_scanned": len(cdx_active),
                "eligible_count": len(cdx_active),
                "deployed_count": len(positions_dict),
                "total_trades": getattr(self, "_trade_count", len(positions_dict)),
                "active_positions": positions_dict,
                "positions": positions_dict,
                "max_concurrent_positions": config.MAX_CONCURRENT_POSITIONS,
                "coin_states": coin_states_dict,
                "paper_mode": config.PAPER_TRADE,
                "cycle_execution_time_seconds": 0,
            }
            with open(config.MULTI_STATE_FILE, "w") as f:
                json.dump(multi_state, f, indent=2)
        except Exception as e:
            logger.debug("Failed to save multi_bot_state during sync: %s", e)

    # ─── State Persistence ───────────────────────────────────────────────────

    def _save_multi_state(self, symbols_scanned, eligible, deployed_count):
        """Save multi-coin bot state for the dashboard."""
        # Also save legacy single-coin state (backward compat)
        top_coin = self._coin_states.get(config.PRIMARY_SYMBOL, {})
        legacy_state = {
            "timestamp":    datetime.utcnow().isoformat(),
            "symbol":       config.PRIMARY_SYMBOL,
            "regime":       top_coin.get("regime", "SCANNING"),
            "confidence":   top_coin.get("confidence", 0),
            "action":       top_coin.get("action", "MULTI_SCAN"),
            "trade_count":  self._trade_count,
            "paper_mode":   config.PAPER_TRADE,
        }
        try:
            with open(config.STATE_FILE, "w") as f:
                json.dump(legacy_state, f, indent=2)
        except Exception:
            pass

        # Multi-coin state
        multi_state = {
            "timestamp":        datetime.utcnow().isoformat(),
            "cycle":            self._cycle_count,
            "coins_scanned":    len(symbols_scanned),
            "eligible_count":   len(eligible),
            "deployed_count":   deployed_count,
            "total_trades":     self._trade_count,
            "active_positions": self._active_positions,
            "max_concurrent_positions": config.MAX_CONCURRENT_POSITIONS,
            "coin_states":      self._coin_states,
            "paper_mode":       config.PAPER_TRADE,
            "cycle_execution_time_seconds": getattr(self, '_last_cycle_duration', 0),
        }
        try:
            with open(config.MULTI_STATE_FILE, "w") as f:
                json.dump(multi_state, f, indent=2)
        except Exception as e:
            logger.error("Failed to save multi state: %s", e)

    def _process_commands(self):
        """Check for external commands (from dashboard kill switch)."""
        import os
        try:
            if not os.path.exists(config.COMMANDS_FILE):
                return
            with open(config.COMMANDS_FILE, "r") as f:
                cmd = json.load(f)

            if cmd.get("command") == "KILL":
                logger.warning("🚨 External KILL command received!")
                self.risk._killed = True
                for sym in list(self._active_positions.keys()):
                    tradebook.close_trade(symbol=sym, reason="EXTERNAL_KILL")
                    self.executor.close_all_positions(sym)
                self._active_positions.clear()
                os.remove(config.COMMANDS_FILE)

            elif cmd.get("command") == "RESET":
                logger.info("🔄 External RESET command received.")
                self.risk.reset_kill_switch()
                os.remove(config.COMMANDS_FILE)

        except (json.JSONDecodeError, KeyError):
            pass
        except Exception as e:
            logger.error("Error processing commands: %s", e)


# ─── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = RegimeMasterBot()
    bot.run()
