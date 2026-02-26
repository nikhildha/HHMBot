"""
Project Regime-Master — Realistic Backtester v2
Walk-forward testing with proper SL/TP, trailing stops, position sizing, and fees.

Fixes all critical issues from hmm_diagnostic.py:
  ✅ Walk-forward: trains on window N, tests on window N+1 (no look-ahead)
  ✅ Per-trade SL/TP using ATR multipliers (matches live trading)
  ✅ Trailing SL: locks in profit as price moves in favor
  ✅ Trailing TP: extends target when momentum continues
  ✅ Position sizing: risk_pct of capital per trade
  ✅ Fees: taker + slippage on entry AND exit
  ✅ Per-trade log: entry, exit, SL, TP, PnL for every trade
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import config
from hmm_brain import HMMBrain, HMM_FEATURES
from feature_engine import compute_all_features, compute_hmm_features
from risk_manager import RiskManager
try:
    from data_pipeline import enrich_with_oi_funding
except ImportError:
    enrich_with_oi_funding = None

logger = logging.getLogger("BacktesterV2")

# ═══════════════════════════════════════════════════════════════════════════════
#  TRADE CLASS — Tracks a single position through its lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class Trade:
    """Represents a single trade with SL/TP/trailing logic."""
    
    def __init__(self, entry_bar, entry_price, side, leverage, atr, capital_allocated,
                 regime, confidence):
        self.entry_bar = entry_bar
        self.entry_price = entry_price
        self.side = side            # 'LONG' or 'SHORT'
        self.leverage = leverage
        self.atr = atr
        self.capital = capital_allocated
        self.regime = regime
        self.confidence = confidence
        
        # Position size (in units of the asset)
        self.quantity = (capital_allocated * leverage) / entry_price
        
        # ATR-based SL/TP
        sl_mult, tp_mult = config.get_atr_multipliers(leverage)
        sl_dist = atr * sl_mult
        tp_dist = atr * tp_mult
        
        if side == 'LONG':
            self.stop_loss = entry_price - sl_dist
            self.take_profit = entry_price + tp_dist
        else:
            self.stop_loss = entry_price + sl_dist
            self.take_profit = entry_price - tp_dist
        
        # Trailing state
        self.peak_price = entry_price    # Best price in our direction
        self.trail_active = False
        self.breakeven_locked = False
        self.tp_extensions = 0
        self.original_tp = self.take_profit
        
        # Result
        self.exit_bar = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl_pct = None
        self.pnl_value = None
    
    def update(self, bar_idx, high, low, close):
        """
        Check if SL/TP/trailing triggers on this bar.
        Returns True if trade is closed, False if still open.
        """
        if self.exit_bar is not None:
            return True  # Already closed
        
        # ─── Check Stop Loss ─────────────────────────────────────────────
        if self.side == 'LONG':
            if low <= self.stop_loss:
                self._close(bar_idx, self.stop_loss, 'SL_HIT')
                return True
        else:
            if high >= self.stop_loss:
                self._close(bar_idx, self.stop_loss, 'SL_HIT')
                return True
        
        # ─── Check Take Profit ────────────────────────────────────────────
        if self.side == 'LONG':
            if high >= self.take_profit:
                self._close(bar_idx, self.take_profit, 'TP_HIT')
                return True
        else:
            if low <= self.take_profit:
                self._close(bar_idx, self.take_profit, 'TP_HIT')
                return True
        
        # ─── BREAKEVEN STOP — Move SL to entry+fees when profit ≥ 10% ────
        total_fees = (config.TAKER_FEE + config.SLIPPAGE_BUFFER) * 2
        be_buffer = self.entry_price * (total_fees / self.leverage + 0.001)  # tiny extra margin
        
        if self.side == 'LONG':
            unrealized_pct = (close - self.entry_price) / self.entry_price * self.leverage
            if unrealized_pct >= 0.10:  # +10% leveraged profit
                be_price = self.entry_price + be_buffer
                if self.stop_loss < be_price:
                    self.stop_loss = be_price
                    self.breakeven_locked = True
        else:
            unrealized_pct = (self.entry_price - close) / self.entry_price * self.leverage
            if unrealized_pct >= 0.10:  # +10% leveraged profit
                be_price = self.entry_price - be_buffer
                if self.stop_loss > be_price:
                    self.stop_loss = be_price
                    self.breakeven_locked = True
        
        # ─── Trailing Stop Loss ───────────────────────────────────────────
        if config.TRAILING_SL_ENABLED:
            trail_activation_dist = self.atr * config.TRAILING_SL_ACTIVATION_ATR
            trail_distance = self.atr * config.TRAILING_SL_DISTANCE_ATR
            
            if self.side == 'LONG':
                self.peak_price = max(self.peak_price, high)
                if self.peak_price - self.entry_price >= trail_activation_dist:
                    self.trail_active = True
                    new_sl = self.peak_price - trail_distance
                    self.stop_loss = max(self.stop_loss, new_sl)
            else:
                self.peak_price = min(self.peak_price, low)
                if self.entry_price - self.peak_price >= trail_activation_dist:
                    self.trail_active = True
                    new_sl = self.peak_price + trail_distance
                    self.stop_loss = min(self.stop_loss, new_sl)
        
        # ─── Trailing Take Profit ─────────────────────────────────────────
        if config.TRAILING_TP_ENABLED and self.tp_extensions < config.TRAILING_TP_MAX_EXTENSIONS:
            tp_dist = abs(self.original_tp - self.entry_price)
            activation_threshold = tp_dist * config.TRAILING_TP_ACTIVATION_PCT
            
            if self.side == 'LONG':
                if high - self.entry_price >= activation_threshold:
                    extension = self.atr * config.TRAILING_TP_EXTENSION_ATR
                    self.take_profit += extension
                    self.tp_extensions += 1
            else:
                if self.entry_price - low >= activation_threshold:
                    extension = self.atr * config.TRAILING_TP_EXTENSION_ATR
                    self.take_profit -= extension
                    self.tp_extensions += 1
        
        return False
    
    def force_close(self, bar_idx, price, reason='FORCED'):
        """Force close at a given price."""
        self._close(bar_idx, price, reason)
    
    def _close(self, bar_idx, exit_price, reason):
        """Calculate PnL and close the trade."""
        self.exit_bar = bar_idx
        self.exit_price = exit_price
        self.exit_reason = reason
        
        # PnL calculation
        if self.side == 'LONG':
            raw_pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            raw_pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        # Apply leverage
        leveraged_pnl_pct = raw_pnl_pct * self.leverage
        
        # Apply fees (entry + exit)
        total_fees = (config.TAKER_FEE + config.SLIPPAGE_BUFFER) * 2
        leveraged_pnl_pct -= total_fees
        
        self.pnl_pct = leveraged_pnl_pct
        self.pnl_value = self.capital * leveraged_pnl_pct
    
    def to_dict(self):
        return {
            'entry_bar': self.entry_bar,
            'exit_bar': self.exit_bar,
            'side': self.side,
            'regime': self.regime,
            'confidence': round(self.confidence, 4),
            'leverage': self.leverage,
            'entry_price': round(self.entry_price, 2),
            'exit_price': round(self.exit_price, 2) if self.exit_price else None,
            'stop_loss': round(self.stop_loss, 2),
            'take_profit': round(self.take_profit, 2),
            'trail_active': self.trail_active,
            'breakeven_locked': self.breakeven_locked,
            'tp_extensions': self.tp_extensions,
            'exit_reason': self.exit_reason,
            'pnl_pct': round(self.pnl_pct * 100, 3) if self.pnl_pct else None,
            'pnl_value': round(self.pnl_value, 2) if self.pnl_value else None,
            'capital': round(self.capital, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  REALISTIC BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_realistic(df, brain, initial_capital=10000, risk_per_trade=None,
                       max_positions=1, verbose=False):
    """
    Realistic per-trade backtest with SL/TP, trailing stops, fees, and position sizing.
    
    Parameters
    ----------
    df : pd.DataFrame with OHLCV + features
    brain : trained HMMBrain
    initial_capital : float
    risk_per_trade : float (fraction, e.g. 0.04 for 4%)
    max_positions : int (max concurrent trades)
    verbose : bool
    
    Returns
    -------
    dict with trades, equity curve, and metrics
    """
    risk_per_trade = risk_per_trade or config.RISK_PER_TRADE
    
    df = df.copy()
    if "log_return" not in df.columns:
        df = compute_all_features(df)
    
    df_hmm = compute_hmm_features(df)
    
    # Get per-bar predictions
    states = brain.predict_all(df_hmm)
    feats = df_hmm[HMM_FEATURES].dropna().values
    feats_scaled = (feats - brain._feat_mean) / brain._feat_std
    confidences = np.max(brain.model.predict_proba(feats_scaled), axis=1)
    
    # Pad to match df length
    n = len(df)
    if len(states) < n:
        pad = n - len(states)
        states = np.concatenate([np.full(pad, config.REGIME_CHOP), states])
        confidences = np.concatenate([np.full(pad, 0.5), confidences])
    
    # ─── Simulation loop ─────────────────────────────────────────────────
    capital = initial_capital
    equity_curve = [capital]
    open_trades = []
    closed_trades = []
    prev_state = None
    
    for i in range(1, n):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        close = df['close'].iloc[i]
        state = int(states[i])
        conf = confidences[i]
        
        # ─── Update open trades (check SL/TP) ────────────────────────────
        still_open = []
        for trade in open_trades:
            closed = trade.update(i, high, low, close)
            if closed:
                capital += trade.capital + trade.pnl_value
                closed_trades.append(trade)
                if verbose and trade.pnl_value is not None:
                    print(f"  Bar {i}: CLOSE {trade.side} @ {trade.exit_price:.2f} | "
                          f"{trade.exit_reason} | PnL: {trade.pnl_pct*100:+.2f}% (${trade.pnl_value:+.2f}) | "
                          f"Capital: ${capital:.2f}")
            else:
                still_open.append(trade)
        open_trades = still_open
        
        # ─── Check for new entry ─────────────────────────────────────────
        if len(open_trades) < max_positions and state != prev_state:
            # Fix #1: Skip CHOP regime entirely
            if config.SKIP_CHOP_TRADES and state == config.REGIME_CHOP:
                prev_state = state
                continue
            
            leverage = RiskManager.get_dynamic_leverage(conf, state)
            
            if leverage > 0 and state != config.REGIME_CRASH:
                # Determine side
                if state == config.REGIME_BULL:
                    side = 'LONG'
                elif state == config.REGIME_BEAR:
                    side = 'SHORT'
                else:
                    side = None
                
                # Fix #4: Counter-trend confidence penalty
                # LONGs in bear → penalize, SHORTs in bull → penalize
                if side:
                    is_counter_trend = False
                    if side == 'LONG' and state == config.REGIME_BEAR:
                        is_counter_trend = True
                    elif side == 'SHORT' and state == config.REGIME_BULL:
                        is_counter_trend = True
                    
                    if is_counter_trend:
                        conf_penalty = getattr(config, 'COUNTER_TREND_CONF_PENALTY', 0.04)
                        conf -= conf_penalty
                        leverage = RiskManager.get_dynamic_leverage(conf, state)
                        if leverage <= 0:
                            prev_state = state
                            continue
                
                if side:
                    # ATR for stops
                    atr = df['atr'].iloc[i] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else close * 0.01
                    
                    # ─── S/R + VWAP Confidence Bias ──────────────────────
                    # Boost confidence when entry aligns with S/R + VWAP
                    # Penalize confidence when entry is at unfavorable levels
                    adj_conf = conf
                    if 'support' in df.columns and 'resistance' in df.columns and 'vwap' in df.columns:
                        sup = df['support'].iloc[i] if not pd.isna(df['support'].iloc[i]) else close
                        res = df['resistance'].iloc[i] if not pd.isna(df['resistance'].iloc[i]) else close
                        vwap_val = df['vwap'].iloc[i] if not pd.isna(df['vwap'].iloc[i]) else close
                        sr_range = res - sup if res > sup else atr
                        
                        if side == 'LONG':
                            # Boost if near support, penalize if near resistance
                            sr_dist = (close - sup) / sr_range  # 0=support, 1=resistance
                            sr_bias = 0.03 * (1 - sr_dist * 2)  # +3% at support, -3% at resistance
                            # Boost if below VWAP (buying cheap), penalize if above
                            vwap_bias = 0.02 if close <= vwap_val else -0.02
                        else:  # SHORT
                            sr_dist = (res - close) / sr_range  # 0=resistance, 1=support
                            sr_bias = 0.03 * (1 - sr_dist * 2)  # +3% at resistance, -3% at support
                            vwap_bias = 0.02 if close >= vwap_val else -0.02
                        
                        adj_conf = min(1.0, max(0.0, conf + sr_bias + vwap_bias))
                    
                    # Recalculate leverage with adjusted confidence
                    leverage = RiskManager.get_dynamic_leverage(adj_conf, state)
                    if leverage <= 0:
                        prev_state = state
                        continue
                    
                    # Capital allocation
                    trade_capital = min(capital * risk_per_trade * 5, capital * 0.2)
                    if trade_capital < 10:
                        continue
                    
                    trade = Trade(
                        entry_bar=i,
                        entry_price=close,
                        side=side,
                        leverage=leverage,
                        atr=atr,
                        capital_allocated=trade_capital,
                        regime=config.REGIME_NAMES.get(state, "?"),
                        confidence=conf,
                    )
                    
                    capital -= trade_capital
                    open_trades.append(trade)
                    
                    if verbose:
                        print(f"  Bar {i}: OPEN {side} @ {close:.2f} | {trade.regime} "
                              f"(conf={conf:.3f}) | Lev={leverage}x | "
                              f"SL={trade.stop_loss:.2f} TP={trade.take_profit:.2f} | "
                              f"Capital: ${trade_capital:.2f}")
        
        prev_state = state
        
        # ─── Track equity (capital + open position value) ─────────────────
        open_value = 0
        for trade in open_trades:
            if trade.side == 'LONG':
                unrealized = (close - trade.entry_price) / trade.entry_price * trade.leverage
            else:
                unrealized = (trade.entry_price - close) / trade.entry_price * trade.leverage
            open_value += trade.capital * (1 + unrealized)
        equity_curve.append(capital + open_value)
    
    # ─── Close any remaining open trades at last close ────────────────────
    last_close = df['close'].iloc[-1]
    for trade in open_trades:
        trade.force_close(n - 1, last_close, 'END_OF_DATA')
        capital += trade.capital + trade.pnl_value
        closed_trades.append(trade)
    
    # ═══════════════════════════════════════════════════════════════════════
    #  METRICS
    # ═══════════════════════════════════════════════════════════════════════
    
    equity = np.array(equity_curve)
    final_capital = equity[-1]
    total_return = (final_capital / initial_capital - 1) * 100
    
    # Max drawdown
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd = drawdowns.min() * 100
    
    # Win rate
    wins = [t for t in closed_trades if t.pnl_pct and t.pnl_pct > 0]
    losses = [t for t in closed_trades if t.pnl_pct and t.pnl_pct <= 0]
    win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
    
    # Profit factor
    gross_profit = sum(t.pnl_value for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_value for t in losses)) if losses else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Sharpe (daily)
    equity_returns = np.diff(equity) / equity[:-1]
    sharpe = (equity_returns.mean() / equity_returns.std() * np.sqrt(365 * 24)) if equity_returns.std() > 0 else 0
    
    # Average trade metrics
    avg_pnl = np.mean([t.pnl_pct * 100 for t in closed_trades]) if closed_trades else 0
    avg_win = np.mean([t.pnl_pct * 100 for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct * 100 for t in losses]) if losses else 0
    avg_bars = np.mean([t.exit_bar - t.entry_bar for t in closed_trades]) if closed_trades else 0
    
    # SL/TP/Trail statistics
    sl_hits = len([t for t in closed_trades if t.exit_reason == 'SL_HIT'])
    tp_hits = len([t for t in closed_trades if t.exit_reason == 'TP_HIT'])
    trail_count = len([t for t in closed_trades if t.trail_active])
    tp_ext_count = len([t for t in closed_trades if t.tp_extensions > 0])
    
    results = {
        'initial_capital': initial_capital,
        'final_capital': round(final_capital, 2),
        'total_return': round(total_return, 2),
        'max_drawdown': round(max_dd, 2),
        'sharpe_ratio': round(sharpe, 3),
        'profit_factor': round(profit_factor, 3),
        'total_trades': len(closed_trades),
        'win_rate': round(win_rate, 1),
        'avg_pnl_pct': round(avg_pnl, 3),
        'avg_win_pct': round(avg_win, 3),
        'avg_loss_pct': round(avg_loss, 3),
        'avg_hold_bars': round(avg_bars, 1),
        'sl_hits': sl_hits,
        'tp_hits': tp_hits,
        'trail_activated': trail_count,
        'tp_extended': tp_ext_count,
        'trades': closed_trades,
        'equity_curve': equity,
    }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_backtest(df_raw, train_window=500, test_window=100, step=100,
                          initial_capital=10000, verbose=False):
    """
    Walk-forward backtesting: train on N bars, test on next M bars, slide forward.
    Eliminates look-ahead bias entirely.
    
    Parameters
    ----------
    df_raw : pd.DataFrame with OHLCV data
    train_window : int, bars to train HMM on
    test_window : int, bars to test on (OOS)
    step : int, bars to slide forward each iteration
    
    Returns
    -------
    dict with aggregated results and per-fold details
    """
    # Enrich with OI + Funding Rate before computing features
    if enrich_with_oi_funding is not None and 'timestamp' in df_raw.columns:
        try:
            df_raw = enrich_with_oi_funding(df_raw)
        except Exception as e:
            import logging
            logging.getLogger('Backtester').warning('OI/funding enrichment failed: %s', e)
    
    df = compute_all_features(df_raw)
    n = len(df)
    
    all_trades = []
    equity_segments = []
    fold_results = []
    capital = initial_capital
    
    fold = 0
    start = 0
    
    while start + train_window + test_window <= n:
        fold += 1
        train_end = start + train_window
        test_end = min(train_end + test_window, n)
        
        df_train = df.iloc[start:train_end]
        df_test = df.iloc[train_end:test_end]
        
        # Train HMM on this window
        brain = HMMBrain(n_states=4)
        df_hmm_train = compute_hmm_features(df_train)
        brain.train(df_hmm_train)
        
        if not brain.is_trained:
            start += step
            continue
        
        # Test on OOS window
        results = backtest_realistic(
            df_test, brain, initial_capital=capital,
            max_positions=1, verbose=verbose,
        )
        
        capital = results['final_capital']
        all_trades.extend(results['trades'])
        equity_segments.append(results['equity_curve'])
        
        fold_results.append({
            'fold': fold,
            'train_range': f"{start}–{train_end}",
            'test_range': f"{train_end}–{test_end}",
            'trades': results['total_trades'],
            'return': results['total_return'],
            'max_dd': results['max_drawdown'],
            'win_rate': results['win_rate'],
        })
        
        start += step
    
    # ─── Aggregate results ────────────────────────────────────────────────
    final_capital = capital
    total_return = (final_capital / initial_capital - 1) * 100
    
    # Build full equity curve
    full_equity = [initial_capital]
    for seg in equity_segments:
        # Scale segment to continue from where last left off
        scale = full_equity[-1] / seg[0] if seg[0] != 0 else 1
        full_equity.extend([s * scale for s in seg[1:]])
    
    full_equity = np.array(full_equity)
    rolling_max = np.maximum.accumulate(full_equity)
    drawdowns = (full_equity - rolling_max) / rolling_max
    max_dd = drawdowns.min() * 100
    
    equity_returns = np.diff(full_equity) / full_equity[:-1]
    sharpe = (equity_returns.mean() / equity_returns.std() * np.sqrt(365 * 24)) if equity_returns.std() > 0 else 0
    
    wins = [t for t in all_trades if t.pnl_pct and t.pnl_pct > 0]
    losses = [t for t in all_trades if t.pnl_pct and t.pnl_pct <= 0]
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
    
    gross_profit = sum(t.pnl_value for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_value for t in losses)) if losses else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    sl_hits = len([t for t in all_trades if t.exit_reason == 'SL_HIT'])
    tp_hits = len([t for t in all_trades if t.exit_reason == 'TP_HIT'])
    trail_count = len([t for t in all_trades if t.trail_active])
    
    return {
        'initial_capital': initial_capital,
        'final_capital': round(final_capital, 2),
        'total_return': round(total_return, 2),
        'max_drawdown': round(max_dd, 2),
        'sharpe_ratio': round(sharpe, 3),
        'profit_factor': round(profit_factor, 3),
        'total_trades': len(all_trades),
        'win_rate': round(win_rate, 1),
        'sl_hits': sl_hits,
        'tp_hits': tp_hits,
        'trail_activated': trail_count,
        'folds': len(fold_results),
        'fold_details': fold_results,
        'trades': all_trades,
        'equity_curve': full_equity,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PRETTY PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

W = 100
DIV = "=" * W

def print_report(results, title="BACKTEST REPORT"):
    """Pretty-print backtest results."""
    print(f"\n{DIV}")
    print(f"  📊 {title}")
    print(DIV)
    
    print(f"  💰 Capital:       ${results['initial_capital']:,.2f} → ${results['final_capital']:,.2f}")
    print(f"  📈 Total Return:  {results['total_return']:+.2f}%")
    print(f"  📉 Max Drawdown:  {results['max_drawdown']:.2f}%")
    print(f"  📊 Sharpe Ratio:  {results['sharpe_ratio']:.3f}")
    print(f"  🎯 Profit Factor: {results['profit_factor']:.3f}")
    print(f"  🔢 Total Trades:  {results['total_trades']}")
    print(f"  ✅ Win Rate:      {results['win_rate']:.1f}%")
    
    if 'avg_pnl_pct' in results:
        print(f"\n  Avg PnL/trade:   {results['avg_pnl_pct']:+.3f}%")
        print(f"  Avg Win:         {results['avg_win_pct']:+.3f}%")
        print(f"  Avg Loss:        {results['avg_loss_pct']:+.3f}%")
        print(f"  Avg Hold:        {results['avg_hold_bars']:.1f} bars")
    
    print(f"\n  🛑 SL Hits:       {results['sl_hits']}")
    print(f"  🎯 TP Hits:       {results['tp_hits']}")
    print(f"  📐 Trail Active:  {results.get('trail_activated', 0)}")
    
    if 'fold_details' in results:
        print(f"\n  📋 Walk-Forward Folds ({results['folds']}):")
        print(f"  {'Fold':>6} {'Train':>12} {'Test':>12} {'Trades':>8} {'Return':>10} {'MaxDD':>10} {'WinRate':>10}")
        print(f"  {'-'*70}")
        for f in results['fold_details']:
            print(f"  {f['fold']:>6} {f['train_range']:>12} {f['test_range']:>12} "
                  f"{f['trades']:>8} {f['return']:>9.2f}% {f['max_dd']:>9.2f}% {f['win_rate']:>9.1f}%")
    
    print(DIV)


def print_trade_log(trades, max_trades=30):
    """Print detailed per-trade log."""
    print(f"\n  📋 TRADE LOG (showing {min(len(trades), max_trades)} of {len(trades)} trades)")
    print(f"  {'#':>4} {'Side':>6} {'Regime':>12} {'Lev':>5} {'Entry':>10} {'Exit':>10} "
          f"{'SL':>10} {'TP':>10} {'Reason':>8} {'PnL%':>8} {'PnL$':>10} {'Trail':>6}")
    print(f"  {'-'*105}")
    
    for i, t in enumerate(trades[:max_trades]):
        d = t.to_dict()
        trail = "✓" if d['trail_active'] else ""
        print(f"  {i+1:>4} {d['side']:>6} {d['regime'][:12]:>12} {d['leverage']:>4}x "
              f"{d['entry_price']:>10.2f} {d['exit_price']:>10.2f} "
              f"{d['stop_loss']:>10.2f} {d['take_profit']:>10.2f} "
              f"{d['exit_reason']:>8} {d['pnl_pct']:>+7.2f}% {d['pnl_value']:>+9.2f} {trail:>6}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Run full validation test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_pipeline import fetch_futures_klines
    
    print(f"\n{DIV}")
    print(f"  🔬 SENTINEL — Realistic Backtester v2 — Full Validation")
    print(f"{DIV}")
    
    # ─── Fetch real data ──────────────────────────────────────────────────
    print("\n  📡 Fetching BTCUSDT 1h (1000 candles ≈ 42 days)...")
    df_raw = fetch_futures_klines("BTCUSDT", "1h", limit=1000)
    if df_raw is None or len(df_raw) < 600:
        print("  ❌ Could not fetch enough data.")
        sys.exit(1)
    print(f"  ✅ Got {len(df_raw)} candles | {df_raw['close'].iloc[0]:.0f} → {df_raw['close'].iloc[-1]:.0f}")
    
    # ═══════════════════════════════════════════════════════════════════════
    #  TEST A: In-Sample (old method) — to show the contrast
    # ═══════════════════════════════════════════════════════════════════════
    
    print(f"\n{'─'*W}")
    print(f"  🔴 TEST A: In-Sample (old method — trains on ALL, tests on ALL)")
    print(f"{'─'*W}")
    
    df_full = compute_all_features(df_raw)
    brain_is = HMMBrain(n_states=4)
    brain_is.train(compute_hmm_features(df_raw))
    
    results_is = backtest_realistic(df_full, brain_is, initial_capital=10000, max_positions=1)
    print_report(results_is, "IN-SAMPLE (BIASED) — Train on ALL, Test on ALL")
    print_trade_log(results_is['trades'])
    
    # ═══════════════════════════════════════════════════════════════════════
    #  TEST B: Walk-Forward (proper OOS)
    # ═══════════════════════════════════════════════════════════════════════
    
    print(f"\n{'─'*W}")
    print(f"  🟢 TEST B: Walk-Forward Out-of-Sample (proper — train on N, test on N+1)")
    print(f"{'─'*W}")
    
    results_wf = walk_forward_backtest(
        df_raw,
        train_window=400,
        test_window=100,
        step=100,
        initial_capital=10000,
    )
    print_report(results_wf, "WALK-FORWARD (TRUE OOS) — Rolling Window")
    print_trade_log(results_wf['trades'])
    
    # ═══════════════════════════════════════════════════════════════════════
    #  COMPARISON
    # ═══════════════════════════════════════════════════════════════════════
    
    print(f"\n{DIV}")
    print(f"  📊 COMPARISON: In-Sample vs Walk-Forward")
    print(DIV)
    
    metrics = [
        ("Total Return", f"{results_is['total_return']:+.2f}%", f"{results_wf['total_return']:+.2f}%"),
        ("Max Drawdown", f"{results_is['max_drawdown']:.2f}%", f"{results_wf['max_drawdown']:.2f}%"),
        ("Sharpe Ratio", f"{results_is['sharpe_ratio']:.3f}", f"{results_wf['sharpe_ratio']:.3f}"),
        ("Profit Factor", f"{results_is['profit_factor']:.3f}", f"{results_wf['profit_factor']:.3f}"),
        ("Win Rate", f"{results_is['win_rate']:.1f}%", f"{results_wf['win_rate']:.1f}%"),
        ("Total Trades", str(results_is['total_trades']), str(results_wf['total_trades'])),
        ("SL Hits", str(results_is['sl_hits']), str(results_wf['sl_hits'])),
        ("TP Hits", str(results_is['tp_hits']), str(results_wf['tp_hits'])),
        ("Trail Active", str(results_is.get('trail_activated', 0)), str(results_wf.get('trail_activated', 0))),
    ]
    
    print(f"\n  {'Metric':<20} {'In-Sample (biased)':>20} {'Walk-Forward (true)':>20}")
    print(f"  {'-'*62}")
    for name, is_val, wf_val in metrics:
        print(f"  {name:<20} {is_val:>20} {wf_val:>20}")
    
    # Verdict
    wf_return = results_wf['total_return']
    wf_dd = results_wf['max_drawdown']
    wf_sharpe = results_wf['sharpe_ratio']
    
    print(f"\n  {'─'*62}")
    if wf_return > 0 and wf_sharpe > 0.5:
        print(f"  🟢 VERDICT: Model has genuine edge on real OOS data!")
        print(f"     Return: {wf_return:+.2f}% | Sharpe: {wf_sharpe:.3f} | MaxDD: {wf_dd:.2f}%")
    elif wf_return > 0:
        print(f"  🟡 VERDICT: Marginal edge — positive but low Sharpe")
        print(f"     Return: {wf_return:+.2f}% | Sharpe: {wf_sharpe:.3f}")
    else:
        print(f"  🔴 VERDICT: No OOS edge — model loses money on unseen data")
        print(f"     Return: {wf_return:+.2f}% | Sharpe: {wf_sharpe:.3f}")
    
    print(f"\n{DIV}\n")
