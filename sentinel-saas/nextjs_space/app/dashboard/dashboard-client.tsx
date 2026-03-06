'use client';

import { useState, useEffect, useCallback } from 'react';
import { Header } from '@/components/header';
import { StatsCard } from '@/components/stats-card';
import { BotCard } from '@/components/bot-card';
import { RegimeCard, PnlCard, ActivePositionsCard, SignalSummaryTable } from '@/components/dashboard/command-center';
import { Bot, TrendingUp, Activity, DollarSign, RefreshCw, Zap } from 'lucide-react';
import Link from 'next/link';
import { motion } from 'framer-motion';

interface DashboardClientProps {
  user: {
    id: string;
    name: string;
    email: string;
    subscription: any;
  };
  stats: {
    activeBots: number;
    totalBots: number;
    activeTrades: number;
    totalTrades: number;
    totalPnl: number;
    activePnl: number;
  };
  bots: any[];
  recentTrades: any[];
}

interface BotState {
  state: { regime: string; confidence: number; symbol: string; timestamp: string | null };
  multi: {
    coins_scanned: number;
    eligible_count: number;
    deployed_count: number;
    total_trades: number;
    active_positions: Record<string, any>;
    coin_states: Record<string, any>;
    cycle: number;
    timestamp: string | null;
  };
  tradebook: { trades: any[]; summary: any };
}

export function DashboardClient({ user, stats, bots, recentTrades }: DashboardClientProps) {
  const [mounted, setMounted] = useState(false);
  const [botState, setBotState] = useState<BotState | null>(null);
  const [lastRefresh, setLastRefresh] = useState('');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [feedHealth, setFeedHealth] = useState<any>(null);

  const fetchBotState = useCallback(async () => {
    try {
      setIsRefreshing(true);
      const res = await fetch('/api/bot-state', { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        setBotState(data);
        setLastRefresh(new Date().toLocaleTimeString());
      }
    } catch {
      // silent
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    setMounted(true);
    fetchBotState();
    const interval = setInterval(fetchBotState, 15000); // refresh every 15s

    // Fetch feed health for admin
    if ((user as any)?.role === 'admin') {
      const fetchHealth = async () => {
        try {
          const [liveRes, fgRes] = await Promise.all([
            fetch('/api/live-market', { cache: 'no-store' }),
            fetch('https://api.alternative.me/fng/?limit=1'),
          ]);
          setFeedHealth({
            liveMarket: liveRes.ok ? 'ok' : 'error',
            liveMarketTime: new Date().toISOString(),
            fearGreed: fgRes.ok ? 'ok' : 'error',
            fearGreedTime: new Date().toISOString(),
          });
        } catch {
          setFeedHealth({ liveMarket: 'error', fearGreed: 'error' });
        }
      };
      fetchHealth();
      const healthInterval = setInterval(fetchHealth, 60000);
      return () => { clearInterval(interval); clearInterval(healthInterval); };
    }

    return () => clearInterval(interval);
  }, [fetchBotState]);

  const handleBotToggle = async (botId: string, currentStatus: boolean) => {
    try {
      const response = await fetch('/api/bots/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ botId, isActive: !currentStatus }),
      });

      if (response.ok) {
        window.location.reload();
      }
    } catch (error) {
      console.error('Error toggling bot:', error);
    }
  };

  if (!mounted) {
    return null;
  }

  const formatCurrency = (value: number) => `$${value.toFixed(2)}`;

  const multi = botState?.multi;
  const trades = botState?.tradebook?.trades || [];

  // Extract BTC multi-timeframe data for regime card — prefer coin_states over stale state
  const btcState = multi?.coin_states?.['BTCUSDT'] || {};
  const regime = btcState?.regime || botState?.state?.regime || 'WAITING';
  const confidence = btcState?.confidence || botState?.state?.confidence || 0;
  const symbol = btcState?.symbol || botState?.state?.symbol || 'BTCUSDT';
  const macroRegime = btcState?.macro_regime || undefined;
  const trend15m = btcState?.ta_multi?.['15m']?.trend || undefined;

  // Live stats from engine data (overrides stale DB stats)
  const liveTrades = trades || [];
  const liveActiveTrades = liveTrades.filter((t: any) => (t.status || '').toUpperCase() === 'ACTIVE');
  const liveClosedTrades = liveTrades.filter((t: any) => (t.status || '').toUpperCase() === 'CLOSED');
  const liveTotalPnl = liveClosedTrades.reduce((sum: number, t: any) => sum + (t.realized_pnl || t.pnl || t.total_pnl || 0), 0);
  const liveActivePnl = liveActiveTrades.reduce((sum: number, t: any) => sum + (t.unrealized_pnl || t.active_pnl || 0), 0);

  // Paper vs Live PNL split
  const paperActiveTrades = liveActiveTrades.filter((t: any) => (t.mode || 'paper').toUpperCase() === 'PAPER');
  const liveModeTrades = liveActiveTrades.filter((t: any) => (t.mode || '').toUpperCase() === 'LIVE');
  const paperActivePnl = paperActiveTrades.reduce((sum: number, t: any) => sum + (t.unrealized_pnl || t.active_pnl || 0), 0);
  const liveActiveModePnl = liveModeTrades.reduce((sum: number, t: any) => sum + (t.unrealized_pnl || t.active_pnl || 0), 0);

  const CAPITAL_PER_TRADE = 100;
  const MAX_CAPITAL = 2500;
  const MAX_SLOTS = 25;
  const usedCapital = liveActiveTrades.length * CAPITAL_PER_TRADE;

  const liveStats = {
    activeBots: stats?.activeBots ?? (bots?.filter((b: any) => b?.isActive)?.length ?? 0),
    activeTrades: liveActiveTrades.length || stats?.activeTrades || 0,
    totalPnl: liveTotalPnl + liveActivePnl,
    paperActivePnl,
    liveActivePnl: liveActiveModePnl,
    usedCapital,
  };

  return (
    <div className="min-h-screen">
      <Header />

      <main className="pt-24 pb-12 px-4">
        <div className="max-w-7xl mx-auto">
          {/* Welcome + Status Bar */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold mb-1">
                  Welcome back, <span className="text-gradient">{user?.name ?? 'Trader'}</span>
                </h1>
                <p className="text-[var(--color-text-secondary)] text-sm">
                  AI Trading Command Center — Monitor your bots and market signals
                </p>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                {/* Engine Status Indicator */}
                {(() => {
                  const engineTs = botState?.multi?.timestamp || botState?.state?.timestamp;
                  const cycle = botState?.multi?.cycle || 0;
                  const coinsScanned = botState?.multi?.coins_scanned || 0;
                  const isEngineRunning = engineTs && (Date.now() - new Date(engineTs).getTime()) < 600000; // 10 min
                  return (
                    <div style={{
                      display: 'flex', alignItems: 'center', gap: '10px',
                      padding: '8px 16px', borderRadius: '12px',
                      background: isEngineRunning ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)',
                      border: `1px solid ${isEngineRunning ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)'}`,
                    }}>
                      <div style={{
                        width: '8px', height: '8px', borderRadius: '50%',
                        background: isEngineRunning ? '#22C55E' : '#EF4444',
                        boxShadow: isEngineRunning ? '0 0 8px rgba(34,197,94,0.6)' : 'none',
                        animation: isEngineRunning ? 'pulse 2s ease-in-out infinite' : 'none',
                      }} />
                      <div>
                        <div style={{ fontSize: '12px', fontWeight: 700, color: isEngineRunning ? '#22C55E' : '#EF4444' }}>
                          <Zap size={11} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-1px' }} />
                          {isEngineRunning ? 'Engine ON' : 'Engine OFF'}
                        </div>
                        {isEngineRunning && (
                          <div style={{ fontSize: '10px', color: '#6B7280', marginTop: '1px' }}>
                            Cycle #{cycle} · {coinsScanned} coins scanned
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}
                {(() => {
                  const engineTs = botState?.multi?.timestamp || botState?.state?.timestamp;
                  if (!engineTs) return null;
                  const lastRun = new Date(engineTs).getTime();
                  const intervalMs = 300000; // 5 minutes
                  const nextRun = new Date(lastRun + intervalMs);
                  const now = Date.now();
                  const diff = nextRun.getTime() - now;
                  const nextLabel = diff > 0
                    ? `Next cycle: ${nextRun.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'Asia/Kolkata' })} IST`
                    : 'Cycle running...';
                  return (
                    <span className="text-xs text-[var(--color-text-secondary)]">
                      {nextLabel}
                    </span>
                  );
                })()}
              </div>
            </div>
          </motion.div>

          {/* ═══ Row 1: Regime + P&L ═══ */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-8"
          >
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '20px',
            }}>
              <RegimeCard regime={regime} confidence={confidence} symbol={symbol} macroRegime={macroRegime} trend15m={trend15m} coinStates={multi?.coin_states} />
              <PnlCard trades={trades} />
            </div>
          </motion.div>

          {/* ═══ Row 2: Quick SaaS Stats ═══ */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
          >
            <StatsCard
              title="Active Bots"
              value={liveStats.activeBots}
              icon={Bot}
              animated
            />
            <StatsCard
              title="Active Trades"
              value={`${liveStats.activeTrades} · $${liveStats.usedCapital} of $${MAX_CAPITAL}`}
              icon={Activity}
              animated
            />
            <StatsCard
              title="Paper Active PNL"
              value={formatCurrency(liveStats.paperActivePnl)}
              icon={DollarSign}
              trend={liveStats.paperActivePnl >= 0 ? 'up' : 'down'}
            />
            <StatsCard
              title="Live Active PNL"
              value={formatCurrency(liveStats.liveActivePnl)}
              icon={TrendingUp}
              trend={liveStats.liveActivePnl >= 0 ? 'up' : 'down'}
            />
          </motion.div>

          {/* ═══ Row 3: Bots Section ═══ */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
            className="mb-12"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-cyan-400">Your Bots</h2>
              <Link
                href="/bots"
                className="px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary-dark)] transition-colors"
              >
                Manage Bots
              </Link>
            </div>

            {bots && bots.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {bots.map((bot) => (
                  <BotCard key={bot?.id} bot={bot} onToggle={handleBotToggle} liveTradeCount={liveActiveTrades.length} />
                ))}
              </div>
            ) : (
              <div className="card-gradient p-12 rounded-xl text-center">
                <Bot className="w-16 h-16 text-[var(--color-primary)] mx-auto mb-4" />
                <h3 className="text-xl font-semibold mb-2">No Bots Yet</h3>
                <p className="text-[var(--color-text-secondary)] mb-4">
                  Deploy your first trading bot to get started
                </p>
                <Link
                  href="/bots"
                  className="inline-block px-6 py-3 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary-dark)] transition-colors"
                >
                  Deploy Bot
                </Link>
              </div>
            )}
          </motion.div>

          {/* ═══ Row 5: Recent Trades ═══ */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <div style={{
              background: 'rgba(17, 24, 39, 0.85)',
              backdropFilter: 'blur(16px)',
              border: '1px solid rgba(6, 182, 212, 0.15)',
              borderRadius: '16px',
              overflow: 'hidden',
            }}>
              {/* Header bar */}
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '16px 24px',
                background: 'linear-gradient(135deg, rgba(6,182,212,0.08) 0%, rgba(139,92,246,0.06) 100%)',
                borderBottom: '1px solid rgba(255,255,255,0.06)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{
                    width: '8px', height: '8px', borderRadius: '50%',
                    background: liveTrades.length > 0 ? '#22C55E' : '#6B7280',
                    boxShadow: liveTrades.length > 0 ? '0 0 8px rgba(34,197,94,0.5)' : 'none',
                  }} />
                  <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#06B6D4', margin: 0 }}>
                    Recent Trades
                  </h2>
                  <span style={{
                    fontSize: '11px', padding: '2px 8px', borderRadius: '10px',
                    background: 'rgba(6,182,212,0.15)', color: '#06B6D4', fontWeight: 600
                  }}>
                    {liveTrades.length} total
                  </span>
                </div>
                <Link href="/trades" style={{
                  fontSize: '13px', color: '#06B6D4', fontWeight: 500,
                  textDecoration: 'none', padding: '4px 12px', borderRadius: '8px',
                  background: 'rgba(6,182,212,0.1)', border: '1px solid rgba(6,182,212,0.2)',
                }}>
                  View All →
                </Link>
              </div>

              {liveTrades.length > 0 ? (
                <div style={{ overflowX: 'auto', maxHeight: '480px', overflowY: 'auto', padding: '0' }}>
                  <table style={{ width: '100%', minWidth: '1200px', borderCollapse: 'collapse', fontSize: '12px' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                        {['Bot', 'Coin', 'Side', 'Entry', 'SL Price', 'Target', 'SL Type', 'Target Type', 'Status', 'PNL', 'Entry Time'].map(h => (
                          <th key={h} style={{
                            textAlign: 'left', padding: '12px 14px',
                            fontSize: '10px', fontWeight: 600, textTransform: 'uppercase',
                            letterSpacing: '0.8px', color: '#6B7280',
                            background: 'rgba(255,255,255,0.02)',
                            position: 'sticky', top: 0, zIndex: 1,
                          }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {[...liveTrades]
                        .sort((a: any, b: any) => {
                          const ta = a.entry_time || a.entryTime || a.timestamp || '';
                          const tb = b.entry_time || b.entryTime || b.timestamp || '';
                          return tb.localeCompare(ta); // latest first
                        })
                        .slice(0, 10)
                        .map((trade: any, i: number) => {
                          const sym = (trade.symbol || trade.coin || '').replace('USDT', '');
                          const side = (trade.side || trade.position || '').toUpperCase();
                          const entry = trade.entry_price || trade.entryPrice || 0;
                          const sl = trade.stop_loss || trade.stopLoss || 0;
                          const tp = trade.take_profit || trade.takeProfit || 0;
                          const slType = trade.sl_type || trade.slType || 'Default';
                          const tpType = trade.tp_type || trade.targetType || 'T1';
                          const status = (trade.status || '').toUpperCase();
                          const pnl = trade.unrealized_pnl || trade.active_pnl || trade.pnl || 0;
                          const isLong = side === 'BUY' || side === 'LONG';
                          const entryTime = trade.entry_time || trade.entryTime || trade.timestamp || '';
                          const fmtTime = (() => {
                            try {
                              // Trim Python microsecond precision (.123456 → .123) for cross-browser compat
                              const sanitized = String(entryTime).replace(/(\.\d{3})\d+/, '$1');
                              const d = new Date(sanitized);
                              if (isNaN(d.getTime())) return '—';
                              return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', timeZone: 'Asia/Kolkata' }) + ' ' +
                                d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: true, timeZone: 'Asia/Kolkata' });
                            } catch { return '—'; }
                          })();
                          return (
                            <tr key={trade.trade_id || i} style={{
                              borderBottom: '1px solid rgba(255,255,255,0.04)',
                              transition: 'background 0.2s',
                            }}
                              onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(6,182,212,0.04)')}
                              onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                            >
                              <td style={{ padding: '10px 14px', color: '#0891B2', fontWeight: 600, fontSize: '11px' }}>
                                {trade.bot_name || trade.profile_id || 'SM-Standard'}
                              </td>
                              <td style={{ padding: '10px 14px', fontWeight: 700, color: '#F0F4F8' }}>{sym}</td>
                              <td style={{ padding: '10px 14px' }}>
                                <span style={{
                                  padding: '3px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: 700,
                                  background: isLong ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
                                  color: isLong ? '#22C55E' : '#EF4444',
                                  border: `1px solid ${isLong ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'}`,
                                }}>
                                  {side}
                                </span>
                              </td>
                              <td style={{ padding: '10px 14px', fontFamily: 'monospace', color: '#D1D5DB' }}>
                                ${Number(entry).toFixed(4)}
                              </td>
                              <td style={{ padding: '10px 14px', fontFamily: 'monospace', color: '#EF4444' }}>
                                ${Number(sl).toFixed(4)}
                              </td>
                              <td style={{ padding: '10px 14px', fontFamily: 'monospace', color: '#22C55E' }}>
                                ${Number(tp).toFixed(4)}
                              </td>
                              <td style={{ padding: '10px 14px' }}>
                                <span style={{
                                  padding: '2px 8px', borderRadius: '4px', fontSize: '10px', fontWeight: 600,
                                  background: slType.includes('Trail') ? 'rgba(245,158,11,0.15)' : 'rgba(107,114,128,0.15)',
                                  color: slType.includes('Trail') ? '#F59E0B' : '#9CA3AF',
                                }}>
                                  {slType}
                                </span>
                              </td>
                              <td style={{ padding: '10px 14px' }}>
                                <span style={{
                                  padding: '2px 8px', borderRadius: '4px', fontSize: '10px', fontWeight: 600,
                                  background: 'rgba(6,182,212,0.12)', color: '#06B6D4',
                                }}>
                                  {tpType}
                                </span>
                              </td>
                              <td style={{ padding: '10px 14px' }}>
                                <span style={{
                                  padding: '3px 10px', borderRadius: '10px', fontSize: '10px', fontWeight: 700,
                                  background: status === 'ACTIVE' ? 'rgba(34,197,94,0.15)' : 'rgba(107,114,128,0.15)',
                                  color: status === 'ACTIVE' ? '#22C55E' : '#9CA3AF',
                                  boxShadow: status === 'ACTIVE' ? '0 0 6px rgba(34,197,94,0.2)' : 'none',
                                }}>
                                  ● {status}
                                </span>
                              </td>
                              <td style={{
                                padding: '10px 14px', fontWeight: 700, fontFamily: 'monospace',
                                color: pnl >= 0 ? '#22C55E' : '#EF4444',
                              }}>
                                {pnl >= 0 ? '+' : ''}${Number(pnl).toFixed(2)}
                              </td>
                              <td style={{ padding: '10px 14px', fontSize: '11px', color: '#9CA3AF' }}>
                                {fmtTime}
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ padding: '48px 24px', textAlign: 'center' }}>
                  <Activity className="w-12 h-12 mx-auto mb-3" style={{ color: '#06B6D4', opacity: 0.5 }} />
                  <p style={{ color: '#6B7280', fontSize: '14px' }}>
                    No trades yet. Start the engine to begin trading.
                  </p>
                </div>
              )}
            </div>
          </motion.div>

          {/* ═══ Row 5: Signal Summary Table ═══ */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
            className="mt-8"
          >
            <SignalSummaryTable coinStates={multi?.coin_states || {}} multi={multi} />
          </motion.div>
        </div>
      </main>

      {/* ═══ Admin: Data Feed Health ═══ */}
      {(user as any)?.role === 'admin' && (
        <div className="max-w-7xl mx-auto px-4 pb-12">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
            <div style={{
              background: 'rgba(17, 24, 39, 0.85)', backdropFilter: 'blur(16px)',
              border: '1px solid rgba(245,158,11,0.2)', borderRadius: '16px', overflow: 'hidden',
            }}>
              <div style={{
                padding: '16px 24px',
                background: 'linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(239,68,68,0.04) 100%)',
                borderBottom: '1px solid rgba(255,255,255,0.06)',
              }}>
                <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#F59E0B', margin: 0 }}>🛡️ Data Feed Health</h2>
                <p style={{ fontSize: '12px', color: '#6B7280', marginTop: '2px' }}>Admin-only monitoring of external data sources</p>
              </div>
              <div style={{ padding: '20px 24px', display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '16px' }}>
                {[
                  {
                    name: 'Engine Cycle',
                    status: botState?.multi?.cycle ? 'ok' : 'waiting',
                    detail: botState?.multi?.cycle ? `Cycle #${botState.multi.cycle}` : 'No cycles yet',
                    sub: botState?.multi?.timestamp ? `Last: ${new Date(botState.multi.timestamp).toLocaleTimeString()}` : 'Waiting…',
                  },
                  {
                    name: 'Binance API',
                    status: feedHealth?.liveMarket || 'checking',
                    detail: feedHealth?.liveMarket === 'ok' ? 'Connected' : 'Error',
                    sub: 'Funding rates & prices',
                  },
                  {
                    name: 'Fear & Greed',
                    status: feedHealth?.fearGreed || 'checking',
                    detail: feedHealth?.fearGreed === 'ok' ? 'Connected' : 'Error',
                    sub: 'alternative.me API',
                  },
                  {
                    name: 'Coin Scanner',
                    status: (botState?.multi?.coins_scanned || 0) > 0 ? 'ok' : 'waiting',
                    detail: `${botState?.multi?.coins_scanned || 0} coins`,
                    sub: `${botState?.multi?.eligible_count || 0} eligible`,
                  },
                  {
                    name: 'Tradebook',
                    status: (botState?.tradebook?.trades?.length || 0) > 0 ? 'ok' : 'waiting',
                    detail: `${botState?.tradebook?.trades?.length || 0} trades`,
                    sub: 'tradebook.json',
                  },
                ].map(feed => {
                  const color = feed.status === 'ok' ? '#22C55E' : feed.status === 'error' ? '#EF4444' : '#F59E0B';
                  return (
                    <div key={feed.name} style={{
                      background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
                      borderRadius: '12px', padding: '16px', textAlign: 'center',
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', marginBottom: '8px' }}>
                        <div style={{
                          width: '8px', height: '8px', borderRadius: '50%', background: color,
                          boxShadow: `0 0 8px ${color}44`,
                        }} />
                        <span style={{ fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.8px', color: '#9CA3AF' }}>{feed.name}</span>
                      </div>
                      <div style={{ fontSize: '16px', fontWeight: 700, color }}>{feed.detail}</div>
                      <div style={{ fontSize: '10px', color: '#6B7280', marginTop: '4px' }}>{feed.sub}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}