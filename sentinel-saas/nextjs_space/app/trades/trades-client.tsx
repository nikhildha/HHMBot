'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import { Header } from '@/components/header';
import { Download, TrendingUp, TrendingDown, Clock, Search, X, BarChart3, RefreshCw } from 'lucide-react';
import { motion } from 'framer-motion';

/* ═══ Types ═══ */
interface Trade {
  id: string; coin: string; symbol?: string; position: string; regime: string;
  confidence: number; leverage: number; capital: number;
  entryPrice: number; currentPrice?: number | null;
  exitPrice?: number | null; stopLoss: number; takeProfit: number;
  slType: string; targetType: string; status: string; mode?: string;
  activePnl: number; activePnlPercent: number;
  totalPnl: number; totalPnlPercent: number;
  exitPercent?: number | null; exitReason?: string | null;
  entryTime: string; exitTime?: string | null;
  botName?: string;
}

/* ═══ Utilities ═══ */
const fmt$ = (v: number) => (v >= 0 ? '+' : '') + v.toFixed(2);
const fmtPct = (v: number) => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
const fmtPrice = (v: number) => v >= 100 ? v.toFixed(2) : v >= 1 ? v.toFixed(4) : v.toFixed(6);
const pnlColor = (v: number) => v > 0 ? '#22C55E' : v < 0 ? '#EF4444' : '#6B7280';

/* ═══ Map raw engine trade to typed Trade ═══ */
function mapTrade(t: any): Trade {
  return {
    id: t.trade_id || t.id || `T-${Math.random().toString(36).slice(2, 8)}`,
    coin: (t.symbol || t.coin || '').replace('USDT', ''),
    symbol: t.symbol || t.coin || '',
    position: (t.side || t.position || '').toLowerCase(),
    regime: t.regime || '',
    confidence: t.confidence || 0,
    leverage: t.leverage || 1,
    capital: t.capital || t.position_size || 0,
    entryPrice: t.entry_price || t.entryPrice || 0,
    currentPrice: t.current_price || t.currentPrice || null,
    exitPrice: t.exit_price || t.exitPrice || null,
    stopLoss: t.stop_loss || t.stopLoss || 0,
    takeProfit: t.take_profit || t.takeProfit || 0,
    slType: t.sl_type || t.slType || 'Default',
    targetType: t.target_type || t.tp_type || t.targetType || 'T1',
    status: (t.status || '').toLowerCase(),
    mode: t.mode || 'paper',
    activePnl: t.unrealized_pnl || t.active_pnl || t.activePnl || 0,
    activePnlPercent: t.unrealized_pnl_pct || t.activePnlPercent || 0,
    totalPnl: t.realized_pnl || t.pnl || t.total_pnl || t.totalPnl || 0,
    totalPnlPercent: t.realized_pnl_pct || t.pnl_pct || t.totalPnlPercent || 0,
    exitPercent: t.exit_percent || null,
    exitReason: t.exit_reason || t.exitReason || null,
    entryTime: t.entry_time || t.entryTime || t.timestamp || new Date().toISOString(),
    exitTime: t.exit_time || t.exitTime || null,
    botName: 'SM-Standard',
  };
}

/* ═══ Card Wrapper ═══ */
function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={className} style={{
      background: 'rgba(17, 24, 39, 0.8)', backdropFilter: 'blur(12px)',
      border: '1px solid rgba(255,255,255,0.06)', borderRadius: '16px', padding: '20px',
    }}>{children}</div>
  );
}

function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <Card>
      <div style={{ fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px', color: '#6B7280', marginBottom: '6px' }}>{label}</div>
      <div style={{ fontSize: '22px', fontWeight: 700, color: color || '#F0F4F8' }}>{value}</div>
      {sub && <div style={{ fontSize: '11px', color: '#6B7280', marginTop: '4px' }}>{sub}</div>}
    </Card>
  );
}

/* ═══════════════════════════════════════════════════════════════════ */
/*  MAIN COMPONENT                                                   */
/* ═══════════════════════════════════════════════════════════════════ */

interface TradesClientProps { trades: Trade[]; }

export function TradesClient({ trades: initialTrades }: TradesClientProps) {
  const [mounted, setMounted] = useState(false);
  const [trades, setTrades] = useState<Trade[]>(initialTrades);
  const [statusFilter, setStatusFilter] = useState<'all' | 'active' | 'closed'>('active');
  const [posFilter, setPosFilter] = useState<string>('all');
  const [regimeFilter, setRegimeFilter] = useState<string>('all');
  const [coinSearch, setCoinSearch] = useState('');
  const [pnlFilter, setPnlFilter] = useState<'all' | 'profit' | 'loss'>('all');
  const [modeFilter, setModeFilter] = useState<'all' | 'paper' | 'live'>('all');
  const [lastRefresh, setLastRefresh] = useState<string>('');

  useEffect(() => { setMounted(true); }, []);

  /* ── 15-second auto-refresh from bot-state API ── */
  const refreshTrades = useCallback(async () => {
    try {
      const res = await fetch('/api/bot-state', { cache: 'no-store' });
      if (res.ok) {
        const d = await res.json();
        const rawTrades = d?.tradebook?.trades || [];
        if (rawTrades.length > 0) {
          setTrades(rawTrades.map(mapTrade));
          setLastRefresh(new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
        }
      }
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    refreshTrades(); // initial fetch
    const timer = setInterval(refreshTrades, 15000);
    return () => clearInterval(timer);
  }, [refreshTrades]);

  /* ── Filter trades — case-insensitive matching ── */
  const filtered = useMemo(() => {
    return (trades ?? []).filter(t => {
      const tStatus = (t.status || '').toLowerCase();
      const tMode = (t.mode || '').toLowerCase();
      const tPos = (t.position || '').toLowerCase();
      const tRegime = (t.regime || '').toLowerCase();

      if (statusFilter === 'active' && tStatus !== 'active') return false;
      if (statusFilter === 'closed' && tStatus === 'active') return false;
      if (modeFilter !== 'all' && tMode !== modeFilter) return false;
      if (posFilter !== 'all') {
        const posMatch = posFilter === 'long'
          ? ['long', 'buy'].includes(tPos)
          : ['short', 'sell'].includes(tPos);
        if (!posMatch) return false;
      }
      if (regimeFilter !== 'all' && !tRegime.includes(regimeFilter)) return false;
      if (coinSearch && !t.coin.toLowerCase().includes(coinSearch.toLowerCase())) return false;
      if (pnlFilter === 'profit') {
        const pnl = tStatus === 'active' ? t.activePnl : t.totalPnl;
        if (pnl <= 0) return false;
      }
      if (pnlFilter === 'loss') {
        const pnl = tStatus === 'active' ? t.activePnl : t.totalPnl;
        if (pnl >= 0) return false;
      }
      return true;
    });
  }, [trades, statusFilter, modeFilter, posFilter, regimeFilter, coinSearch, pnlFilter]);

  /* ── Portfolio Stats ── */
  const CAPITAL_PER_TRADE = 100;
  const stats = useMemo(() => {
    const all = trades ?? [];
    const active = all.filter(t => (t.status || '').toLowerCase() === 'active');
    const closed = all.filter(t => (t.status || '').toLowerCase() !== 'active');
    const wins = closed.filter(t => t.totalPnl > 0);
    const losses = closed.filter(t => t.totalPnl <= 0);
    const winRate = closed.length > 0 ? (wins.length / closed.length) * 100 : 0;
    const realizedPnl = closed.reduce((s, t) => s + (t.totalPnl || 0), 0);
    const unrealizedPnl = active.reduce((s, t) => s + (t.activePnl || 0), 0);
    const combinedPnl = realizedPnl + unrealizedPnl;

    const allPnlPcts = [
      ...closed.map(t => t.totalPnlPercent || 0),
      ...active.map(t => t.activePnlPercent || 0),
    ];
    const bestTrade = allPnlPcts.length > 0 ? Math.max(...allPnlPcts) : 0;
    const worstTrade = allPnlPcts.length > 0 ? Math.min(...allPnlPcts) : 0;

    // Max drawdown as % of total deployed capital
    const totalDeployedCapital = all.length * CAPITAL_PER_TRADE;
    let peak = 0, maxDD = 0, cumPnl = 0;
    const sortedClosed = [...closed].sort((a, b) => (a.entryTime || '').localeCompare(b.entryTime || ''));
    sortedClosed.forEach(t => {
      cumPnl += t.totalPnl || 0;
      if (cumPnl > peak) peak = cumPnl;
      const dd = peak - cumPnl;
      if (dd > maxDD) maxDD = dd;
    });
    const totalEquity = cumPnl + unrealizedPnl;
    if (totalEquity < peak) {
      const dd = peak - totalEquity;
      if (dd > maxDD) maxDD = dd;
    }
    const maxDDPct = totalDeployedCapital > 0 ? (maxDD / totalDeployedCapital * 100) : 0;

    const grossProfit = wins.reduce((s, t) => s + t.totalPnl, 0);
    const grossLoss = Math.abs(losses.reduce((s, t) => s + t.totalPnl, 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;

    const avgWin = wins.length > 0 ? grossProfit / wins.length : 0;
    const avgLoss = losses.length > 0 ? grossLoss / losses.length : 1;
    const riskReward = avgLoss > 0 ? avgWin / avgLoss : 0;

    return {
      total: all.length, active: active.length, closed: closed.length,
      wins: wins.length, losses: losses.length, winRate,
      realizedPnl, unrealizedPnl, combinedPnl,
      bestTrade, worstTrade,
      maxDD, maxDDPct, profitFactor, riskReward,
    };
  }, [trades]);

  /* ── CSV Export ── */
  const exportCSV = () => {
    const headers = ['Bot', 'Type', 'Coin', 'Side', 'Leverage', 'Capital', 'Entry Price', 'Exit Price', 'SL', 'TP', 'SL Type', 'Target Type', 'P&L $', 'P&L %', 'Status', 'Entry Time', 'Exit Time'];
    const rows = filtered.map(t => [
      'SM-Standard', t.mode || 'paper', t.coin, t.position, t.leverage, t.capital,
      t.entryPrice, t.exitPrice || t.currentPrice || '', t.stopLoss, t.takeProfit,
      t.slType, t.targetType,
      t.status === 'active' ? t.activePnl : t.totalPnl,
      t.status === 'active' ? t.activePnlPercent : t.totalPnlPercent,
      t.status, t.entryTime, t.exitTime || '',
    ]);
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `tradebook_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click(); URL.revokeObjectURL(url);
  };

  const uniqueRegimes = useMemo(() => [...new Set(trades?.map(t => t.regime?.toLowerCase()).filter(Boolean))], [trades]);

  if (!mounted) return null;

  return (
    <div className="min-h-screen">
      <Header />
      <main className="pt-24 pb-12 px-4">
        <div className="max-w-7xl mx-auto">

          {/* ─── Hero ─── */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold mb-1">Trade Journal</h1>
                <p className="text-sm text-[var(--color-text-secondary)]">
                  Complete history · Portfolio analytics · Auto-refreshes every 15s
                  {lastRefresh && <span style={{ marginLeft: '8px', color: '#06B6D4' }}>Last: {lastRefresh}</span>}
                </p>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={refreshTrades} style={{
                  display: 'flex', alignItems: 'center', gap: '6px',
                  padding: '10px 14px', borderRadius: '12px', border: 'none',
                  background: 'rgba(34,197,94,0.1)', color: '#22C55E',
                  fontSize: '13px', fontWeight: 600, cursor: 'pointer',
                }}>
                  <RefreshCw size={14} /> Refresh
                </button>
                <button onClick={exportCSV} style={{
                  display: 'flex', alignItems: 'center', gap: '6px',
                  padding: '10px 14px', borderRadius: '12px', border: 'none',
                  background: 'rgba(8, 145, 178, 0.15)', color: '#0EA5E9',
                  fontSize: '13px', fontWeight: 600, cursor: 'pointer',
                }}>
                  <Download size={14} /> Export CSV
                </button>
              </div>
            </div>
          </motion.div>

          {/* ═══ Portfolio Summary Stats ═══ */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="mb-6">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '12px' }}>
              <StatCard label="Total Trades" value={String(stats.total)} sub={`${stats.active} active · ${stats.closed} closed`} />
              <StatCard label="Win Rate" value={stats.winRate.toFixed(1) + '%'} sub={`${stats.wins}W / ${stats.losses}L`} color={stats.winRate >= 50 ? '#22C55E' : '#EF4444'} />
              <StatCard label="Total PNL" value={'$' + fmt$(stats.combinedPnl)} sub={`Realized: $${fmt$(stats.realizedPnl)} · Active: $${fmt$(stats.unrealizedPnl)}`} color={pnlColor(stats.combinedPnl)} />
              <StatCard label="Active PNL" value={'$' + fmt$(stats.unrealizedPnl)} sub={`${stats.active} open positions`} color={pnlColor(stats.unrealizedPnl)} />
              <StatCard label="Best / Worst" value={fmtPct(stats.bestTrade)} sub={fmtPct(stats.worstTrade) + ' worst'} color={pnlColor(stats.bestTrade)} />
              <StatCard label="Max Drawdown" value={stats.maxDDPct.toFixed(2) + '%'} sub={`$${stats.maxDD.toFixed(2)} · PF: ${stats.profitFactor === Infinity ? '∞' : stats.profitFactor.toFixed(2)}`} color="#EF4444" />
            </div>
          </motion.div>

          {/* ═══ P&L Equity Curve ═══ */}
          {(() => {
            const closed = (trades ?? [])
              .filter(t => (t.status || '').toLowerCase() !== 'active' && t.exitTime)
              .sort((a, b) => new Date(a.exitTime!).getTime() - new Date(b.exitTime!).getTime());
            if (closed.length < 2) return null;

            let cum = 0;
            const points = closed.map(t => { cum += t.totalPnl || 0; return cum; });
            const minV = Math.min(0, ...points);
            const maxV = Math.max(0, ...points);
            const range = maxV - minV || 1;
            const W = 800, H = 140, PAD = 8;

            const toX = (i: number) => PAD + (i / (points.length - 1)) * (W - PAD * 2);
            const toY = (v: number) => PAD + (1 - (v - minV) / range) * (H - PAD * 2);
            const zeroY = toY(0);
            const polyline = points.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');
            const areaPath = `M${toX(0)},${zeroY} L${points.map((v, i) => `${toX(i)},${toY(v)}`).join(' L')} L${toX(points.length - 1)},${zeroY} Z`;
            const lastPnl = points[points.length - 1];
            const color = lastPnl >= 0 ? '#22C55E' : '#EF4444';

            return (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }} className="mb-6">
                <Card>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1px', color: '#6B7280' }}>
                      Cumulative P&L Curve
                    </div>
                    <div style={{ fontSize: '14px', fontWeight: 700, color }}>{fmt$(lastPnl)} USD</div>
                  </div>
                  <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: '140px' }}>
                    <line x1={PAD} y1={zeroY} x2={W - PAD} y2={zeroY} stroke="rgba(255,255,255,0.1)" strokeDasharray="4,4" />
                    <path d={areaPath} fill={color} fillOpacity="0.08" />
                    <polyline points={polyline} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
                    <circle cx={toX(points.length - 1)} cy={toY(lastPnl)} r="3.5" fill={color} />
                    <text x={PAD + 4} y={H - 4} fontSize="9" fill="#6B7280">{fmt$(minV)}</text>
                    <text x={PAD + 4} y={12} fontSize="9" fill="#6B7280">{fmt$(maxV)}</text>
                  </svg>
                </Card>
              </motion.div>
            );
          })()}

          {/* ═══ Filter Bar ═══ */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="mb-6">
            <Card>
              <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '12px' }}>
                {(['all', 'active', 'closed'] as const).map(s => (
                  <button key={s} onClick={() => setStatusFilter(s)} style={{
                    padding: '6px 14px', borderRadius: '8px', border: 'none', cursor: 'pointer',
                    fontSize: '13px', fontWeight: 600,
                    background: statusFilter === s ? '#0891B2' : 'rgba(255,255,255,0.05)',
                    color: statusFilter === s ? '#fff' : '#9CA3AF',
                    transition: 'all 0.2s',
                  }}>
                    {s === 'all' ? `All (${stats.total})` : s === 'active' ? `Active (${stats.active})` : `Closed (${stats.closed})`}
                  </button>
                ))}

                <div style={{ width: '1px', height: '20px', background: 'rgba(255,255,255,0.1)' }} />

                {(['all', 'paper', 'live'] as const).map(m => (
                  <button key={m} onClick={() => setModeFilter(m)} style={{
                    padding: '6px 14px', borderRadius: '8px', border: 'none', cursor: 'pointer',
                    fontSize: '13px', fontWeight: 600,
                    background: modeFilter === m
                      ? m === 'paper' ? 'rgba(34,197,94,0.2)' : m === 'live' ? 'rgba(239,68,68,0.2)' : 'rgba(255,255,255,0.1)'
                      : 'rgba(255,255,255,0.05)',
                    color: modeFilter === m
                      ? m === 'paper' ? '#22C55E' : m === 'live' ? '#EF4444' : '#D1D5DB'
                      : '#6B7280',
                    transition: 'all 0.2s',
                  }}>
                    {m === 'all' ? 'All Modes' : m === 'paper' ? '🟢 Paper' : '🔴 Live'}
                  </button>
                ))}

                <div style={{ width: '1px', height: '20px', background: 'rgba(255,255,255,0.1)' }} />

                <select value={posFilter} onChange={e => setPosFilter(e.target.value)} style={{
                  padding: '6px 10px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)',
                  background: 'rgba(255,255,255,0.04)', color: '#D1D5DB', fontSize: '13px',
                }}>
                  <option value="all">All Positions</option>
                  <option value="long">Long / Buy</option>
                  <option value="short">Short / Sell</option>
                </select>

                <select value={regimeFilter} onChange={e => setRegimeFilter(e.target.value)} style={{
                  padding: '6px 10px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)',
                  background: 'rgba(255,255,255,0.04)', color: '#D1D5DB', fontSize: '13px',
                }}>
                  <option value="all">All Regimes</option>
                  {uniqueRegimes.map(r => <option key={r} value={r}>{r}</option>)}
                </select>

                <select value={pnlFilter} onChange={e => setPnlFilter(e.target.value as any)} style={{
                  padding: '6px 10px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)',
                  background: 'rgba(255,255,255,0.04)', color: '#D1D5DB', fontSize: '13px',
                }}>
                  <option value="all">All P&L</option>
                  <option value="profit">Profit Only</option>
                  <option value="loss">Loss Only</option>
                </select>

                <div style={{ position: 'relative', marginLeft: 'auto' }}>
                  <Search size={14} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: '#6B7280' }} />
                  <input value={coinSearch} onChange={e => setCoinSearch(e.target.value)}
                    placeholder="Search coin..."
                    style={{
                      padding: '6px 10px 6px 30px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)',
                      background: 'rgba(255,255,255,0.04)', color: '#D1D5DB', fontSize: '13px', width: '150px',
                    }} />
                  {coinSearch && (
                    <X size={12} onClick={() => setCoinSearch('')}
                      style={{ position: 'absolute', right: '8px', top: '50%', transform: 'translateY(-50%)', cursor: 'pointer', color: '#6B7280' }} />
                  )}
                </div>
              </div>
            </Card>
          </motion.div>

          {/* ═══ Trade Journal Table ═══ */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
            {filtered.length > 0 ? (
              <Card>
                <div style={{ overflowX: 'auto', maxHeight: '600px', overflowY: 'auto' }}>
                  <table style={{ width: '100%', minWidth: '1300px', borderCollapse: 'collapse', fontSize: '13px' }}>
                    <thead>
                      <tr style={{ borderBottom: '2px solid rgba(255,255,255,0.08)' }}>
                        {['Bot', 'Type', 'Coin', 'Side', 'Lev', 'Capital', 'Entry', 'Current / Exit', 'SL', 'TP', 'SL Type', 'Target', 'P&L $', 'P&L %', 'Status', 'Duration'].map(h => (
                          <th key={h} style={{
                            padding: '10px 10px', textAlign: h === 'Bot' || h === 'Coin' ? 'left' : 'center',
                            fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.8px',
                            color: '#6B7280', position: 'sticky', top: 0, background: 'rgba(17, 24, 39, 0.95)',
                          }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.map(t => {
                        const isActive = (t.status || '').toLowerCase() === 'active';
                        const pnl = isActive ? t.activePnl : t.totalPnl;
                        const pnlPct = isActive ? t.activePnlPercent : t.totalPnlPercent;
                        const price = isActive ? t.currentPrice : t.exitPrice;
                        const duration = getDuration(t.entryTime, t.exitTime);
                        const pos = (t.position || '').toLowerCase();
                        const isLong = pos === 'long' || pos === 'buy';

                        return (
                          <tr key={t.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                            <td style={{ padding: '10px', color: '#8B5CF6', fontWeight: 600, fontSize: '12px' }}>
                              SM-Standard
                            </td>
                            <td style={{ padding: '10px', textAlign: 'center' }}>
                              <span style={{
                                padding: '2px 8px', borderRadius: '6px', fontSize: '10px', fontWeight: 700,
                                background: (t.mode || '').toLowerCase() === 'live' ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.15)',
                                color: (t.mode || '').toLowerCase() === 'live' ? '#EF4444' : '#22C55E',
                              }}>
                                {(t.mode || 'paper').toUpperCase()}
                              </span>
                            </td>
                            <td style={{ padding: '10px', fontWeight: 700, color: '#F0F4F8' }}>
                              {t.coin.replace('USDT', '')}
                            </td>
                            <td style={{ padding: '10px', textAlign: 'center' }}>
                              <span style={{
                                padding: '2px 8px', borderRadius: '6px', fontSize: '10px', fontWeight: 700,
                                color: isLong ? '#22C55E' : '#EF4444',
                                background: isLong ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
                              }}>
                                {isLong ? '▲ LONG' : '▼ SHORT'}
                              </span>
                            </td>
                            <td style={{ padding: '10px', textAlign: 'center', color: '#D1D5DB' }}>{t.leverage}×</td>
                            <td style={{ padding: '10px', textAlign: 'center', color: '#D1D5DB' }}>${t.capital}</td>
                            <td style={{ padding: '10px', textAlign: 'center', color: '#D1D5DB', fontFamily: 'monospace', fontSize: '12px' }}>{fmtPrice(t.entryPrice)}</td>
                            <td style={{ padding: '10px', textAlign: 'center', color: pnlColor(pnl), fontFamily: 'monospace', fontSize: '12px' }}>
                              {price ? fmtPrice(price) : '—'}
                            </td>
                            <td style={{ padding: '10px', textAlign: 'center', color: '#EF4444', fontFamily: 'monospace', fontSize: '12px' }}>{fmtPrice(t.stopLoss)}</td>
                            <td style={{ padding: '10px', textAlign: 'center', color: '#22C55E', fontFamily: 'monospace', fontSize: '12px' }}>{fmtPrice(t.takeProfit)}</td>
                            <td style={{ padding: '10px', textAlign: 'center' }}>
                              <span style={{
                                padding: '2px 8px', borderRadius: '4px', fontSize: '10px', fontWeight: 600,
                                background: (t.slType || '').includes('Trail') ? 'rgba(34,197,94,0.15)' : 'rgba(107,114,128,0.15)',
                                color: (t.slType || '').includes('Trail') ? '#22C55E' : '#9CA3AF',
                              }}>
                                {(t.slType || '').includes('Trail') ? '🛡️ ' : ''}{t.slType || 'Default'}
                              </span>
                            </td>
                            <td style={{ padding: '10px', textAlign: 'center' }}>
                              <span style={{
                                padding: '2px 8px', borderRadius: '4px', fontSize: '10px', fontWeight: 600,
                                background: 'rgba(6,182,212,0.12)', color: '#06B6D4',
                              }}>
                                {t.targetType || 'T1'}
                              </span>
                            </td>
                            <td style={{ padding: '10px', textAlign: 'center', fontWeight: 700, color: pnlColor(pnl) }}>
                              {fmt$(pnl)}
                            </td>
                            <td style={{ padding: '10px', textAlign: 'center', fontWeight: 700, color: pnlColor(pnlPct) }}>
                              {fmtPct(pnlPct)}
                            </td>
                            <td style={{ padding: '10px', textAlign: 'center' }}>
                              <span style={{
                                padding: '2px 8px', borderRadius: '6px', fontSize: '10px', fontWeight: 700,
                                color: isActive ? '#22C55E' : '#6B7280',
                                background: isActive ? 'rgba(34,197,94,0.15)' : 'rgba(255,255,255,0.05)',
                              }}>
                                {isActive ? '● LIVE' : t.exitReason || 'CLOSED'}
                              </span>
                            </td>
                            <td style={{ padding: '10px', textAlign: 'center', color: '#6B7280', fontSize: '12px' }}>
                              {duration}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div style={{ marginTop: '12px', fontSize: '12px', color: '#6B7280', textAlign: 'right' }}>
                  Showing {filtered.length} of {trades.length} trades
                </div>
              </Card>
            ) : (
              <Card>
                <div style={{ textAlign: 'center', padding: '60px 0' }}>
                  <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(8,145,178,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px' }}><BarChart3 size={24} style={{ color: '#0891B2' }} /></div>
                  <div style={{ fontSize: '18px', fontWeight: 600, color: '#D1D5DB', marginBottom: '8px' }}>No Trades Found</div>
                  <div style={{ fontSize: '14px', color: '#6B7280' }}>
                    {statusFilter === 'all' ? 'Deploy a bot to start trading' : `No ${statusFilter} trades match your filters`}
                  </div>
                </div>
              </Card>
            )}
          </motion.div>

          {/* ═══ Strategy Quality Section ═══ */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="mt-8">
            <div style={{
              background: 'rgba(17, 24, 39, 0.85)', backdropFilter: 'blur(16px)',
              border: '1px solid rgba(139,92,246,0.15)', borderRadius: '16px', overflow: 'hidden',
            }}>
              <div style={{
                padding: '18px 24px',
                background: 'linear-gradient(135deg, rgba(139,92,246,0.08) 0%, rgba(6,182,212,0.06) 100%)',
                borderBottom: '1px solid rgba(255,255,255,0.06)',
              }}>
                <h2 style={{ fontSize: '20px', fontWeight: 700, color: '#8B5CF6', margin: 0 }}>Bot Performance</h2>
                <p style={{ fontSize: '13px', color: '#6B7280', marginTop: '4px' }}>Performance metrics and risk analysis</p>
              </div>
              <div style={{ padding: '20px 24px' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                      {['Bot Name', 'Deployed', 'Last Closed', 'Total', 'Active', 'Closed', 'Win Rate', 'Profit Factor', 'Risk/Reward', 'Max DD'].map(h => (
                        <th key={h} style={{
                          padding: '12px 14px', textAlign: h === 'Bot Name' ? 'left' : 'center',
                          fontSize: '11px', fontWeight: 600, textTransform: 'uppercase',
                          letterSpacing: '0.8px', color: '#6B7280',
                        }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(() => {
                      const all = trades ?? [];
                      if (all.length === 0) return (
                        <tr><td colSpan={10} style={{ padding: '40px', textAlign: 'center', color: '#6B7280', fontSize: '14px' }}>No strategy data yet</td></tr>
                      );
                      const firstEntry = all.reduce((min, t) => t.entryTime < min ? t.entryTime : min, all[0]?.entryTime || '');
                      const closedTrades = all.filter(t => (t.status || '').toLowerCase() !== 'active');
                      const lastClosed = closedTrades.length > 0
                        ? closedTrades.reduce((max, t) => (t.exitTime || '') > max ? (t.exitTime || '') : max, '')
                        : null;
                      const fmtDate = (d: string | null) => {
                        if (!d) return '—';
                        try { return new Date(d).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: '2-digit', hour: '2-digit', minute: '2-digit' }); }
                        catch { return '—'; }
                      };
                      return (
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                          <td style={{ padding: '14px', color: '#8B5CF6', fontWeight: 700, fontSize: '14px' }}>SM-Standard</td>
                          <td style={{ padding: '14px', textAlign: 'center', color: '#D1D5DB', fontSize: '12px' }}>{fmtDate(firstEntry)}</td>
                          <td style={{ padding: '14px', textAlign: 'center', color: '#D1D5DB', fontSize: '12px' }}>{fmtDate(lastClosed)}</td>
                          <td style={{ padding: '14px', textAlign: 'center', fontWeight: 700, color: '#F0F4F8', fontSize: '15px' }}>{stats.total}</td>
                          <td style={{ padding: '14px', textAlign: 'center' }}>
                            <span style={{ padding: '3px 10px', borderRadius: '6px', fontSize: '12px', fontWeight: 700, background: 'rgba(34,197,94,0.15)', color: '#22C55E' }}>{stats.active}</span>
                          </td>
                          <td style={{ padding: '14px', textAlign: 'center', color: '#9CA3AF', fontSize: '14px' }}>{stats.closed}</td>
                          <td style={{ padding: '14px', textAlign: 'center', fontWeight: 700, fontSize: '14px', color: stats.winRate >= 50 ? '#22C55E' : '#EF4444' }}>
                            {stats.winRate.toFixed(1)}%
                          </td>
                          <td style={{ padding: '14px', textAlign: 'center', fontWeight: 700, fontSize: '14px', color: stats.profitFactor >= 1.5 ? '#22C55E' : stats.profitFactor >= 1 ? '#F59E0B' : '#EF4444' }}>
                            {stats.profitFactor === Infinity ? '∞' : stats.profitFactor.toFixed(2)}
                          </td>
                          <td style={{ padding: '14px', textAlign: 'center', fontWeight: 700, fontSize: '14px', color: stats.riskReward >= 1.5 ? '#22C55E' : stats.riskReward >= 1 ? '#F59E0B' : '#EF4444' }}>
                            {stats.riskReward.toFixed(2)}
                          </td>
                          <td style={{ padding: '14px', textAlign: 'center', fontWeight: 700, fontSize: '14px', color: '#EF4444' }}>
                            {stats.maxDDPct.toFixed(2)}%
                          </td>
                        </tr>
                      );
                    })()}
                  </tbody>
                </table>
              </div>
            </div>
          </motion.div>

        </div>
      </main>
    </div>
  );
}

/* ─── Duration Helper ─── */
function getDuration(entry: string, exit?: string | null): string {
  try {
    const start = new Date(entry);
    const end = exit ? new Date(exit) : new Date();
    const diffMs = end.getTime() - start.getTime();
    const mins = Math.floor(diffMs / 60000);
    if (mins < 60) return `${mins}m`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ${mins % 60}m`;
    const days = Math.floor(hrs / 24);
    return `${days}d ${hrs % 24}h`;
  } catch { return '—'; }
}