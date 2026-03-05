'use client';

import { useState, useEffect, useCallback } from 'react';
import { Header } from '@/components/header';
import { BotCard } from '@/components/bot-card';
import {
  Plus, Trash2, Shield, TrendingUp, FlaskConical, Play, Rocket,
  ChevronDown, ChevronUp, Power, PowerOff, Zap, Activity, Radio
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSession } from 'next-auth/react';

/* ═══ Bot Model Definitions ═══ */
const BOT_MODELS = [
  {
    id: 'standard',
    name: 'Standard',
    color: '#22C55E',
    description: 'Balanced risk-reward, full HMM signals',
    badge: '⚡',
  },
  {
    id: 'conservative',
    name: 'Conservative',
    color: '#0EA5E9',
    description: 'Lower risk, tighter stops, moderate leverage',
    badge: '🛡️',
  },
];

/* ═══ Strategy Lab Default Config ═══ */
const DEFAULT_LAB = {
  // Portfolio
  balance: 1000,
  capitalPerTrade: 100,
  coins: 'BTCUSDT, ETHUSDT, SOLUSDT',
  // HMM
  hmmStates: 3,
  lookback: 120,
  // Timeframes
  tfExecution: '5m',
  tfPrimary: '1h',
  tfMacro: '4h',
  // Leverage Tiers
  highConf: 75, highLev: 5,
  medConf: 60, medLev: 3,
  lowConf: 50, lowLev: 2,
  // Risk
  slMultiplier: 0.8,
  tpMultiplier: 1.0,
  maxLossPct: 15,
  killSwitchPct: 25,
  // Trailing
  trailingSL: true,
  trailingTP: false,
  capitalProtect: true,
  capitalProtectPct: 10,
};

interface BotsClientProps { bots: any[]; }

export function BotsClient({ bots: initialBots }: BotsClientProps) {
  const { data: session } = useSession();
  const isAdmin = (session?.user as any)?.role === 'admin';
  const [mounted, setMounted] = useState(false);
  const [showDeployModal, setShowDeployModal] = useState(false);
  const [bots, setBots] = useState(initialBots);
  const [loading, setLoading] = useState(false);
  const [engineOn, setEngineOn] = useState(false);
  const [engineLoading, setEngineLoading] = useState(false);

  // Deploy modal state
  const [deployModel, setDeployModel] = useState('standard');
  const [deployExchange, setDeployExchange] = useState('binance');
  const [deployMode, setDeployMode] = useState('paper');
  const [deployMaxTrades, setDeployMaxTrades] = useState(25);
  const [deployCapitalPerTrade, setDeployCapitalPerTrade] = useState(100);

  // Strategy Lab
  const [labOpen, setLabOpen] = useState(false);
  const [labConfig, setLabConfig] = useState(DEFAULT_LAB);
  const [labRunning, setLabRunning] = useState(false);
  const [labResults, setLabResults] = useState<any>(null);

  useEffect(() => { setMounted(true); }, []);

  // Live active trade count from bot-state
  const [liveTradeCount, setLiveTradeCount] = useState(0);
  const fetchLiveCount = useCallback(async () => {
    try {
      const res = await fetch('/api/bot-state', { cache: 'no-store' });
      if (res.ok) {
        const d = await res.json();
        const trades = d?.tradebook?.trades || [];
        setLiveTradeCount(trades.filter((t: any) => (t.status || '').toUpperCase() === 'ACTIVE').length);
      }
    } catch { /* silent */ }
  }, []);
  useEffect(() => {
    fetchLiveCount();
    const timer = setInterval(fetchLiveCount, 15000);
    return () => clearInterval(timer);
  }, [fetchLiveCount]);

  // Check engine status on load
  useEffect(() => {
    checkEngineStatus();
  }, []);

  const checkEngineStatus = async () => {
    try {
      const res = await fetch('/api/admin/orchestrator/health');
      setEngineOn(res.ok);
    } catch {
      setEngineOn(false);
    }
  };

  const toggleEngine = async () => {
    setEngineLoading(true);
    try {
      // Just toggle local state for now — will connect to orchestrator later
      setEngineOn(!engineOn);
    } catch (error) {
      console.error('Engine toggle error:', error);
    }
    setEngineLoading(false);
  };

  const handleBotToggle = async (botId: string, currentStatus: boolean) => {
    try {
      const res = await fetch('/api/bots/toggle', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ botId, isActive: !currentStatus }),
      });
      if (res.ok) window.location.reload();
    } catch (error) { console.error('Error toggling bot:', error); }
  };

  const handleDeployBot = async () => {
    setLoading(true);
    try {
      const selectedModel = BOT_MODELS.find(m => m.id === deployModel);
      const botName = `Sentinel Marshal — ${selectedModel?.name || 'Standard'}`;
      const res = await fetch('/api/bots/create', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: botName,
          exchange: deployExchange,
          mode: deployMode,
          maxTrades: deployMaxTrades,
          capitalPerTrade: deployCapitalPerTrade,
        }),
      });
      if (res.ok) {
        setShowDeployModal(false);
        window.location.reload();
      } else {
        const data = await res.json();
        alert(data.error || 'Failed to deploy bot');
      }
    } catch (error) { console.error('Error deploying bot:', error); }
    finally { setLoading(false); }
  };

  const handleDeleteBot = async (botId: string) => {
    if (!confirm('Are you sure you want to delete this bot?')) return;
    try {
      const res = await fetch('/api/bots/delete', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ botId }),
      });
      if (res.ok) window.location.reload();
    } catch (error) { console.error('Error deleting bot:', error); }
  };

  const getModel = (botName: string) =>
    BOT_MODELS.find(m => botName?.toLowerCase().includes(m.id)) || BOT_MODELS[0];

  const updateLab = (patch: Partial<typeof DEFAULT_LAB>) => setLabConfig(prev => ({ ...prev, ...patch }));

  /* ── Run Backtest (simulation for now) ── */
  const runBacktest = async () => {
    setLabRunning(true);
    setLabResults(null);
    await new Promise(r => setTimeout(r, 2500));
    const trades = Math.floor(Math.random() * 40 + 20);
    const wins = Math.floor(trades * (0.45 + Math.random() * 0.25));
    const totalPnl = ((Math.random() * 2 - 0.5) * labConfig.balance * 0.3);
    const maxDD = Math.random() * 15 + 5;
    setLabResults({
      trades, wins, losses: trades - wins,
      winRate: ((wins / trades) * 100).toFixed(1),
      totalPnl: totalPnl.toFixed(2),
      maxDrawdown: maxDD.toFixed(1),
      sharpe: (Math.random() * 2 + 0.3).toFixed(2),
      profitFactor: (Math.random() * 2 + 0.5).toFixed(2),
    });
    setLabRunning(false);
  };

  /* ── Deploy from Lab Results ── */
  const deployFromLab = () => {
    setShowDeployModal(true);
  };

  if (!mounted) return null;

  /* ── Helpers ── */
  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '8px 12px', borderRadius: '10px',
    border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.04)',
    color: '#D1D5DB', fontSize: '13px',
  };
  const labelStyle: React.CSSProperties = {
    fontSize: '11px', fontWeight: 600, color: '#6B7280', marginBottom: '4px', display: 'block',
  };
  const groupLabel: React.CSSProperties = {
    fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1.5px',
    color: '#4B5563', marginBottom: '10px', marginTop: '16px',
  };

  return (
    <div className="min-h-screen">
      <Header />
      <main className="pt-24 pb-12 px-4">
        <div className="max-w-7xl mx-auto">

          {/* ═══ ENGINE STATUS BAR (visible to admin, read-only for users) ═══ */}
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '14px 20px', borderRadius: '14px', marginBottom: '24px',
              background: engineOn
                ? 'linear-gradient(135deg, rgba(34,197,94,0.08), rgba(16,185,129,0.04))'
                : 'linear-gradient(135deg, rgba(239,68,68,0.06), rgba(249,115,22,0.03))',
              border: `1px solid ${engineOn ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.15)'}`,
            }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: '10px', height: '10px', borderRadius: '50%',
                background: engineOn ? '#22C55E' : '#EF4444',
                boxShadow: engineOn ? '0 0 12px rgba(34,197,94,0.6)' : 'none',
                animation: engineOn ? 'pulse 2s infinite' : 'none',
              }} />
              <div>
                <div style={{ fontSize: '14px', fontWeight: 700, color: '#E5E7EB' }}>
                  {engineOn ? '🟢 Engine Online' : '🔴 Engine Offline'}
                </div>
                <div style={{ fontSize: '11px', color: '#6B7280' }}>
                  {engineOn
                    ? 'Sentinel Marshal — HMM analysis active, ready for deployments'
                    : 'Engine is not running. Start the engine to enable bot deployment.'}
                </div>
              </div>
            </div>
            <button
              onClick={toggleEngine}
              disabled={engineLoading}
              style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                padding: '8px 16px', borderRadius: '10px', border: 'none',
                background: engineOn
                  ? 'rgba(239,68,68,0.15)'
                  : 'linear-gradient(135deg, #22C55E, #16A34A)',
                color: engineOn ? '#EF4444' : '#fff',
                fontSize: '12px', fontWeight: 700, cursor: 'pointer',
                transition: 'all 0.2s',
              }}>
              {engineOn ? <PowerOff size={14} /> : <Power size={14} />}
              {engineLoading ? 'Processing...' : engineOn ? 'Stop Engine' : 'Start Engine'}
            </button>
          </motion.div>

          {/* ═══ SECTION 1: BOT MANAGEMENT ═══ */}
          <div className="flex items-center justify-between mb-6">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
              <h1 className="text-3xl font-bold mb-1">
                <span className="text-gradient">Bot Management</span>
              </h1>
              <p className="text-sm text-[var(--color-text-secondary)]">
                Deploy and manage your automated trading bots
              </p>
            </motion.div>
            <button
              onClick={() => setShowDeployModal(true)}
              style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: '10px 20px', borderRadius: '12px', border: 'none',
                background: 'linear-gradient(135deg, var(--color-primary), var(--color-primary-dark, #0284c7))',
                color: '#fff',
                fontSize: '14px', fontWeight: 600, cursor: 'pointer',
                transition: 'all 0.2s',
              }}>
              <Rocket size={16} />
              Deploy Bot
            </button>
          </div>

          {/* ── Engine offline info (informational only, doesn't block bot creation) ── */}
          {!engineOn && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              style={{
                background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)',
                borderRadius: '12px', padding: '16px 20px', marginBottom: '24px',
                display: 'flex', alignItems: 'center', gap: '12px',
                fontSize: '13px', color: '#F59E0B',
              }}>
              <Activity size={18} />
              <span>Engine is offline. You can still create bots — they will start receiving trades once the engine comes online.</span>
            </motion.div>
          )}

          {/* ── Sentinel Marshal Card (shows when no bots yet) ── */}
          {bots.length === 0 && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              style={{
                background: 'linear-gradient(135deg, rgba(17,24,39,0.8), rgba(30,41,59,0.5))',
                backdropFilter: 'blur(12px)',
                border: '1px solid rgba(34,197,94,0.15)',
                borderRadius: '16px', padding: '32px', textAlign: 'center', marginBottom: '32px',
              }}>
              <div style={{
                width: '56px', height: '56px', borderRadius: '14px',
                background: 'linear-gradient(135deg, rgba(34,197,94,0.15), rgba(16,185,129,0.1))',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                margin: '0 auto 16px',
              }}>
                <Shield size={28} color="#22C55E" />
              </div>
              <h3 style={{ fontSize: '20px', fontWeight: 700, marginBottom: '6px', color: '#E5E7EB' }}>
                Sentinel Marshal
              </h3>
              <p style={{ fontSize: '12px', color: '#6B7280', marginBottom: '6px' }}>
                HMM-Powered Crypto Trading Engine
              </p>
              <div style={{
                display: 'inline-flex', alignItems: 'center', gap: '6px',
                padding: '4px 12px', borderRadius: '20px', fontSize: '11px',
                background: engineOn ? 'rgba(34,197,94,0.1)' : 'rgba(245,158,11,0.1)',
                color: engineOn ? '#22C55E' : '#F59E0B', fontWeight: 600,
              }}>
                <Activity size={12} /> {engineOn ? 'Engine Ready — Deploy your first bot' : 'Deploy a bot — it will activate when engine starts'}
              </div>
              <div style={{ marginTop: '20px' }}>
                <button
                  onClick={() => setShowDeployModal(true)}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: '8px',
                    padding: '12px 28px', borderRadius: '12px', border: 'none',
                    background: 'linear-gradient(135deg, #22C55E, #16A34A)',
                    color: '#fff', fontSize: '14px', fontWeight: 700, cursor: 'pointer',
                    transition: 'all 0.2s',
                  }}>
                  <Rocket size={16} /> Deploy Sentinel Marshal
                </button>
              </div>
            </motion.div>
          )}

          {/* Engine OFF empty state removed — bots can be created anytime */}

          {/* ── Deployed Bots Grid ── */}
          {bots && bots.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
              {bots.map((bot) => {
                const model = getModel(bot?.name || '');
                return (
                  <div key={bot?.id} className="relative">
                    <div style={{
                      position: 'absolute', top: '12px', left: '12px', zIndex: 10,
                      padding: '3px 10px', borderRadius: '8px', fontSize: '10px', fontWeight: 700,
                      color: model.color, background: model.color + '22', letterSpacing: '0.5px',
                    }}>
                      {model.badge} {model.name}
                    </div>
                    <BotCard bot={bot} onToggle={handleBotToggle} liveTradeCount={liveTradeCount} />
                    <button onClick={() => handleDeleteBot(bot?.id)}
                      title="Delete bot"
                      style={{
                        position: 'absolute', top: '56px', right: '16px', zIndex: 10,
                        padding: '6px', borderRadius: '8px',
                        background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)',
                        color: '#EF4444', cursor: 'pointer',
                        transition: 'all 0.2s',
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239,68,68,0.3)'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(239,68,68,0.15)'; }}
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })}
            </div>
          )}

          {/* ═══ SECTION 2: STRATEGY LAB ═══ */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
            <div style={{
              background: 'rgba(17, 24, 39, 0.7)', backdropFilter: 'blur(12px)',
              border: '1px solid rgba(139, 92, 246, 0.15)', borderRadius: '16px',
              overflow: 'hidden',
            }}>
              {/* Lab Header */}
              <div
                onClick={() => setLabOpen(!labOpen)}
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '18px 24px', cursor: 'pointer',
                  borderBottom: labOpen ? '1px solid rgba(255,255,255,0.06)' : 'none',
                }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{
                    width: '36px', height: '36px', borderRadius: '10px',
                    background: 'rgba(139, 92, 246, 0.15)', display: 'flex',
                    alignItems: 'center', justifyContent: 'center',
                  }}>
                    <FlaskConical size={18} color="#8B5CF6" />
                  </div>
                  <div>
                    <div style={{ fontSize: '16px', fontWeight: 700, color: '#06B6D4' }}>Strategy Lab</div>
                    <div style={{ fontSize: '11px', color: '#6B7280' }}>
                      Experiment with parameters · Run backtests · Deploy winning strategies
                    </div>
                  </div>
                </div>
                {labOpen ? <ChevronUp size={18} color="#6B7280" /> : <ChevronDown size={18} color="#6B7280" />}
              </div>

              {/* Lab Content */}
              <AnimatePresence>
                {labOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.2 }}
                    style={{ overflow: 'hidden' }}>
                    <div style={{ padding: '20px 24px' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px' }}>

                        {/* Column 1: Portfolio & HMM */}
                        <div>
                          <div style={groupLabel}>Portfolio</div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            <div>
                              <span style={labelStyle}>Starting Balance ($)</span>
                              <input type="number" value={labConfig.balance} onChange={e => updateLab({ balance: +e.target.value })} style={inputStyle} />
                            </div>
                            <div>
                              <span style={labelStyle}>Capital per Trade ($)</span>
                              <input type="number" value={labConfig.capitalPerTrade} onChange={e => updateLab({ capitalPerTrade: +e.target.value })} style={inputStyle} />
                            </div>
                            <div>
                              <span style={labelStyle}>Coins (comma-separated)</span>
                              <input value={labConfig.coins} onChange={e => updateLab({ coins: e.target.value })} style={inputStyle} />
                            </div>
                          </div>

                          <div style={groupLabel}>HMM Brain</div>
                          <div style={{ display: 'flex', gap: '8px' }}>
                            <div style={{ flex: 1 }}>
                              <span style={labelStyle}>States</span>
                              <input type="number" value={labConfig.hmmStates} onChange={e => updateLab({ hmmStates: +e.target.value })} style={inputStyle} />
                            </div>
                            <div style={{ flex: 1 }}>
                              <span style={labelStyle}>Lookback</span>
                              <input type="number" value={labConfig.lookback} onChange={e => updateLab({ lookback: +e.target.value })} style={inputStyle} />
                            </div>
                          </div>

                          <div style={groupLabel}>Timeframes</div>
                          <div style={{ display: 'flex', gap: '8px' }}>
                            {(['tfExecution', 'tfPrimary', 'tfMacro'] as const).map((key, i) => (
                              <div key={key} style={{ flex: 1 }}>
                                <span style={labelStyle}>{['Execution', 'Primary', 'Macro'][i]}</span>
                                <select value={labConfig[key]} onChange={e => updateLab({ [key]: e.target.value } as any)} style={inputStyle}>
                                  {['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d'].map(tf => (
                                    <option key={tf} value={tf}>{tf}</option>
                                  ))}
                                </select>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Column 2: Leverage & Risk */}
                        <div>
                          <div style={groupLabel}>Leverage Tiers</div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            {([
                              { label: 'High', confKey: 'highConf', levKey: 'highLev', dot: '#22C55E' },
                              { label: 'Med', confKey: 'medConf', levKey: 'medLev', dot: '#F59E0B' },
                              { label: 'Low', confKey: 'lowConf', levKey: 'lowLev', dot: '#EF4444' },
                            ] as const).map(tier => (
                              <div key={tier.label} style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: tier.dot, flexShrink: 0 }} />
                                <span style={{ fontSize: '11px', color: '#9CA3AF', width: '32px' }}>{tier.label}</span>
                                <div style={{ flex: 1 }}>
                                  <input type="number" value={labConfig[tier.confKey]} onChange={e => updateLab({ [tier.confKey]: +e.target.value } as any)}
                                    style={{ ...inputStyle, padding: '5px 8px', fontSize: '12px' }} placeholder="Conf%" />
                                </div>
                                <div style={{ flex: 1 }}>
                                  <input type="number" value={labConfig[tier.levKey]} onChange={e => updateLab({ [tier.levKey]: +e.target.value } as any)}
                                    style={{ ...inputStyle, padding: '5px 8px', fontSize: '12px' }} placeholder="Lev×" />
                                </div>
                              </div>
                            ))}
                          </div>

                          <div style={groupLabel}>Risk Management</div>
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                            <div><span style={labelStyle}>SL (ATR×)</span><input type="number" step="0.1" value={labConfig.slMultiplier} onChange={e => updateLab({ slMultiplier: +e.target.value })} style={inputStyle} /></div>
                            <div><span style={labelStyle}>TP (ATR×)</span><input type="number" step="0.1" value={labConfig.tpMultiplier} onChange={e => updateLab({ tpMultiplier: +e.target.value })} style={inputStyle} /></div>
                            <div><span style={labelStyle}>Max Loss %</span><input type="number" value={labConfig.maxLossPct} onChange={e => updateLab({ maxLossPct: +e.target.value })} style={inputStyle} /></div>
                            <div><span style={labelStyle}>Kill Switch %</span><input type="number" value={labConfig.killSwitchPct} onChange={e => updateLab({ killSwitchPct: +e.target.value })} style={inputStyle} /></div>
                          </div>
                        </div>

                        {/* Column 3: Trailing + Run */}
                        <div>
                          <div style={groupLabel}>Trailing Settings</div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {([
                              { label: 'Trailing SL', key: 'trailingSL' as const },
                              { label: 'Trailing TP', key: 'trailingTP' as const },
                              { label: 'Capital Protect', key: 'capitalProtect' as const },
                            ]).map(toggle => (
                              <div key={toggle.key} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <span style={{ fontSize: '12px', color: '#9CA3AF' }}>{toggle.label}</span>
                                <div onClick={() => updateLab({ [toggle.key]: !labConfig[toggle.key] } as any)}
                                  style={{
                                    width: '36px', height: '20px', borderRadius: '10px', cursor: 'pointer',
                                    background: labConfig[toggle.key] ? '#8B5CF6' : 'rgba(255,255,255,0.1)',
                                    position: 'relative', transition: 'all 0.2s',
                                  }}>
                                  <div style={{
                                    width: '16px', height: '16px', borderRadius: '50%', background: '#fff',
                                    position: 'absolute', top: '2px', transition: 'all 0.2s',
                                    left: labConfig[toggle.key] ? '18px' : '2px',
                                  }} />
                                </div>
                              </div>
                            ))}
                            {labConfig.capitalProtect && (
                              <div>
                                <span style={labelStyle}>Protect at % profit</span>
                                <input type="number" value={labConfig.capitalProtectPct} onChange={e => updateLab({ capitalProtectPct: +e.target.value })} style={inputStyle} />
                              </div>
                            )}
                          </div>

                          {/* Run Button */}
                          <div style={{ marginTop: '24px' }}>
                            <button onClick={runBacktest} disabled={labRunning}
                              style={{
                                width: '100%', padding: '12px', borderRadius: '12px', border: 'none',
                                background: labRunning ? 'rgba(139, 92, 246, 0.3)' : 'linear-gradient(135deg, #8B5CF6, #6D28D9)',
                                color: '#fff', fontSize: '14px', fontWeight: 700, cursor: labRunning ? 'wait' : 'pointer',
                                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
                                transition: 'all 0.2s',
                              }}>
                              {labRunning ? (
                                <>
                                  <div style={{ width: '16px', height: '16px', border: '2px solid rgba(255,255,255,0.3)', borderTop: '2px solid #fff', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                                  Running Backtest...
                                </>
                              ) : (
                                <><Play size={16} /> Run Backtest</>
                              )}
                            </button>
                          </div>
                        </div>
                      </div>

                      {/* ── Lab Results ── */}
                      <AnimatePresence>
                        {labResults && (
                          <motion.div
                            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                            style={{ marginTop: '20px', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '20px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
                              <div style={{ fontSize: '13px', fontWeight: 700, color: '#E5E7EB' }}>📊 Backtest Results</div>
                              <button onClick={deployFromLab} disabled={loading}
                                style={{
                                  display: 'flex', alignItems: 'center', gap: '6px',
                                  padding: '8px 16px', borderRadius: '10px', border: 'none',
                                  background: 'linear-gradient(135deg, #22C55E, #16A34A)',
                                  color: '#fff',
                                  fontSize: '12px', fontWeight: 700,
                                  cursor: loading ? 'wait' : 'pointer',
                                }}>
                                <Rocket size={14} />
                                {loading ? 'Deploying...' : 'Deploy This Strategy'}
                              </button>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                              {[
                                { label: 'Total Trades', value: labResults.trades, color: '#E5E7EB' },
                                { label: 'Win Rate', value: labResults.winRate + '%', color: parseFloat(labResults.winRate) >= 50 ? '#22C55E' : '#EF4444' },
                                { label: 'Total P&L', value: '$' + labResults.totalPnl, color: parseFloat(labResults.totalPnl) >= 0 ? '#22C55E' : '#EF4444' },
                                { label: 'Sharpe Ratio', value: labResults.sharpe, color: parseFloat(labResults.sharpe) >= 1 ? '#22C55E' : '#F59E0B' },
                                { label: 'Wins / Losses', value: `${labResults.wins} / ${labResults.losses}`, color: '#D1D5DB' },
                                { label: 'Profit Factor', value: labResults.profitFactor, color: parseFloat(labResults.profitFactor) >= 1 ? '#22C55E' : '#EF4444' },
                                { label: 'Max Drawdown', value: labResults.maxDrawdown + '%', color: '#EF4444' },
                                { label: 'Strategy', value: labConfig.slMultiplier + ' / ' + labConfig.tpMultiplier, color: '#8B5CF6' },
                              ].map((s, i) => (
                                <div key={i} style={{
                                  padding: '12px', borderRadius: '10px',
                                  background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)',
                                }}>
                                  <div style={{ fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: '#6B7280', marginBottom: '4px' }}>{s.label}</div>
                                  <div style={{ fontSize: '18px', fontWeight: 700, color: s.color }}>{s.value}</div>
                                </div>
                              ))}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>

        </div>
      </main >

      {/* ═══ DEPLOY BOT MODAL ═══ */}
      <AnimatePresence>
        {
          showDeployModal && (
            <div
              className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
              onClick={(e) => { if (e.target === e.currentTarget) setShowDeployModal(false); }}
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.9, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9, y: 20 }}
                style={{
                  background: 'linear-gradient(135deg, rgba(17,24,39,0.98), rgba(30,41,59,0.95))',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '20px', padding: '32px', maxWidth: '520px', width: '100%',
                }}
              >
                {/* Modal Header */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                  <div style={{
                    width: '44px', height: '44px', borderRadius: '12px',
                    background: 'linear-gradient(135deg, rgba(34,197,94,0.2), rgba(16,185,129,0.1))',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <Rocket size={22} color="#22C55E" />
                  </div>
                  <div>
                    <h2 style={{ fontSize: '20px', fontWeight: 700, color: '#E5E7EB' }}>Deploy Sentinel Marshal</h2>
                    <p style={{ fontSize: '12px', color: '#6B7280' }}>Configure and launch your trading bot</p>
                  </div>
                </div>

                {/* Step 1: Select Model */}
                <div style={{ marginBottom: '20px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '1.5px', marginBottom: '10px' }}>
                    1. Select Model
                  </div>
                  <div style={{ display: 'flex', gap: '12px' }}>
                    {BOT_MODELS.map(model => (
                      <div key={model.id}
                        onClick={() => setDeployModel(model.id)}
                        style={{
                          flex: 1, padding: '16px', borderRadius: '14px', cursor: 'pointer',
                          background: deployModel === model.id ? model.color + '12' : 'rgba(255,255,255,0.03)',
                          border: `2px solid ${deployModel === model.id ? model.color : 'rgba(255,255,255,0.06)'}`,
                          transition: 'all 0.2s', textAlign: 'center',
                        }}>
                        <div style={{ fontSize: '28px', marginBottom: '8px' }}>{model.badge}</div>
                        <div style={{ fontSize: '14px', fontWeight: 700, color: deployModel === model.id ? model.color : '#9CA3AF' }}>{model.name}</div>
                        <div style={{ fontSize: '10px', color: '#6B7280', marginTop: '4px' }}>{model.description}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Step 2: Select Exchange */}
                <div style={{ marginBottom: '20px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '1.5px', marginBottom: '10px' }}>
                    2. Select Exchange
                  </div>
                  <div style={{ display: 'flex', gap: '12px' }}>
                    {[
                      { id: 'binance', name: 'Binance', icon: '🔶', desc: 'Largest crypto exchange' },
                      { id: 'coindcx', name: 'CoinDCX', icon: '🇮🇳', desc: 'India\'s crypto exchange' },
                    ].map(ex => (
                      <div key={ex.id}
                        onClick={() => setDeployExchange(ex.id)}
                        style={{
                          flex: 1, padding: '14px', borderRadius: '12px', cursor: 'pointer',
                          background: deployExchange === ex.id ? 'rgba(14,165,233,0.1)' : 'rgba(255,255,255,0.03)',
                          border: `2px solid ${deployExchange === ex.id ? '#0EA5E9' : 'rgba(255,255,255,0.06)'}`,
                          transition: 'all 0.2s', textAlign: 'center',
                        }}>
                        <div style={{ fontSize: '24px', marginBottom: '6px' }}>{ex.icon}</div>
                        <div style={{ fontSize: '13px', fontWeight: 700, color: deployExchange === ex.id ? '#0EA5E9' : '#9CA3AF' }}>{ex.name}</div>
                        <div style={{ fontSize: '10px', color: '#6B7280', marginTop: '2px' }}>{ex.desc}</div>
                      </div>
                    ))}
                  </div>
                  <p style={{ fontSize: '10px', color: '#4B5563', marginTop: '6px' }}>
                    ℹ️ Make sure your API key is configured in Settings for the selected exchange
                  </p>
                </div>

                {/* Step 3: Trading Mode */}
                <div style={{ marginBottom: '24px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '1.5px', marginBottom: '10px' }}>
                    3. Trading Mode
                  </div>
                  <div style={{ display: 'flex', gap: '12px' }}>
                    {[
                      { id: 'paper', name: 'Paper Trading', icon: '📝', desc: 'Simulated trades, no real money', color: '#0EA5E9' },
                      { id: 'live', name: 'Live Trading', icon: '💰', desc: 'Real trades with your capital', color: '#EF4444' },
                    ].map(mode => (
                      <div key={mode.id}
                        onClick={() => setDeployMode(mode.id)}
                        style={{
                          flex: 1, padding: '14px', borderRadius: '12px', cursor: 'pointer',
                          background: deployMode === mode.id ? mode.color + '10' : 'rgba(255,255,255,0.03)',
                          border: `2px solid ${deployMode === mode.id ? mode.color : 'rgba(255,255,255,0.06)'}`,
                          transition: 'all 0.2s', textAlign: 'center',
                        }}>
                        <div style={{ fontSize: '24px', marginBottom: '6px' }}>{mode.icon}</div>
                        <div style={{ fontSize: '13px', fontWeight: 700, color: deployMode === mode.id ? mode.color : '#9CA3AF' }}>{mode.name}</div>
                        <div style={{ fontSize: '10px', color: '#6B7280', marginTop: '2px' }}>{mode.desc}</div>
                      </div>
                    ))}
                  </div>
                  {deployMode === 'live' && (
                    <div style={{
                      marginTop: '8px', padding: '8px 12px', borderRadius: '8px',
                      background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)',
                      fontSize: '11px', color: '#F87171',
                    }}>
                      ⚠️ Live trading uses real capital. Ensure your risk settings are configured in Settings.
                    </div>
                  )}
                </div>

                {/* Step 4: Trade Limits */}
                <div style={{ marginBottom: '24px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '1.5px', marginBottom: '10px' }}>
                    4. Trade Settings
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                    <div>
                      <label style={{ display: 'block', fontSize: '12px', color: '#9CA3AF', marginBottom: '6px', fontWeight: 600 }}>
                        Max Concurrent Trades
                      </label>
                      <input
                        type="number"
                        min={1}
                        max={100}
                        value={deployMaxTrades}
                        onChange={(e) => setDeployMaxTrades(Math.max(1, parseInt(e.target.value) || 1))}
                        style={{
                          width: '100%', padding: '10px 12px', borderRadius: '10px',
                          background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                          color: '#F0F4F8', fontSize: '14px', fontWeight: 600,
                          outline: 'none',
                        }}
                      />
                      <p style={{ fontSize: '10px', color: '#4B5563', marginTop: '4px' }}>
                        Max positions open at the same time
                      </p>
                    </div>
                    <div>
                      <label style={{ display: 'block', fontSize: '12px', color: '#9CA3AF', marginBottom: '6px', fontWeight: 600 }}>
                        Capital Per Trade ($)
                      </label>
                      <input
                        type="number"
                        min={10}
                        max={10000}
                        step={10}
                        value={deployCapitalPerTrade}
                        onChange={(e) => setDeployCapitalPerTrade(Math.max(10, parseInt(e.target.value) || 10))}
                        style={{
                          width: '100%', padding: '10px 12px', borderRadius: '10px',
                          background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                          color: '#F0F4F8', fontSize: '14px', fontWeight: 600,
                          outline: 'none',
                        }}
                      />
                      <p style={{ fontSize: '10px', color: '#4B5563', marginTop: '4px' }}>
                        Amount allocated per trade entry
                      </p>
                    </div>
                  </div>
                  <div style={{
                    marginTop: '10px', padding: '8px 12px', borderRadius: '8px',
                    background: 'rgba(8,145,178,0.08)', border: '1px solid rgba(8,145,178,0.2)',
                    fontSize: '11px', color: '#06B6D4',
                  }}>
                    💡 Max capital exposure: ${deployMaxTrades * deployCapitalPerTrade} ({deployMaxTrades} × ${deployCapitalPerTrade})
                  </div>
                </div>

                {/* Actions */}
                <div style={{ display: 'flex', gap: '12px' }}>
                  <button
                    onClick={() => setShowDeployModal(false)}
                    style={{
                      flex: 1, padding: '12px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)',
                      background: 'rgba(255,255,255,0.05)', color: '#9CA3AF',
                      fontSize: '14px', fontWeight: 600, cursor: 'pointer', transition: 'all 0.2s',
                    }}>
                    Cancel
                  </button>
                  <button
                    onClick={handleDeployBot}
                    disabled={loading}
                    style={{
                      flex: 1, padding: '12px', borderRadius: '12px', border: 'none',
                      background: 'linear-gradient(135deg, #22C55E, #16A34A)',
                      color: '#fff', fontSize: '14px', fontWeight: 700, cursor: 'pointer',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
                      transition: 'all 0.2s', opacity: loading ? 0.6 : 1,
                    }}>
                    <Rocket size={16} />
                    {loading ? 'Deploying...' : 'Deploy Bot'}
                  </button>
                </div>
              </motion.div>
            </div>
          )
        }
      </AnimatePresence >

      <style jsx>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div >
  );
}