'use client';

import { useState, useEffect, useCallback } from 'react';
import {
    RefreshCw, Terminal, Activity, TrendingUp, TrendingDown,
    AlertTriangle, CheckCircle2, XCircle, Clock, Cpu
} from 'lucide-react';

interface CoinAnalysis {
    symbol: string;
    regime: string;
    confidence: number;
    action: string;
    sentiment_score: number;
    features: {
        log_return: number;
        volatility: number;
        volume_change: number;
        rsi_norm: number;
        oi_change?: number;
        funding?: number;
    };
}

export default function KernelView() {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [lastRefresh, setLastRefresh] = useState('');
    const [autoRefresh, setAutoRefresh] = useState(true);

    const fetchData = useCallback(async () => {
        try {
            setLoading(true);
            const res = await fetch('/api/bot-state', { cache: 'no-store' });
            if (res.ok) {
                const d = await res.json();
                setData(d);
                setLastRefresh(new Date().toLocaleTimeString());
            }
        } catch (e) {
            console.error('Kernel fetch error:', e);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData();
        if (autoRefresh) {
            const interval = setInterval(fetchData, 10000);
            return () => clearInterval(interval);
        }
    }, [fetchData, autoRefresh]);

    const multi = data?.multi || {};
    const state = data?.state || {};
    const coinStates = multi?.coin_states || {};
    const coins = Object.values(coinStates) as CoinAnalysis[];
    const sorted = [...coins].sort((a, b) => (a.symbol || '').localeCompare(b.symbol || ''));

    const eligible = coins.filter(c => (c.action || '').includes('ELIGIBLE'));
    const skipped = coins.filter(c => (c.action || '').includes('SKIP') || (c.action || '').includes('VETO'));
    const lastCycle = state?.last_analysis_time || multi?.last_cycle_time || null;

    const getActionIcon = (action: string) => {
        if (action.includes('ELIGIBLE')) return <CheckCircle2 size={14} className="text-green-400" />;
        if (action.includes('SKIP') || action.includes('VETO')) return <XCircle size={14} className="text-red-400" />;
        return <Clock size={14} className="text-gray-400" />;
    };

    const getActionColor = (action: string) => {
        if (action.includes('ELIGIBLE')) return '#22C55E';
        if (action.includes('SKIP') || action.includes('VETO')) return '#EF4444';
        return '#6B7280';
    };

    const getRegimeColor = (regime: string) => {
        if (regime.includes('BULL')) return '#22C55E';
        if (regime.includes('BEAR')) return '#EF4444';
        if (regime.includes('CHOP') || regime.includes('SIDE')) return '#F59E0B';
        if (regime.includes('CRASH')) return '#DC2626';
        return '#6B7280';
    };

    const fmt = (v: any, d = 4) => v != null ? Number(v).toFixed(d) : '—';

    return (
        <div className="space-y-6">
            {/* Header Bar */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-purple-500/10">
                        <Terminal size={20} className="text-purple-400" />
                    </div>
                    <div>
                        <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#06B6D4' }}>
                            Engine Kernel
                        </h2>
                        <p className="text-xs text-gray-500">
                            Real-time HMM analysis cycle · Trade eligibility decisions
                        </p>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={autoRefresh}
                            onChange={(e) => setAutoRefresh(e.target.checked)}
                            className="rounded"
                        />
                        Auto-refresh (10s)
                    </label>
                    <button
                        onClick={fetchData}
                        className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 text-xs transition"
                    >
                        <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
                        Refresh
                    </button>
                </div>
            </div>

            {/* Status Summary Cards */}
            <div className="grid grid-cols-4 gap-3">
                <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Cpu size={14} className="text-cyan-400" />
                        <span className="text-xs text-gray-400 uppercase tracking-wider">Coins Tracked</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{coins.length}</p>
                </div>
                <div className="bg-white/5 border border-green-500/20 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <CheckCircle2 size={14} className="text-green-400" />
                        <span className="text-xs text-gray-400 uppercase tracking-wider">Eligible</span>
                    </div>
                    <p className="text-2xl font-bold text-green-400">{eligible.length}</p>
                </div>
                <div className="bg-white/5 border border-red-500/20 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <XCircle size={14} className="text-red-400" />
                        <span className="text-xs text-gray-400 uppercase tracking-wider">Skipped</span>
                    </div>
                    <p className="text-2xl font-bold text-red-400">{skipped.length}</p>
                </div>
                <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Clock size={14} className="text-amber-400" />
                        <span className="text-xs text-gray-400 uppercase tracking-wider">Last Cycle</span>
                    </div>
                    <p className="text-sm font-bold text-white mt-1">
                        {lastRefresh || '—'}
                    </p>
                </div>
            </div>

            {/* Per-Coin Analysis Table */}
            {coins.length === 0 ? (
                <div className="bg-white/5 border border-white/10 rounded-xl p-12 text-center">
                    <Cpu size={32} className="text-gray-500 mx-auto mb-3" />
                    <p className="text-gray-400 text-sm">
                        No engine data available. Make sure the Python engine is running.
                    </p>
                    <p className="text-gray-500 text-xs mt-1">
                        Run: <code className="bg-white/10 px-2 py-0.5 rounded">python main.py</code>
                    </p>
                </div>
            ) : (
                <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', minWidth: '900px', borderCollapse: 'collapse', fontSize: '12px' }}>
                            <thead>
                                <tr style={{ borderBottom: '2px solid rgba(255,255,255,0.08)' }}>
                                    {['Coin', 'Regime', 'Confidence', 'Action', 'Reason', 'Log Ret', 'Volatility', 'Vol Chg', 'RSI', 'Sentiment'].map(h => (
                                        <th key={h} style={{
                                            padding: '12px 14px',
                                            textAlign: h === 'Coin' ? 'left' : 'center',
                                            fontSize: '10px', fontWeight: 600,
                                            textTransform: 'uppercase',
                                            letterSpacing: '1px',
                                            color: '#6B7280',
                                            background: 'rgba(17, 24, 39, 0.95)',
                                            position: 'sticky' as const, top: 0,
                                        }}>{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {sorted.map((c) => {
                                    const action = (c.action || '').replace(/_/g, ' ');
                                    const regime = c.regime || 'WAITING';
                                    const conf = c.confidence != null
                                        ? (c.confidence <= 1 ? c.confidence * 100 : c.confidence) : 0;
                                    const f = c.features || {} as any;

                                    // Derive skip reason
                                    let reason = '';
                                    if (action.includes('VETO')) {
                                        if (regime.includes('BEAR')) reason = 'Bearish regime';
                                        else if (regime.includes('CHOP') || regime.includes('SIDE')) reason = 'Choppy market';
                                        else if (regime.includes('CRASH')) reason = 'Crash detected';
                                        else reason = 'Regime veto';
                                    } else if (action.includes('SKIP')) {
                                        if (conf < 60) reason = `Low confidence (${conf.toFixed(0)}%)`;
                                        else if ((f.volatility || 0) > 0.05) reason = 'High volatility';
                                        else reason = 'Below threshold';
                                    } else if (action.includes('ELIGIBLE')) {
                                        reason = `✓ ${regime} @ ${conf.toFixed(0)}%`;
                                    } else {
                                        reason = 'Awaiting analysis';
                                    }

                                    const regBg = regime.includes('BULL') ? 'rgba(34,197,94,0.15)'
                                        : regime.includes('BEAR') ? 'rgba(239,68,68,0.15)'
                                            : regime.includes('CHOP') || regime.includes('SIDE') ? 'rgba(245,158,11,0.15)'
                                                : 'rgba(107,114,128,0.15)';

                                    return (
                                        <tr key={c.symbol} style={{
                                            borderBottom: '1px solid rgba(255,255,255,0.04)',
                                            background: action.includes('ELIGIBLE') ? 'rgba(34,197,94,0.03)' : 'transparent',
                                        }}>
                                            <td style={{ padding: '12px 14px', fontWeight: 700, color: '#F0F4F8' }}>
                                                {(c.symbol || '').replace('USDT', '')}
                                            </td>
                                            <td style={{ padding: '12px 14px', textAlign: 'center' }}>
                                                <span style={{
                                                    background: regBg, color: getRegimeColor(regime),
                                                    padding: '3px 10px', borderRadius: '10px',
                                                    fontSize: '10px', fontWeight: 700,
                                                }}>{regime}</span>
                                            </td>
                                            <td style={{
                                                padding: '12px 14px', textAlign: 'center',
                                                fontWeight: 600,
                                                color: conf > 70 ? '#22C55E' : conf > 50 ? '#F59E0B' : '#6B7280',
                                            }}>{conf.toFixed(1)}%</td>
                                            <td style={{ padding: '12px 14px', textAlign: 'center' }}>
                                                <span style={{
                                                    display: 'inline-flex', alignItems: 'center', gap: '4px',
                                                    fontWeight: 700, fontSize: '10px',
                                                    color: getActionColor(action),
                                                }}>
                                                    {getActionIcon(action)}
                                                    {action || '—'}
                                                </span>
                                            </td>
                                            <td style={{
                                                padding: '12px 14px', textAlign: 'center',
                                                fontSize: '11px', color: '#9CA3AF',
                                            }}>{reason}</td>
                                            <td style={{
                                                padding: '12px 14px', textAlign: 'center',
                                                color: (f.log_return || 0) > 0 ? '#22C55E' : (f.log_return || 0) < 0 ? '#EF4444' : '#6B7280',
                                            }}>{fmt(f.log_return)}</td>
                                            <td style={{
                                                padding: '12px 14px', textAlign: 'center',
                                                color: (f.volatility || 0) > 0.02 ? '#F59E0B' : '#6B7280',
                                            }}>{fmt(f.volatility)}</td>
                                            <td style={{
                                                padding: '12px 14px', textAlign: 'center',
                                                color: (f.volume_change || 0) > 0 ? '#22C55E' : '#EF4444',
                                            }}>{fmt(f.volume_change, 2)}</td>
                                            <td style={{
                                                padding: '12px 14px', textAlign: 'center',
                                                color: (f.rsi_norm || 0) > 0.8 ? '#EF4444' : (f.rsi_norm || 0) < 0.2 ? '#22C55E' : '#6B7280',
                                            }}>{fmt(f.rsi_norm, 2)}</td>
                                            <td style={{
                                                padding: '12px 14px', textAlign: 'center',
                                                color: (c.sentiment_score || 0) > 0 ? '#22C55E' : (c.sentiment_score || 0) < 0 ? '#EF4444' : '#6B7280',
                                            }}>{fmt(c.sentiment_score, 2)}</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Engine State JSON (collapsible) */}
            <details className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                <summary className="px-5 py-3 cursor-pointer text-sm text-gray-400 hover:text-gray-300 transition">
                    <span className="ml-1">Raw Engine State (JSON)</span>
                </summary>
                <pre className="px-5 py-4 text-xs text-gray-500 overflow-auto max-h-80 border-t border-white/5" style={{ fontFamily: 'monospace' }}>
                    {JSON.stringify({ state, multi: { ...multi, coin_states: '...' } }, null, 2)}
                </pre>
            </details>
        </div>
    );
}
