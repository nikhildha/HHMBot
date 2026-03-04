'use client';

import { useState, useEffect } from 'react';

const REGIME_MAP: Record<string, { emoji: string; color: string; bgGlow: string }> = {
    'BULLISH': { emoji: '🟢', color: '#22C55E', bgGlow: 'rgba(34, 197, 94, 0.15)' },
    'BEARISH': { emoji: '🔴', color: '#EF4444', bgGlow: 'rgba(239, 68, 68, 0.15)' },
    'SIDEWAYS/CHOP': { emoji: '🟡', color: '#F59E0B', bgGlow: 'rgba(245, 158, 11, 0.15)' },
    'CRASH/PANIC': { emoji: '💀', color: '#DC2626', bgGlow: 'rgba(220, 38, 38, 0.2)' },
    'WAITING': { emoji: '⏳', color: '#F59E0B', bgGlow: 'rgba(245, 158, 11, 0.1)' },
    'SCANNING': { emoji: '🔍', color: '#3B82F6', bgGlow: 'rgba(59, 130, 246, 0.15)' },
    'OFFLINE': { emoji: '⚫', color: '#6B7280', bgGlow: 'rgba(107, 114, 128, 0.1)' },
};

function getRegimeInfo(regime: string) {
    return REGIME_MAP[regime] || REGIME_MAP['WAITING'];
}

interface RegimeCardProps {
    regime: string;
    confidence: number;
    symbol: string;
    macroRegime?: string;
    trend15m?: string;
    coinStates?: Record<string, any>;
}

export function RegimeCard({ regime, confidence, symbol, macroRegime, trend15m, coinStates }: RegimeCardProps) {
    const info = getRegimeInfo(regime);
    let conf = confidence;
    if (conf <= 1) conf *= 100;
    const pct = Math.round(conf);

    let gaugeColor = '#EF4444';
    if (pct >= 85) gaugeColor = '#22C55E';
    else if (pct >= 65) gaugeColor = '#0EA5E9';
    else if (pct >= 50) gaugeColor = '#F59E0B';

    // Live BTC price with fast refresh
    const [btcPrice, setBtcPrice] = useState<number | null>(null);
    const [btcChange, setBtcChange] = useState<number>(0);

    useEffect(() => {
        const fetchBtc = async () => {
            try {
                const res = await fetch('https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT');
                if (res.ok) {
                    const d = await res.json();
                    setBtcPrice(parseFloat(d.lastPrice));
                    setBtcChange(parseFloat(d.priceChangePercent));
                }
            } catch { /* silent */ }
        };
        fetchBtc();
        const timer = setInterval(fetchBtc, 2000);
        return () => clearInterval(timer);
    }, []);

    // Group coins by regime
    const regimeCoins: Record<string, string[]> = { bullish: [], bearish: [], sideways: [], crash: [] };
    if (coinStates) {
        Object.values(coinStates).forEach((c: any) => {
            const r = (c.regime || '').toUpperCase();
            const name = (c.symbol || '').replace('USDT', '');
            if (!name) return;
            if (r.includes('BULL')) regimeCoins.bullish.push(name);
            else if (r.includes('CRASH') || r.includes('PANIC')) regimeCoins.crash.push(name);
            else if (r.includes('BEAR')) regimeCoins.bearish.push(name);
            else if (r.includes('CHOP') || r.includes('SIDE')) regimeCoins.sideways.push(name);
        });
    }

    const categories = [
        { label: 'Bullish', coins: regimeCoins.bullish, color: '#22C55E', emoji: '🟢' },
        { label: 'Bearish', coins: regimeCoins.bearish, color: '#EF4444', emoji: '🔴' },
        { label: 'Sideways', coins: regimeCoins.sideways, color: '#F59E0B', emoji: '🟡' },
        { label: 'Crash', coins: regimeCoins.crash, color: '#DC2626', emoji: '💀' },
    ].filter(c => c.coins.length > 0);

    return (
        <div style={{
            background: 'rgba(17, 24, 39, 0.8)',
            backdropFilter: 'blur(12px)',
            border: `1px solid ${info.color}33`,
            borderRadius: '16px',
            padding: '20px 24px',
            position: 'relative',
            overflow: 'hidden',
        }}>
            {/* Top accent line */}
            <div style={{
                position: 'absolute', top: 0, left: 0, right: 0, height: '3px',
                background: `linear-gradient(90deg, ${info.color}, transparent)`,
            }} />

            <div style={{
                fontSize: '10px', fontWeight: 600, textTransform: 'uppercase' as const,
                letterSpacing: '1.5px', color: '#9CA3AF', marginBottom: '12px',
            }}>BTC Regime</div>

            {/* Regime + Confidence row */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ fontSize: '36px' }}>{info.emoji}</span>
                    <div>
                        <div style={{
                            fontSize: '22px', fontWeight: 700, color: info.color,
                            letterSpacing: '0.5px',
                        }}>{regime}</div>
                    </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                    <div style={{
                        fontSize: '28px', fontWeight: 700, color: gaugeColor,
                    }}>{pct}%</div>
                    <div style={{
                        fontSize: '9px', textTransform: 'uppercase' as const,
                        letterSpacing: '1px', color: '#6B7280',
                    }}>Confidence</div>
                </div>
            </div>

            {/* Live BTC Price */}
            <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px',
                padding: '10px 0', marginBottom: '12px',
                borderTop: '1px solid rgba(255,255,255,0.06)',
                borderBottom: '1px solid rgba(255,255,255,0.06)',
            }}>
                <span style={{ fontSize: '11px', color: '#6B7280', fontWeight: 600, textTransform: 'uppercase' as const, letterSpacing: '1px' }}>BTC</span>
                <span style={{
                    fontSize: '20px', fontWeight: 700, fontFamily: 'monospace',
                    color: '#F0F4F8',
                }}>
                    {btcPrice ? `$${btcPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '...'}
                </span>
                {btcPrice && (
                    <span style={{
                        fontSize: '12px', fontWeight: 700,
                        padding: '2px 8px', borderRadius: '6px',
                        background: btcChange >= 0 ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
                        color: btcChange >= 0 ? '#22C55E' : '#EF4444',
                    }}>
                        {btcChange >= 0 ? '▲' : '▼'} {Math.abs(btcChange).toFixed(2)}%
                    </span>
                )}
            </div>

            {/* Coin categories by regime */}
            {categories.length > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(categories.length, 4)}, 1fr)`, gap: '8px' }}>
                    {categories.map(cat => (
                        <div key={cat.label} style={{
                            padding: '8px', borderRadius: '10px',
                            background: `${cat.color}0D`,
                            border: `1px solid ${cat.color}22`,
                        }}>
                            <div style={{ fontSize: '9px', fontWeight: 700, color: cat.color, textTransform: 'uppercase' as const, letterSpacing: '0.5px', marginBottom: '4px' }}>
                                {cat.emoji} {cat.label}
                            </div>
                            <div style={{ fontSize: '10px', color: '#D1D5DB', lineHeight: '1.5' }}>
                                {cat.coins.join(', ')}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Background glow */}
            <div style={{
                position: 'absolute', inset: 0, zIndex: -1,
                background: `radial-gradient(circle at center, ${info.bgGlow}, transparent 70%)`,
            }} />
        </div>
    );
}

interface PnlCardProps {
    trades: any[];
}

export function PnlCard({ trades }: PnlCardProps) {
    const MAX_CAPITAL = 2500;
    const CAPITAL_PER_TRADE = 100;

    // Categorize trades
    const allTrades = trades || [];

    const paperTrades = allTrades.filter((t: any) => (t.mode || 'paper').toUpperCase() === 'PAPER');
    const liveTrades = allTrades.filter((t: any) => (t.mode || '').toUpperCase() === 'LIVE');

    const calcPnl = (list: any[]) => {
        let realized = 0, unrealized = 0, activeCount = 0;
        list.forEach((t: any) => {
            const status = (t.status || '').toUpperCase();
            if (status === 'CLOSED') {
                realized += (t.pnl || t.realized_pnl || t.total_pnl || 0);
            } else if (status === 'ACTIVE') {
                unrealized += (t.unrealized_pnl || t.active_pnl || 0);
                activeCount++;
            }
        });
        return { realized, unrealized, total: realized + unrealized, activeCount, count: list.length };
    };

    const paperPnl = calcPnl(paperTrades);
    const livePnl = calcPnl(liveTrades);
    const totalPnl = paperPnl.total + livePnl.total;
    const totalActiveCount = paperPnl.activeCount + livePnl.activeCount;
    const deployedCapital = totalActiveCount * CAPITAL_PER_TRADE;
    const totalRoi = MAX_CAPITAL > 0 ? (totalPnl / MAX_CAPITAL * 100) : 0;

    const sign = totalPnl >= 0 ? '+' : '';
    const mainColor = totalPnl >= 0 ? '#22C55E' : '#EF4444';

    const PnlRow = ({ label, pnl, color: labelColor, count }: { label: string; pnl: { total: number; realized: number; unrealized: number; activeCount: number; count: number }; color: string; count: number }) => {
        const capital = pnl.activeCount * CAPITAL_PER_TRADE;
        const roi = capital > 0 ? (pnl.total / capital * 100) : (pnl.total !== 0 ? 100 : 0);
        const pColor = pnl.total >= 0 ? '#22C55E' : '#EF4444';
        const s = pnl.total >= 0 ? '+' : '';
        return (
            <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '8px 12px', borderRadius: '10px',
                background: `${labelColor}0A`, border: `1px solid ${labelColor}18`,
            }}>
                <div>
                    <span style={{ fontSize: '10px', fontWeight: 700, color: labelColor, textTransform: 'uppercase' as const, letterSpacing: '0.5px' }}>
                        {label}
                    </span>
                    <span style={{ fontSize: '10px', color: '#6B7280', marginLeft: '6px' }}>
                        {count} trades · {pnl.activeCount} active
                    </span>
                </div>
                <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: '16px', fontWeight: 700, color: pColor, fontFamily: 'monospace' }}>
                        {s}${pnl.total.toFixed(2)}
                    </span>
                    <span style={{ fontSize: '11px', color: pColor, marginLeft: '8px', fontWeight: 600 }}>
                        ({s}{roi.toFixed(1)}%)
                    </span>
                </div>
            </div>
        );
    };

    return (
        <div style={{
            background: 'rgba(17, 24, 39, 0.8)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: '16px',
            padding: '24px 28px',
        }}>
            <div style={{
                fontSize: '11px', fontWeight: 600, textTransform: 'uppercase' as const,
                letterSpacing: '1.5px', color: '#9CA3AF', marginBottom: '12px',
            }}>Total PNL</div>

            {/* Headline number */}
            <div style={{ textAlign: 'center', marginBottom: '16px' }}>
                <div style={{ fontSize: '42px', fontWeight: 700, color: mainColor }}>
                    {sign}${totalPnl.toFixed(2)}
                </div>
                <div style={{ fontSize: '13px', color: '#9CA3AF', marginTop: '4px' }}>
                    {sign}{totalRoi.toFixed(2)}% ROI on ${MAX_CAPITAL}
                </div>
            </div>

            {/* Paper / Live breakdown */}
            <div style={{ display: 'flex', flexDirection: 'column' as const, gap: '8px' }}>
                {paperTrades.length > 0 && (
                    <PnlRow label="🟢 Paper" pnl={paperPnl} color="#22C55E" count={paperTrades.length} />
                )}
                {liveTrades.length > 0 && (
                    <PnlRow label="🔴 Live" pnl={livePnl} color="#EF4444" count={liveTrades.length} />
                )}
                {paperTrades.length === 0 && liveTrades.length === 0 && (
                    <div style={{ fontSize: '12px', color: '#6B7280', textAlign: 'center', padding: '8px' }}>
                        No trades yet
                    </div>
                )}
            </div>
        </div>
    );
}

interface ActivePositionsProps {
    deployedCount: number;
    activePositions: Record<string, any>;
    trades: any[];
}

export function ActivePositionsCard({ deployedCount, activePositions, trades }: ActivePositionsProps) {
    const activeTrades = (trades || []).filter((t: any) => t.status === 'ACTIVE');
    const count = activeTrades.length || deployedCount || 0;
    const coinList = activeTrades.length > 0
        ? activeTrades.map((t: any) => t.symbol?.replace('USDT', '')).join(', ')
        : Object.keys(activePositions || {}).map(s => s.replace('USDT', '')).join(', ') || 'No coins deployed';

    const capital = count * 100;

    return (
        <div style={{
            background: 'rgba(17, 24, 39, 0.8)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: '16px',
            padding: '28px',
            textAlign: 'center',
        }}>
            <div style={{
                fontSize: '11px', fontWeight: 600, textTransform: 'uppercase' as const,
                letterSpacing: '1.5px', color: '#9CA3AF', marginBottom: '16px',
            }}>Deployment</div>

            <div style={{
                fontSize: '42px', fontWeight: 700, color: '#F0F4F8',
            }}>{count}</div>

            <div style={{ fontSize: '13px', color: '#9CA3AF', marginTop: '4px' }}>
                Active Positions
            </div>

            <div style={{
                fontSize: '12px', color: '#6B7280', marginTop: '8px',
                maxWidth: '200px', margin: '8px auto 0',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' as const,
            }}>
                {coinList}
            </div>

            <div style={{ fontSize: '11px', color: '#6B7280', marginTop: '4px' }}>
                Capital: ${capital}
            </div>
        </div>
    );
}

interface SignalSummaryProps {
    coinStates: Record<string, any>;
    multi?: any;
}

function formatPrice(price: number): string {
    if (!price || isNaN(price)) return '$0';
    if (price >= 1000) return '$' + price.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (price >= 1) return '$' + price.toFixed(4);
    return '$' + price.toFixed(6);
}

export function SignalSummaryTable({ coinStates, multi }: SignalSummaryProps) {
    const [selectedCoins, setSelectedCoins] = useState<string[]>([]);
    const [filterOpen, setFilterOpen] = useState(false);
    const [liveMulti, setLiveMulti] = useState<any>(multi);
    const [liveCoinStates, setLiveCoinStates] = useState<Record<string, any>>(coinStates || {});

    // Auto-refresh: poll bot-state at engine interval or every 60s
    const refreshMs = Math.min(Math.max((liveMulti?.analysis_interval_seconds || 60) * 1000, 30000), 900000);

    useEffect(() => {
        const fetchLatest = async () => {
            try {
                const res = await fetch('/api/bot-state', { cache: 'no-store' });
                if (res.ok) {
                    const d = await res.json();
                    if (d?.multi?.coin_states) setLiveCoinStates(d.multi.coin_states);
                    if (d?.multi) setLiveMulti(d.multi);
                }
            } catch { /* silent */ }
        };
        const timer = setInterval(fetchLatest, refreshMs);
        return () => clearInterval(timer);
    }, [refreshMs]);

    useEffect(() => { if (coinStates) setLiveCoinStates(coinStates); }, [coinStates]);
    useEffect(() => { if (multi) setLiveMulti(multi); }, [multi]);

    const coins = liveCoinStates ? Object.values(liveCoinStates) : [];
    const allSymbols = coins.map((c: any) => c.symbol || '').filter(Boolean).sort();
    const lastCycle = liveMulti?.last_analysis_time || null;
    const intervalSec = liveMulti?.analysis_interval_seconds || 0;

    const formatIST = (iso: string | null) => {
        if (!iso) return '—';
        try {
            // Engine stores local IST but appends Z — strip Z to avoid double-conversion
            const clean = iso.replace(/Z$/, '');
            const d = new Date(clean);
            return d.toLocaleTimeString('en-IN', {
                hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true,
            }) + ' IST';
        } catch { return '—'; }
    };

    if (coins.length === 0) {
        return (
            <div className="card-gradient rounded-xl p-12 text-center">
                <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(8,145,178,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px', fontSize: '20px' }}>📡</div>
                <div style={{ color: '#9CA3AF', fontSize: '14px' }}>Waiting for engine analysis cycle...</div>
            </div>
        );
    }

    const filtered = selectedCoins.length > 0 ? coins.filter((c: any) => selectedCoins.includes(c.symbol)) : coins;
    const sorted = [...filtered].sort((a: any, b: any) => {
        const ae = (a.action || '').includes('ELIGIBLE') ? 1 : 0;
        const be = (b.action || '').includes('ELIGIBLE') ? 1 : 0;
        if (ae !== be) return be - ae;
        const ac = a.confidence != null ? (a.confidence <= 1 ? a.confidence * 100 : a.confidence) : 0;
        const bc = b.confidence != null ? (b.confidence <= 1 ? b.confidence * 100 : b.confidence) : 0;
        return bc - ac;
    });

    const eligible = coins.filter((c: any) => (c.action || '').includes('ELIGIBLE'));
    const skipped = coins.filter((c: any) => {
        const a = c.action || '';
        return a.includes('SKIP') || a.includes('VETO') || a.includes('CONFLICT') || a.includes('CRASH');
    });

    const actStyle = (action: string) => {
        if (action.includes('ELIGIBLE')) return { bg: 'rgba(34,197,94,0.12)', color: '#22C55E', icon: '✓' };
        if (action.includes('CRASH')) return { bg: 'rgba(220,38,38,0.12)', color: '#DC2626', icon: '✕' };
        if (action.includes('SKIP') || action.includes('VETO') || action.includes('CONFLICT')) return { bg: 'rgba(239,68,68,0.12)', color: '#EF4444', icon: '✕' };
        if (action.includes('CHOP') || action.includes('MEAN_REV')) return { bg: 'rgba(245,158,11,0.12)', color: '#F59E0B', icon: '~' };
        return { bg: 'rgba(107,114,128,0.08)', color: '#6B7280', icon: '•' };
    };

    const regColor = (r: string) => {
        if (r.includes('BULL')) return '#22C55E';
        if (r.includes('BEAR')) return '#EF4444';
        if (r.includes('CHOP') || r.includes('SIDE')) return '#F59E0B';
        if (r.includes('CRASH')) return '#DC2626';
        return '#6B7280';
    };

    const getReason = (c: any) => {
        const a = c.action || '', r = c.regime || '';
        const pct = c.confidence != null ? (c.confidence <= 1 ? c.confidence * 100 : c.confidence) : 0;
        if (a.includes('ELIGIBLE_BUY')) return `Bullish @ ${pct.toFixed(0)}% — LONG ready`;
        if (a.includes('ELIGIBLE_SELL')) return `Bearish @ ${pct.toFixed(0)}% — SHORT ready`;
        if (a.includes('ELIGIBLE')) return `${r} @ ${pct.toFixed(0)}% — trade ready`;
        if (a.includes('CRASH_SKIP') || a.includes('MACRO_CRASH')) return 'Crash regime — safety skip';
        if (a.includes('MTF_CONFLICT')) return '1H vs 4H regime conflict';
        if (a.includes('15M_FILTER')) return '15m momentum opposes direction';
        if (a.includes('SENTIMENT_VETO') || a.includes('SENTIMENT_ALERT')) return 'Sentiment filter — vetoed';
        if (a.includes('CHOP_NO_SIGNAL')) return 'Sideways — no mean-rev signal';
        if (a.includes('MEAN_REV')) return 'Mean-reversion in choppy market';
        if (a.includes('LOW_CONVICTION')) return 'Conviction too low';
        if (a.includes('VOL_TOO_HIGH')) return 'ATR too high — risky';
        if (a.includes('VOL_TOO_LOW')) return 'ATR too low — no opportunity';
        return 'Awaiting analysis';
    };

    const toggleCoin = (sym: string) => setSelectedCoins(prev => prev.includes(sym) ? prev.filter(s => s !== sym) : [...prev, sym]);

    return (
        <div>
            {/* Header */}
            <div style={{ marginBottom: '16px' }}>
                <h2 style={{ fontSize: '20px', fontWeight: 700, color: '#06B6D4', margin: 0 }}>Bot Scan Summary</h2>
                <p style={{ fontSize: '12px', color: '#6B7280', marginTop: '4px' }}>SM-Standard · Auto-refreshes every {Math.round(refreshMs / 1000)}s</p>
            </div>

            {/* Stats Bar */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px', marginBottom: '12px' }}>
                {[
                    { label: 'Coins Scanned', value: coins.length, color: '#06B6D4' },
                    { label: 'Eligible', value: eligible.length, color: '#22C55E' },
                    { label: 'Filtered Out', value: skipped.length, color: '#EF4444' },
                    { label: 'Last Cycle (IST)', value: formatIST(lastCycle), color: '#9CA3AF', isText: true },
                    { label: 'Interval', value: intervalSec ? `${Math.round(intervalSec / 60)}m` : '—', color: '#9CA3AF', isText: true },
                ].map((s, i) => (
                    <div key={i} style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '10px', padding: '10px 12px', textAlign: 'center' }}>
                        <div style={{ fontSize: '9px', textTransform: 'uppercase', letterSpacing: '1px', color: '#6B7280', marginBottom: '4px' }}>{s.label}</div>
                        <div style={{ fontSize: (s as any).isText ? '12px' : '20px', fontWeight: 700, color: s.color, fontFamily: (s as any).isText ? 'monospace' : 'inherit' }}>{s.value}</div>
                    </div>
                ))}
            </div>

            {/* Coin Filter Dropdown */}
            <div style={{ marginBottom: '12px', position: 'relative' }}>
                <div onClick={() => setFilterOpen(!filterOpen)} style={{ padding: '8px 14px', borderRadius: '10px', cursor: 'pointer', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: selectedCoins.length > 0 ? '#06B6D4' : '#6B7280' }}>
                    <span>🔍</span>
                    {selectedCoins.length === 0 ? 'Filter by coin (all shown)' : `Showing ${selectedCoins.length}: ${selectedCoins.map(s => s.replace('USDT', '')).join(', ')}`}
                    <span style={{ marginLeft: 'auto', fontSize: '10px' }}>{filterOpen ? '▲' : '▼'}</span>
                </div>
                {filterOpen && (
                    <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 50, marginTop: '4px', padding: '10px', background: 'rgba(17,24,39,0.98)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', maxHeight: '200px', overflowY: 'auto', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        <button onClick={() => setSelectedCoins([])} style={{ padding: '4px 10px', borderRadius: '8px', fontSize: '10px', fontWeight: 700, border: 'none', cursor: 'pointer', background: selectedCoins.length === 0 ? '#06B6D422' : 'rgba(255,255,255,0.05)', color: selectedCoins.length === 0 ? '#06B6D4' : '#6B7280' }}>ALL</button>
                        {allSymbols.map((sym: string) => (
                            <button key={sym} onClick={() => toggleCoin(sym)} style={{ padding: '4px 10px', borderRadius: '8px', fontSize: '10px', fontWeight: 700, border: 'none', cursor: 'pointer', background: selectedCoins.includes(sym) ? '#06B6D422' : 'rgba(255,255,255,0.05)', color: selectedCoins.includes(sym) ? '#06B6D4' : '#9CA3AF' }}>{sym.replace('USDT', '')}</button>
                        ))}
                    </div>
                )}
            </div>

            {/* Table */}
            <div className="card-gradient rounded-xl overflow-hidden">
                <div style={{ overflowX: 'auto', maxHeight: '480px', overflowY: 'auto' }}>
                    <table style={{ width: '100%', minWidth: '900px', borderCollapse: 'collapse', fontSize: '12px' }}>
                        <thead>
                            <tr style={{ borderBottom: '2px solid rgba(255,255,255,0.08)' }}>
                                {['#', 'Bot', 'Coin', 'Regime', 'Conf %', 'Action', 'Deploy', 'Reason', 'Scan Time'].map(h => (
                                    <th key={h} style={{ padding: '10px 8px', textAlign: h === '#' || h === 'Coin' || h === 'Bot' || h === 'Reason' ? 'left' : 'center', fontSize: '9px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1px', color: '#4B5563', position: 'sticky' as const, top: 0, background: 'var(--color-surface, rgba(17,24,39,0.98))' }}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {sorted.map((c: any, idx: number) => {
                                const regime = c.regime || 'WAITING';
                                const conf = c.confidence != null ? (c.confidence <= 1 ? c.confidence * 100 : c.confidence) : 0;
                                const action = (c.action || '').replace(/_/g, ' ');
                                const as = actStyle(action);
                                const isE = action.includes('ELIGIBLE');
                                const regBg = regime.includes('BULL') ? 'rgba(34,197,94,0.12)' : regime.includes('BEAR') ? 'rgba(239,68,68,0.12)' : regime.includes('CHOP') || regime.includes('SIDE') ? 'rgba(245,158,11,0.12)' : 'rgba(107,114,128,0.10)';
                                let dLabel = '⏳ PENDING', dColor = '#6B7280', dBg = 'rgba(107,114,128,0.08)';
                                if (isE) { dLabel = '🟢 READY'; dColor = '#22C55E'; dBg = 'rgba(34,197,94,0.12)'; }
                                else if (action.includes('SKIP') || action.includes('VETO') || action.includes('CONFLICT') || action.includes('CRASH')) { dLabel = '🔴 FILTERED'; dColor = '#EF4444'; dBg = 'rgba(239,68,68,0.08)'; }

                                return (
                                    <tr key={c.symbol} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: isE ? 'rgba(34,197,94,0.04)' : 'transparent' }}>
                                        <td style={{ padding: '8px 8px', color: '#4B5563', fontSize: '10px', fontWeight: 600 }}>{idx + 1}</td>
                                        <td style={{ padding: '8px 8px' }}><span style={{ fontSize: '10px', color: '#06B6D4', fontWeight: 600 }}>SM-Standard</span></td>
                                        <td style={{ padding: '8px 8px' }}><div style={{ fontWeight: 700, color: '#F0F4F8', fontSize: '13px' }}>{(c.symbol || '').replace('USDT', '')}</div></td>
                                        <td style={{ padding: '8px 8px', textAlign: 'center' }}><span style={{ background: regBg, color: regColor(regime), padding: '3px 10px', borderRadius: '10px', fontSize: '9px', fontWeight: 700 }}>{regime}</span></td>
                                        <td style={{ padding: '8px 8px', textAlign: 'center', fontWeight: 700, fontSize: '13px', color: conf > 80 ? '#22C55E' : conf > 60 ? '#0EA5E9' : conf > 40 ? '#F59E0B' : '#6B7280' }}>{conf.toFixed(1)}%</td>
                                        <td style={{ padding: '8px 8px', textAlign: 'center' }}><span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', background: as.bg, color: as.color, padding: '3px 10px', borderRadius: '10px', fontSize: '9px', fontWeight: 700, whiteSpace: 'nowrap' as const }}><span style={{ fontSize: '10px' }}>{as.icon}</span>{action || '—'}</span></td>
                                        <td style={{ padding: '8px 8px', textAlign: 'center' }}><span style={{ background: dBg, color: dColor, padding: '4px 12px', borderRadius: '10px', fontSize: '10px', fontWeight: 700 }}>{dLabel}</span></td>
                                        <td style={{ padding: '8px 8px', fontSize: '11px', color: '#9CA3AF', maxWidth: '200px' }}>{getReason(c)}</td>
                                        <td style={{ padding: '8px 8px', textAlign: 'center', fontFamily: 'monospace', fontSize: '10px', color: '#6B7280' }}>{formatIST(lastCycle)}</td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

