'use client';

import { useState, useEffect, useCallback } from 'react';
import { Header } from '@/components/header';
import { RefreshCw, Newspaper, TrendingUp, BarChart3, Coins, Brain } from 'lucide-react';
import { motion } from 'framer-motion';

/* ─── Regime Colors ─── */
const REGIME_COLORS: Record<string, string> = {
    'BULLISH': '#22C55E', 'BEARISH': '#EF4444', 'SIDEWAYS/CHOP': '#F59E0B',
    'CRASH/PANIC': '#DC2626', 'WAITING': '#6B7280', 'SCANNING': '#3B82F6',
};
function regimeColor(r: string) { return REGIME_COLORS[r] || '#6B7280'; }

/* ─── Card wrapper ─── */
function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) {
    return (
        <div className={className} style={{
            background: 'rgba(17, 24, 39, 0.8)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: '16px',
            padding: '24px',
        }}>{children}</div>
    );
}

function CardTitle({ children, sub }: { children: React.ReactNode; sub?: string }) {
    return (
        <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '15px', fontWeight: 700, color: '#F0F4F8', display: 'flex', alignItems: 'center', gap: '8px' }}>
                {children}
            </div>
            {sub && <div style={{ fontSize: '11px', color: '#6B7280', marginTop: '4px' }}>{sub}</div>}
        </div>
    );
}

function SectionHeader({ icon, title, sub }: { icon: React.ReactNode; title: string; sub: string }) {
    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '20px' }}>
            <div style={{
                width: '42px', height: '42px', borderRadius: '12px',
                background: 'rgba(8, 145, 178, 0.15)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>{icon}</div>
            <div>
                <div style={{ fontSize: '18px', fontWeight: 700, color: '#06B6D4' }}>{title}</div>
                <div style={{ fontSize: '12px', color: '#6B7280' }}>{sub}</div>
            </div>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════ */
/*  MAIN COMPONENT                                                   */
/* ═══════════════════════════════════════════════════════════════════ */

export function IntelligenceClient() {
    const [data, setData] = useState<any>(null);
    const [fearGreed, setFearGreed] = useState<any>(null);
    const [lastRefresh, setLastRefresh] = useState('');
    const [isRefreshing, setIsRefreshing] = useState(false);

    const fetchData = useCallback(async () => {
        try {
            setIsRefreshing(true);
            const res = await fetch('/api/bot-state', { cache: 'no-store' });
            if (res.ok) {
                const d = await res.json();
                setData(d);
                setLastRefresh(new Date().toLocaleTimeString());
            }
        } catch { /* silent */ } finally { setIsRefreshing(false); }
    }, []);

    const fetchFearGreed = useCallback(async () => {
        try {
            const res = await fetch('https://api.alternative.me/fng/?limit=1');
            if (res.ok) {
                const d = await res.json();
                if (d?.data?.[0]) setFearGreed(d.data[0]);
            }
        } catch { /* silent */ }
    }, []);

    useEffect(() => {
        fetchData();
        fetchFearGreed();
        const interval = setInterval(fetchData, 15000);
        return () => clearInterval(interval);
    }, [fetchData, fetchFearGreed]);

    const multi = data?.multi || {};
    const state = data?.state || {};
    const coinStates = multi?.coin_states || {};
    const coins = Object.values(coinStates) as any[];
    const trades = data?.tradebook?.trades || [];
    const activePositions = multi?.active_positions || {};
    const activeCount = Object.keys(activePositions).length;

    // Compute aggregates
    const avgSentiment = coins.length > 0
        ? coins.reduce((s: number, c: any) => s + (c.sentiment_score || 0), 0) / coins.length
        : 0;
    const topConviction = coins.length > 0
        ? coins.reduce((best: any, c: any) => (!best || (c.confidence || 0) > (best.confidence || 0)) ? c : best, null)
        : null;

    return (
        <div className="min-h-screen">
            <Header />
            <main className="pt-24 pb-12 px-4">
                <div className="max-w-7xl mx-auto">

                    {/* ─── Hero ─── */}
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <div className="flex items-center gap-3">
                                    <h1 className="text-3xl font-bold">Market Intelligence</h1>
                                    <span style={{
                                        fontSize: '10px', fontWeight: 700, padding: '3px 10px',
                                        borderRadius: '20px', background: 'rgba(34, 197, 94, 0.2)',
                                        color: '#22C55E', letterSpacing: '1px',
                                    }}>LIVE</span>
                                </div>
                                <p className="text-sm text-[var(--color-text-secondary)] mt-1">
                                    Sentiment · Order Flow · Funding Rates · Regime Drivers — all tracked coins
                                </p>
                            </div>
                            {lastRefresh && <span className="text-xs text-[var(--color-text-secondary)]">Updated: {lastRefresh}</span>}
                        </div>
                    </motion.div>

                    {/* ═══ Command Bar ═══ */}
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="mb-8">
                        <CommandBar
                            state={state}
                            fearGreed={fearGreed}
                            avgSentiment={avgSentiment}
                            activeCount={activeCount}
                            topConviction={topConviction}
                        />
                    </motion.div>

                    {/* ═══ Section 1: Sentiment Intelligence ═══ */}
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="mb-8">
                        <SectionHeader icon={<Newspaper className="w-5 h-5" style={{ color: '#0891B2' }} />} title="Sentiment Intelligence" sub="VADER NLP · Fear & Greed Index · CryptoPanic · Multi-Source Analysis" />
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                            {/* Left: Bias + Conviction */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                                <SentimentBiasCard avgSentiment={avgSentiment} />
                                <ConvictionDistribution coins={coins} />
                            </div>
                            {/* Right: Per-Coin Sentiment Bars */}
                            <PerCoinSentiment coins={coins} />
                        </div>
                    </motion.div>

                    {/* ═══ Section 2: Technical Analysis ═══ */}
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="mb-8">
                        <SectionHeader icon={<TrendingUp className="w-5 h-5" style={{ color: '#0891B2' }} />} title="Technical Analysis" sub="Multi-Timeframe Support/Resistance · RSI · Volatility" />
                        <TechnicalAnalysisTable coins={coins} />
                    </motion.div>

                    {/* ═══ Section 3: Order Flow Intelligence ═══ */}
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="mb-8">
                        <SectionHeader icon={<BarChart3 className="w-5 h-5" style={{ color: '#0891B2' }} />} title="Order Flow Intelligence" sub="L2 Order Book · Taker Buy/Sell · Cumulative Delta · Long/Short Ratio" />
                        <OrderFlowTable coins={coins} />
                    </motion.div>

                    {/* ═══ Section 4: Funding Rates ═══ */}
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }} className="mb-8">
                        <SectionHeader icon={<Coins className="w-5 h-5" style={{ color: '#0891B2' }} />} title="Funding Rates & Open Interest" sub="Binance Futures perpetual swap rates · Smart-money signals" />
                        <FundingRatesGrid coins={coins} />
                    </motion.div>

                    {/* ═══ Section 5: Regime Drivers Heatmap ═══ */}
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="mb-8">
                        <SectionHeader icon={<Brain className="w-5 h-5" style={{ color: '#0891B2' }} />} title="Regime Drivers — Feature Heatmap" sub="HMM feature values driving regime classification per coin" />
                        <RegimeHeatmap coins={coins} />
                    </motion.div>

                </div>
            </main>
        </div>
    );
}


/* ═══════════════════════════════════════════════════════════════════ */
/*  COMMAND BAR                                                      */
/* ═══════════════════════════════════════════════════════════════════ */

function CommandBar({ state, fearGreed, avgSentiment, activeCount, topConviction }: any) {
    const btcPrice = state?.btc_price
        ? '$' + Number(state.btc_price).toLocaleString('en-US', { maximumFractionDigits: 0 })
        : '—';
    const fgVal = fearGreed?.value || '—';
    const fgLabel = fearGreed?.value_classification || 'Loading…';
    const fgColor = (fearGreed?.value || 0) >= 55 ? '#22C55E' : (fearGreed?.value || 0) <= 30 ? '#EF4444' : '#F59E0B';

    const biasVal = avgSentiment.toFixed(3);
    const biasColor = avgSentiment > 0.1 ? '#22C55E' : avgSentiment < -0.1 ? '#EF4444' : '#F59E0B';

    const topConvVal = topConviction
        ? (topConviction.confidence <= 1 ? (topConviction.confidence * 100).toFixed(0) : topConviction.confidence.toFixed(0)) + '%'
        : '—';
    const topConvName = topConviction?.symbol?.replace('USDT', '') || '';

    const items = [
        { label: 'BTC Price', val: btcPrice, sub: `Regime: ${state?.regime || '—'}`, color: regimeColor(state?.regime) },
        { label: 'Fear & Greed', val: fgVal, sub: fgLabel, color: fgColor },
        { label: 'Market Sentiment', val: biasVal, sub: 'Avg across coins', color: biasColor },
        { label: 'Active Positions', val: String(activeCount), sub: 'paper + live', color: '#0EA5E9' },
        { label: 'Top Conviction', val: topConvVal, sub: topConvName, color: '#22C55E' },
    ];

    return (
        <div style={{
            display: 'grid', gridTemplateColumns: `repeat(${items.length}, 1fr)`, gap: '12px',
        }}>
            {items.map(it => (
                <div key={it.label} style={{
                    background: 'rgba(17, 24, 39, 0.8)', backdropFilter: 'blur(12px)',
                    border: '1px solid rgba(255,255,255,0.06)', borderRadius: '14px',
                    padding: '18px 20px', textAlign: 'center',
                }}>
                    <div style={{ fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1.2px', color: '#6B7280', marginBottom: '8px' }}>{it.label}</div>
                    <div style={{ fontSize: '24px', fontWeight: 700, color: it.color }}>{it.val}</div>
                    <div style={{ fontSize: '11px', color: '#9CA3AF', marginTop: '4px' }}>{it.sub}</div>
                </div>
            ))}
        </div>
    );
}


/* ═══════════════════════════════════════════════════════════════════ */
/*  SENTIMENT BIAS CARD                                              */
/* ═══════════════════════════════════════════════════════════════════ */

function SentimentBiasCard({ avgSentiment }: { avgSentiment: number }) {
    const pct = ((avgSentiment + 1) / 2) * 100;
    const color = avgSentiment > 0.1 ? '#22C55E' : avgSentiment < -0.1 ? '#EF4444' : '#F59E0B';

    return (
        <Card>
            <CardTitle sub="Aggregated VADER NLP score across all tracked coins">Market Sentiment Bias</CardTitle>
            <div style={{ textAlign: 'center', marginBottom: '16px' }}>
                <span style={{ fontSize: '36px', fontWeight: 700, color }}>{avgSentiment.toFixed(3)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#6B7280', marginBottom: '6px' }}>
                <span>Bearish -1</span><span>0</span><span>Bullish +1</span>
            </div>
            <div style={{ height: '10px', borderRadius: '5px', background: 'rgba(255,255,255,0.06)', overflow: 'hidden', position: 'relative' }}>
                <div style={{
                    position: 'absolute', top: 0, left: 0, bottom: 0,
                    width: `${Math.max(2, Math.min(98, pct))}%`,
                    background: `linear-gradient(90deg, #EF4444, #F59E0B 50%, #22C55E)`,
                    borderRadius: '5px',
                    transition: 'width 1s ease',
                }} />
            </div>
        </Card>
    );
}


/* ═══════════════════════════════════════════════════════════════════ */
/*  CONVICTION DISTRIBUTION                                          */
/* ═══════════════════════════════════════════════════════════════════ */

function ConvictionDistribution({ coins }: { coins: any[] }) {
    const buckets = { high: 0, medium: 0, low: 0 };
    coins.forEach(c => {
        const conf = c.confidence != null ? (c.confidence <= 1 ? c.confidence * 100 : c.confidence) : 0;
        if (conf >= 75) buckets.high++;
        else if (conf >= 50) buckets.medium++;
        else buckets.low++;
    });
    const total = coins.length || 1;

    return (
        <Card>
            <CardTitle>Conviction Distribution</CardTitle>
            <div style={{ display: 'flex', gap: '12px' }}>
                {[
                    { label: 'High (≥75%)', count: buckets.high, color: '#22C55E' },
                    { label: 'Medium (50-74%)', count: buckets.medium, color: '#F59E0B' },
                    { label: 'Low (<50%)', count: buckets.low, color: '#EF4444' },
                ].map(b => (
                    <div key={b.label} style={{ flex: 1, textAlign: 'center' }}>
                        <div style={{ fontSize: '28px', fontWeight: 700, color: b.color }}>{b.count}</div>
                        <div style={{ fontSize: '10px', color: '#6B7280', marginTop: '2px' }}>{b.label}</div>
                        <div style={{
                            height: '4px', borderRadius: '2px', marginTop: '8px',
                            background: 'rgba(255,255,255,0.06)', overflow: 'hidden',
                        }}>
                            <div style={{
                                height: '100%', borderRadius: '2px', background: b.color,
                                width: `${(b.count / total) * 100}%`, transition: 'width 0.8s ease',
                            }} />
                        </div>
                    </div>
                ))}
            </div>
        </Card>
    );
}


/* ═══════════════════════════════════════════════════════════════════ */
/*  PER-COIN SENTIMENT BARS                                          */
/* ═══════════════════════════════════════════════════════════════════ */

function PerCoinSentiment({ coins }: { coins: any[] }) {
    const sorted = [...coins].sort((a, b) => (b.sentiment_score || 0) - (a.sentiment_score || 0));

    return (
        <Card>
            <CardTitle sub="Sorted by sentiment score, all tracked coins">Per-Coin Sentiment Scores</CardTitle>
            <div style={{ maxHeight: '360px', overflowY: 'auto' }}>
                {sorted.length === 0 ? (
                    <div style={{ color: '#6B7280', textAlign: 'center', padding: '40px 0', fontSize: '13px' }}>
                        Waiting for bot analysis cycle…
                    </div>
                ) : sorted.map((c: any) => {
                    const score = c.sentiment_score || 0;
                    const pct = ((score + 1) / 2) * 100;
                    const color = score > 0.1 ? '#22C55E' : score < -0.1 ? '#EF4444' : '#F59E0B';
                    return (
                        <div key={c.symbol} style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
                            <div style={{ width: '60px', fontSize: '12px', fontWeight: 700, color: '#F0F4F8' }}>
                                {(c.symbol || '').replace('USDT', '')}
                            </div>
                            <div style={{ flex: 1, height: '8px', borderRadius: '4px', background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                                <div style={{
                                    height: '100%', borderRadius: '4px', background: color,
                                    width: `${Math.max(2, pct)}%`, transition: 'width 0.8s ease',
                                }} />
                            </div>
                            <div style={{ width: '50px', textAlign: 'right', fontSize: '11px', fontWeight: 600, color }}>
                                {score.toFixed(3)}
                            </div>
                        </div>
                    );
                })}
            </div>
        </Card>
    );
}


/* ═══════════════════════════════════════════════════════════════════ */
/*  TECHNICAL ANALYSIS TABLE                                         */
/* ═══════════════════════════════════════════════════════════════════ */

function TechnicalAnalysisTable({ coins }: { coins: any[] }) {
    if (coins.length === 0) return <EmptyState />;
    const sorted = [...coins].sort((a, b) => (a.symbol || '').localeCompare(b.symbol || ''));
    const headers = ['Coin', 'Regime', 'Confidence', 'RSI', 'Volatility', 'Log Return', 'Volume Chg'];

    return (
        <Card>
            <div style={{ overflowX: 'auto', maxHeight: '400px', overflowY: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                    <thead>
                        <tr style={{ borderBottom: '2px solid rgba(255,255,255,0.08)' }}>
                            {headers.map(h => (
                                <th key={h} style={{
                                    padding: '10px 14px', textAlign: h === 'Coin' ? 'left' : 'center',
                                    fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px',
                                    color: '#6B7280', position: 'sticky', top: 0, background: 'rgba(17, 24, 39, 0.95)',
                                }}>{h}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {sorted.map((c: any) => {
                            const features = c.features || {};
                            const conf = c.confidence != null ? (c.confidence <= 1 ? c.confidence * 100 : c.confidence) : 0;
                            const rsi = features.rsi_norm != null ? (features.rsi_norm * 100).toFixed(1) : '—';
                            const rsiColor = (features.rsi_norm || 0) > 0.7 ? '#EF4444' : (features.rsi_norm || 0) < 0.3 ? '#22C55E' : '#6B7280';
                            return (
                                <tr key={c.symbol} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                                    <td style={{ padding: '10px 14px', fontWeight: 700, color: '#F0F4F8' }}>{(c.symbol || '').replace('USDT', '')}</td>
                                    <td style={{ padding: '10px 14px', textAlign: 'center' }}>
                                        <span style={{
                                            padding: '2px 10px', borderRadius: '10px', fontSize: '10px', fontWeight: 700,
                                            color: regimeColor(c.regime), background: regimeColor(c.regime) + '22',
                                        }}>{c.regime || '—'}</span>
                                    </td>
                                    <td style={{ padding: '10px 14px', textAlign: 'center', fontWeight: 600, color: conf > 70 ? '#22C55E' : '#6B7280' }}>{conf.toFixed(1)}%</td>
                                    <td style={{ padding: '10px 14px', textAlign: 'center', color: rsiColor, fontWeight: 600 }}>{rsi}</td>
                                    <td style={{ padding: '10px 14px', textAlign: 'center', color: (features.volatility || 0) > 0.02 ? '#F59E0B' : '#6B7280' }}>
                                        {features.volatility != null ? Number(features.volatility).toFixed(4) : '—'}
                                    </td>
                                    <td style={{ padding: '10px 14px', textAlign: 'center', color: valColor(features.log_return) }}>
                                        {features.log_return != null ? Number(features.log_return).toFixed(4) : '—'}
                                    </td>
                                    <td style={{ padding: '10px 14px', textAlign: 'center', color: valColor(features.volume_change) }}>
                                        {features.volume_change != null ? Number(features.volume_change).toFixed(2) : '—'}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </Card>
    );
}


/* ═══════════════════════════════════════════════════════════════════ */
/*  ORDER FLOW TABLE                                                 */
/* ═══════════════════════════════════════════════════════════════════ */

function OrderFlowTable({ coins }: { coins: any[] }) {
    if (coins.length === 0) return <EmptyState />;
    const sorted = [...coins].sort((a, b) => (a.symbol || '').localeCompare(b.symbol || ''));
    const headers = ['Coin', 'Regime', 'Imbalance', 'Taker Buy%', 'Cum Delta', 'L/S Ratio', 'Bid Walls', 'Ask Walls', 'Signal'];

    return (
        <Card>
            <div style={{ overflowX: 'auto', maxHeight: '400px', overflowY: 'auto' }}>
                <table style={{ width: '100%', minWidth: '750px', borderCollapse: 'collapse', fontSize: '12px' }}>
                    <thead>
                        <tr style={{ borderBottom: '2px solid rgba(255,255,255,0.08)' }}>
                            {headers.map(h => (
                                <th key={h} style={{
                                    padding: '10px 12px', textAlign: h === 'Coin' ? 'left' : 'center',
                                    fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px',
                                    color: '#6B7280', position: 'sticky', top: 0, background: 'rgba(17, 24, 39, 0.95)',
                                }}>{h}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {sorted.map((c: any) => {
                            const of = c.order_flow || {};
                            const fmt = (v: any, d = 3) => v != null ? Number(v).toFixed(d) : '—';
                            const takerBuy = of.taker_buy_pct != null ? (of.taker_buy_pct * 100).toFixed(1) : '—';
                            const takerColor = (of.taker_buy_pct || 0.5) > 0.55 ? '#22C55E' : (of.taker_buy_pct || 0.5) < 0.45 ? '#EF4444' : '#6B7280';
                            const signal = of.signal || c.order_flow_signal || '—';
                            const sigColor = signal.toLowerCase().includes('bull') ? '#22C55E' : signal.toLowerCase().includes('bear') ? '#EF4444' : '#6B7280';

                            return (
                                <tr key={c.symbol} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                                    <td style={{ padding: '10px 12px', fontWeight: 700, color: '#F0F4F8' }}>{(c.symbol || '').replace('USDT', '')}</td>
                                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                                        <span style={{
                                            padding: '2px 10px', borderRadius: '10px', fontSize: '10px', fontWeight: 700,
                                            color: regimeColor(c.regime), background: regimeColor(c.regime) + '22',
                                        }}>{c.regime || '—'}</span>
                                    </td>
                                    <td style={{ padding: '10px 12px', textAlign: 'center', color: valColor(of.imbalance) }}>{fmt(of.imbalance)}</td>
                                    <td style={{ padding: '10px 12px', textAlign: 'center', color: takerColor, fontWeight: 600 }}>{takerBuy}%</td>
                                    <td style={{ padding: '10px 12px', textAlign: 'center', color: valColor(of.cum_delta) }}>{fmt(of.cum_delta)}</td>
                                    <td style={{ padding: '10px 12px', textAlign: 'center', color: valColor((of.long_short_ratio || 1) - 1) }}>{fmt(of.long_short_ratio, 2)}</td>
                                    <td style={{ padding: '10px 12px', textAlign: 'center', color: '#22C55E' }}>{fmt(of.bid_walls, 0)}</td>
                                    <td style={{ padding: '10px 12px', textAlign: 'center', color: '#EF4444' }}>{fmt(of.ask_walls, 0)}</td>
                                    <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 700, fontSize: '10px', color: sigColor }}>{signal}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </Card>
    );
}


/* ═══════════════════════════════════════════════════════════════════ */
/*  FUNDING RATES GRID                                               */
/* ═══════════════════════════════════════════════════════════════════ */

function FundingRatesGrid({ coins }: { coins: any[] }) {
    if (coins.length === 0) return <EmptyState />;
    const sorted = [...coins].sort((a, b) => Math.abs(b.features?.funding || 0) - Math.abs(a.features?.funding || 0));

    return (
        <Card>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '12px' }}>
                {sorted.map((c: any) => {
                    const funding = c.features?.funding || 0;
                    const oiChange = c.features?.oi_change || 0;
                    const color = funding > 0.0001 ? '#22C55E' : funding < -0.0001 ? '#EF4444' : '#6B7280';
                    const bgColor = funding > 0.0001 ? 'rgba(34,197,94,0.08)' : funding < -0.0001 ? 'rgba(239,68,68,0.08)' : 'rgba(255,255,255,0.03)';

                    return (
                        <div key={c.symbol} style={{
                            background: bgColor,
                            border: '1px solid rgba(255,255,255,0.06)',
                            borderRadius: '12px', padding: '14px', textAlign: 'center',
                        }}>
                            <div style={{ fontSize: '12px', fontWeight: 700, color: '#F0F4F8', marginBottom: '6px' }}>
                                {(c.symbol || '').replace('USDT', '')}
                            </div>
                            <div style={{ fontSize: '18px', fontWeight: 700, color }}>
                                {(funding * 100).toFixed(4)}%
                            </div>
                            <div style={{ fontSize: '10px', color: '#6B7280', marginTop: '4px' }}>
                                OI Δ: <span style={{ color: valColor(oiChange) }}>{oiChange != null ? Number(oiChange).toFixed(2) : '—'}</span>
                            </div>
                        </div>
                    );
                })}
            </div>
        </Card>
    );
}


/* ═══════════════════════════════════════════════════════════════════ */
/*  REGIME HEATMAP                                                   */
/* ═══════════════════════════════════════════════════════════════════ */

function RegimeHeatmap({ coins }: { coins: any[] }) {
    if (coins.length === 0) return <EmptyState />;
    const features = ['log_return', 'volatility', 'volume_change', 'rsi_norm'];
    const sorted = [...coins].sort((a, b) => (a.symbol || '').localeCompare(b.symbol || ''));

    function heatColor(feature: string, val: number): string {
        if (val == null || isNaN(val)) return 'rgba(255,255,255,0.03)';
        if (feature === 'rsi_norm') {
            if (val > 0.7) return 'rgba(239, 68, 68, 0.3)';
            if (val < 0.3) return 'rgba(34, 197, 94, 0.3)';
            return 'rgba(245, 158, 11, 0.15)';
        }
        if (feature === 'volatility') {
            if (val > 0.03) return 'rgba(239, 68, 68, 0.3)';
            if (val > 0.01) return 'rgba(245, 158, 11, 0.2)';
            return 'rgba(34, 197, 94, 0.15)';
        }
        // log_return, volume_change
        if (val > 0.01) return 'rgba(34, 197, 94, 0.3)';
        if (val < -0.01) return 'rgba(239, 68, 68, 0.3)';
        return 'rgba(255,255,255,0.05)';
    }

    return (
        <Card>
            <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                    <thead>
                        <tr style={{ borderBottom: '2px solid rgba(255,255,255,0.08)' }}>
                            <th style={{ padding: '10px 14px', textAlign: 'left', fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px', color: '#6B7280' }}>Coin</th>
                            <th style={{ padding: '10px 14px', textAlign: 'center', fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px', color: '#6B7280' }}>Regime</th>
                            {features.map(f => (
                                <th key={f} style={{ padding: '10px 14px', textAlign: 'center', fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px', color: '#6B7280' }}>
                                    {f.replace(/_/g, ' ').replace('rsi norm', 'RSI')}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {sorted.map((c: any) => (
                            <tr key={c.symbol} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                                <td style={{ padding: '8px 14px', fontWeight: 700, color: '#F0F4F8' }}>{(c.symbol || '').replace('USDT', '')}</td>
                                <td style={{ padding: '8px 14px', textAlign: 'center' }}>
                                    <span style={{
                                        padding: '2px 10px', borderRadius: '10px', fontSize: '10px', fontWeight: 700,
                                        color: regimeColor(c.regime), background: regimeColor(c.regime) + '22',
                                    }}>{c.regime || '—'}</span>
                                </td>
                                {features.map(f => {
                                    const val = c.features?.[f];
                                    return (
                                        <td key={f} style={{
                                            padding: '8px 14px', textAlign: 'center',
                                            background: heatColor(f, val),
                                            fontWeight: 600, fontSize: '11px',
                                            color: '#F0F4F8',
                                        }}>
                                            {val != null ? Number(val).toFixed(4) : '—'}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </Card>
    );
}


/* ─── Utilities ─── */

function valColor(v: any): string {
    if (v == null || Math.abs(v) < 0.0001) return '#6B7280';
    return v > 0 ? '#22C55E' : '#EF4444';
}

function EmptyState() {
    return (
        <Card>
            <div style={{ textAlign: 'center', padding: '40px 0' }}>
                <div style={{ fontSize: '36px', marginBottom: '12px' }}>📊</div>
                <div style={{ color: '#9CA3AF', fontSize: '14px' }}>Waiting for engine analysis cycle…</div>
            </div>
        </Card>
    );
}
