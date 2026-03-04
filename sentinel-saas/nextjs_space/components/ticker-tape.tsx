'use client';

import { useState, useEffect, useRef } from 'react';

interface CoinTicker {
    symbol: string;
    price: string;
    change: number;
}

const TOP_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
    'MATICUSDT', 'SHIBUSDT', 'LTCUSDT', 'ATOMUSDT', 'NEARUSDT',
    'APTUSDT', 'OPUSDT', 'ARBUSDT', 'SUIUSDT', 'INJUSDT',
];

function formatPrice(price: number): string {
    if (price >= 1000) return '$' + price.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (price >= 1) return '$' + price.toFixed(4);
    return '$' + price.toFixed(6);
}

export function TickerTape() {
    const [coins, setCoins] = useState<CoinTicker[]>([]);
    const [mounted, setMounted] = useState(false);
    const trackRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        setMounted(true);
    }, []);

    useEffect(() => {
        if (!mounted) return;

        const fetchPrices = async () => {
            try {
                const res = await fetch('https://api.binance.com/api/v3/ticker/24hr?symbols=' +
                    encodeURIComponent(JSON.stringify(TOP_COINS)));
                if (!res.ok) throw new Error('Failed to fetch');
                const data = await res.json();
                const tickers: CoinTicker[] = data.map((t: any) => ({
                    symbol: t.symbol,
                    price: t.lastPrice,
                    change: parseFloat(t.priceChangePercent),
                }));
                setCoins(tickers);
            } catch {
                // Fallback: fetch individually if batch fails
                try {
                    const res = await fetch('https://api.binance.com/api/v3/ticker/price');
                    if (!res.ok) return;
                    const all = await res.json();
                    const filtered = all
                        .filter((t: any) => TOP_COINS.includes(t.symbol))
                        .map((t: any) => ({
                            symbol: t.symbol,
                            price: t.price,
                            change: 0,
                        }));
                    if (filtered.length > 0) setCoins(filtered);
                } catch {
                    /* silent */
                }
            }
        };

        fetchPrices();
        const interval = setInterval(fetchPrices, 15000); // refresh every 15s
        return () => clearInterval(interval);
    }, [mounted]);

    if (!mounted || coins.length === 0) {
        return (
            <div style={styles.tape}>
                <div style={styles.loading}>Loading market data...</div>
            </div>
        );
    }

    // Duplicate for seamless loop
    const items = [...coins, ...coins];

    return (
        <div style={styles.tape}>
            <div ref={trackRef} style={styles.track}>
                {items.map((coin, i) => {
                    const symbol = coin.symbol.replace('USDT', '');
                    const price = parseFloat(coin.price);
                    const isUp = coin.change >= 0;
                    return (
                        <span key={`${coin.symbol}-${i}`} style={styles.item}>
                            <span style={styles.symbol}>{symbol}</span>
                            <span style={styles.price}>{formatPrice(price)}</span>
                            {coin.change !== 0 && (
                                <span style={{
                                    ...styles.changeBadge,
                                    background: isUp ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                                    color: isUp ? '#4ADE80' : '#FCA5A5',
                                }}>
                                    {isUp ? '▲' : '▼'} {Math.abs(coin.change).toFixed(2)}%
                                </span>
                            )}
                        </span>
                    );
                })}
            </div>

            <style jsx global>{`
        @keyframes sentinel-ticker-scroll {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
      `}</style>
        </div>
    );
}

const styles: Record<string, React.CSSProperties> = {
    tape: {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 60,
        height: '38px',
        background: 'linear-gradient(90deg, #0F172A, #1E293B, #0F172A)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
    },
    loading: {
        color: 'rgba(255, 255, 255, 0.4)',
        fontSize: '12px',
        fontWeight: 500,
        paddingLeft: '24px',
        fontFamily: 'Inter, system-ui, sans-serif',
    },
    track: {
        display: 'flex',
        whiteSpace: 'nowrap' as const,
        animation: 'sentinel-ticker-scroll 90s linear infinite',
    },
    item: {
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: '0 22px',
        fontSize: '12px',
        fontWeight: 500,
        color: 'rgba(240, 244, 248, 0.6)',
        fontFamily: 'Inter, system-ui, sans-serif',
    },
    symbol: {
        color: '#F0F4F8',
        fontWeight: 700,
        letterSpacing: '0.5px',
    },
    price: {
        color: '#60C5F1',
        fontWeight: 600,
    },
    changeBadge: {
        fontSize: '10px',
        padding: '1px 7px',
        borderRadius: '20px',
        fontWeight: 600,
        letterSpacing: '0.3px',
    },
};
