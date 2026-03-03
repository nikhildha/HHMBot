/**
 * SENTINEL — Charts Page Engine
 * TradingView Lightweight Charts with active trade visualization
 */

const API_BASE = window.location.origin;
const socket = io(API_BASE);

let multiState = {};
let tradebookData = {};
let selectedSymbol = null;
let selectedTimeframe = '1h';
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let entryLine = null;
let slLine = null;
let tpLine = null;

// ─── Socket Events ──────────────────────────────────────────────────────────
socket.on('connect', () => {
    console.log('🔌 Connected to SENTINEL API');
    document.getElementById('statusDot')?.classList.add('online');
    document.getElementById('lastUpdate').textContent = 'Online';
});

socket.on('disconnect', () => {
    document.getElementById('statusDot')?.classList.remove('online');
    document.getElementById('lastUpdate').textContent = 'Offline';
});

socket.on('full-update', (data) => {
    multiState = data.multi || {};
    if (!isLiveMode()) tradebookData = data.tradebook || {};
    renderCoinList();
    if (selectedSymbol) updateTradeInfo();
});

socket.on('multi-update', (multi) => {
    multiState = multi || multiState;
    renderCoinList();
    if (selectedSymbol) updateTradeInfo();
});

socket.on('tradebook-update', (tb) => {
    if (isLiveMode()) return;  // In LIVE mode, positions come from CoinDCX
    tradebookData = tb || tradebookData;
    if (selectedSymbol) updateTradeInfo();
});

// Live price updates for overlay
socket.on('price-tick', (data) => {
    if (!data?.prices || !selectedSymbol) return;
    const livePrice = data.prices[selectedSymbol];
    if (livePrice === undefined) return;
    lastPrice = livePrice;

    // Update overlay current price and P&L in-place (no re-render)
    const container = document.getElementById('chartContainer');
    if (!container) return;
    const overlay = container.querySelector('.trade-info-overlay');
    if (!overlay) return;

    const trade = (tradebookData?.trades || []).find(t =>
        t.symbol === selectedSymbol && t.status === 'ACTIVE'
    );
    if (!trade) return;

    const entry = parseFloat(trade.entry_price) || 0;
    const qty = trade.quantity || 0;
    const lev = trade.leverage || 1;
    const capital = trade.capital || 100;
    let rawPnl = trade.position === 'LONG'
        ? (livePrice - entry) * qty
        : (entry - livePrice) * qty;
    const pnl = parseFloat((rawPnl * lev).toFixed(4));
    const pnlPct = capital > 0 ? parseFloat((pnl / capital * 100).toFixed(2)) : 0;
    const pnlClass = pnl >= 0 ? 'green' : 'red';
    const pnlSign = pnl >= 0 ? '+' : '';

    // Update current price value
    const rows = overlay.querySelectorAll('.info-row');
    rows.forEach(row => {
        const label = row.querySelector('.info-label');
        const value = row.querySelector('.info-value');
        if (!label || !value) return;
        if (label.textContent === 'Current') {
            value.textContent = `$${livePrice}`;
        }
        if (label.textContent === 'P&L') {
            value.textContent = `${pnlSign}$${pnl.toFixed(2)} (${pnlSign}${pnlPct.toFixed(2)}%)`;
            value.className = `info-value ${pnlClass}`;
            value.style.fontSize = '14px';
            value.style.fontWeight = '700';
        }
    });
});

// ─── Render Coin Sidebar ────────────────────────────────────────────────────
function renderCoinList() {
    const list = document.getElementById('coinList');
    const activeTrades = (tradebookData?.trades || []).filter(t => t.status === 'ACTIVE');

    if (activeTrades.length === 0) {
        list.innerHTML = '<div class="no-coins">No active positions</div>';
        return;
    }

    list.innerHTML = activeTrades.map(trade => {
        const side = trade.position || (trade.side === 'BUY' ? 'LONG' : 'SHORT');
        const sideClass = side === 'LONG' ? 'buy' : 'sell';
        const isActive = trade.symbol === selectedSymbol ? 'active' : '';

        return `<div class="coin-item ${isActive}" onclick="selectCoin('${trade.symbol}')">
            <span class="coin-name">${trade.symbol.replace('USDT', '')}</span>
            <span class="coin-side ${sideClass}">${side}</span>
        </div>`;
    }).join('');
}

// ─── Select Coin ────────────────────────────────────────────────────────────
function selectCoin(symbol) {
    selectedSymbol = symbol;
    renderCoinList();
    loadChart(symbol, selectedTimeframe);
    updateRegimeBanner(symbol);
}

// ─── Regime Banner ───────────────────────────────────────────────────────────
function updateRegimeBanner(symbol) {
    fetch('/api/multi-state')
        .then(r => r.json())
        .then(data => {
            const states = data.coin_states || data.active_positions || {};
            const coin = states[symbol] || states[symbol + 'USDT'];
            const banner = document.getElementById('regimeBanner');
            if (!banner) return;
            const regime = (coin?.regime || 'UNKNOWN').toUpperCase();
            const conf = coin?.confidence ? (coin.confidence * 100).toFixed(0) + '% confidence' : '';
            const map = {
                'BULLISH': 'bull',
                'BEARISH': 'bear',
                'SIDEWAYS/CHOP': 'chop',
                'CRASH/PANIC': 'crash'
            };
            const cls = map[regime] || '';
            banner.className = 'regime-banner' + (cls ? ' ' + cls : '');
            banner.style.display = 'flex';
            document.getElementById('regimeDot').className = 'regime-dot' + (cls ? ' ' + cls : '');
            document.getElementById('regimeBannerText').textContent = regime + ' REGIME';
            document.getElementById('regimeBannerConf').textContent = conf;
        })
        .catch(() => {});
}

// ─── Timeframe Selection ────────────────────────────────────────────────────
document.getElementById('timeframeBar').addEventListener('click', (e) => {
    const btn = e.target.closest('.tf-btn');
    if (!btn) return;

    document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    selectedTimeframe = btn.dataset.tf;
    if (selectedSymbol) {
        loadChart(selectedSymbol, selectedTimeframe);
    }
});

// ─── Load Chart ─────────────────────────────────────────────────────────────
async function loadChart(symbol, timeframe) {
    const container = document.getElementById('chartContainer');
    const toolbar = document.getElementById('chartToolbar');
    const noSel = document.getElementById('noSelection');

    // Show toolbar, hide placeholder
    toolbar.style.display = 'flex';
    if (noSel) noSel.style.display = 'none';

    // Update toolbar
    document.getElementById('chartSymbol').textContent = symbol;

    // Fetch klines
    try {
        const res = await fetch(`${API_BASE}/api/klines?symbol=${symbol}&interval=${timeframe}&limit=300`);
        const data = await res.json();

        if (!data.candles || data.candles.length === 0) {
            container.innerHTML = '<div class="no-selection">No candle data available</div>';
            return;
        }

        renderChart(container, data.candles, symbol);
    } catch (err) {
        console.error('Failed to load klines:', err);
        container.innerHTML = '<div class="no-selection">Failed to load chart data</div>';
    }
}

// ─── Render TradingView Chart ───────────────────────────────────────────────
function renderChart(container, candles, symbol) {
    // Remove old chart
    if (chart) {
        chart.remove();
        chart = null;
    }

    // Remove trade info overlay if exists
    const oldOverlay = container.querySelector('.trade-info-overlay');
    if (oldOverlay) oldOverlay.remove();

    chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            background: { color: '#FFFFFF' },
            textColor: '#5A6A7E',
            fontFamily: 'Inter, sans-serif',
        },
        grid: {
            vertLines: { color: 'rgba(0,0,0,0.04)' },
            horzLines: { color: 'rgba(0,0,0,0.04)' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {
                color: 'rgba(59,130,246,0.3)',
                labelBackgroundColor: '#3B82F6',
            },
            horzLine: {
                color: 'rgba(59,130,246,0.3)',
                labelBackgroundColor: '#3B82F6',
            },
        },
        rightPriceScale: {
            borderColor: 'rgba(0,0,0,0.08)',
        },
        timeScale: {
            borderColor: 'rgba(0,0,0,0.08)',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    // Candlestick series
    candleSeries = chart.addCandlestickSeries({
        upColor: '#22C55E',
        downColor: '#EF4444',
        borderUpColor: '#22C55E',
        borderDownColor: '#EF4444',
        wickUpColor: '#22C55E',
        wickDownColor: '#EF4444',
    });

    candleSeries.setData(candles);

    // Volume series
    volumeSeries = chart.addHistogramSeries({
        color: 'rgba(59, 130, 246, 0.15)',
        priceFormat: { type: 'volume' },
        priceScaleId: '',
    });

    volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
    });

    if (candles[0]?.volume !== undefined) {
        volumeSeries.setData(candles.map(c => ({
            time: c.time,
            value: c.volume,
            color: c.close >= c.open ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
        })));
    }

    // Update price display
    const lastCandle = candles[candles.length - 1];
    if (lastCandle) {
        document.getElementById('chartPrice').textContent = `$${Number(lastCandle.close).toLocaleString()}`;
        const firstCandle = candles[0];
        const change = ((lastCandle.close - firstCandle.open) / firstCandle.open * 100).toFixed(2);
        const changeEl = document.getElementById('chartChange');
        changeEl.textContent = `${change > 0 ? '+' : ''}${change}%`;
        changeEl.className = `change ${change >= 0 ? 'positive' : 'negative'}`;
    }

    // Add trade markers (entry, SL, TP, current price)
    addTradeMarkers(symbol, candles);

    // Auto-resize
    const resizeObserver = new ResizeObserver(() => {
        chart?.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    });
    resizeObserver.observe(container);

    // Fit content
    chart.timeScale().fitContent();
}

// ─── Add Trade Entry/SL/TP + Current Price Lines ────────────────────────────
let lastPrice = 0;

function addTradeMarkers(symbol, candles) {
    if (!candleSeries) return;

    // Get trade details from tradebook
    const trade = (tradebookData?.trades || []).find(t =>
        t.symbol === symbol && t.status === 'ACTIVE'
    );
    if (!trade) return;

    const entry = parseFloat(trade.entry_price) || 0;
    const sl = parseFloat(trade.stop_loss) || 0;
    const tp = parseFloat(trade.take_profit) || 0;

    // Current price from last candle
    const lastCandle = candles?.[candles.length - 1];
    lastPrice = lastCandle ? lastCandle.close : (parseFloat(trade.current_price) || 0);

    // Entry line (blue solid)
    if (entry) {
        candleSeries.createPriceLine({
            price: entry,
            color: '#3B82F6',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            axisLabelVisible: true,
            title: `Entry $${entry}`,
        });
    }

    // Stop loss line (red dashed)
    if (sl) {
        candleSeries.createPriceLine({
            price: sl,
            color: '#EF4444',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            axisLabelVisible: true,
            title: `SL $${sl}`,
        });
    }

    // Take profit line (green dashed)
    if (tp) {
        candleSeries.createPriceLine({
            price: tp,
            color: '#22C55E',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            axisLabelVisible: true,
            title: `TP $${tp}`,
        });
    }

    // Current price line (orange dotted)
    if (lastPrice) {
        candleSeries.createPriceLine({
            price: lastPrice,
            color: '#F59E0B',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            axisLabelVisible: true,
            title: `Now $${lastPrice}`,
        });
    }

    // Add trade info overlay
    updateTradeInfo();
}

// ─── Trade Info Overlay ─────────────────────────────────────────────────────
function updateTradeInfo() {
    const container = document.getElementById('chartContainer');
    if (!container || !selectedSymbol) return;

    // Remove old
    const old = container.querySelector('.trade-info-overlay');
    if (old) old.remove();

    // Get trade details from tradebook (single source of truth)
    const trade = (tradebookData?.trades || []).find(t =>
        t.symbol === selectedSymbol && t.status === 'ACTIVE'
    );
    if (!trade) return;

    const side = trade.position || (trade.side === 'BUY' ? 'LONG' : 'SHORT');
    const regime = trade.regime || '—';
    const conf = ((trade.confidence || 0) * 100).toFixed(1);
    const leverage = trade.leverage || 1;
    const entry = parseFloat(trade.entry_price) || 0;
    const sl = parseFloat(trade.stop_loss) || 0;
    const tp = parseFloat(trade.take_profit) || 0;
    const currentPrice = lastPrice || parseFloat(trade.current_price) || 0;

    // Use tradebook's stored PNL (calculated correctly in Python with qty * leverage)
    const pnl = parseFloat(trade.unrealized_pnl) || 0;
    const pnlPct = parseFloat(trade.unrealized_pnl_pct) || 0;
    const pnlClass = pnl >= 0 ? 'green' : 'red';
    const pnlSign = pnl >= 0 ? '+' : '';

    const overlay = document.createElement('div');
    overlay.className = 'trade-info-overlay';
    overlay.innerHTML = `
        <div class="info-title">${side} — ${selectedSymbol}</div>
        <div class="info-row">
            <span class="info-label">Regime</span>
            <span class="info-value">${regime}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Confidence</span>
            <span class="info-value">${conf}%</span>
        </div>
        <div class="info-row">
            <span class="info-label">Leverage</span>
            <span class="info-value">${leverage}x</span>
        </div>
        <div class="info-row" style="margin-top:6px; border-top:1px solid rgba(0,0,0,0.08); padding-top:6px;">
            <span class="info-label">Entry</span>
            <span class="info-value">$${entry}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Current</span>
            <span class="info-value" style="color:#F59E0B;font-weight:700;">$${currentPrice}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Stop Loss</span>
            <span class="info-value red">$${sl}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Take Profit</span>
            <span class="info-value green">$${tp}</span>
        </div>
        <div class="info-row" style="margin-top:6px; border-top:1px solid rgba(0,0,0,0.08); padding-top:6px;">
            <span class="info-label">P&L</span>
            <span class="info-value ${pnlClass}" style="font-size:14px;font-weight:700;">${pnlSign}$${pnl.toFixed(2)} (${pnlSign}${pnlPct.toFixed(2)}%)</span>
        </div>
    `;

    container.appendChild(overlay);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  LIVE MODE — CoinDCX Integration
// ═══════════════════════════════════════════════════════════════════════════════

let liveRefreshInterval = null;

function isLiveMode() {
    return typeof sentinelGetMode === 'function' && sentinelGetMode() === 'LIVE';
}

async function fetchLivePositions() {
    try {
        const res = await fetch(`${API_BASE}/api/coindcx/positions`);
        const data = await res.json();
        if (!data.success) return null;
        return data;
    } catch (e) {
        console.error('[LIVE] Failed to fetch CoinDCX positions:', e);
        return null;
    }
}

function mapLiveToTradebook(cdxData) {
    const trades = (cdxData.positions || []).map(p => ({
        symbol: p.symbol,
        side: p.side === 'LONG' ? 'BUY' : 'SELL',
        position: p.side,
        status: 'ACTIVE',
        entry_price: p.entry_price,
        current_price: p.mark_price,
        leverage: p.leverage,
        quantity: p.quantity,
        unrealized_pnl: p.pnl,
        unrealized_pnl_pct: p.pnl_pct,
        stop_loss: p.stop_loss || 0,
        take_profit: p.take_profit || 0,
        liquidation_price: p.liquidation_price,
        locked_margin: p.locked_margin,
        capital: p.locked_margin,
        regime: '—',
        confidence: 0,
        mode: 'LIVE',
    }));
    return { trades };
}

async function refreshLiveData() {
    const cdxData = await fetchLivePositions();
    if (!cdxData) return;
    tradebookData = mapLiveToTradebook(cdxData);
    renderCoinList();
    if (selectedSymbol) updateTradeInfo();
}

function startLiveRefresh() {
    stopLiveRefresh();
    refreshLiveData();
    liveRefreshInterval = setInterval(refreshLiveData, 3000);
}

function stopLiveRefresh() {
    if (liveRefreshInterval) {
        clearInterval(liveRefreshInterval);
        liveRefreshInterval = null;
    }
}

// Listen for master toggle changes
window.addEventListener('mode-change', (e) => {
    const mode = e.detail?.mode || 'PAPER';
    if (mode === 'LIVE') {
        startLiveRefresh();
    } else {
        stopLiveRefresh();
    }
});

// On initial load, check mode
if (isLiveMode()) {
    startLiveRefresh();
}
