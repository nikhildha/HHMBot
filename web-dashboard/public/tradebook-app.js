/**
 * Project SENTINEL — Tradebook Page Engine
 * Full trade journal with filtering, P&L curve, CSV export,
 * and real-time WebSocket updates.
 */

const API_BASE = window.location.hostname === 'localhost'
    ? 'http://localhost:3001'
    : window.location.origin;

const socket = io(API_BASE);

// ─── State ───────────────────────────────────────────────────────────────────
let tradebookData = { trades: [], summary: {} };
let pnlChart = null;
let pnlTimelineChart = null;
let livePositionsData = null;  // cached CoinDCX positions

// Live price cache — prevents flickering when file-based updates
// overwrite with stale prices between price-tick events
const livePriceCache = {};
let updateAllTimer = null;

function isLiveMode() { return typeof sentinelGetMode === 'function' && sentinelGetMode() === 'LIVE'; }
function tradebookUrl() { return `${API_BASE}/api/tradebook`; }  // Paper only

// ─── Helpers ─────────────────────────────────────────────────────────────────
function formatPrice(p) {
    p = parseFloat(p);
    if (isNaN(p)) return '—';
    if (p >= 1000) return '$' + p.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (p >= 1) return '$' + p.toFixed(4);
    return '$' + p.toFixed(6);
}

function formatPnl(val) {
    val = parseFloat(val) || 0;
    const sign = val >= 0 ? '+' : '';
    return sign + '$' + val.toFixed(2);
}

function formatPnlPct(val) {
    val = parseFloat(val) || 0;
    const sign = val >= 0 ? '+' : '';
    return sign + val.toFixed(2) + '%';
}

function formatTime(ts) {
    if (!ts) return '—';
    // Treat timestamps without timezone info as UTC (Python uses datetime.utcnow())
    if (typeof ts === 'string' && !ts.endsWith('Z') && !ts.includes('+')) ts += 'Z';
    const d = new Date(ts);
    return d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'Asia/Kolkata' });
}

function formatDateTime(ts) {
    if (!ts) return '—';
    if (typeof ts === 'string' && !ts.endsWith('Z') && !ts.includes('+')) ts += 'Z';
    const d = new Date(ts);
    return d.toLocaleDateString('en-IN', { month: 'short', day: 'numeric', timeZone: 'Asia/Kolkata' }) + ' ' + formatTime(ts);
}

function pnlClass(val) {
    val = parseFloat(val) || 0;
    if (val > 0) return 'green';
    if (val < 0) return 'red';
    return '';
}

function pnlPctBadge(val) {
    val = parseFloat(val) || 0;
    return val >= 0 ? 'positive' : 'negative';
}

function showToast(msg, type = 'success') {
    const t = document.createElement('div');
    t.className = `toast toast-${type}`;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3500);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SUMMARY UPDATE
// ═══════════════════════════════════════════════════════════════════════════════

function updateSummary(summary) {
    if (!summary) return;

    document.getElementById('summTotalTrades').textContent = summary.total_trades ?? 0;
    document.getElementById('summActiveTrades').textContent = summary.active_trades ?? 0;
    document.getElementById('summClosedTrades').textContent = summary.closed_trades ?? 0;
    document.getElementById('summWins').textContent = summary.wins ?? 0;
    document.getElementById('summLosses').textContent = summary.losses ?? 0;
    document.getElementById('summWinRate').textContent = (summary.win_rate_pct ?? 0) + '%';

    // ── Compute P&L fresh from trades (matches dashboard logic) ──
    const MAX_CAPITAL = 1500;
    const CAPITAL_PER_TRADE = 100;
    let realizedPnl = 0, unrealizedPnl = 0, activeCount = 0;
    (tradebookData?.trades || []).forEach(t => {
        if (t.status === 'CLOSED') realizedPnl += (t.realized_pnl || 0);
        else if (t.status === 'ACTIVE') { unrealizedPnl += (t.unrealized_pnl || 0); activeCount++; }
    });
    const deployedCapital = activeCount * CAPITAL_PER_TRADE;
    const totalPnl = realizedPnl + unrealizedPnl;

    const realizedPct = MAX_CAPITAL > 0 ? (realizedPnl / MAX_CAPITAL * 100) : 0;
    const unrealizedPct = deployedCapital > 0 ? (unrealizedPnl / deployedCapital * 100) : 0;
    const cumulativePct = MAX_CAPITAL > 0 ? (totalPnl / MAX_CAPITAL * 100) : 0;

    const rPnl = document.getElementById('summRealizedPnl');
    const rPnlPct = document.getElementById('summRealizedPnlPct');
    rPnl.textContent = formatPnl(realizedPnl);
    rPnl.className = 'pnl-value ' + pnlClass(realizedPnl);
    rPnlPct.textContent = formatPnlPct(realizedPct);
    rPnlPct.className = 'pnl-pct ' + pnlPctBadge(realizedPct);

    const uPnl = document.getElementById('summUnrealizedPnl');
    const uPnlPct = document.getElementById('summUnrealizedPnlPct');
    uPnl.textContent = formatPnl(unrealizedPnl);
    uPnl.className = 'pnl-value ' + pnlClass(unrealizedPnl);
    uPnlPct.textContent = formatPnlPct(unrealizedPct);
    uPnlPct.className = 'pnl-pct ' + pnlPctBadge(unrealizedPct);

    const cPnl = document.getElementById('summCumulativePnl');
    const cPnlPct = document.getElementById('summCumulativePnlPct');
    cPnl.textContent = formatPnl(totalPnl);
    cPnl.className = 'pnl-value ' + pnlClass(totalPnl);
    cPnlPct.textContent = formatPnlPct(cumulativePct);
    cPnlPct.className = 'pnl-pct ' + pnlPctBadge(cumulativePct);

    const best = document.getElementById('summBestTrade');
    const worst = document.getElementById('summWorstTrade');
    best.textContent = formatPnl(summary.best_trade);
    worst.textContent = formatPnl(summary.worst_trade);

    // ── Strategy Stats (computed from closed trades) ──
    const closedTrades = (tradebookData?.trades || []).filter(t => t.status === 'CLOSED');
    const closedPnls = closedTrades.map(t => t.realized_pnl || 0);

    // Profit Factor
    const grossWin = closedPnls.filter(p => p > 0).reduce((s, p) => s + p, 0);
    const grossLoss = Math.abs(closedPnls.filter(p => p < 0).reduce((s, p) => s + p, 0));
    const profitFactor = grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Infinity : 0;
    const pfEl = document.getElementById('summProfitFactor');
    pfEl.textContent = profitFactor === Infinity ? '∞' : profitFactor.toFixed(2);
    pfEl.className = profitFactor >= 1.5 ? 'green' : profitFactor < 1 ? 'red' : '';

    // Sharpe Ratio
    let sharpe = 0;
    if (closedPnls.length > 1) {
        const mean = closedPnls.reduce((s, v) => s + v, 0) / closedPnls.length;
        const variance = closedPnls.reduce((s, v) => s + (v - mean) ** 2, 0) / closedPnls.length;
        sharpe = Math.sqrt(variance) > 0 ? mean / Math.sqrt(variance) : 0;
    }
    const shEl = document.getElementById('summSharpe');
    shEl.textContent = sharpe.toFixed(3);
    shEl.className = sharpe >= 1 ? 'green' : sharpe < 0 ? 'red' : '';

    // Risk:Reward
    const winPnls = closedPnls.filter(p => p > 0);
    const lossPnls = closedPnls.filter(p => p < 0);
    const avgW = winPnls.length > 0 ? winPnls.reduce((s, v) => s + v, 0) / winPnls.length : 0;
    const avgL = lossPnls.length > 0 ? Math.abs(lossPnls.reduce((s, v) => s + v, 0) / lossPnls.length) : 0;
    const rr = avgL > 0 ? avgW / avgL : avgW > 0 ? Infinity : 0;
    const rrEl = document.getElementById('summRiskReward');
    rrEl.textContent = (rr === Infinity ? '∞' : rr.toFixed(2)) + ':1';
    rrEl.className = rr >= 1.5 ? 'green' : rr < 1 ? 'red' : '';

    // Max Drawdown (% only)
    let peak = 0, maxDD = 0, eq = 0;
    [...closedTrades].sort((a, b) => new Date(a.exit_timestamp || 0) - new Date(b.exit_timestamp || 0))
        .forEach(t => {
            eq += (t.realized_pnl || 0);
            if (eq > peak) peak = eq;
            const dd = peak - eq;
            if (dd > maxDD) maxDD = dd;
        });
    const ddEl = document.getElementById('summMaxDrawdown');
    ddEl.textContent = '-' + (MAX_CAPITAL > 0 ? (maxDD / MAX_CAPITAL * 100).toFixed(1) : '0') + '%';
    ddEl.className = 'red';

    // Win/Loss Streak
    let maxWinStreak = 0, maxLossStreak = 0, curWin = 0, curLoss = 0;
    [...closedTrades].sort((a, b) => new Date(a.exit_timestamp || 0) - new Date(b.exit_timestamp || 0))
        .forEach(t => {
            if ((t.realized_pnl || 0) > 0) { curWin++; curLoss = 0; maxWinStreak = Math.max(maxWinStreak, curWin); }
            else { curLoss++; curWin = 0; maxLossStreak = Math.max(maxLossStreak, curLoss); }
        });
    document.getElementById('summWinStreak').textContent = maxWinStreak;
    document.getElementById('summLossStreak').textContent = maxLossStreak;

    document.getElementById('lastUpdate').textContent = summary.last_updated
        ? `Last: ${formatTime(summary.last_updated)}`
        : 'Live';
}

// ═══════════════════════════════════════════════════════════════════════════════
//  P&L CURVE CHART
// ═══════════════════════════════════════════════════════════════════════════════

function updatePnlChart(trades) {
    if (!trades || trades.length === 0) return;

    // Build cumulative P&L curve from closed trades (chronological)
    const closed = trades
        .filter(t => t.status === 'CLOSED')
        .sort((a, b) => new Date(a.exit_timestamp) - new Date(b.exit_timestamp));

    if (closed.length === 0) return;

    let cumulative = 0;
    const labels = [];
    const dataPnl = [];
    const colors = [];

    closed.forEach((t, i) => {
        cumulative += t.realized_pnl || 0;
        labels.push(t.trade_id);
        dataPnl.push(cumulative.toFixed(4));
        colors.push(cumulative >= 0 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(239, 68, 68, 0.8)');
    });

    const ctx = document.getElementById('pnlChart');
    if (!ctx) return;

    if (pnlChart) pnlChart.destroy();

    pnlChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Cumulative P&L ($)',
                data: dataPnl,
                borderColor: '#3B82F6',
                backgroundColor: 'rgba(59, 130, 246, 0.06)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointBackgroundColor: colors,
                pointBorderColor: colors,
                pointRadius: 4,
                pointHoverRadius: 7,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#5A6A7E', font: { family: 'Inter', size: 12 } }
                },
                tooltip: {
                    backgroundColor: '#FFFFFF',
                    borderColor: '#E2E8F0',
                    borderWidth: 1,
                    titleColor: '#1A2332',
                    bodyColor: '#5A6A7E',
                    cornerRadius: 10,
                    padding: 12,
                    callbacks: {
                        label: (ctx) => `Cumulative: ${formatPnl(ctx.parsed.y)}`
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#5A6A7E', font: { family: 'Inter', size: 10 } },
                    grid: { color: 'rgba(0,0,0,0.05)' },
                },
                y: {
                    ticks: {
                        color: '#5A6A7E',
                        font: { family: 'Inter', size: 11 },
                        callback: v => '$' + parseFloat(v).toFixed(2),
                    },
                    grid: { color: 'rgba(0,0,0,0.05)' },
                }
            }
        }
    });
}

// ─── P&L Timeline Chart (time-based x-axis) ─────────────────────────────────
function updatePnlTimelineChart(trades) {
    if (!trades || trades.length === 0) return;

    // Include both closed and active trades, sorted by timestamp
    const allTrades = trades
        .filter(t => t.entry_timestamp)
        .sort((a, b) => {
            const tA = a.exit_timestamp || a.entry_timestamp;
            const tB = b.exit_timestamp || b.entry_timestamp;
            return new Date(tA) - new Date(tB);
        });

    if (allTrades.length === 0) return;

    let cumulative = 0;
    const dataPoints = [];

    allTrades.forEach(t => {
        const pnl = t.status === 'CLOSED' ? (t.realized_pnl || 0) : (t.unrealized_pnl || 0);
        cumulative += pnl;
        const ts = t.exit_timestamp || t.entry_timestamp;
        dataPoints.push({
            x: new Date(ts),
            y: parseFloat(cumulative.toFixed(4)),
            color: cumulative >= 0 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(239, 68, 68, 0.8)',
        });
    });

    const ctx = document.getElementById('pnlTimelineChart');
    if (!ctx) return;

    if (pnlTimelineChart) pnlTimelineChart.destroy();

    pnlTimelineChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Cumulative P&L ($)',
                data: dataPoints,
                borderColor: '#8B5CF6',
                backgroundColor: 'rgba(139, 92, 246, 0.06)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointBackgroundColor: dataPoints.map(d => d.color),
                pointBorderColor: dataPoints.map(d => d.color),
                pointRadius: 4,
                pointHoverRadius: 7,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#5A6A7E', font: { family: 'Inter', size: 12 } }
                },
                tooltip: {
                    backgroundColor: '#FFFFFF',
                    borderColor: '#E2E8F0',
                    borderWidth: 1,
                    titleColor: '#1A2332',
                    bodyColor: '#5A6A7E',
                    cornerRadius: 10,
                    padding: 12,
                    callbacks: {
                        title: (items) => {
                            const d = new Date(items[0].parsed.x);
                            return d.toLocaleString('en-IN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
                        },
                        label: (ctx) => `Cumulative: ${formatPnl(ctx.parsed.y)}`
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'hour',
                        displayFormats: { hour: 'MMM d, HH:mm' },
                        tooltipFormat: 'MMM d, HH:mm',
                    },
                    ticks: { color: '#5A6A7E', font: { family: 'Inter', size: 10 }, maxRotation: 45 },
                    grid: { color: 'rgba(0,0,0,0.05)' },
                },
                y: {
                    ticks: {
                        color: '#5A6A7E',
                        font: { family: 'Inter', size: 11 },
                        callback: v => '$' + parseFloat(v).toFixed(2),
                    },
                    grid: { color: 'rgba(0,0,0,0.05)' },
                }
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TRADE TABLE
// ═══════════════════════════════════════════════════════════════════════════════

function getFilteredTrades() {
    const status = document.getElementById('filterStatus').value;
    const position = document.getElementById('filterPosition').value;
    const regime = document.getElementById('filterRegime').value;
    const pnl = document.getElementById('filterPnl').value;
    const symbolSearch = document.getElementById('filterSymbol').value.toUpperCase();

    let trades = [...(tradebookData.trades || [])];

    if (status !== 'ALL') trades = trades.filter(t => t.status === status);
    if (position !== 'ALL') trades = trades.filter(t => t.position === position);
    if (regime !== 'ALL') trades = trades.filter(t => t.regime === regime);
    if (pnl === 'PROFIT') trades = trades.filter(t => {
        const v = t.status === 'ACTIVE' ? t.unrealized_pnl : t.realized_pnl;
        return (v || 0) > 0;
    });
    if (pnl === 'LOSS') trades = trades.filter(t => {
        const v = t.status === 'ACTIVE' ? t.unrealized_pnl : t.realized_pnl;
        return (v || 0) < 0;
    });
    if (symbolSearch) trades = trades.filter(t =>
        (t.symbol || '').toUpperCase().includes(symbolSearch)
    );

    return trades;
}

function renderTable(trades) {
    const area = document.getElementById('tradebookArea');

    if (!trades || trades.length === 0) {
        area.innerHTML = `<div class="empty-state">
      <div class="icon">📝</div>
      <p>No trades match the current filters.</p>
    </div>`;
        return;
    }

    // Sort: Active first by unrealized P&L (highest→lowest), then Closed by time (newest→oldest)
    const sorted = [...trades].sort((a, b) => {
        // Active trades come first
        if (a.status === 'ACTIVE' && b.status !== 'ACTIVE') return -1;
        if (a.status !== 'ACTIVE' && b.status === 'ACTIVE') return 1;

        if (a.status === 'ACTIVE' && b.status === 'ACTIVE') {
            // Both active: sort by unrealized P&L descending
            return (parseFloat(b.unrealized_pnl) || 0) - (parseFloat(a.unrealized_pnl) || 0);
        }

        // Both closed: sort by exit time descending (newest first)
        const tA = new Date(a.exit_timestamp || a.entry_timestamp || 0).getTime();
        const tB = new Date(b.exit_timestamp || b.entry_timestamp || 0).getTime();
        return tB - tA;
    });

    let html = `<table class="tradebook-table">
    <thead><tr>
      <th style="width:32px"><input type="checkbox" id="selectAllTrades" onchange="toggleSelectAll(this)" title="Select All"></th>
      <th>Symbol</th>
      <th>Position</th>
      <th>Regime</th>
      <th>Confidence</th>
      <th>Leverage</th>
      <th>Capital</th>
      <th>Entry Price</th>
      <th>Current</th>
      <th>SL / TP</th>
      <th>Status</th>
      <th>Unrealized P&L</th>
      <th>Realized P&L</th>
      <th>Duration</th>
      <th>Exit Reason</th>
      <th>Exit Price</th>
      <th>Time</th>
      <th>ID</th>
      <th>Commission</th>
      <th>Actions</th>
    </tr></thead><tbody>`;

    // For cumulative P&L
    let cumPnl = 0;

    // We compute cumulative in chronological order (entry timestamp)
    const chronological = [...trades].sort((a, b) =>
        new Date(a.entry_timestamp) - new Date(b.entry_timestamp)
    );
    const cumMap = {};
    chronological.forEach(t => {
        const pnl = t.status === 'CLOSED' ? (t.realized_pnl || 0) : (t.unrealized_pnl || 0);
        cumPnl += pnl;
        cumMap[t.trade_id] = cumPnl;
    });

    sorted.forEach(t => {
        const isActive = t.status === 'ACTIVE';
        const posClass = t.position === 'LONG' ? 'side-buy' : 'side-sell';
        const posIcon = t.position === 'LONG' ? '▲' : '▼';
        const conf = ((t.confidence || 0) * 100).toFixed(1);

        // Regime badge
        const regimeColors = {
            'BULLISH': 'badge-bull',
            'BEARISH': 'badge-bear',
            'SIDEWAYS/CHOP': 'badge-chop',
            'CRASH/PANIC': 'badge-crash',
        };
        const regimeBadge = regimeColors[t.regime] || 'badge-chop';

        // P&L values
        const uPnl = t.unrealized_pnl || 0;
        const uPnlPct = t.unrealized_pnl_pct || 0;
        const rPnl = t.realized_pnl || 0;
        const rPnlPct = t.realized_pnl_pct || 0;

        // Duration
        const dur = t.duration_minutes || 0;
        const durStr = dur >= 60 ? `${(dur / 60).toFixed(1)}h` : `${dur.toFixed(0)}m`;

        // Exit reason badge
        let exitBadge = '';
        if (t.exit_reason === 'TAKE_PROFIT') exitBadge = '<span class="exit-reason-badge tp">TP</span>';
        else if (t.exit_reason === 'TRAILING_TP') exitBadge = '<span class="exit-reason-badge tp">TRAIL TP</span>';
        else if (t.exit_reason === 'STOP_LOSS') exitBadge = '<span class="exit-reason-badge sl">SL</span>';
        else if (t.exit_reason === 'TRAILING_SL') exitBadge = '<span class="exit-reason-badge sl">TRAIL SL</span>';
        else if (t.exit_reason === 'MANUAL') exitBadge = '<span class="exit-reason-badge manual">MANUAL</span>';
        else if (t.exit_reason) exitBadge = `<span class="exit-reason-badge manual">${t.exit_reason}</span>`;

        html += `<tr>
      <td><input type="checkbox" class="trade-select" data-tradeid="${t.trade_id}" ${isActive ? '' : 'disabled'}></td>
      <td><strong>${t.symbol || '—'}</strong></td>
      <td class="${posClass}">${posIcon} ${t.position}</td>
      <td><span class="regime-badge ${regimeBadge}">${(t.regime || '—').split('/')[0]}</span></td>
      <td>${conf}%</td>
      <td><span class="leverage-badge">${t.leverage || 1}x</span></td>
      <td class="col-price">$${(t.capital || 100).toFixed(0)}</td>
      <td class="col-price">${formatPrice(t.entry_price)}</td>
      <td class="col-price ${isActive ? pnlClass(uPnl) : ''}">${isActive ? formatPrice(t.current_price) : '—'}</td>
      <td class="col-price col-sltp" style="font-size:10px">${formatPrice(t.trailing_sl || t.stop_loss)} / ${formatPrice(t.trailing_tp || t.take_profit)}${t.trailing_active ? ' <span class="trail-indicator">⟟</span>' : ''}</td>
      <td><span class="status-badge ${isActive ? 'active' : 'closed'}">${t.status}</span></td>
      <td class="${pnlClass(uPnl)}">${isActive ? `${formatPnl(uPnl)} (${formatPnlPct(uPnlPct)})` : '—'}</td>
      <td class="${pnlClass(rPnl)}">${!isActive ? `${formatPnl(rPnl)} (${formatPnlPct(rPnlPct)})` : '—'}</td>
      <td>${durStr}</td>
      <td>${exitBadge || '—'}</td>
      <td class="col-price">${t.exit_price ? formatPrice(t.exit_price) : '—'}</td>
      <td>${formatDateTime(t.entry_timestamp)}</td>
      <td class="col-id">${t.trade_id}</td>
      <td class="col-price">$${(t.commission || ((t.entry_price || 0) * (t.quantity || 0) + ((t.current_price || t.entry_price || 0) * (t.quantity || 0))) * 0.0005).toFixed(2)}</td>
      <td><button class="btn-delete-trade" onclick="deleteTrade('${t.trade_id}')" title="Delete trade">Delete</button></td>
    </tr>`;
    });

    html += '</tbody></table>';
    area.innerHTML = html;

    // Listen for checkbox changes to show/hide Close Selected button
    area.querySelectorAll('.trade-select').forEach(cb => {
        cb.addEventListener('change', updateCloseSelectedBar);
    });
}

function applyFilters() {
    const filtered = getFilteredTrades();
    renderTable(filtered);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CSV EXPORT
// ═══════════════════════════════════════════════════════════════════════════════

function exportCSV() {
    const trades = getFilteredTrades();
    if (trades.length === 0) {
        showToast('No trades to export', 'error');
        return;
    }

    const headers = [
        'Trade ID', 'Entry Time', 'Exit Time', 'Symbol', 'Position', 'Side',
        'Regime', 'Confidence', 'Leverage', 'Capital', 'Quantity',
        'Entry Price', 'Exit Price', 'Current Price',
        'Stop Loss', 'Take Profit', 'ATR at Entry',
        'Status', 'Exit Reason',
        'Realized P&L ($)', 'Realized P&L (%)',
        'Unrealized P&L ($)', 'Unrealized P&L (%)',
        'Max Favorable ($)', 'Max Adverse ($)',
        'Duration (min)', 'Mode',
    ];

    const rows = trades.map(t => [
        t.trade_id, t.entry_timestamp, t.exit_timestamp || '',
        t.symbol, t.position, t.side,
        t.regime, t.confidence, t.leverage, t.capital, t.quantity,
        t.entry_price, t.exit_price || '', t.current_price,
        t.stop_loss, t.take_profit, t.atr_at_entry,
        t.status, t.exit_reason || '',
        t.realized_pnl, t.realized_pnl_pct,
        t.unrealized_pnl, t.unrealized_pnl_pct,
        t.max_favorable, t.max_adverse,
        t.duration_minutes, t.mode,
    ]);

    let csv = headers.join(',') + '\n';
    rows.forEach(r => csv += r.join(',') + '\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tradebook_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Tradebook exported!');
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MASTER UPDATE
// ═══════════════════════════════════════════════════════════════════════════════

function updateAll(data) {
    if (!data) return;
    tradebookData = data;

    // Merge cached live prices into trades to prevent stale file data flickering
    if (tradebookData.trades && Object.keys(livePriceCache).length > 0) {
        tradebookData.trades.forEach(trade => {
            if (trade.status !== 'ACTIVE') return;
            const cached = livePriceCache[trade.symbol];
            if (cached === undefined) return;

            trade.current_price = cached;
            // Recalculate unrealized P&L with live price
            const entry = trade.entry_price;
            const qty = trade.quantity;
            const lev = trade.leverage;
            const capital = trade.capital || 100;
            let rawPnl = trade.position === 'LONG'
                ? (cached - entry) * qty
                : (entry - cached) * qty;
            const isLive = trade.mode === 'LIVE';
            trade.unrealized_pnl = parseFloat((isLive ? rawPnl : rawPnl * lev).toFixed(4));
            trade.unrealized_pnl_pct = capital > 0
                ? parseFloat((trade.unrealized_pnl / capital * 100).toFixed(2))
                : 0;
        });
    }

    updateSummary(data.summary);
    updatePnlChart(data.trades);
    updatePnlTimelineChart(data.trades);
    applyFilters();
    renderReportHistory();
}

// Debounced version for socket/file-watcher events to avoid rapid re-renders
function debouncedUpdateAll(data) {
    clearTimeout(updateAllTimer);
    updateAllTimer = setTimeout(() => updateAll(data), 300);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SOCKET.IO EVENTS
// ═══════════════════════════════════════════════════════════════════════════════

function setConnectionStatus(online) {
    const pill = document.getElementById('statusPill');
    const txt = document.getElementById('statusText');
    if (pill && txt) {
        pill.classList.toggle('online', online);
        pill.classList.toggle('offline', !online);
        txt.textContent = online ? 'LIVE' : 'OFFLINE';
    }
}

/**
 * Build tradebook entries from multi_bot_state active_positions + trade_log
 * when tradebook.json has no data yet.
 */
function buildFromMultiState(data) {
    const multi = data?.multi;
    const tradeLog = data?.trades || [];
    if (!multi?.active_positions) return null;

    const positions = multi.active_positions;
    const trades = [];
    let idx = 1;

    for (const [symbol, pos] of Object.entries(positions)) {
        // Find matching entry in trade_log for richer data
        const logEntry = tradeLog.find(t => t.symbol === symbol) || {};

        const side = pos.side || logEntry.side || 'BUY';
        const position = side === 'BUY' ? 'LONG' : 'SHORT';
        const entryPrice = parseFloat(logEntry.entry_price) || pos.entry_price || 0;
        const leverage = parseInt(pos.leverage || logEntry.leverage) || 1;
        const confidence = pos.confidence || parseFloat(logEntry.confidence) || 0;
        const regime = pos.regime || logEntry.regime || 'UNKNOWN';
        const entryTime = pos.entry_time || logEntry.timestamp || new Date().toISOString();

        // Find scanner data for current price
        const scannerCoin = data?.scanner?.coins_scanned
            ? null
            : (data?.scanner?.coins || []).find(c => c.symbol === symbol);
        const currentPrice = scannerCoin?.price || entryPrice;

        const capital = 100;
        const qty = entryPrice > 0 ? (capital * leverage) / entryPrice : 0;

        // Compute unrealized P&L
        let uPnl = 0;
        let uPnlPct = 0;
        if (entryPrice > 0 && currentPrice > 0) {
            if (position === 'LONG') {
                uPnl = (currentPrice - entryPrice) * qty;
            } else {
                uPnl = (entryPrice - currentPrice) * qty;
            }
            uPnlPct = capital > 0 ? (uPnl / capital) * 100 : 0;
        }

        trades.push({
            trade_id: `T${String(idx).padStart(4, '0')}`,
            symbol,
            side,
            position,
            regime,
            confidence,
            leverage,
            capital,
            quantity: qty,
            entry_price: entryPrice,
            current_price: currentPrice,
            entry_timestamp: entryTime,
            exit_timestamp: null,
            exit_price: null,
            stop_loss: parseFloat(logEntry.stop_loss) || 0,
            take_profit: parseFloat(logEntry.take_profit) || 0,
            atr_at_entry: 0,
            status: 'ACTIVE',
            exit_reason: null,
            realized_pnl: 0,
            realized_pnl_pct: 0,
            unrealized_pnl: uPnl,
            unrealized_pnl_pct: uPnlPct,
            max_favorable: Math.max(0, uPnl),
            max_adverse: Math.min(0, uPnl),
            duration_minutes: ((Date.now() - new Date(entryTime).getTime()) / 60000),
            mode: 'PAPER',
            reason: logEntry.reason || '',
        });
        idx++;
    }

    const activeTrades = trades.filter(t => t.status === 'ACTIVE');
    const totalU = activeTrades.reduce((s, t) => s + (t.unrealized_pnl || 0), 0);
    const totalUPct = activeTrades.length > 0 ? totalU / (activeTrades.length * 100) * 100 : 0;

    return {
        trades,
        summary: {
            total_trades: trades.length,
            active_trades: trades.length,
            closed_trades: 0,
            wins: 0,
            losses: 0,
            win_rate_pct: 0,
            total_realized_pnl: 0,
            total_realized_pnl_pct: 0,
            total_unrealized_pnl: totalU,
            total_unrealized_pnl_pct: totalUPct,
            cumulative_pnl: totalU,
            cumulative_pnl_pct: totalUPct,
            best_trade: 0,
            worst_trade: 0,
            last_updated: new Date().toISOString(),
        }
    };
}

// ─── Dialog guard: prevents socket re-renders from dismissing confirm() ──────
let dialogOpen = false;

socket.on('connect', () => {
    console.log('🔌 Connected to SENTINEL API');
    setConnectionStatus(true);
    showToast('Connected', 'success');
});

socket.on('disconnect', () => {
    setConnectionStatus(false);
    showToast('Disconnected', 'error');
});

socket.on('full-update', (data) => {
    if (dialogOpen) return;  // Skip re-render while confirm is open
    if (data?.tradebook && data.tradebook.trades && data.tradebook.trades.length > 0) {
        debouncedUpdateAll(data.tradebook);
    } else {
        // Fallback: build tradebook from multi-state + trade log
        const built = buildFromMultiState(data);
        if (built && built.trades.length > 0) {
            console.log('📗 Built tradebook from active positions:', built.trades.length, 'trades');
            debouncedUpdateAll(built);
        }
    }
});

socket.on('tradebook-update', (data) => {
    if (dialogOpen) return;  // Skip re-render while confirm is open
    if (data) debouncedUpdateAll(data);
});

// Also listen to multi-update to keep tradebook synced when no tradebook.json
socket.on('multi-update', async (multiData) => {
    if (dialogOpen) return;  // Skip while confirm is open
    if (tradebookData.trades.length === 0 && multiData?.active_positions) {
        // Re-fetch all data to rebuild
        try {
            const res = await fetch(`${API_BASE}/api/all`);
            const allData = await res.json();
            if (!allData.tradebook?.trades?.length) {
                const built = buildFromMultiState(allData);
                if (built && built.trades.length > 0) updateAll(built);
            }
        } catch (e) { /* silent */ }
    }
});

// ─── Refresh button ──────────────────────────────────────────────────────────
async function refreshData() {
    if (isLiveMode()) {
        // LIVE mode: fetch from CoinDCX
        const data = await fetchLivePositions();
        if (data) {
            renderLiveTable(data.positions);
            updateLiveSummary(data);
            startLiveRefresh();
            showToast(`${data.count} live positions from CoinDCX`);
        }
        return;
    }

    // PAPER mode: existing tradebook.json flow
    if (liveRefreshInterval) { clearInterval(liveRefreshInterval); liveRefreshInterval = null; }
    try {
        const res = await fetch(tradebookUrl());
        const data = await res.json();
        if (data?.trades?.length > 0) {
            updateAll(data);
            showToast('Tradebook refreshed');
            return;
        }
        // Fallback: build from all data
        const allRes = await fetch(`${API_BASE}/api/all`);
        const allData = await allRes.json();
        const built = buildFromMultiState(allData);
        if (built && built.trades.length > 0) {
            updateAll(built);
            showToast('Tradebook synced from active positions');
        } else {
            showToast('No trade data available', 'error');
        }
    } catch (e) {
        showToast('Failed to refresh', 'error');
    }
}

// ─── Initial fetch ───────────────────────────────────────────────────────────
setTimeout(async () => {
    if (isLiveMode()) {
        // LIVE mode initial load
        const syncBtn = document.getElementById('syncCdxBtn');
        if (syncBtn) syncBtn.style.display = '';
        const data = await fetchLivePositions();
        if (data) {
            renderLiveTable(data.positions);
            updateLiveSummary(data);
            startLiveRefresh();
        }
        return;
    }

    // PAPER mode initial load
    try {
        const res = await fetch(tradebookUrl());
        const data = await res.json();
        if (data?.trades?.length > 0) {
            updateAll(data);
            return;
        }
        const allRes = await fetch(`${API_BASE}/api/all`);
        const allData = await allRes.json();
        const built = buildFromMultiState(allData);
        if (built && built.trades.length > 0) {
            console.log('📗 Initial load: built tradebook from active positions');
            updateAll(built);
        }
    } catch (e) {
        console.log('Initial tradebook fetch failed, waiting for WebSocket...');
    }
}, 800);

// ─── Live Price Tick (1-second updates) ──────────────────────────────────────
socket.on('price-tick', (data) => {
    if (dialogOpen) return;  // Skip while confirm is open
    if (!data?.prices || !tradebookData?.trades) return;

    // Cache live prices to prevent flickering on file-based re-renders
    Object.assign(livePriceCache, data.prices);

    let changed = false;
    tradebookData.trades.forEach(trade => {
        if (trade.status !== 'ACTIVE') return;
        const livePrice = data.prices[trade.symbol];
        if (livePrice === undefined) return;

        trade.current_price = livePrice;

        // Recalculate unrealized P&L
        const entry = trade.entry_price;
        const qty = trade.quantity;
        const lev = trade.leverage;
        const capital = trade.capital || 100;

        let rawPnl;
        if (trade.position === 'LONG') {
            rawPnl = (livePrice - entry) * qty;
        } else {
            rawPnl = (entry - livePrice) * qty;
        }
        const isLive = trade.mode === 'LIVE';
        trade.unrealized_pnl = parseFloat((isLive ? rawPnl : rawPnl * lev).toFixed(4));
        trade.unrealized_pnl_pct = capital > 0
            ? parseFloat((trade.unrealized_pnl / capital * 100).toFixed(2))
            : 0;

        changed = true;
    });

    if (changed) {
        // Update table cells in-place (without full re-render for smooth 1s updates)
        document.querySelectorAll('.tradebook-table tbody tr').forEach(row => {
            const idCell = row.querySelector('.col-id');
            if (!idCell) return;
            const tradeId = idCell.textContent.trim();
            const trade = tradebookData.trades.find(t => t.trade_id === tradeId);
            if (!trade || trade.status !== 'ACTIVE') return;

            // Current price column (index 8, shifted +1 for checkbox)
            const cells = row.querySelectorAll('td');
            if (cells[8]) {
                cells[8].textContent = formatPrice(trade.current_price);
                cells[8].className = 'col-price ' + pnlClass(trade.unrealized_pnl);
            }
            // SL / TP column (index 9)
            if (cells[9]) {
                const sl = trade.trailing_sl || trade.stop_loss;
                const tp = trade.trailing_tp || trade.take_profit;
                const trailIcon = trade.trailing_active ? ' <span class="trail-indicator">⟟</span>' : '';
                cells[9].innerHTML = `${formatPrice(sl)} / ${formatPrice(tp)}${trailIcon}`;
            }
            // Unrealized P&L column (index 11)
            if (cells[11]) {
                cells[11].textContent = `${formatPnl(trade.unrealized_pnl)} (${formatPnlPct(trade.unrealized_pnl_pct)})`;
                cells[11].className = pnlClass(trade.unrealized_pnl);
            }
        });

        // Recompute summary
        const active = tradebookData.trades.filter(t => t.status === 'ACTIVE');
        const totalU = active.reduce((s, t) => s + (t.unrealized_pnl || 0), 0);
        const totalUPct = active.length > 0 ? totalU / (active.length * 100) * 100 : 0;
        if (tradebookData.summary) {
            tradebookData.summary.total_unrealized_pnl = parseFloat(totalU.toFixed(4));
            tradebookData.summary.total_unrealized_pnl_pct = parseFloat(totalUPct.toFixed(2));
            const realized = tradebookData.summary.total_realized_pnl || 0;
            tradebookData.summary.cumulative_pnl = parseFloat((realized + totalU).toFixed(4));
            updateSummary(tradebookData.summary);
        }
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
//  CUSTOM CONFIRM MODAL (replaces native confirm() which gets auto-dismissed)
// ═══════════════════════════════════════════════════════════════════════════════

function showConfirm(title, msg, icon = '⚠️', confirmLabel = 'Confirm') {
    return new Promise((resolve) => {
        const modal = document.getElementById('confirmModal');
        document.getElementById('confirmIcon').textContent = icon;
        document.getElementById('confirmTitle').textContent = title;
        document.getElementById('confirmMsg').textContent = msg;
        modal.style.display = 'flex';

        const yesBtn = document.getElementById('confirmYes');
        const noBtn = document.getElementById('confirmNo');
        yesBtn.textContent = confirmLabel;

        function cleanup() {
            modal.style.display = 'none';
            yesBtn.removeEventListener('click', onYes);
            noBtn.removeEventListener('click', onNo);
        }
        function onYes() { cleanup(); resolve(true); }
        function onNo() { cleanup(); resolve(false); }

        yesBtn.addEventListener('click', onYes);
        noBtn.addEventListener('click', onNo);
    });
}

async function deleteTrade(tradeId) {
    const yes = await showConfirm(
        `Delete Trade ${tradeId}?`,
        'This will permanently remove this trade. This cannot be undone.',
        '🗑️'
    );
    if (!yes) return;
    try {
        const res = await fetch(`${API_BASE}/api/tradebook/trade/${tradeId}`, { method: 'DELETE' });
        const result = await res.json();
        if (result.success) {
            showToast(`Trade ${tradeId} deleted`);
            refreshData();
        } else {
            showToast(result.error || 'Failed to delete', 'error');
        }
    } catch (e) {
        showToast('Failed to delete trade', 'error');
    }
}

async function deleteAllTrades() {
    const yes = await showConfirm(
        'Delete ALL Trades?',
        'This will permanently wipe the entire tradebook including all active and closed trades. This cannot be undone.',
        '🗑️'
    );
    if (!yes) return;
    try {
        const res = await fetch(`${API_BASE}/api/tradebook/all`, { method: 'DELETE' });
        const result = await res.json();
        if (result.success) {
            showToast('All trades deleted');
            refreshData();
        } else {
            showToast(result.error || 'Failed to delete', 'error');
        }
    } catch (e) {
        showToast('Failed to delete trades', 'error');
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MULTI-SELECT CLOSE TRADES
// ═══════════════════════════════════════════════════════════════════════════════

function toggleSelectAll(masterCb) {
    document.querySelectorAll('.trade-select:not(:disabled)').forEach(cb => {
        cb.checked = masterCb.checked;
    });
    updateCloseSelectedBar();
}

function updateCloseSelectedBar() {
    const selected = document.querySelectorAll('.trade-select:checked');
    const btn = document.getElementById('closeSelectedBtn');
    const countEl = document.getElementById('selectedCount');
    if (btn) btn.style.display = selected.length > 0 ? 'inline-flex' : 'none';
    if (countEl) countEl.textContent = selected.length > 0 ? `${selected.length} trade(s) selected` : '';
}

async function closeSelectedTrades() {
    const selected = document.querySelectorAll('.trade-select:checked');
    const tradeIds = Array.from(selected).map(cb => cb.dataset.tradeid);
    if (tradeIds.length === 0) return;

    console.log('[Close Selected] Trade IDs:', tradeIds);

    const yes = await showConfirm(
        `Close ${tradeIds.length} Trade(s)?`,
        `This will close ${tradeIds.length} active trade(s) at their current market price with MANUAL exit reason.`,
        '⏹️',
        'Close Trades'
    );
    if (!yes) return;

    try {
        console.log('[Close Selected] Sending POST to /api/tradebook/close');
        const res = await fetch(`${API_BASE}/api/tradebook/close`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ trade_ids: tradeIds }),
        });
        const result = await res.json();
        console.log('[Close Selected] Response:', result);
        if (result.success) {
            showToast(`Closed ${result.closed} trade(s)`);
            refreshData();
        } else {
            showToast(result.error || 'Failed to close', 'error');
        }
    } catch (e) {
        console.error('[Close Selected] Error:', e);
        showToast('Failed to close trades: ' + e.message, 'error');
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PDF REPORT GENERATOR + REPORT HISTORY
// ═══════════════════════════════════════════════════════════════════════════════

function computeReportStats(trades) {
    const closed = trades.filter(t => t.status === 'CLOSED');
    const active = trades.filter(t => t.status === 'ACTIVE');
    if (closed.length === 0) return null;

    const wins = closed.filter(t => (t.realized_pnl || 0) > 0);
    const losses = closed.filter(t => (t.realized_pnl || 0) < 0);
    const totalPnl = closed.reduce((s, t) => s + (t.realized_pnl || 0), 0);
    const totalWins = wins.reduce((s, t) => s + (t.realized_pnl || 0), 0);
    const totalLosses = Math.abs(losses.reduce((s, t) => s + (t.realized_pnl || 0), 0));
    const totalCommission = closed.reduce((s, t) => s + (t.commission || 0), 0);
    const totalCapital = closed.reduce((s, t) => s + (t.capital || 100), 0);

    const winRate = (wins.length / closed.length * 100);
    const profitFactor = totalLosses > 0 ? (totalWins / totalLosses) : totalWins > 0 ? Infinity : 0;
    const avgWin = wins.length > 0 ? totalWins / wins.length : 0;
    const avgLoss = losses.length > 0 ? totalLosses / losses.length : 0;
    const riskReward = avgLoss > 0 ? (avgWin / avgLoss) : avgWin > 0 ? Infinity : 0;

    // Sharpe Ratio (daily-ish approximation)
    const pnls = closed.map(t => t.realized_pnl || 0);
    const meanPnl = pnls.reduce((a, b) => a + b, 0) / pnls.length;
    const variance = pnls.reduce((s, p) => s + Math.pow(p - meanPnl, 2), 0) / pnls.length;
    const stdDev = Math.sqrt(variance);
    const sharpe = stdDev > 0 ? (meanPnl / stdDev) : 0;

    // Date range
    const exitDates = closed.map(t => new Date(t.exit_timestamp)).filter(d => !isNaN(d));
    const minDate = exitDates.length > 0 ? new Date(Math.min(...exitDates)).toLocaleDateString('en-IN') : 'N/A';
    const maxDate = exitDates.length > 0 ? new Date(Math.max(...exitDates)).toLocaleDateString('en-IN') : 'N/A';

    // Regime breakdown
    const regimes = {};
    closed.forEach(t => {
        const r = t.regime || 'UNKNOWN';
        if (!regimes[r]) regimes[r] = { count: 0, wins: 0, pnl: 0 };
        regimes[r].count++;
        if ((t.realized_pnl || 0) > 0) regimes[r].wins++;
        regimes[r].pnl += (t.realized_pnl || 0);
    });

    // Leverage breakdown
    const leverages = {};
    closed.forEach(t => {
        const l = t.leverage || 1;
        if (!leverages[l]) leverages[l] = { count: 0, wins: 0, pnl: 0 };
        leverages[l].count++;
        if ((t.realized_pnl || 0) > 0) leverages[l].wins++;
        leverages[l].pnl += (t.realized_pnl || 0);
    });

    // Top 5 winners/losers
    const sortedByPnl = [...closed].sort((a, b) => (b.realized_pnl || 0) - (a.realized_pnl || 0));
    const topWinners = sortedByPnl.slice(0, 5);
    const topLosers = sortedByPnl.slice(-5).reverse();

    // Max Drawdown
    let peak = 0, maxDD = 0, eq = 0;
    [...closed].sort((a, b) => new Date(a.exit_timestamp || 0) - new Date(b.exit_timestamp || 0))
        .forEach(t => {
            eq += (t.realized_pnl || 0);
            if (eq > peak) peak = eq;
            const dd = peak - eq;
            if (dd > maxDD) maxDD = dd;
        });
    const maxDrawdownPct = (maxDD / 1500 * 100);

    return {
        closedCount: closed.length, activeCount: active.length,
        wins: wins.length, losses: losses.length,
        totalPnl, totalWins, totalLosses, totalCommission, totalCapital,
        winRate, profitFactor, avgWin, avgLoss, riskReward, sharpe,
        maxDrawdownPct, maxDrawdownDollar: maxDD,
        minDate, maxDate, regimes, leverages, topWinners, topLosers,
    };
}

function generateInsights(stats, trades) {
    const closed = (trades || []).filter(t => t.status === 'CLOSED');
    const insights = [];

    // ── 1. Win Rate Assessment ──
    if (stats.winRate < 40) {
        insights.push({
            title: 'CRITICAL: Low Win Rate',
            body: `Win rate is ${stats.winRate.toFixed(1)}% across ${stats.closedCount} trades. Only ${stats.wins} out of ${stats.closedCount} trades were profitable.`,
            action: 'Increase HMM confidence threshold from current level to 70%+ before entering trades. Filter out low-probability setups.'
        });
    } else if (stats.winRate < 50) {
        insights.push({
            title: 'Win Rate Below Breakeven',
            body: `Win rate is ${stats.winRate.toFixed(1)}%. This requires a risk-reward ratio above 1:1 to be profitable.`,
            action: 'Current R:R is ' + stats.riskReward.toFixed(2) + ':1. ' + (stats.riskReward >= 1 ? 'R:R compensates for low WR — maintain current TP levels.' : 'Both WR and R:R are weak. Widen TP targets or tighten SL distances.')
        });
    } else {
        insights.push({
            title: 'Healthy Win Rate',
            body: `Win rate is ${stats.winRate.toFixed(1)}% (${stats.wins}W / ${stats.losses}L). Strategy has positive edge.`,
            action: 'Maintain current entry criteria. Consider scaling position sizes gradually.'
        });
    }

    // ── 2. Profit Factor & P&L Analysis ──
    const netAfterComm = stats.totalPnl - stats.totalCommission;
    if (stats.profitFactor < 1) {
        insights.push({
            title: 'Negative Expectancy',
            body: `Profit Factor ${stats.profitFactor.toFixed(2)} means every $1 won costs $${(1 / stats.profitFactor).toFixed(2)} in losses. Total wins: $${stats.totalWins.toFixed(2)}, Total losses: $${stats.totalLosses.toFixed(2)}.`,
            action: 'Reduce position sizes by 50% until PF exceeds 1.2. Consider pausing LIVE trading and running extended backtests to validate edge.'
        });
    } else if (stats.profitFactor < 1.5) {
        insights.push({
            title: 'Marginal Profitability',
            body: `PF is ${stats.profitFactor.toFixed(2)}. After $${stats.totalCommission.toFixed(2)} commission, net P&L is $${netAfterComm.toFixed(2)}.`,
            action: 'Tighten SL by 10-20% or extend TP targets. Commission drag of $' + stats.totalCommission.toFixed(2) + ' erodes thin margins.'
        });
    }

    // ── 3. Asset-Level Analysis ──
    const assetPnl = {};
    closed.forEach(t => {
        const sym = t.symbol || 'UNKNOWN';
        if (!assetPnl[sym]) assetPnl[sym] = { count: 0, wins: 0, pnl: 0 };
        assetPnl[sym].count++;
        if ((t.realized_pnl || 0) > 0) assetPnl[sym].wins++;
        assetPnl[sym].pnl += (t.realized_pnl || 0);
    });

    const sortedAssets = Object.entries(assetPnl).sort((a, b) => a[1].pnl - b[1].pnl);
    const worstAsset = sortedAssets[0];
    const bestAsset = sortedAssets[sortedAssets.length - 1];

    if (worstAsset && worstAsset[1].pnl < -5 && worstAsset[1].count >= 2) {
        const wa = worstAsset[1];
        const waWr = (wa.wins / wa.count * 100).toFixed(0);
        insights.push({
            title: `Worst Performer: ${worstAsset[0]}`,
            body: `${worstAsset[0]} lost $${Math.abs(wa.pnl).toFixed(2)} across ${wa.count} trades (${waWr}% WR). This accounts for ${(Math.abs(wa.pnl) / stats.totalLosses * 100).toFixed(0)}% of all losses.`,
            action: `Add ${worstAsset[0]} to the exclusion list in coin_scanner.py, or reduce leverage for this asset specifically.`
        });
    }

    if (bestAsset && bestAsset[1].pnl > 5 && bestAsset[1].count >= 2) {
        const ba = bestAsset[1];
        insights.push({
            title: `Best Performer: ${bestAsset[0]}`,
            body: `${bestAsset[0]} earned $${ba.pnl.toFixed(2)} across ${ba.count} trades (${(ba.wins / ba.count * 100).toFixed(0)}% WR).`,
            action: 'Consider increasing allocation or priority weighting for this asset.'
        });
    }

    // ── 4. Hold Duration Analysis ──
    const durations = closed.map(t => {
        if (!t.entry_timestamp || !t.exit_timestamp) return null;
        const ms = new Date(t.exit_timestamp) - new Date(t.entry_timestamp);
        return { mins: ms / 60000, pnl: t.realized_pnl || 0, win: (t.realized_pnl || 0) > 0 };
    }).filter(Boolean);

    if (durations.length > 5) {
        const short = durations.filter(d => d.mins < 30);
        const medium = durations.filter(d => d.mins >= 30 && d.mins <= 120);
        const long = durations.filter(d => d.mins > 120);

        const shortWR = short.length > 0 ? (short.filter(d => d.win).length / short.length * 100) : 0;
        const medWR = medium.length > 0 ? (medium.filter(d => d.win).length / medium.length * 100) : 0;
        const longWR = long.length > 0 ? (long.filter(d => d.win).length / long.length * 100) : 0;

        let bestWindow = 'medium (30-120 min)';
        let bestWR = medWR;
        if (shortWR > medWR && shortWR > longWR) { bestWindow = 'short (<30 min)'; bestWR = shortWR; }
        if (longWR > medWR && longWR > shortWR) { bestWindow = 'long (>120 min)'; bestWR = longWR; }

        insights.push({
            title: 'Optimal Hold Duration',
            body: `Short (<30m): ${short.length} trades, ${shortWR.toFixed(0)}% WR. Medium (30-120m): ${medium.length} trades, ${medWR.toFixed(0)}% WR. Long (>2h): ${long.length} trades, ${longWR.toFixed(0)}% WR.`,
            action: `Best window is ${bestWindow} at ${bestWR.toFixed(0)}% WR. ${shortWR < 40 && short.length >= 3 ? 'Consider adding minimum hold time before exits.' : ''}`
        });
    }

    // ── 5. Exit Reason Analysis ──
    const exits = {};
    closed.forEach(t => {
        const reason = (t.exit_reason || 'UNKNOWN').split(':')[0];
        if (!exits[reason]) exits[reason] = { count: 0, pnl: 0 };
        exits[reason].count++;
        exits[reason].pnl += (t.realized_pnl || 0);
    });

    const exitEntries = Object.entries(exits);
    if (exitEntries.length > 0) {
        const exitLines = exitEntries.map(([r, d]) => `${r}: ${d.count} trades, P&L $${d.pnl.toFixed(2)}`).join('. ');
        const worstExit = exitEntries.sort((a, b) => a[1].pnl - b[1].pnl)[0];
        insights.push({
            title: 'Exit Reason Breakdown',
            body: exitLines,
            action: worstExit[1].pnl < 0 ? `${worstExit[0]} exits are losing $${Math.abs(worstExit[1].pnl).toFixed(2)}. Review if this exit trigger is too aggressive.` : 'Exit triggers are performing as expected.'
        });
    }

    // ── 6. Regime-Specific Insights ──
    for (const [regime, data] of Object.entries(stats.regimes)) {
        const wr = data.count > 0 ? (data.wins / data.count * 100) : 0;
        if (wr < 35 && data.count >= 3) {
            insights.push({
                title: `Weak Regime: ${regime}`,
                body: `${regime} has ${wr.toFixed(0)}% WR across ${data.count} trades. Net P&L: $${data.pnl.toFixed(2)}.`,
                action: `Disable ${regime} regime trading or increase confidence threshold to 80%+ for this regime.`
            });
        } else if (wr >= 60 && data.count >= 3) {
            insights.push({
                title: `Strong Regime: ${regime}`,
                body: `${regime} has ${wr.toFixed(0)}% WR across ${data.count} trades. Net P&L: $${data.pnl.toFixed(2)}.`,
                action: 'Consider increasing position sizes during this regime.'
            });
        }
    }

    // ── 7. Leverage Efficiency ──
    for (const [lev, data] of Object.entries(stats.leverages)) {
        const wr = data.count > 0 ? (data.wins / data.count * 100) : 0;
        const avgPnl = data.pnl / data.count;
        if (parseInt(lev) >= 20 && wr < 45 && data.count >= 3) {
            insights.push({
                title: `High Leverage Risk: ${lev}x`,
                body: `${lev}x leverage: ${data.count} trades, ${wr.toFixed(0)}% WR, avg P&L $${avgPnl.toFixed(2)}/trade.`,
                action: `Review conviction scoring thresholds. High leverage (${lev}x) is amplifying losses on the ${(100 - wr).toFixed(0)}% losing trades. Consider raising minimum conviction score.`
            });
        }
    }

    // ── 8. Commission Drag ──
    if (stats.totalCommission > 0) {
        const commPctOfPnl = stats.totalPnl !== 0 ? Math.abs(stats.totalCommission / stats.totalPnl * 100) : 100;
        const commPerTrade = stats.totalCommission / stats.closedCount;
        insights.push({
            title: 'Commission Analysis',
            body: `Total commission: $${stats.totalCommission.toFixed(2)} ($${commPerTrade.toFixed(3)}/trade). Commission is ${commPctOfPnl.toFixed(0)}% of total P&L.`,
            action: commPctOfPnl > 30 ? 'Commission is eating significant profits. Switch to limit orders (0.02% maker vs 0.04% taker) and avoid rapid re-entries.' : 'Commission drag is manageable at current levels.'
        });
    }

    // ── 9. Consecutive Loss Streaks ──
    let maxStreak = 0, currentStreak = 0;
    const sortedByExit = [...closed].sort((a, b) => new Date(a.exit_timestamp) - new Date(b.exit_timestamp));
    sortedByExit.forEach(t => {
        if ((t.realized_pnl || 0) < 0) { currentStreak++; maxStreak = Math.max(maxStreak, currentStreak); }
        else { currentStreak = 0; }
    });
    if (maxStreak >= 4) {
        insights.push({
            title: 'Loss Streak Warning',
            body: `Maximum consecutive losing streak: ${maxStreak} trades. Extended streaks can cause emotional overtrading.`,
            action: 'Add a cooldown period: pause new entries for 15 minutes after 3 consecutive losses.'
        });
    }

    return insights;
}

function _buildReportPdf(stats) {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF('p', 'mm', 'a4');
    const pageWidth = doc.internal.pageSize.getWidth();
    let y = 15;

    // ── Header ──
    doc.setFillColor(26, 35, 50);
    doc.rect(0, 0, pageWidth, 28, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(20);
    doc.setFont('helvetica', 'bold');
    doc.text('SENTINEL — Strategy Performance Report', pageWidth / 2, 12, { align: 'center' });
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text(`Generated: ${new Date().toLocaleString('en-IN')} | Period: ${stats.minDate} – ${stats.maxDate}`, pageWidth / 2, 22, { align: 'center' });
    y = 36;

    // ── Summary Statistics ──
    doc.setTextColor(26, 35, 50);
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Summary Statistics', 14, y);
    y += 2;

    doc.autoTable({
        startY: y,
        head: [['Metric', 'Value']],
        body: [
            ['Closed Trades', stats.closedCount],
            ['Active Trades', stats.activeCount],
            ['Wins / Losses', `${stats.wins} / ${stats.losses}`],
            ['Win Rate', `${stats.winRate.toFixed(1)}%`],
            ['Total P&L', `$${stats.totalPnl.toFixed(2)}`],
            ['Profit Factor', stats.profitFactor === Infinity ? '∞' : stats.profitFactor.toFixed(2)],
            ['Sharpe Ratio', stats.sharpe.toFixed(3)],
            ['Risk-Reward', `${stats.riskReward === Infinity ? '∞' : stats.riskReward.toFixed(2)}:1`],
            ['Max Drawdown', `-$${stats.maxDrawdownDollar.toFixed(2)} (${stats.maxDrawdownPct.toFixed(1)}%)`],
            ['Avg Win', `$${stats.avgWin.toFixed(2)}`],
            ['Avg Loss', `-$${stats.avgLoss.toFixed(2)}`],
            ['Total Commission', `$${stats.totalCommission.toFixed(2)}`],
            ['Total Capital Deployed', `$${stats.totalCapital.toFixed(0)}`],
            ['Max Portfolio Capital', '$1,500'],
            ['ROI (on Max Capital)', `${(stats.totalPnl / 1500 * 100).toFixed(2)}%`],
            ['Period', `${stats.minDate} – ${stats.maxDate}`],
        ],
        theme: 'striped',
        headStyles: { fillColor: [26, 35, 50], fontSize: 9 },
        bodyStyles: { fontSize: 9 },
        columnStyles: { 0: { fontStyle: 'bold', cellWidth: 55 } },
        margin: { left: 14, right: 14 },
    });
    y = doc.lastAutoTable.finalY + 8;

    // ── Regime Breakdown ──
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Regime Breakdown', 14, y);
    y += 2;

    const regimeRows = Object.entries(stats.regimes).map(([r, d]) => [
        r, d.count, d.wins, d.count - d.wins,
        `${(d.wins / d.count * 100).toFixed(0)}%`, `$${d.pnl.toFixed(2)}`
    ]);

    doc.autoTable({
        startY: y,
        head: [['Regime', 'Trades', 'Wins', 'Losses', 'Win Rate', 'P&L']],
        body: regimeRows,
        theme: 'striped',
        headStyles: { fillColor: [59, 130, 246], fontSize: 9 },
        bodyStyles: { fontSize: 9 },
        margin: { left: 14, right: 14 },
    });
    y = doc.lastAutoTable.finalY + 8;

    // ── Leverage Breakdown ──
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Leverage Breakdown', 14, y);
    y += 2;

    const levRows = Object.entries(stats.leverages)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
        .map(([l, d]) => [
            `${l}x`, d.count, d.wins, d.count - d.wins,
            `${(d.wins / d.count * 100).toFixed(0)}%`, `$${d.pnl.toFixed(2)}`
        ]);

    doc.autoTable({
        startY: y,
        head: [['Leverage', 'Trades', 'Wins', 'Losses', 'Win Rate', 'P&L']],
        body: levRows,
        theme: 'striped',
        headStyles: { fillColor: [139, 92, 246], fontSize: 9 },
        bodyStyles: { fontSize: 9 },
        margin: { left: 14, right: 14 },
    });
    y = doc.lastAutoTable.finalY + 8;

    // ── Page break for insights ──
    if (y > 220) { doc.addPage(); y = 15; }

    // ── Top 5 Winners ──
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Top 5 Winners', 14, y);
    y += 2;

    doc.autoTable({
        startY: y,
        head: [['Symbol', 'Side', 'Lev', 'P&L', 'P&L %', 'Exit']],
        body: stats.topWinners.map(t => [
            t.symbol, t.position, `${t.leverage}x`,
            `$${(t.realized_pnl || 0).toFixed(2)}`, `${(t.realized_pnl_pct || 0).toFixed(2)}%`,
            t.exit_reason || '-'
        ]),
        theme: 'striped',
        headStyles: { fillColor: [34, 197, 94], fontSize: 9 },
        bodyStyles: { fontSize: 9 },
        margin: { left: 14, right: 14 },
    });
    y = doc.lastAutoTable.finalY + 8;

    // ── Top 5 Losers ──
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Top 5 Losers', 14, y);
    y += 2;

    doc.autoTable({
        startY: y,
        head: [['Symbol', 'Side', 'Lev', 'P&L', 'P&L %', 'Exit']],
        body: stats.topLosers.map(t => [
            t.symbol, t.position, `${t.leverage}x`,
            `$${(t.realized_pnl || 0).toFixed(2)}`, `${(t.realized_pnl_pct || 0).toFixed(2)}%`,
            t.exit_reason || '-'
        ]),
        theme: 'striped',
        headStyles: { fillColor: [239, 68, 68], fontSize: 9 },
        bodyStyles: { fontSize: 9 },
        margin: { left: 14, right: 14 },
    });
    y = doc.lastAutoTable.finalY + 8;

    // ── Page break for insights ──
    if (y > 200) { doc.addPage(); y = 15; }

    // ── Actionable Insights (using autoTable for proper wrapping) ──
    doc.setTextColor(26, 35, 50);
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Actionable Insights & Recommendations', 14, y);
    y += 2;

    const insights = generateInsights(stats, tradebookData?.trades || []);
    const insightRows = insights.map(i => [i.title, i.body, i.action]);

    doc.autoTable({
        startY: y,
        head: [['Finding', 'Analysis', 'Recommended Action']],
        body: insightRows,
        theme: 'striped',
        headStyles: { fillColor: [26, 35, 50], fontSize: 8, cellPadding: 3 },
        bodyStyles: { fontSize: 8, cellPadding: 3, lineHeight: 1.3 },
        columnStyles: {
            0: { fontStyle: 'bold', cellWidth: 35 },
            1: { cellWidth: 75 },
            2: { cellWidth: 72, textColor: [30, 80, 30] },
        },
        margin: { left: 14, right: 14 },
        didParseCell: function (data) {
            if (data.section === 'body' && data.column.index === 0) {
                const title = data.cell.raw || '';
                if (title.includes('CRITICAL') || title.includes('Negative') || title.includes('Worst') || title.includes('Weak') || title.includes('Risk') || title.includes('Warning')) {
                    data.cell.styles.textColor = [220, 38, 38];
                } else if (title.includes('Best') || title.includes('Healthy') || title.includes('Strong')) {
                    data.cell.styles.textColor = [34, 140, 34];
                } else {
                    data.cell.styles.textColor = [60, 60, 60];
                }
            }
        }
    });

    // ── Footer ──
    const pages = doc.internal.getNumberOfPages();
    for (let i = 1; i <= pages; i++) {
        doc.setPage(i);
        doc.setFontSize(8);
        doc.setTextColor(150);
        doc.text(`SENTINEL v2.0 — Page ${i}/${pages}`, pageWidth / 2, 290, { align: 'center' });
    }

    return doc;
}

function generatePdfReport() {
    if (!tradebookData?.trades || tradebookData.trades.length === 0) {
        showToast('No trades available to generate report', 'error');
        return;
    }

    const stats = computeReportStats(tradebookData.trades);
    if (!stats) {
        showToast('No closed trades to analyze', 'error');
        return;
    }

    const doc = _buildReportPdf(stats);

    // Save PDF
    const filename = `strategy_report_${new Date().toISOString().slice(0, 10)}_${Date.now()}.pdf`;
    doc.save(filename);

    // Store PDF base64 for persistent access
    const pdfBase64 = doc.output('datauristring');

    // Save to report history
    const report = {
        timestamp: new Date().toISOString(),
        filename,
        closedTrades: stats.closedCount,
        winRate: stats.winRate.toFixed(1) + '%',
        profitFactor: stats.profitFactor === Infinity ? '∞' : stats.profitFactor.toFixed(2),
        sharpe: stats.sharpe.toFixed(3),
        riskReward: (stats.riskReward === Infinity ? '∞' : stats.riskReward.toFixed(2)) + ':1',
        totalPnl: '$' + stats.totalPnl.toFixed(2),
        totalCommission: '$' + stats.totalCommission.toFixed(2),
        period: `${stats.minDate} – ${stats.maxDate}`,
        pdfData: pdfBase64,
    };

    const history = JSON.parse(localStorage.getItem('reportHistory') || '[]');
    history.unshift(report);
    if (history.length > 50) history.pop();
    localStorage.setItem('reportHistory', JSON.stringify(history));
    renderReportHistory();

    showToast(`Report saved: ${filename}`);
}

// ── Report History Table ──

function renderReportHistory() {
    const area = document.getElementById('reportHistoryArea');
    if (!area) return;

    const history = JSON.parse(localStorage.getItem('reportHistory') || '[]');
    if (history.length === 0) {
        area.innerHTML = '<div class="empty-state" style="padding:24px"><p style="color:#64748B;font-size:13px">No reports generated yet.</p></div>';
        return;
    }

    let html = `<table class="tradebook-table" style="font-size:12px">
    <thead><tr>
      <th>Date</th>
      <th>Closed</th>
      <th>Win Rate</th>
      <th>Profit Factor</th>
      <th>Sharpe</th>
      <th>R:R</th>
      <th>Total P&L</th>
      <th>Commission</th>
      <th>Period</th>
      <th>Action</th>
    </tr></thead><tbody>`;

    history.forEach((r, idx) => {
        const dt = new Date(r.timestamp);
        const dateStr = dt.toLocaleString('en-IN', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
        const pnlVal = parseFloat(r.totalPnl.replace('$', ''));
        const pnlClass = pnlVal >= 0 ? 'pnl-positive' : 'pnl-negative';
        html += `<tr>
          <td>${dateStr}</td>
          <td>${r.closedTrades}</td>
          <td>${r.winRate}</td>
          <td>${r.profitFactor}</td>
          <td>${r.sharpe}</td>
          <td>${r.riskReward}</td>
          <td class="${pnlClass}">${r.totalPnl}</td>
          <td>${r.totalCommission}</td>
          <td style="font-size:10px">${r.period}</td>
          <td>
            <div style="display:flex;flex-direction:column;gap:4px;align-items:flex-start">
              <span style="font-size:10px;color:#64748B;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${r.filename}">${r.filename}</span>
              <div style="display:flex;gap:6px">
                <button onclick="viewReportPdf(${idx})" style="padding:3px 8px;font-size:10px;border:1px solid #CBD5E1;border-radius:6px;background:#F8FAFC;cursor:pointer;color:#334155">View</button>
                <button onclick="downloadReportPdf(${idx})" style="padding:3px 8px;font-size:10px;border:1px solid #8B5CF6;border-radius:6px;background:#F5F3FF;cursor:pointer;color:#6D28D9">Download</button>
                <button onclick="deleteReportEntry(${idx})" style="padding:3px 8px;font-size:10px;border:1px solid #FCA5A5;border-radius:6px;background:#FEF2F2;cursor:pointer;color:#DC2626">Delete</button>
              </div>
            </div>
          </td>
        </tr>`;
    });

    html += '</tbody></table>';
    area.innerHTML = html;
}

function viewReportPdf(idx) {
    const history = JSON.parse(localStorage.getItem('reportHistory') || '[]');
    const r = history[idx];
    if (!r) return;

    if (r.pdfData) {
        // Open stored PDF
        const byteString = atob(r.pdfData.split(',')[1]);
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) ia[i] = byteString.charCodeAt(i);
        const blob = new Blob([ab], { type: 'application/pdf' });
        window.open(URL.createObjectURL(blob), '_blank');
    } else {
        // Fallback: regenerate from current data
        const stats = computeReportStats(tradebookData.trades);
        if (!stats) { showToast('No closed trades to regenerate', 'error'); return; }
        const doc = _buildReportPdf(stats);
        window.open(URL.createObjectURL(doc.output('blob')), '_blank');
    }
}

function downloadReportPdf(idx) {
    const history = JSON.parse(localStorage.getItem('reportHistory') || '[]');
    const r = history[idx];
    if (!r) return;

    if (r.pdfData) {
        // Download stored PDF
        const link = document.createElement('a');
        link.href = r.pdfData;
        link.download = r.filename;
        link.click();
    } else {
        // Fallback: regenerate from current data
        const stats = computeReportStats(tradebookData.trades);
        if (!stats) { showToast('No closed trades to regenerate', 'error'); return; }
        const doc = _buildReportPdf(stats);
        doc.save(r.filename);
    }
}

function deleteReportEntry(idx) {
    const history = JSON.parse(localStorage.getItem('reportHistory') || '[]');
    history.splice(idx, 1);
    localStorage.setItem('reportHistory', JSON.stringify(history));
    renderReportHistory();
    showToast('Report deleted');
}

// Load report history on startup
document.addEventListener('DOMContentLoaded', renderReportHistory);

// ═══════════════════════════════════════════════════════════════════════════════
//  MASTER MODE TOGGLE INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

// Listen for master toggle changes
window.addEventListener('mode-change', (e) => {
    const mode = e.detail?.mode || 'PAPER';
    const syncBtn = document.getElementById('syncCdxBtn');
    if (syncBtn) syncBtn.style.display = mode === 'LIVE' ? '' : 'none';
    showToast(`Switched to ${mode} mode`);
    refreshData();
});

// ─── CoinDCX Live Positions Fetch ────────────────────────────────────────────
async function fetchLivePositions() {
    try {
        const res = await fetch(`${API_BASE}/api/coindcx/positions`);
        const data = await res.json();
        if (!data.success) {
            showToast('Failed to fetch CoinDCX positions: ' + (data.error || 'Unknown error'), 'error');
            return null;
        }
        livePositionsData = data;
        return data;
    } catch (e) {
        showToast('CoinDCX API unreachable', 'error');
        return null;
    }
}

// ─── Render Live Positions Table ─────────────────────────────────────────────
function renderLiveTable(positions) {
    const area = document.getElementById('tradebookArea');
    if (!positions || positions.length === 0) {
        area.innerHTML = `<div class="empty-state">
            <div class="icon">🔌</div>
            <p>No active positions on CoinDCX.</p>
        </div>`;
        return;
    }

    let html = `<table class="tradebook-table">
    <thead><tr>
      <th>Symbol</th>
      <th>Side</th>
      <th>Leverage</th>
      <th>Quantity</th>
      <th>Entry Price</th>
      <th>Mark Price</th>
      <th>Margin</th>
      <th>Unrealized PnL</th>
      <th>PnL %</th>
      <th>Liq. Price</th>
      <th>Opened</th>
      <th>Status</th>
    </tr></thead><tbody>`;

    positions.forEach(p => {
        const posClass = p.side === 'LONG' ? 'side-buy' : 'side-sell';
        const posIcon = p.side === 'LONG' ? '▲' : '▼';
        const pnlCls = p.pnl >= 0 ? 'green' : 'red';
        const pnlSign = p.pnl >= 0 ? '+' : '';
        const opened = p.created_at ? new Date(p.created_at).toLocaleString('en-IN', {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        }) : '—';

        html += `<tr>
      <td><strong>${p.symbol || '—'}</strong></td>
      <td class="${posClass}">${posIcon} ${p.side}</td>
      <td><span class="leverage-badge">${p.leverage}x</span></td>
      <td>${p.quantity?.toFixed(4) || '—'}</td>
      <td class="col-price">${formatPrice(p.entry_price)}</td>
      <td class="col-price">${formatPrice(p.mark_price)}</td>
      <td class="col-price">$${p.locked_margin?.toFixed(4) || '0'}</td>
      <td class="${pnlCls}" style="font-weight:600">${pnlSign}$${p.pnl?.toFixed(4) || '0'}</td>
      <td class="${pnlCls}" style="font-weight:600">${pnlSign}${p.pnl_pct?.toFixed(2) || '0'}%</td>
      <td class="col-price">${formatPrice(p.liquidation_price)}</td>
      <td>${opened}</td>
      <td><span class="status-badge active">ACTIVE</span></td>
    </tr>`;
    });

    html += '</tbody></table>';
    area.innerHTML = html;
}

// ─── Update Live Summary ─────────────────────────────────────────────────────
function updateLiveSummary(data) {
    const positions = data.positions || [];
    const totalPnl = positions.reduce((s, p) => s + (p.pnl || 0), 0);
    const totalMargin = positions.reduce((s, p) => s + (p.locked_margin || 0), 0);
    const pnlPct = totalMargin > 0 ? (totalPnl / totalMargin * 100) : 0;

    // Portfolio stats
    const el = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
    el('summTotalTrades', positions.length);
    el('summActiveTrades', positions.length);
    el('summClosedTrades', '0');
    el('summWins', '—');
    el('summLosses', '—');
    el('summWinRate', '—');
    el('summProfitFactor', '—');
    el('summSharpe', '—');
    el('summRiskReward', '—');
    el('summMaxDrawdown', '—');

    // PnL cards
    el('summRealizedPnl', '$0.00');
    el('summRealizedPnlPct', '0%');
    el('summUnrealizedPnl', formatPnl(totalPnl));
    el('summUnrealizedPnlPct', `${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%`);
    el('summCumulativePnl', formatPnl(totalPnl));
    el('summCumulativePnlPct', `${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%`);

    // Color them
    ['summUnrealizedPnl', 'summUnrealizedPnlPct', 'summCumulativePnl', 'summCumulativePnlPct'].forEach(id => {
        const e = document.getElementById(id);
        if (e) { e.className = totalPnl >= 0 ? 'pnl-value green' : 'pnl-value red'; }
    });

    // Wallet balance
    el('summBestTrade', `$${data.wallet_balance?.toFixed(2) || '0'}`);
    el('summWorstTrade', 'Wallet');
    el('summWinStreak', positions.length);
    el('summLossStreak', '—');
}

// ─── Auto-refresh live positions every 10 seconds ────────────────────────────
let liveRefreshInterval = null;
function startLiveRefresh() {
    if (liveRefreshInterval) clearInterval(liveRefreshInterval);
    liveRefreshInterval = setInterval(async () => {
        if (!isLiveMode()) { clearInterval(liveRefreshInterval); return; }
        const data = await fetchLivePositions();
        if (data) {
            renderLiveTable(data.positions);
            updateLiveSummary(data);
        }
    }, 3000);
}
