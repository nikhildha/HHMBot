/**
 * Project SENTINEL — Premium Dashboard UI Engine
 * Handles WebSocket real-time updates, Chart.js rendering,
 * and all dynamic UI components.
 */

// ─── Socket.IO Connection ────────────────────────────────────────────────────
const API_BASE = window.location.hostname === 'localhost'
    ? `http://localhost:3001`
    : window.location.origin;

const socket = io(API_BASE);

// ─── State ───────────────────────────────────────────────────────────────────
let currentState = {};
let multiState = {};
let scannerData = {};
let tradeLog = [];
let tradebookData = { trades: [] };
let regimeChart = null;
let confChart = null;

// ─── Regime Helpers ──────────────────────────────────────────────────────────
const REGIME_MAP = {
    'BULLISH': { emoji: '🟢', class: 'bull', color: '#22C55E' },
    'BEARISH': { emoji: '🔴', class: 'bear', color: '#EF4444' },
    'SIDEWAYS/CHOP': { emoji: '🟡', class: 'chop', color: '#F59E0B' },
    'CRASH/PANIC': { emoji: '💀', class: 'crash', color: '#DC2626' },
    'WAITING': { emoji: '⏳', class: 'chop', color: '#F59E0B' },
    'SCANNING': { emoji: '🔍', class: 'chop', color: '#3B82F6' },
    'OFFLINE': { emoji: '⚫', class: 'chop', color: '#8E9BB3' },
};

function getRegimeInfo(regime) {
    return REGIME_MAP[regime] || REGIME_MAP['WAITING'];
}

// ─── Format Helpers ──────────────────────────────────────────────────────────
function formatPrice(price) {
    const p = parseFloat(price);
    if (isNaN(p)) return '$0';
    if (p >= 1000) return '$' + p.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (p >= 1) return '$' + p.toFixed(4);
    return '$' + p.toFixed(6);
}

function formatTime(ts) {
    if (!ts) return '—';
    const d = new Date(ts);
    return d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'Asia/Kolkata' });
}

function formatDate(ts) {
    if (!ts) return '—';
    const d = new Date(ts);
    return d.toLocaleDateString('en-IN', { month: 'short', day: 'numeric', timeZone: 'Asia/Kolkata' }) + ' ' + formatTime(ts);
}

// ─── Toast Notification ──────────────────────────────────────────────────────
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  UI UPDATE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Header & Status ─────────────────────────────────────────────────────────
function updateHeader(state, multi) {
    const ts = multi?.timestamp || state?.timestamp;
    document.getElementById('lastUpdate').textContent = ts ? `Last: ${formatTime(ts)}` : 'Connecting...';

    const pill = document.getElementById('statusPill');
    const statusText = document.getElementById('statusText');
    const isPaper = state?.paper_mode !== false;

    pill.className = isPaper ? 'status-pill status-paper' : 'status-pill status-live';
    statusText.textContent = isPaper ? 'PAPER MODE' : 'LIVE TRADING';

    if (!isPaper) {
        pill.querySelector('.dot')?.remove();
    }
}

// ─── Stats Row ───────────────────────────────────────────────────────────────
function updateStats(multi, tradebook) {
    // Always pull active count from tradebook (not multi_bot_state) so it updates independently of cycle runs
    const activeCount = tradebook?.trades ? tradebook.trades.filter(t => t.status === 'ACTIVE').length : 0;
    animateValue('activePositions', activeCount);
    animateValue('eligibleCount', multi?.eligible_count ?? '—');
    animateValue('deployedCount', multi?.deployed_count ?? '—');
    // Use tradebook trade count if available
    const tbCount = tradebook?.trades?.length;
    animateValue('totalTrades', tbCount || multi?.total_trades || '—');
    animateValue('cycleNum', multi?.cycle ?? '—');

    // Analysis timestamps
    const lastEl = document.getElementById('lastAnalysis');
    const nextEl = document.getElementById('nextAnalysis');
    if (lastEl && multi?.last_analysis_time) {
        lastEl.textContent = formatTime(multi.last_analysis_time);
    }
    if (nextEl && multi?.next_analysis_time) {
        nextEl.textContent = formatTime(multi.next_analysis_time);
    }

    // ── Cumulative P&L card ──────────────────────────────────────
    if (tradebook?.trades) {
        const MAX_CAPITAL = 2500;  // Total portfolio capital
        const CAPITAL_PER_TRADE = 100;

        let realizedPnl = 0;
        let unrealizedPnl = 0;
        let activeCount = 0;

        tradebook.trades.forEach(t => {
            if (t.status === 'CLOSED') {
                realizedPnl += (t.realized_pnl || 0);
            } else if (t.status === 'ACTIVE') {
                unrealizedPnl += (t.unrealized_pnl || 0);
                activeCount++;
            }
        });

        const totalPnl = realizedPnl + unrealizedPnl;
        const deployedCapital = activeCount * CAPITAL_PER_TRADE;

        // Cumulative ROI = total P&L / max capital
        const cumulativeRoi = MAX_CAPITAL > 0 ? (totalPnl / MAX_CAPITAL * 100) : 0;
        // Unrealized ROI = unrealized P&L / deployed capital
        const unrealizedRoi = deployedCapital > 0 ? (unrealizedPnl / deployedCapital * 100) : 0;

        const pnlEl = document.getElementById('cumulativePnlValue');
        const pctEl = document.getElementById('cumulativePnlPct');
        const capEl = document.getElementById('capitalDeployed');

        if (pnlEl) {
            const sign = totalPnl >= 0 ? '+' : '';
            pnlEl.textContent = `${sign}$${totalPnl.toFixed(2)}`;
            pnlEl.style.color = totalPnl >= 0 ? '#22C55E' : '#EF4444';
        }
        if (pctEl) {
            const sign = cumulativeRoi >= 0 ? '+' : '';
            pctEl.textContent = `${sign}${cumulativeRoi.toFixed(2)}% ROI on $${MAX_CAPITAL}`;
        }
        if (capEl) {
            const uSign = unrealizedPnl >= 0 ? '+' : '';
            const uColor = unrealizedPnl >= 0 ? '#22C55E' : '#EF4444';
            capEl.innerHTML = `Deployed: $${deployedCapital} (${activeCount} trades) | Unrealized: <span style="color:${uColor};font-weight:700">${uSign}$${unrealizedPnl.toFixed(2)} (${uSign}${unrealizedRoi.toFixed(2)}%)</span>`;
        }
    }
}

function animateValue(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = value;
    el.style.animation = 'none';
    el.offsetHeight; // trigger reflow
    el.style.animation = 'count-up 0.3s ease';
}

// ─── Regime Card ─────────────────────────────────────────────────────────────
function updateRegimeCard(state) {
    const regime = state?.regime || 'WAITING';
    const info = getRegimeInfo(regime);
    const card = document.getElementById('regimeCard');

    card.className = `card card-dark regime-card ${info.class}`;
    document.getElementById('regimeEmoji').textContent = info.emoji;
    document.getElementById('regimeName').textContent = regime;
    document.getElementById('regimeSymbol').textContent = state?.symbol || 'BTCUSDT';
}

// ─── Confidence Gauge ────────────────────────────────────────────────────────
function updateGauge(state) {
    let conf = state?.confidence || 0;
    if (conf <= 1) conf *= 100;
    const pct = Math.round(conf);
    const circumference = 2 * Math.PI * 68; // radius = 68
    const offset = circumference - (pct / 100) * circumference;

    const fill = document.getElementById('gaugeFill');
    fill.style.strokeDashoffset = offset;

    // Color based on confidence
    if (pct >= 85) fill.style.stroke = '#22C55E';
    else if (pct >= 65) fill.style.stroke = '#0EA5E9';
    else if (pct >= 50) fill.style.stroke = '#F59E0B';
    else fill.style.stroke = '#EF4444';

    document.getElementById('gaugePct').textContent = `${pct}%`;
}

// ─── Positions Count ─────────────────────────────────────────────────────────
function updatePositionsCount(multi, tradebook) {
    const el = document.getElementById('positionsCount');
    if (!el) return;
    const tbActive = (tradebook?.trades || []).filter(t => t.status === 'ACTIVE').length;
    const multiActive = Object.keys(multi?.active_positions || {}).length;
    const active = tbActive || multiActive;
    el.textContent = `${active}/25`;
}

// ─── Last Action ─────────────────────────────────────────────────────────────
function updateLastAction(state, tradebook) {
    const actionEl = document.getElementById('lastAction');
    const timeEl = document.getElementById('actionTime');
    if (!actionEl || !timeEl) return;
    const trades = tradebook?.trades || [];
    if (trades.length > 0) {
        const sorted = [...trades].sort((a, b) => {
            const aTime = a.exit_timestamp || a.entry_timestamp || '';
            const bTime = b.exit_timestamp || b.entry_timestamp || '';
            return bTime.localeCompare(aTime);
        });
        const latest = sorted[0];
        if (latest) {
            let action = '';
            let time = '';
            if (latest.status === 'CLOSED' && latest.exit_timestamp) {
                const reason = latest.exit_reason || 'CLOSED';
                action = `${reason} ${latest.symbol}`;
                time = latest.exit_timestamp;
            } else if (latest.status === 'ACTIVE') {
                const side = latest.side === 'BUY' ? 'LONG' : 'SHORT';
                action = `OPENED ${side} ${latest.symbol}`;
                time = latest.entry_timestamp;
            }
            if (action) {
                actionEl.textContent = action;
                timeEl.textContent = time ? formatTime(time) : '—';
                return;
            }
        }
    }
    actionEl.textContent = state?.action || '—';
    timeEl.textContent = state?.timestamp ? formatTime(state.timestamp) : '—';
}

// ─── Active Positions Table ──────────────────────────────────────────────────
function updatePositions() {
    const area = document.getElementById('positionsArea');
    if (!area) return;
    const activeTrades = (tradebookData?.trades || []).filter(t => t.status === 'ACTIVE');

    if (activeTrades.length === 0) {
        area.innerHTML = `
      <div class="empty-state">
        <div class="icon">📭</div>
        <p>No active positions. The bot will deploy when eligible coins are found.</p>
      </div>`;
        return;
    }

    let html = `<table class="data-table">
    <thead><tr>
      <th>Symbol</th><th>Side</th><th>Regime</th><th>Leverage</th>
      <th>Entry</th><th>Current</th><th>Unrealized P&L</th><th>Time</th>
    </tr></thead><tbody>`;

    // Sort by unrealized P&L descending
    const sorted = [...activeTrades].sort((a, b) => (b.unrealized_pnl || 0) - (a.unrealized_pnl || 0));

    sorted.forEach(t => {
        const side = t.position || (t.side === 'BUY' ? 'LONG' : 'SHORT');
        const sideClass = side === 'LONG' ? 'side-buy' : 'side-sell';
        const sideIcon = side === 'LONG' ? '▲' : '▼';
        const regimeInfo = getRegimeInfo(t.regime);
        const pnl = parseFloat(t.unrealized_pnl) || 0;
        const pnlPct = parseFloat(t.unrealized_pnl_pct) || 0;
        const pnlSign = pnl >= 0 ? '+' : '';
        const pnlColor = pnl >= 0 ? '#22C55E' : '#EF4444';
        const currentPrice = livePrices[t.symbol] || t.current_price || 0;

        html += `<tr data-symbol="${t.symbol}">
      <td><strong>${t.symbol}</strong></td>
      <td class="${sideClass}">${sideIcon} ${side}</td>
      <td>${regimeInfo.emoji} ${t.regime || '—'}</td>
      <td><span class="leverage-badge">${t.leverage || 1}x</span></td>
      <td>${formatPrice(t.entry_price)}</td>
      <td class="pos-current-price">${formatPrice(currentPrice)}</td>
      <td class="pos-pnl" style="color:${pnlColor};font-weight:700">${pnlSign}$${pnl.toFixed(2)} (${pnlSign}${pnlPct.toFixed(2)}%)</td>
      <td>${t.entry_timestamp ? formatTime(t.entry_timestamp) : '—'}</td>
    </tr>`;
    });

    html += '</tbody></table>';
    area.innerHTML = html;
}

// ─── Coin Heatmap ────────────────────────────────────────────────────────────
function updateHeatmap(scanner) {
    const grid = document.getElementById('heatmapGrid');
    const coins = scanner?.coins || [];

    if (coins.length === 0) {
        grid.innerHTML = `<div class="empty-state"><div class="icon">📊</div><p>Waiting for scanner data...</p></div>`;
        return;
    }

    // Split coins into active (have open position) and inactive
    const activeSymbols = Object.keys(multiState?.active_positions || {});
    const activeCoins = coins.filter(c => activeSymbols.includes(c.symbol));
    const inactiveCoins = coins.filter(c => !activeSymbols.includes(c.symbol));

    const renderCell = (coin, isActive) => {
        const regime = coin.regime_name || 'SIDEWAYS/CHOP';
        const info = getRegimeInfo(regime);
        const conf = ((coin.confidence || 0) * 100).toFixed(0);
        const symbol = (coin.symbol || '').replace('USDT', '');
        const activeClass = isActive ? ' heatmap-active' : '';

        let tooltip = `${coin.symbol}: ${regime} (${conf}%)`;
        if (isActive) {
            const pos = multiState.active_positions[coin.symbol];
            const trade = (tradebookData?.trades || []).find(t => t.symbol === coin.symbol && t.status === 'ACTIVE');
            if (pos && trade) {
                const side = pos.side === 'BUY' ? 'LONG' : 'SHORT';
                tooltip = `${coin.symbol} — ${side} | Entry: $${trade.entry_price} | SL: $${trade.stop_loss} | TP: $${trade.take_profit} | Leverage: ${pos.leverage}x | ${regime} (${conf}%)`;
            } else if (pos) {
                const side = pos.side === 'BUY' ? 'LONG' : 'SHORT';
                tooltip = `${coin.symbol} — ${side} | ${pos.leverage}x | ${regime} (${conf}%)`;
            }
        }

        return `<div class="heatmap-cell cell-${info.class}${activeClass}" data-tooltip="${tooltip}">
      <div class="coin-symbol">${symbol}</div>
      <div class="coin-conf">${info.emoji} ${conf}%</div>
      <div class="coin-price">${formatPrice(coin.price)}</div>
    </div>`;
    };

    let html = '';

    // Active section
    html += `<div class="heatmap-section-label"><span class="heatmap-dot active"></span> Active Positions <span class="heatmap-count">${activeCoins.length}</span></div>`;
    if (activeCoins.length > 0) {
        html += `<div class="heatmap-subgrid">${activeCoins.map(c => renderCell(c, true)).join('')}</div>`;
    } else {
        html += `<div class="heatmap-empty">No active positions</div>`;
    }

    // Divider
    html += `<div class="heatmap-divider"></div>`;

    // Inactive section
    html += `<div class="heatmap-section-label"><span class="heatmap-dot inactive"></span> Watchlist <span class="heatmap-count">${inactiveCoins.length}</span></div>`;
    if (inactiveCoins.length > 0) {
        html += `<div class="heatmap-subgrid">${inactiveCoins.map(c => renderCell(c, false)).join('')}</div>`;
    } else {
        html += `<div class="heatmap-empty">All coins are active</div>`;
    }

    grid.innerHTML = html;
}

// ─── Ticker Tape ─────────────────────────────────────────────────────────────
function updateTicker(scanner) {
    const track = document.getElementById('tickerTrack');
    const coins = scanner?.coins || [];
    if (coins.length === 0) return;

    // Duplicate items for seamless loop
    const items = [...coins, ...coins];
    track.innerHTML = items.map(coin => {
        const regime = coin.regime_name || 'SIDEWAYS/CHOP';
        const info = getRegimeInfo(regime);
        const symbol = (coin.symbol || '').replace('USDT', '');
        const badgeClass = `badge-${info.class}`;

        return `<span class="ticker-item">
      <span class="symbol">${symbol}</span>
      <span class="price">${formatPrice(coin.price)}</span>
      <span class="regime-badge ${badgeClass}">${regime.split('/')[0]}</span>
    </span>`;
    }).join('');
}

// ─── Regime Distribution Chart ───────────────────────────────────────────────
function updateRegimeChart(scanner) {
    const coins = scanner?.coins || [];
    if (coins.length === 0) return;

    const counts = { BULLISH: 0, BEARISH: 0, 'SIDEWAYS/CHOP': 0, 'CRASH/PANIC': 0 };
    const activeCounts = { BULLISH: 0, BEARISH: 0, 'SIDEWAYS/CHOP': 0, 'CRASH/PANIC': 0 };
    const activeSymbols = Object.keys(multiState?.active_positions || {});

    coins.forEach(c => {
        counts[c.regime_name] = (counts[c.regime_name] || 0) + 1;
        if (activeSymbols.includes(c.symbol)) {
            activeCounts[c.regime_name] = (activeCounts[c.regime_name] || 0) + 1;
        }
    });

    const labels = Object.keys(counts);
    const data = Object.values(counts);
    const colors = labels.map(l => getRegimeInfo(l).color);

    const ctx = document.getElementById('regimeChart');
    if (!ctx) return;

    if (regimeChart) regimeChart.destroy();

    // Inline plugin to render value labels on doughnut segments
    const doughnutLabelsPlugin = {
        id: 'doughnutLabels',
        afterDraw(chart) {
            const { ctx: c, data: chartData } = chart;
            const dataset = chartData.datasets[0];
            const total = dataset.data.reduce((a, b) => a + b, 0);
            if (total === 0) return;

            chart.getDatasetMeta(0).data.forEach((arc, i) => {
                const value = dataset.data[i];
                if (value === 0) return;

                const pct = Math.round((value / total) * 100);
                const { x, y } = arc.tooltipPosition();

                c.save();
                c.font = 'bold 12px Inter, sans-serif';
                c.textAlign = 'center';
                c.textBaseline = 'middle';
                c.fillStyle = '#1A2332';
                c.fillText(`${value}`, x, y - 7);
                c.font = '11px Inter, sans-serif';
                c.fillStyle = '#5A6A7E';
                c.fillText(`${pct}%`, x, y + 7);
                c.restore();
            });
        }
    };

    regimeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: colors.map(c => c + '33'),
                borderColor: colors,
                borderWidth: 2,
                hoverBorderWidth: 3,
            }]
        },
        plugins: [doughnutLabelsPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#5A6A7E',
                        font: { family: 'Inter', size: 12 },
                        padding: 16,
                        usePointStyle: true,
                        pointStyleWidth: 12,
                        generateLabels(chart) {
                            const ds = chart.data.datasets[0];
                            const total = ds.data.reduce((a, b) => a + b, 0);
                            return chart.data.labels.map((label, i) => {
                                const val = ds.data[i];
                                const pct = total > 0 ? Math.round((val / total) * 100) : 0;
                                const active = activeCounts[label] || 0;
                                const activeStr = active > 0 ? ` (${active} active)` : '';
                                return {
                                    text: `${label}  ${val} (${pct}%)${activeStr}`,
                                    fillStyle: ds.borderColor[i] + '33',
                                    strokeStyle: ds.borderColor[i],
                                    lineWidth: 2,
                                    pointStyle: 'circle',
                                    hidden: false,
                                    index: i,
                                };
                            });
                        },
                    }
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
                        label(ctx) {
                            const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                            const pct = total > 0 ? Math.round((ctx.raw / total) * 100) : 0;
                            const active = activeCounts[ctx.label] || 0;
                            return ` ${ctx.label}: ${ctx.raw} coins (${pct}%) — ${active} active`;
                        }
                    }
                }
            },
            cutout: '65%',
        }
    });
}

// ─── Confidence Distribution Chart ───────────────────────────────────────────
function updateConfChart(scanner) {
    const coins = scanner?.coins || [];
    if (coins.length === 0) return;

    // Group by confidence ranges
    const ranges = { '90-100%': 0, '80-90%': 0, '70-80%': 0, '60-70%': 0, '<60%': 0 };
    coins.forEach(c => {
        const conf = (c.confidence || 0) * 100;
        if (conf >= 90) ranges['90-100%']++;
        else if (conf >= 80) ranges['80-90%']++;
        else if (conf >= 70) ranges['70-80%']++;
        else if (conf >= 60) ranges['60-70%']++;
        else ranges['<60%']++;
    });

    const ctx = document.getElementById('confChart');
    if (!ctx) return;

    if (confChart) confChart.destroy();

    // Inline plugin to render value labels on top of bars
    const barLabelsPlugin = {
        id: 'barLabels',
        afterDraw(chart) {
            const { ctx: c } = chart;
            chart.getDatasetMeta(0).data.forEach((bar, i) => {
                const value = chart.data.datasets[0].data[i];
                if (value === 0) return;
                c.save();
                c.font = 'bold 12px Inter, sans-serif';
                c.textAlign = 'center';
                c.textBaseline = 'bottom';
                c.fillStyle = '#1A2332';
                c.fillText(value, bar.x, bar.y - 4);
                c.restore();
            });
        }
    };

    confChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(ranges),
            datasets: [{
                data: Object.values(ranges),
                backgroundColor: [
                    'rgba(34, 197, 94, 0.2)',
                    'rgba(59, 130, 246, 0.2)',
                    'rgba(14, 165, 233, 0.2)',
                    'rgba(245, 158, 11, 0.2)',
                    'rgba(239, 68, 68, 0.2)',
                ],
                borderColor: [
                    '#22C55E', '#3B82F6', '#0EA5E9', '#F59E0B', '#EF4444',
                ],
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        plugins: [barLabelsPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#FFFFFF',
                    borderColor: '#E2E8F0',
                    borderWidth: 1,
                    titleColor: '#1A2332',
                    bodyColor: '#5A6A7E',
                    cornerRadius: 10,
                    padding: 12,
                }
            },
            scales: {
                x: {
                    ticks: { color: '#5A6A7E', font: { family: 'Inter', size: 11 } },
                    grid: { color: 'rgba(0,0,0,0.06)' },
                },
                y: {
                    ticks: { color: '#5A6A7E', font: { family: 'Inter', size: 11 }, stepSize: 5 },
                    grid: { color: 'rgba(0,0,0,0.06)' },
                }
            }
        }
    });
}

// ─── Trade Log ───────────────────────────────────────────────────────────────
function updateTradeLog(trades) {
    const area = document.getElementById('tradeLogArea');
    if (!area) return;

    if (!trades || trades.length === 0) {
        area.innerHTML = `
      <div class="empty-state">
        <div class="icon">📝</div>
        <p>Trade history will appear here once the bot executes trades.</p>
      </div>`;
        return;
    }

    let html = `<table class="data-table">
    <thead><tr>
      <th>Time</th><th>Symbol</th><th>Side</th><th>Price</th><th>Qty</th><th>Leverage</th><th>Regime</th>
    </tr></thead><tbody>`;

    // Show most recent first
    const sorted = [...trades].reverse();
    sorted.forEach(t => {
        const sideClass = t.side === 'BUY' ? 'side-buy' : 'side-sell';
        const sideIcon = t.side === 'BUY' ? '▲' : '▼';

        html += `<tr>
      <td>${formatDate(t.timestamp)}</td>
      <td><strong>${t.symbol || '—'}</strong></td>
      <td class="${sideClass}">${sideIcon} ${t.side || '—'}</td>
      <td>${formatPrice(t.entry_price)}</td>
      <td>${parseFloat(t.quantity || 0).toFixed(4)}</td>
      <td><span class="leverage-badge">${t.leverage || 1}x</span></td>
      <td>${t.regime || '—'}</td>
    </tr>`;
    });

    html += '</tbody></table>';
    area.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MASTER UPDATE
// ═══════════════════════════════════════════════════════════════════════════════

function updateAll(data) {
    if (!data) return;

    currentState = data.state || currentState;
    multiState = data.multi || multiState;
    scannerData = data.scanner || scannerData;
    tradeLog = data.trades || tradeLog;
    tradebookData = data.tradebook || tradebookData;

    // Prefer fresh coin_states from multi over stale scanner_state.json
    let liveScanner = scannerData;
    if (multiState?.coin_states && Object.keys(multiState.coin_states).length > 0) {
        const coins = Object.values(multiState.coin_states).map((c, i) => ({
            ...c,
            regime_name: c.regime,
            rank: i + 1,
        }));
        liveScanner = { coins };
    }

    updateHeader(currentState, multiState);
    updateStats(multiState, tradebookData);
    updateRegimeCard(currentState);
    updateGauge(currentState);
    updatePositionsCount(multiState, tradebookData);
    updateLastAction(currentState, tradebookData);
    updatePositions();
    updateHeatmap(liveScanner);
    updateTicker(liveScanner);
    updateRegimeChart(liveScanner);
    updateConfChart(liveScanner);
    updateTradeLog(tradeLog);
    addExecLogEntry(multiState);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SOCKET.IO EVENTS
// ═══════════════════════════════════════════════════════════════════════════════

socket.on('connect', () => {
    console.log('🔌 Connected to SENTINEL API');
    showToast('Connected to bot server', 'success');
});

socket.on('disconnect', () => {
    console.log('🔌 Disconnected');
    showToast('Disconnected from server', 'error');
});

socket.on('full-update', (data) => {
    if (dialogOpen) return;
    // In LIVE mode, accept everything except tradebook (positions come from CoinDCX)
    if (isLiveMode()) {
        currentState = data.state || currentState;
        multiState = data.multi || multiState;
        scannerData = data.scanner || scannerData;
        tradeLog = data.trades || tradeLog;
        // Don't overwrite tradebookData — LIVE mode manages it
        updateHeader(currentState, multiState);
        updateRegimeCard(currentState);
        updateGauge(currentState);
        let liveScanner = scannerData;
        if (multiState?.coin_states && Object.keys(multiState.coin_states).length > 0) {
            const coins = Object.values(multiState.coin_states).map((c, i) => ({ ...c, regime_name: c.regime, rank: i + 1 }));
            liveScanner = { coins };
        }
        updateHeatmap(liveScanner);
        updateTicker(liveScanner);
        updateRegimeChart(liveScanner);
        updateConfChart(liveScanner);
        updateTradeLog(tradeLog);
        addExecLogEntry(multiState);
        return;
    }
    updateAll(data);
});

socket.on('state-update', (state) => {
    currentState = state || currentState;
    updateHeader(currentState, multiState);
    updateRegimeCard(currentState);
    updateGauge(currentState);
    updateLastAction(currentState);
});

socket.on('multi-update', (multi) => {
    if (dialogOpen) return;
    multiState = multi || multiState;
    updateStats(multiState, tradebookData);
    updatePositionsCount(multiState, tradebookData);
    updatePositions();
    updateHeader(currentState, multiState);
    addExecLogEntry(multiState);

    // Also update heatmap from coin_states if available
    if (multi?.coin_states) {
        const coins = Object.values(multi.coin_states).map((c, i) => ({
            ...c,
            regime_name: c.regime,
            rank: i + 1,
        }));
        if (coins.length > 0) {
            updateHeatmap({ coins });
            updateTicker({ coins });
            updateRegimeChart({ coins });
            updateConfChart({ coins });
        }
    }
});

socket.on('scanner-update', (scanner) => {
    scannerData = scanner || scannerData;
    updateHeatmap(scannerData);
    updateTicker(scannerData);
    updateRegimeChart(scannerData);
    updateConfChart(scannerData);
});

socket.on('trades-update', (trades) => {
    tradeLog = trades || tradeLog;
    updateTradeLog(tradeLog);
    showToast('New trade executed!', 'success');
});

socket.on('tradebook-update', (data) => {
    if (dialogOpen) return;
    if (isLiveMode()) return;  // In LIVE mode, positions come from CoinDCX
    tradebookData = data || tradebookData;
    updatePositionsCount(multiState, tradebookData);
    updateLastAction(currentState, tradebookData);
    updateStats(multiState, tradebookData);
    updatePositions();
});

socket.on('command', (cmd) => {
    if (cmd.command === 'KILL') {
        showToast('🚨 Kill switch activated!', 'error');
    } else if (cmd.command === 'RESET') {
        showToast('Kill switch reset', 'success');
    }
});

// ─── Live Price Tick (1-second updates) ──────────────────────────────────────
let livePrices = {};

socket.on('price-tick', (data) => {
    if (dialogOpen) return;
    if (!data?.prices) return;
    livePrices = data.prices;

    // Update heatmap cell prices
    document.querySelectorAll('.heatmap-cell').forEach(cell => {
        const symbolEl = cell.querySelector('.coin-symbol');
        const priceEl = cell.querySelector('.coin-price');
        if (symbolEl && priceEl) {
            const sym = symbolEl.textContent.trim() + 'USDT';
            if (livePrices[sym] !== undefined) {
                priceEl.textContent = formatPrice(livePrices[sym]);
            }
        }
    });

    // Update ticker tape prices
    document.querySelectorAll('.ticker-item').forEach(item => {
        const symbolEl = item.querySelector('.symbol');
        const priceEl = item.querySelector('.price');
        if (symbolEl && priceEl) {
            const sym = symbolEl.textContent.trim() + 'USDT';
            if (livePrices[sym] !== undefined) {
                priceEl.textContent = formatPrice(livePrices[sym]);
            }
        }
    });

    // Update active positions table with live prices and recalculated P&L
    const posTable = document.querySelector('#positionsArea .data-table');
    if (posTable) {
        posTable.querySelectorAll('tbody tr').forEach(row => {
            const sym = row.dataset.symbol;
            if (!sym || !livePrices[sym]) return;

            const priceCell = row.querySelector('.pos-current-price');
            const pnlCell = row.querySelector('.pos-pnl');
            if (priceCell) priceCell.textContent = formatPrice(livePrices[sym]);

            // Recalculate P&L with live price
            const trade = (tradebookData?.trades || []).find(t => t.symbol === sym && t.status === 'ACTIVE');
            if (trade && pnlCell) {
                const entry = parseFloat(trade.entry_price) || 0;
                const qty = trade.quantity || 0;
                const lev = trade.leverage || 1;
                const capital = trade.capital || 100;
                const rawPnl = trade.position === 'LONG'
                    ? (livePrices[sym] - entry) * qty
                    : (entry - livePrices[sym]) * qty;
                const pnl = parseFloat((rawPnl * lev).toFixed(4));
                const pnlPct = capital > 0 ? parseFloat((pnl / capital * 100).toFixed(2)) : 0;
                const sign = pnl >= 0 ? '+' : '';
                pnlCell.textContent = `${sign}$${pnl.toFixed(2)} (${sign}${pnlPct.toFixed(2)}%)`;
                pnlCell.style.color = pnl >= 0 ? '#22C55E' : '#EF4444';
            }
        });
    }

    // Update cumulative P&L with live prices
    updateStats(multiState, tradebookData);
});

// ═══════════════════════════════════════════════════════════════════════════════
//  USER ACTIONS
// ═══════════════════════════════════════════════════════════════════════════════

let dialogOpen = false;

function showConfirm(title, msg, icon = '⚠️') {
    return new Promise((resolve) => {
        const modal = document.getElementById('confirmModal');
        document.getElementById('confirmIcon').textContent = icon;
        document.getElementById('confirmTitle').textContent = title;
        document.getElementById('confirmMsg').textContent = msg;
        modal.style.display = 'flex';

        const yesBtn = document.getElementById('confirmYes');
        const noBtn = document.getElementById('confirmNo');

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

async function sendCommand(command) {
    try {
        const res = await fetch(`${API_BASE}/api/command`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command }),
        });
        const data = await res.json();
        if (data.success) {
            showToast(`${command} command sent`, command === 'KILL' ? 'error' : 'success');
        }
    } catch (e) {
        showToast('Failed to send command', 'error');
    }
}

async function confirmKill() {
    const yes = await showConfirm(
        'Emergency Kill Switch',
        'This will close ALL positions immediately. Are you sure?',
        '🚨'
    );
    if (yes) sendCommand('KILL');
}

async function deleteAllTrades() {
    const yes = await showConfirm(
        'Delete ALL Trades?',
        'This will permanently remove all active and closed trades from the tradebook. This cannot be undone.',
        '🗑️'
    );
    if (!yes) return;
    try {
        const res = await fetch(`${API_BASE}/api/tradebook/all`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            showToast('All trades deleted — tradebook reset', 'success');
            refreshData();
        } else {
            showToast('Failed to delete trades', 'error');
        }
    } catch (e) {
        showToast('Error deleting trades', 'error');
    }
}

// ─── Manual Cycle Trigger ────────────────────────────────────────────────────
function triggerCycle() {
    socket.emit('trigger-cycle');
    showToast('⚡ Cycle trigger sent — running now...', 'success');
}

socket.on('trigger-ack', (data) => {
    console.log('Trigger acknowledged:', data);
});

// ─── Engine Pause / Start Toggle ─────────────────────────────────────────────
function updateEngineButton(state) {
    const btn = document.getElementById('engineToggleBtn');
    if (!btn) return;
    if (state.status === 'paused') {
        btn.innerHTML = '▶️ Start Engine';
        btn.style.background = 'linear-gradient(135deg, #22C55E, #16A34A)';
    } else {
        btn.innerHTML = '⏸️ Pause Engine';
        btn.style.background = 'linear-gradient(135deg, #F59E0B, #D97706)';
    }
}

async function toggleEngine() {
    try {
        const res = await fetch(`${API_BASE}/api/engine/toggle`, { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            updateEngineButton(data.state);
            const action = data.state.status === 'paused' ? '⏸️ Engine PAUSED' : '▶️ Engine RUNNING';
            showToast(action, data.state.status === 'paused' ? 'error' : 'success');
        }
    } catch (e) {
        showToast('Failed to toggle engine', 'error');
    }
}

// Listen for engine state changes from other clients
socket.on('engine_state', (state) => {
    updateEngineButton(state);
});

// Load initial engine state
(async function loadEngineState() {
    try {
        const res = await fetch(`${API_BASE}/api/engine/state`);
        const state = await res.json();
        updateEngineButton(state);
    } catch (e) { }
})();

async function refreshData() {
    try {
        const res = await fetch(`${API_BASE}/api/all`);
        const data = await res.json();
        updateAll(data);
        showToast('Data refreshed', 'success');
    } catch (e) {
        showToast('Failed to refresh', 'error');
    }
}

// ─── Initial HTTP fetch (fallback if WebSocket hasn't loaded yet) ────────────
setTimeout(async () => {
    try {
        const res = await fetch(`${API_BASE}/api/all`);
        const data = await res.json();
        updateAll(data);
    } catch (e) {
        console.log('Initial fetch failed, waiting for WebSocket...');
    }
}, 1000);

// ═══════════════════════════════════════════════════════════════════════════════
//  LIVE LOG VIEWER
// ═══════════════════════════════════════════════════════════════════════════════

let logAutoScroll = true;
const MAX_LOG_LINES = 500;

function getLogLevel(line) {
    if (line.includes(' ERROR:') || line.includes(' ERROR ')) return 'error';
    if (line.includes(' WARNING:') || line.includes(' WARNING ')) return 'warning';
    if (line.includes(' DEBUG:') || line.includes(' DEBUG ')) return 'debug';
    return 'info';
}

function appendLogLines(lines) {
    const output = document.getElementById('logOutput');
    if (!output) return;

    // Remove empty state placeholder
    const empty = output.querySelector('.log-empty');
    if (empty) empty.remove();

    lines.forEach(line => {
        const div = document.createElement('div');
        div.className = `log-line log-${getLogLevel(line)}`;
        div.textContent = line;
        output.appendChild(div);
    });

    // Trim old lines
    while (output.children.length > MAX_LOG_LINES) {
        output.removeChild(output.firstChild);
    }

    if (logAutoScroll) {
        output.scrollTop = output.scrollHeight;
    }
}

function clearLogs() {
    const output = document.getElementById('logOutput');
    if (output) {
        output.innerHTML = '<div class="log-empty">Logs cleared</div>';
    }
}

function toggleAutoScroll() {
    logAutoScroll = !logAutoScroll;
    const btn = document.getElementById('autoScrollBtn');
    if (btn) {
        btn.classList.toggle('active', logAutoScroll);
        btn.textContent = logAutoScroll ? '⬇ Auto-scroll ON' : '⬇ Auto-scroll OFF';
    }
    if (logAutoScroll) {
        const output = document.getElementById('logOutput');
        if (output) output.scrollTop = output.scrollHeight;
    }
}

// Initial log load
socket.on('log-init', (lines) => {
    const output = document.getElementById('logOutput');
    if (output) output.innerHTML = '';
    appendLogLines(lines);
});

// Live log stream
socket.on('log-lines', (lines) => {
    appendLogLines(lines);
});

// Set auto-scroll button initial state
document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('autoScrollBtn');
    if (btn) btn.classList.add('active');
});

// ═══════════════════════════════════════════════════════════════════════════════
//  REGIME ENGINE EXECUTION LOG TABLE
// ═══════════════════════════════════════════════════════════════════════════════

let execLogHistory = [];
let lastLoggedCycle = -1;
const MAX_EXEC_LOG_ROWS = 200;  // Keep more history for pagination
const EXEC_LOG_PAGE_SIZE = 10;
let execLogCurrentPage = 0;

// Load persisted execution log history on page load
(async function loadExecLogHistory() {
    try {
        const res = await fetch(`${window.location.origin}/api/execution-log`);
        const history = await res.json();
        if (Array.isArray(history) && history.length > 0) {
            execLogHistory = history.slice(0, MAX_EXEC_LOG_ROWS);
            lastLoggedCycle = execLogHistory[0]?.cycle ?? -1;
            renderExecLog();
        }
    } catch (e) {
        console.warn('Failed to load execution log history:', e);
    }
})();

function addExecLogEntry(multi) {
    if (!multi || multi.cycle == null) return;
    // Only log a new row if the cycle number changed
    if (multi.cycle === lastLoggedCycle) return;
    lastLoggedCycle = multi.cycle;

    const closedCount = (tradebookData?.trades || []).filter(t => t.status === 'CLOSED').length;

    execLogHistory.unshift({
        timestamp: multi.timestamp || new Date().toISOString(),
        cycle: multi.cycle,
        coins_scanned: multi.coins_scanned ?? 0,
        eligible: multi.eligible_count ?? 0,
        deployed: multi.deployed_count ?? 0,
        total_trades: multi.total_trades ?? 0,
        closed_trades: closedCount,
        exec_time: multi.cycle_execution_time_seconds ?? 0,
    });

    // Trim
    if (execLogHistory.length > MAX_EXEC_LOG_ROWS) {
        execLogHistory = execLogHistory.slice(0, MAX_EXEC_LOG_ROWS);
    }

    // Reset to page 0 on new entry so user sees latest
    execLogCurrentPage = 0;
    renderExecLog();

    // Update last/next run timestamps
    const lastEl = document.getElementById('execLastRun');
    const nextEl = document.getElementById('execNextRun');
    if (lastEl && multi.last_analysis_time) lastEl.textContent = formatTime(multi.last_analysis_time);
    if (nextEl && multi.next_analysis_time) nextEl.textContent = formatTime(multi.next_analysis_time);
}

function renderExecLog() {
    const tbody = document.getElementById('execLogBody');
    if (!tbody) return;

    if (execLogHistory.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#8899AA;padding:24px;">Waiting for first analysis cycle...</td></tr>';
        renderExecLogPagination();
        return;
    }

    // Sort descending by timestamp (most recent first) — already in order from unshift
    const totalPages = Math.ceil(execLogHistory.length / EXEC_LOG_PAGE_SIZE);
    if (execLogCurrentPage >= totalPages) execLogCurrentPage = totalPages - 1;
    if (execLogCurrentPage < 0) execLogCurrentPage = 0;

    const start = execLogCurrentPage * EXEC_LOG_PAGE_SIZE;
    const pageData = execLogHistory.slice(start, start + EXEC_LOG_PAGE_SIZE);

    tbody.innerHTML = pageData.map(e => {
        const secs = e.exec_time || 0;
        const mins = (secs / 60).toFixed(1);
        const timeStr = secs > 0 ? `${mins} min` : '—';
        return `
        <tr>
            <td>${formatTime(e.timestamp)}</td>
            <td><strong>${e.cycle}</strong></td>
            <td>${e.coins_scanned}</td>
            <td>${e.eligible}</td>
            <td>${e.deployed}</td>
            <td>${e.total_trades}</td>
            <td>${e.closed_trades ?? 0}</td>
            <td>${timeStr}</td>
        </tr>
    `;
    }).join('');

    renderExecLogPagination();
}

function renderExecLogPagination() {
    const container = document.getElementById('execLogPagination');
    if (!container) return;

    const total = execLogHistory.length;
    const totalPages = Math.ceil(total / EXEC_LOG_PAGE_SIZE);

    if (totalPages <= 1) {
        container.innerHTML = total > 0 ? `<span style="opacity:0.6">${total} entries</span>` : '';
        return;
    }

    const prevDisabled = execLogCurrentPage === 0 ? 'opacity:0.4;pointer-events:none;' : 'cursor:pointer;';
    const nextDisabled = execLogCurrentPage >= totalPages - 1 ? 'opacity:0.4;pointer-events:none;' : 'cursor:pointer;';

    container.innerHTML = `
        <button onclick="execLogGoToPage(0)" style="padding:4px 10px;border:1px solid #E2E8F0;border-radius:6px;background:#F8FAFC;font-size:12px;${execLogCurrentPage === 0 ? 'opacity:0.4;pointer-events:none;' : 'cursor:pointer;'}">⏮ First</button>
        <button onclick="execLogGoToPage(${execLogCurrentPage - 1})" style="padding:4px 10px;border:1px solid #E2E8F0;border-radius:6px;background:#F8FAFC;font-size:12px;${prevDisabled}">◀ Prev</button>
        <span style="font-weight:600;">Page ${execLogCurrentPage + 1} of ${totalPages}</span>
        <button onclick="execLogGoToPage(${execLogCurrentPage + 1})" style="padding:4px 10px;border:1px solid #E2E8F0;border-radius:6px;background:#F8FAFC;font-size:12px;${nextDisabled}">Next ▶</button>
        <button onclick="execLogGoToPage(${totalPages - 1})" style="padding:4px 10px;border:1px solid #E2E8F0;border-radius:6px;background:#F8FAFC;font-size:12px;${execLogCurrentPage >= totalPages - 1 ? 'opacity:0.4;pointer-events:none;' : 'cursor:pointer;'}">Last ⏭</button>
        <span style="opacity:0.6;margin-left:8px;">${total} total</span>
    `;
}

function execLogGoToPage(page) {
    execLogCurrentPage = page;
    renderExecLog();
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
    // Convert CoinDCX positions to tradebook format so all existing UI functions work
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
        entry_timestamp: p.updated_at ? new Date(p.updated_at).toISOString() : null,
        mode: 'LIVE',
    }));
    return { trades, wallet_balance: cdxData.wallet_balance };
}

async function refreshLiveData() {
    const cdxData = await fetchLivePositions();
    if (!cdxData) return;
    tradebookData = mapLiveToTradebook(cdxData);
    updateStats(multiState, tradebookData);
    updatePositionsCount(multiState, tradebookData);
    updateLastAction(currentState, tradebookData);
    updatePositions();
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

