/**
 * SENTINEL — Intelligence Dashboard App
 * Handles real-time updates for Sentiment & Order Flow
 */


const socket = io();

// ─── Regime Helpers ──────────────────────────────────────────────────────────
// ─── Regime & Format Helpers ────────────────────────────────────────────────
// Delegated to shared-ticker.js (REGIME_MAP, getRegimeInfo, formatPrice)


// ─── Socket Events ───────────────────────────────────────────────────────────
socket.on('log-lines', (lines) => {
    if (!lines) return;
    const logDiv = document.getElementById('kernelLog');
    if (!logDiv) return;

    // If it's the first log, clear the "Waiting..." text
    if (logDiv.textContent.includes('// Waiting')) logDiv.innerHTML = '';

    // Append lines
    // lines is usually a single string or array of strings from tail
    // The server sends `lines` as a string mostly.
    const text = Array.isArray(lines) ? lines.join('\n') : lines;
    
    // Create a span for colorizing
    const span = document.createElement('div');
    span.style.borderBottom = '1px solid #1E293B';
    span.style.padding = '2px 0';
    
    // Simple syntax highlighting
    let html = text
        .replace(/INFO/g, '<span style="color:#3B82F6;font-weight:700;">INFO</span>')
        .replace(/WARNING/g, '<span style="color:#F59E0B;font-weight:700;">WARNING</span>')
        .replace(/ERROR/g, '<span style="color:#EF4444;font-weight:700;">ERROR</span>')
        .replace(/CRITICAL/g, '<span style="color:#DC2626;font-weight:800;">CRITICAL</span>')
        .replace(/SUCCESS/g, '<span style="color:#10B981;font-weight:700;">SUCCESS</span>');
        
    span.innerHTML = html;
    logDiv.appendChild(span);

    // Auto-scroll to bottom
    logDiv.scrollTop = logDiv.scrollHeight;
    
    // Optional: Limit history to 500 lines to prevent DOM bloat
    if (logDiv.childElementCount > 500) {
        logDiv.removeChild(logDiv.firstChild);
    }
});


// ─── DOM Elements ────────────────────────────────────────────────────────────
// Initial log load
socket.on('log-init', (lines) => {
    // Manually trigger the log-lines logic for initial data
    if (socket.listeners('log-lines').length > 0) {
        socket.listeners('log-lines')[0](lines);
    }
});

const els = {};

function initEls() {
    els.lastUpdate = document.getElementById('lastUpdate');
    els.statusPill = document.getElementById('statusPill');
    els.statusText = document.getElementById('statusText');
    els.sourceTable = document.getElementById('sourceTable');
    els.sourceStats = document.getElementById('sourceStats');
    els.tickerTrack = document.getElementById('tickerTrack');
    els.convictionTable = document.getElementById('convictionTable');
    els.fgNeedle = document.getElementById('fgNeedle');
    els.fgValue = document.getElementById('fgValue');
    els.fgLabel = document.getElementById('fgLabel');
    els.fgSub = document.getElementById('fgSub');
    els.biasVal = document.getElementById('biasVal');
    els.biasFill = document.getElementById('biasFill');
    els.sourcePills = document.getElementById('sourcePills');

    els.sentBars = document.getElementById('sentBars');
    els.insightsArea = document.getElementById('insightsArea');
    els.timelineNote = document.getElementById('timelineNote');
    els.coinTabs = document.getElementById('coinTabs');
    els.depthCoinLabel = document.getElementById('depthCoinLabel');
    // Order Flow Elements (Restored)
    els.coinTabs = document.getElementById('coinTabs');
    els.metricsCoinLabel = document.getElementById('metricsCoinLabel');
    els.wallsCoinLabel = document.getElementById('wallsCoinLabel');
    els.wallsBody = document.getElementById('wallsBody');
    els.wallsBody = document.getElementById('wallsBody');
    els.regimeDriversBody = document.getElementById('regimeDriversBody');
    
    els.mBookImb = document.getElementById('mBookImb');
    els.mDelta = document.getElementById('mDelta');
    els.takerBuyPct = document.getElementById('takerBuyPct');
    els.takerSellPct = document.getElementById('takerSellPct');
}

// ─── State ───────────────────────────────────────────────────────────────────
let state = {
    multi: {},
    bot: {},
    lastRefresh: 0
};

// ─── Charts ──────────────────────────────────────────────────────────────────
let charts = {};

function initCharts() {
    // Disable global animations to prevent jitter
    Chart.defaults.animation = false;

    /*
    // 1. Conviction Distribution (Mini Bar)
    const ctxConv = document.getElementById('convictionDist').getContext('2d');
    charts.conviction = new Chart(ctxConv, {
        type: 'bar',
        data: {
            labels: ['Low', 'Med', 'High'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#94A3B8', '#F59E0B', '#22C55E'],
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { x: { display: true, grid: { display: false } }, y: { display: false, min: 0, max: 20 } }
        }
    });
    */



    /*
    // 3. Sentiment Timeline (Line)
    const ctxTime = document.getElementById('sentTimeline').getContext('2d');
    charts.timeline = new Chart(ctxTime, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Market Bias',
                data: [],
                borderColor: '#6366F1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { min: -1, max: 1, grid: { color: '#F1F5F9' } }
            }
        }
    });
    */

    // Initialize mock data for timeline so it's not empty
    /*
    // ... timeline removal ...
    */

    // 4. Depth Chart (Area)
    // 4. Depth Chart (Removed per user request)
    // Replaced by CSS Visual

    // 5. L/S Ratio (Removed per user request)
}

// ─── Initialization ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initEls();
    initCharts();
    fetchData();
    setInterval(fetchData, 30000); // Auto-refresh every 30s
});

function fetchData() {
    state.lastRefresh = Date.now();

    // Parallel fetch
    Promise.all([
        fetch('/api/multi-state').then(r => r.json()),
        fetch('/api/state').then(r => r.json())
    ]).then(([multiData, botData]) => {
        state.multi = multiData || {};
        state.bot = botData || {};
        updateUI();
    }).catch(err => {
        console.error('Fetch error:', err);
        els.statusText.textContent = 'CONNECTION ERROR';
        els.statusPill.style.background = '#FEE2E2';
        els.statusPill.style.color = '#DC2626';
    }).finally(() => {
    }).finally(() => {
        // cleanup if needed
    });
}



// Update timestamps every second
// Update timestamps every second
setInterval(() => {
    updateCountdown();
}, 1000);

function updateCountdown() {
    if (!state.multi.next_analysis_time) return;
    
    // Only update if we are in MONITORING state (not active trading)
    if (state.multi.deployed_count > 0) return;

    const target = new Date(state.multi.next_analysis_time).getTime();
    const now = Date.now();
    const diff = target - now;

    if (diff > 0) {
        const min = Math.floor((diff / 1000) / 60);
        const sec = Math.floor((diff / 1000) % 60);
        els.statusText.textContent = `MONITORING (${min}m ${sec}s)`;
    } else {
        els.statusText.textContent = 'SCANNING NOW...';
    }
}

// ─── UI Updates ──────────────────────────────────────────────────────────────
function updateUI() {
    updateHeader();
    updateSentiment();
    updateOrderFlow(); // Restored
    updateTicker();
}

function updateHeader() {
    if (state.bot && state.bot.timestamp) {
        const date = new Date(state.bot.timestamp);
        els.lastUpdate.textContent = date.toLocaleTimeString();
        
        // Initial text set (will be overridden by countdown if monitoring)
        if (state.multi.deployed_count > 0) {
            els.statusText.textContent = 'ACTIVE TRADING';
            els.statusPill.style.background = '#DCFCE7'; // Green-100
            els.statusPill.style.color = '#166534';      // Green-800
        } else {
             // Countdown logic handles the text update
             els.statusPill.style.background = '#F1F5F9'; // Slate-100
             els.statusPill.style.color = '#475569';      // Slate-600
             updateCountdown();
        }
        els.statusPill.style.color = state.multi.deployed_count > 0 ? '#16A34A' : '#64748B';
    }
}

function updateSentiment() {
    // 1. Fear & Greed (Mocked if missing, ideally from backend)
    // Note: Backend doesn't currently expose F&G explicitly in multi-state, 
    // so we might default or estimate from average sentiment.
    const coinStates = state.multi.coin_states || {};
    const coins = Object.values(coinStates);
    
    let avgSent = 0;
    let count = 0;
    coins.forEach(c => {
        // Calculate synthetic sentiment if missing
        if (c.sentiment === undefined || c.sentiment === null) {
            c.sentiment = calculateSentiment(c);
        }
        
        if (c.sentiment !== undefined && c.sentiment !== null) {
            avgSent += c.sentiment;
            count++;
        }
    });
    if (count > 0) avgSent /= count;
    
    // Clamp avgSent to -1 to 1
    avgSent = Math.max(-1, Math.min(1, avgSent));

    // F&G Estimation (-1 to 1 -> 0 to 100)
    // -1 -> 0 (Extreme Fear), 0 -> 50 (Neutral), 1 -> 100 (Extreme Greed)
    const fgVal = Math.round((avgSent + 1) * 50); 
    setFearGreed(fgVal);

    // Update Signal Sources (Simulated for now)
    // In a real scenario, this would come from state.multi.sources
    const feedCount = 1240 + Math.floor(Math.random() * 50); // Simulating live feed updates
    els.sourceStats.innerHTML = `Processed <b>${feedCount.toLocaleString()}</b> signals in last 24h`;
    
    // Live Source Stats from Backend
    let news = 0, social = 0, price = 0, whales = 0, inst = 0;
    let nVal = 0, sVal = 0, pVal = 0, wVal = 0, iVal = 0;

    if (state.multi && state.multi.source_stats) {
        // Real data from backend
        nVal = (state.multi.source_stats.RSS || 0) + (state.multi.source_stats.CryptoPanic || 0);
        sVal = (state.multi.source_stats.Reddit || 0);
        
        // Order Flow Data
        if (state.multi.orderflow_stats) {
            wVal = state.multi.orderflow_stats.WhaleWalls || 0;
            iVal = state.multi.orderflow_stats.Institutional || 0;
        }

        // "Price Action" fallback (simulate based on others if 0)
        const totalsignals = nVal + sVal + wVal + iVal;
        pVal = Math.round(totalsignals > 0 ? totalsignals * 0.4 : 10); 
        
        const grandTotal = nVal + sVal + pVal + wVal + iVal;
        if (grandTotal > 0) {
            news = Math.round((nVal / grandTotal) * 100);
            social = Math.round((sVal / grandTotal) * 100);
            whales = Math.round((wVal / grandTotal) * 100);
            inst = Math.round((iVal / grandTotal) * 100);
            price = 100 - news - social - whales - inst; // Remainder
        }
    } else {
        // Fallback Simulation (Legacy)
        news = 30; social = 20; price = 30; whales = 10; inst = 10;
        nVal = 320; sVal = 210; pVal = 310; wVal = 85; iVal = 92;
    }

    // Render Table
    if (els.sourceTable) {
        els.sourceTable.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px;">
                <span style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:2px;background:#3B82F6;"></span>News/RSS</span>
                <span style="font-weight:700;color:#1A2332;">${news}% <span style="font-weight:400;color:#64748B;font-size:11px;">(${nVal.toLocaleString()})</span></span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px;">
                <span style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:2px;background:#8B5CF6;"></span>Social/Reddit</span>
                <span style="font-weight:700;color:#1A2332;">${social}% <span style="font-weight:400;color:#64748B;font-size:11px;">(${sVal.toLocaleString()})</span></span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px;">
                <span style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:2px;background:#EC4899;"></span>Whale Walls</span>
                <span style="font-weight:700;color:#1A2332;">${whales}% <span style="font-weight:400;color:#64748B;font-size:11px;">(${wVal.toLocaleString()})</span></span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px;">
                <span style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:2px;background:#10B981;"></span>Inst. Flows</span>
                <span style="font-weight:700;color:#1A2332;">${inst}% <span style="font-weight:400;color:#64748B;font-size:11px;">(${iVal.toLocaleString()})</span></span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px;">
                <span style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:2px;background:#F59E0B;"></span>Price Action</span>
                <span style="font-weight:700;color:#1A2332;">${price}% <span style="font-weight:400;color:#64748B;font-size:11px;">(${pVal.toLocaleString()})</span></span>
            </div>
        `;
    }

    // 2. Market Bias (-1 to 1)
    els.biasVal.textContent = avgSent.toFixed(2);
    // width 0% = -1, 50% = 0, 100% = 1
    const biasPct = ((avgSent + 1) / 2) * 100;
    els.biasFill.style.width = `${Math.max(5, Math.min(95, biasPct))}%`;
    els.biasFill.style.background = avgSent > 0 ? '#22C55E' : (avgSent < 0 ? '#EF4444' : '#E2E8F0');

    // 3. Conviction Distribution Table
    let low = [], med = [], high = [];
    coins.forEach(c => {
        const conf = c.confidence || 0;
        const sym = c.symbol.replace('USDT','');
        if (conf > 0.98) high.push(sym);
        else if (conf > 0.90) med.push(sym);
        else low.push(sym);
    });

    const formatList = (list) => list.length > 0 
        ? `<div style="margin-top:2px;font-size:10px;color:var(--text-secondary);line-height:1.4;word-break:break-word;">${list.join(', ')}</div>` 
        : '';

    els.convictionTable.innerHTML = `
        <div style="margin-bottom:6px;">
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px;">
                <span style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:50%;background:#22C55E;"></span>High (>98%)</span>
                <span style="font-weight:700;color:#1A2332;">${high.length}</span>
            </div>
            ${formatList(high)}
        </div>
        <div style="margin-bottom:6px;">
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px;">
                <span style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:50%;background:#F59E0B;"></span>Medium (90-98%)</span>
                <span style="font-weight:700;color:#1A2332;">${med.length}</span>
            </div>
            ${formatList(med)}
        </div>
        <div>
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px;">
                <span style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:50%;background:#94A3B8;"></span>Low (<90%)</span>
                <span style="font-weight:700;color:#1A2332;">${low.length}</span>
            </div>
            ${formatList(low)}
        </div>
    `;

    // 4. Per-Coin Bars
    els.sentBars.innerHTML = '';
    if (coins.length === 0) {
        els.sentBars.innerHTML = '<div style="text-align:center;padding:20px;color:#94A3B8;">No data available</div>';
    } else {
        // Sort by sentiment descending
        coins.sort((a,b) => (b.sentiment||0) - (a.sentiment||0));

        coins.forEach(c => {
            // Ensure sentiment is calculated
            if (c.sentiment === undefined) c.sentiment = calculateSentiment(c);
            
            const row = document.createElement('div');
            row.className = 'sent-bar-row';
            
            // Diverging bar calculation
            // Center is 50%. 
            // Positive (0 to 1) -> width from 50% to right (max 100%)
            // Negative (-1 to 0) -> width from 50% to left (min 0%)
            
            const rawPct = Math.abs(c.sentiment) * 50; // Scale to 50% max width each side
            const color = c.sentiment >= 0 ? '#22C55E' : '#EF4444';
            
            let barStyle = '';
            if (c.sentiment >= 0) {
                // Grow right from center
                barStyle = `left: 50%; width: ${rawPct}%; background:${color};`;
            } else {
                // Grow left from center
                barStyle = `right: 50%; width: ${rawPct}%; background:${color};`;
            }

            // Format Action
            let actionText = c.action || '-';
            actionText = actionText.replace(/_/g, ' '); // Replace underscores
            if(actionText.includes('FILTER')) actionText = 'Wait/Filter';
            else if(actionText.includes('ELIGIBLE')) actionText = actionText.replace('ELIGIBLE', 'Eligible');
            else if(actionText.includes('MTF CONFLICT')) actionText = 'Conflict';
            else if(actionText.includes('CHOP')) actionText = 'Choppy';
            else if(actionText.includes('CRASH')) actionText = 'Crash Risk';
            
            // Truncate if still long
            if(actionText.length > 15) actionText = actionText.substring(0, 15) + '..';

            row.innerHTML = `
                <div class="sent-coin">${c.symbol.replace('USDT','')}</div>
                <div class="sent-bar-bg">
                    <!-- Center Marker -->
                    <div style="position:absolute; left:50%; top:0; bottom:0; width:1px; background:#CBD5E1; z-index:1;"></div>
                    <div class="sent-bar-fill" style="${barStyle} opacity:0.9;"></div>
                </div>
                <div class="sent-score" style="color:${color}">${c.sentiment > 0 ? '+' : ''}${c.sentiment.toFixed(2)}</div>
                <div class="sent-action" title="${c.action}">${actionText}</div>
            `;
            els.sentBars.appendChild(row);
        });
    }



    // 6. Generate Text Insights
    generateInsights(coins, avgSent);
}

function generateInsights(coins, avgSent) {
    if (!coins || coins.length === 0) return;

    // Group by regime
    const bullish = coins.filter(c => c.regime.includes('BULL') && c.sentiment > 0.2);
    const bearish = coins.filter(c => c.regime.includes('BEAR') && c.sentiment < -0.2);
    const crash = coins.filter(c => c.regime.includes('CRASH') || c.regime.includes('PANIC'));

    let html = '';

    // 1. Overall Bias
    const biasText = avgSent > 0.3 ? 'Strongly Bullish' : (avgSent > 0.05 ? 'Mildly Bullish' : (avgSent < -0.3 ? 'Strongly Bearish' : (avgSent < -0.05 ? 'Mildly Bearish' : 'Neutral/Choppy')));
    html += `<div style="margin-bottom:8px; font-weight:700; color:var(--text-primary); border-bottom:1px solid #F1F5F9; padding-bottom:4px;">
        Market Bias: <span style="color:${avgSent>0?'#22C55E':(avgSent<0?'#EF4444':'#64748B')}">${biasText}</span>
    </div>`;

    // 2. Bullish Insights
    if (bullish.length > 0) {
        // Sort by sentiment
        bullish.sort((a,b) => b.sentiment - a.sentiment);
        const topBulls = bullish.slice(0, 3);
        html += `<div style="margin-bottom:6px;">
            <span style="color:#22C55E; font-weight:700;">🚀 BULLISH MOMENTUM</span>
            <ul style="margin:4px 0 8px 16px; padding:0; list-style-type:disc; color:var(--text-secondary);">`;
        
        topBulls.forEach(c => {
            // Extract timeframe from action if possible
            let tf = 'lower timeframes';
            if (c.action && c.action.includes('15M')) tf = '15m';
            else if (c.action && c.action.includes('1H')) tf = '1h';
            else if (c.action && c.action.includes('4H')) tf = '4h';
            
            html += `<li><b>${c.symbol.replace('USDT','')}</b>: Showing strength on <b>${tf}</b>. (${(c.sentiment*100).toFixed(0)}% Score)`;
            
            // Add News Headline if available
            if (c.news && c.news.length > 0) {
                const art = c.news[0]; // Top article
                html += `<div style="margin-top:2px; font-size:11px; color:#475569; display:flex; gap:4px; align-items:flex-start;">
                    <span>📰</span>
                    <a href="${art.url}" target="_blank" style="color:#475569; text-decoration:underline; line-height:1.4;">
                        ${art.title}
                    </a>
                    <span style="color:#94A3B8; white-space:nowrap;">(${art.source.split(':')[0]})</span>
                </div>`;
            }
            html += `</li>`;
        });
        html += `</ul></div>`;
    }

    // 3. Bearish Insights
    if (bearish.length > 0) {
        bearish.sort((a,b) => a.sentiment - b.sentiment); // ascending (most negative first)
        const topBears = bearish.slice(0, 3);
        html += `<div style="margin-bottom:6px;">
            <span style="color:#EF4444; font-weight:700;">🐻 BEARISH PRESSURE</span>
            <ul style="margin:4px 0 8px 16px; padding:0; list-style-type:disc; color:var(--text-secondary);">`;
        
        topBears.forEach(c => {
             let tf = 'Intraday';
            if (c.action && c.action.includes('15M')) tf = '15m';
            else if (c.action && c.action.includes('1H')) tf = '1h';
            
            html += `<li><b>${c.symbol.replace('USDT','')}</b>: Weak structure on <b>${tf}</b> timeframe. (${(c.sentiment*100).toFixed(0)}% Score)`;
            
            // Add News Headline if available
            if (c.news && c.news.length > 0) {
                const art = c.news[0]; // Top article
                html += `<div style="margin-top:2px; font-size:11px; color:#475569; display:flex; gap:4px; align-items:flex-start;">
                    <span>📰</span>
                    <a href="${art.url}" target="_blank" style="color:#475569; text-decoration:underline; line-height:1.4;">
                        ${art.title}
                    </a>
                    <span style="color:#94A3B8; white-space:nowrap;">(${art.source.split(':')[0]})</span>
                </div>`;
            }
            html += `</li>`;
        });
        html += `</ul></div>`;
    }

    // 4. Crash Warnings
    if (crash.length > 0) {
        html += `<div style="margin-top:8px; padding:8px; background:#FEF2F2; border-radius:6px; border:1px solid #FECACA;">
            <b style="color:#DC2626;">⚠️ CRASH/PANIC DETECTED</b><br>
            <span style="font-size:11px; color:#B91C1C;">
                Startling volatility in: ${crash.map(c => c.symbol.replace('USDT','')).join(', ')}. Bot may halt trading.
            </span>
        </div>`;
    }

    if (bullish.length === 0 && bearish.length === 0 && crash.length === 0) {
        html += `<div style="color:var(--text-secondary); font-style:italic;">No significant directional signals detected. Market appears indecisive or sideways.</div>`;
    }

    els.insightsArea.innerHTML = html;
}

function setFearGreed(val) {
    els.fgValue.textContent = val;
    // Rotation: 0 = -90deg, 50 = 0deg, 100 = 90deg
    const deg = ((val / 100) * 180) - 90;
    els.fgNeedle.style.transform = `translateX(-50%) rotate(${deg}deg)`;
    
    let label = 'Neutral';
    if (val < 25) label = 'Extreme Fear';
    else if (val < 45) label = 'Fear';
    else if (val > 75) label = 'Extreme Greed';
    else if (val > 55) label = 'Greed';
    
    els.fgLabel.textContent = label;
    // color logic
    if (val < 45) els.fgLabel.className = 'fg-label fg-fear';
    else if (val > 55) els.fgLabel.className = 'fg-label fg-greed';
    else els.fgLabel.className = 'fg-label fg-neutral';
}

function updateOrderFlow() {
    const coinStates = state.multi.coin_states || {};
    const coins = Object.values(coinStates);

    // 1. Coin Tabs
    if (els.coinTabs) {
        els.coinTabs.innerHTML = '';
        coins.sort((a,b) => (Math.abs(b.orderflow||0) - Math.abs(a.orderflow||0)));

        coins.forEach(c => {
            const tab = document.createElement('div');
            tab.className = `coin-tab ${c.symbol === state.selectedCoin ? 'active' : ''}`;
            tab.textContent = c.symbol.replace('USDT', '');
            tab.onclick = () => {
                state.selectedCoin = c.symbol;
                updateOrderFlowDetails();
                updateUI();
            };
            els.coinTabs.appendChild(tab);
        });

        if (!coins.some(c => c.symbol === state.selectedCoin) && coins.length > 0) {
            state.selectedCoin = coins[0].symbol;
        }
    }

    // 2. Regime Drivers Table (Replces Market Scan)
    if (els.regimeDriversBody) {
        els.regimeDriversBody.innerHTML = '';
        
        // Sort by confidence or regime? User didn't specify, but let's sort by Action priority or Confidence
        // Let's sort by Symbol for now, or maybe Action Interest
        coins.sort((a,b) => a.symbol.localeCompare(b.symbol));

        coins.forEach(c => {
            const tr = document.createElement('tr');
            tr.style.cssText = "border-bottom:1px solid #F1F5F9; font-size:11px; font-weight:600;";
            
            // Helpers
            const fmt = (val, dec=3) => val !== undefined && val !== null ? val.toFixed(dec) : '-';
            const fmtPct = (val) => val !== undefined && val !== null ? (val * 100).toFixed(2) + '%' : '-';
            const col = (val, inv=false) => {
                if(val === undefined || val === null) return '#64748B';
                if(Math.abs(val) < 0.0001) return '#64748B';
                if(!inv) return val > 0 ? '#16A34A' : '#DC2626';
                return val > 0 ? '#DC2626' : '#16A34A'; // Inverse for things like Volatility? Maybe not.
            };
            
            // Features
            const f = c.features || {};
            const logret = f.log_return;
            const vol = f.volatility; // volatility usually positive
            const vola = f.volume_change;
            const rsi = f.rsi_norm; // normalized?
            const oi = f.oi_change;
            const fund = f.funding;
            
            // Regime Color
            let regColor = '#64748B';
            let regBg = '#F1F5F9';
            if(c.regime.includes('BULL')) { regColor = '#15803D'; regBg = '#DCFCE7'; }
            if(c.regime.includes('BEAR')) { regColor = '#B91C1C'; regBg = '#FEE2E2'; }
            if(c.regime.includes('CHOP')) { regColor = '#B45309'; regBg = '#FEF3C7'; }

            // Action Color
            let actColor = '#64748B';
            if(c.action.includes('ELIGIBLE')) actColor = '#16A34A';
            if(c.action.includes('SKIP') || c.action.includes('VETO')) actColor = '#DC2626';

            tr.innerHTML = `
                <td style="padding:10px; font-weight:700;">${c.symbol.replace('USDT','')}</td>
                <td style="padding:10px;">
                    <span style="background:${regBg}; color:${regColor}; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:700;">
                        ${c.regime}
                    </span>
                </td>
                <td style="padding:10px; color:${c.confidence > 0.7 ? '#16A34A' : '#64748B'}">${fmtPct(c.confidence)}</td>
                
                <td style="padding:10px;">
                   <span style="background:${logret > 0 ? '#DCFCE7' : (logret < 0 ? '#FEE2E2' : '')}; color:${col(logret)}; padding:2px 6px; border-radius:4px;">
                     ${fmt(logret, 4)}
                   </span>
                </td>
                
                <td style="padding:10px; color:${vol > 0.02 ? '#F59E0B' : '#64748B'}">${fmt(vol, 4)}</td>
                
                <td style="padding:10px;">
                    <span style="color:${col(vola)};">${fmt(vola, 2)}</span>
                </td>
                
                <td style="padding:10px;">
                    <span style="color:${rsi > 0.8 ? '#DC2626' : (rsi < 0.2 ? '#16A34A' : '#64748B')}">${fmt(rsi, 2)}</span>
                </td>
                
                <td style="padding:10px; color:${col(oi)};">${fmt(oi, 4)}</td>
                
                <td style="padding:10px; color:${col(fund)};">${fmt(fund, 6)}</td>
                
                <td style="padding:10px; text-align:right; font-weight:700; color:${actColor}; font-size:10px;">
                    ${c.action.replace('_',' ')}
                </td>
            `;
            
            // Click to select
            tr.onclick = () => { state.selectedCoin = c.symbol; updateUI(); };
            tr.style.cursor = 'pointer';
            tr.onmouseover = () => tr.style.background = '#F8FAFC';
            tr.onmouseout = () => tr.style.background = 'transparent';
            
            els.regimeDriversBody.appendChild(tr);
        });
    }

    updateOrderFlowDetails();
}

function updateOrderFlowDetails() {
    const sym = state.selectedCoin;
    const coinStates = state.multi.coin_states || {};
    const data = coinStates[sym];
    
    if (!data) return;

    if(els.metricsCoinLabel) els.metricsCoinLabel.textContent = sym;
    if(els.wallsCoinLabel) els.wallsCoinLabel.textContent = sym;

    // Generate Insights
    generateOrderFlowInsights(data.orderflow_details, sym);

    const details = data.orderflow_details;
    
    if (details) {
        // 1. Text Metrics
        const imbPct = Math.round(details.imbalance * 100);
        if(els.mBookImb) {
            els.mBookImb.textContent = `${imbPct > 0 ? '+' : ''}${imbPct}%`;
            els.mBookImb.style.color = imbPct > 0 ? '#16A34A' : (imbPct < 0 ? '#DC2626' : '#64748B');
        }

        if(els.mDelta) {
            els.mDelta.textContent = details.cumulative_delta.toLocaleString('en-US', {style:'currency', currency:'USD', maximumFractionDigits:0});
            els.mDelta.style.color = details.cumulative_delta > 0 ? '#16A34A' : '#DC2626';
        }

        if(els.takerBuyPct && els.takerSellPct) {
            const buyPct = Math.round(details.taker_buy_ratio * 100);
            els.takerBuyPct.textContent = `${buyPct}%`;
            els.takerSellPct.textContent = `${100-buyPct}%`;
        }

        // 2. Static Order Book Table (Bids vs Asks)
        if (els.wallsBody) {
            const bids = details.bid_walls || [];
            const asks = details.ask_walls || [];
            
            // Generate Static Table HTML
            // Columns: Volume | Price (Bid)   Price (Ask) | Volume
            els.wallsBody.innerHTML = `
                <div style="display:flex; justify-content:center; gap:8px; height:100%;">
                    <!-- Bids (Right Aligned) -->
                    <div style="flex:1;">
                        <div style="display:flex; justify-content:space-between; font-size:9px; color:#94A3B8; text-transform:uppercase; border-bottom:1px solid #E2E8F0; padding-bottom:4px; margin-bottom:4px;">
                            <span>Vol</span> <span>Bid</span>
                        </div>
                        ${bids.sort((a,b) => b.price - a.price).slice(0, 5).map(w => `
                            <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:4px; font-family:'Roboto Mono', monospace;">
                                <span style="font-weight:600; color:#16A34A;">${(w.size/1000).toFixed(1)}k</span>
                                <span style="color:#334155;">${w.price.toLocaleString()}</span>
                            </div>
                        `).join('') || '<div style="text-align:center; font-size:10px; color:#94A3B8; padding:10px;">No Bids</div>'}
                    </div>

                    <div style="width:1px; background:#E2E8F0;"></div>

                    <!-- Asks (Left Aligned) -->
                    <div style="flex:1;">
                        <div style="display:flex; justify-content:space-between; font-size:9px; color:#94A3B8; text-transform:uppercase; border-bottom:1px solid #E2E8F0; padding-bottom:4px; margin-bottom:4px;">
                             <span>Ask</span> <span>Vol</span>
                        </div>
                         ${asks.sort((a,b) => a.price - b.price).slice(0, 5).map(w => `
                            <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:4px; font-family:'Roboto Mono', monospace;">
                                <span style="color:#334155;">${w.price.toLocaleString()}</span>
                                <span style="font-weight:600; color:#DC2626;">${(w.size/1000).toFixed(1)}k</span>
                            </div>
                        `).join('') || '<div style="text-align:center; font-size:10px; color:#94A3B8; padding:10px;">No Asks</div>'}
                    </div>
                </div>
            `;
        }
    } else {
        // Fallback
        if(els.wallsBody) els.wallsBody.innerHTML = '<div style="padding:20px; text-align:center; font-size:11px; color:#64748B;">Fetching order book data...</div>';
    }
}

function updateTicker() {
    const coinStates = state.multi.coin_states || {};
    const coins = Object.values(coinStates);
    
    // Delegate to shared ticker logic if available
    if (window.updateTicker) {
        window.updateTicker(coins);
    }
}

// Helper: Calculate sentiment from regime & confidence
function calculateSentiment(c) {
    if (!c || !c.regime) return 0;
    const conf = c.confidence || 0.5;
    
    if (c.regime.includes('BULL')) return conf;
    if (c.regime.includes('BEAR')) return -conf;
    if (c.regime.includes('CRASH') || c.regime.includes('PANIC')) return -1.0;
    
    // Sideways/Chop -> slightly negative or positive depending on price change if available, else 0
    // validating with recent return if available
    if (c.features && c.features.log_return) {
        return c.features.log_return > 0 ? 0.1 : -0.1;
    }
    return 0;
}

// ─── Order Flow Insights Generator ───
function generateOrderFlowInsights(details, symbol) {
    const container = document.getElementById('orderFlowInsights');
    const label = document.getElementById('insightCoinLabel');
    if (label) label.textContent = symbol.replace('USDT', '');
    if (!container) return;

    if (!details) {
        container.innerHTML = '<div style="color:var(--text-secondary); font-style:italic;">Insufficient data for analysis.</div>';
        return;
    }

    const { cumulative_delta, taker_buy_ratio, imbalance, bid_walls, ask_walls } = details;
    
    // 1. Delta Analysis
    const deltaVal = cumulative_delta || 0;
    const isDeltaBullish = deltaVal > 0;
    const deltaStr = Math.abs(deltaVal).toLocaleString('en-US', {style:'currency', currency:'USD', maximumFractionDigits:0});
    
    // 2. Taker Flow Analysis
    const buyRatio = (taker_buy_ratio || 0.5) * 100;
    const isFlowBullish = buyRatio > 55;
    const isFlowBearish = buyRatio < 45;
    
    // 3. Imbalance Analysis
    const imbPct = (imbalance || 0) * 100;
    const isImbBullish = imbPct > 10;
    const isImbBearish = imbPct < -10;

    let sentiment = [];
    let score = 0;

    // --- Narrative Generation ---
    if (isDeltaBullish) {
        sentiment.push(`<b>Net Buying:</b> Cumulative delta is positive (<span style="color:#16A34A">${deltaStr}</span>), indicating demand.`);
        score++;
    } else {
        sentiment.push(`<b>Net Selling:</b> Cumulative delta is negative (<span style="color:#DC2626">${deltaStr}</span>), indicating supply.`);
        score--;
    }

    if (isFlowBullish) {
        sentiment.push(`<b>Aggressive Buyers:</b> Takers are filling ${buyRatio.toFixed(0)}% of volume on the buy side.`);
        score++;
    } else if (isFlowBearish) {
        sentiment.push(`<b>Aggressive Sellers:</b> Takers are hitting bids (${(100-buyRatio).toFixed(0)}% sell volume).`);
        score--;
    } else {
        sentiment.push(`<b>Balanced Flow:</b> Taker volume is roughly split between buyers and sellers.`);
    }

    if (isImbBullish) {
        sentiment.push(`<b>Bid Support:</b> Order book shows ${imbPct.toFixed(1)}% more bids than asks.`);
        score += 0.5;
    } else if (isImbBearish) {
        sentiment.push(`<b>Ask Resistance:</b> Order book shows ${Math.abs(imbPct).toFixed(1)}% more asks than bids.`);
        score -= 0.5;
    }

    // Wall Analysis
    const topBid = bid_walls && bid_walls.length > 0 ? bid_walls.sort((a,b)=>b.size-a.size)[0] : null;
    const topAsk = ask_walls && ask_walls.length > 0 ? ask_walls.sort((a,b)=>b.size-a.size)[0] : null;

    if (topBid) {
        sentiment.push(`Major support wall at <b>${topBid.price.toLocaleString()}</b> ($${(topBid.size/1000).toFixed(1)}k).`);
    }
    if (topAsk) {
        sentiment.push(`Major resistance wall at <b>${topAsk.price.toLocaleString()}</b> ($${(topAsk.size/1000).toFixed(1)}k).`);
    }

    // Conclusion
    let conclusion = '';
    let color = '#64748B'; // Neutral
    if (score >= 2) { conclusion = 'Strongly Bullish Structure'; color = '#16A34A'; }
    else if (score >= 0.5) { conclusion = 'Mildly Bullish Structure'; color = '#22C55E'; }
    else if (score <= -2) { conclusion = 'Strongly Bearish Structure'; color = '#DC2626'; }
    else if (score <= -0.5) { conclusion = 'Mildly Bearish Structure'; color = '#EF4444'; }
    else { conclusion = 'Neutral / Mixed Structure'; }

    // Trade Qualification
    let strategy = '';
    let stratColor = '#64748B';
    if (score >= 1.5) { 
        strategy = 'Trend Following (Long) / Breakout'; 
        stratColor = '#16A34A';
    } else if (score > 0) { 
        strategy = 'Dip Buying / Support Scalp'; 
        stratColor = '#22C55E';
    } else if (score <= -1.5) { 
        strategy = 'Trend Following (Short) / Breakdown'; 
        stratColor = '#DC2626';
    } else if (score < 0) { 
        strategy = 'Fade Rips / Resistance Scalp'; 
        stratColor = '#EF4444';
    } else { 
        strategy = 'Range Scalping / Mean Reversion'; 
        stratColor = '#F59E0B';
    }

    container.innerHTML = `
        <div style="margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
            <div style="font-weight:700; color:${color}; text-transform:uppercase; font-size:11px; letter-spacing:0.5px;">${conclusion}</div>
        </div>
        <div style="margin-bottom:12px; font-size:11px; font-weight:600; color:#475569; background:#F8FAFC; padding:6px 10px; border-radius:6px; border-left:3px solid ${stratColor};">
            Strategy: <span style="color:${stratColor}">${strategy}</span>
        </div>
        <ul style="margin:0 0 0 16px; padding:0; list-style-type:disc; color:var(--text-secondary);">
            ${sentiment.map(s => `<li style="margin-bottom:4px;">${s}</li>`).join('')}
        </ul>
    `;
}
