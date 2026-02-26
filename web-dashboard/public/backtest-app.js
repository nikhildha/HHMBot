/**
 * SENTINEL — Backtest Lab Engine
 * Sends config to server, polls for progress, displays results.
 * Generates branded PDF reports and tracks history.
 */

const API = window.location.origin;
let _lastReport = null;
let _lastParams = null;

function getParams() {
    return {
        initialBalance: parseFloat(document.getElementById('initialBalance').value),
        capitalPerTrade: parseFloat(document.getElementById('capitalPerTrade').value),
        coinCount: parseInt(document.getElementById('coinCount').value),
        tfExecution: document.getElementById('tfExecution').value,
        tfPrimary: document.getElementById('tfPrimary').value,
        tfMacro: document.getElementById('tfMacro').value,
        hmmStates: parseInt(document.getElementById('hmmStates').value),
        hmmLookback: parseInt(document.getElementById('hmmLookback').value),
        hmmCovariance: document.getElementById('hmmCovariance').value,
        trainPeriod: parseInt(document.getElementById('trainPeriod').value),
        levHigh: parseInt(document.getElementById('levHigh').value),
        levModerate: parseInt(document.getElementById('levModerate').value),
        levLow: parseInt(document.getElementById('levLow').value),
        confHigh: parseFloat(document.getElementById('confHigh').value) / 100,
        confMedium: parseFloat(document.getElementById('confMedium').value) / 100,
        confLow: parseFloat(document.getElementById('confLow').value) / 100,
        atrMode: document.getElementById('atrMode').value,
        atrSL: parseFloat(document.getElementById('atrSL').value),
        atrTP: parseFloat(document.getElementById('atrTP').value),
        riskPerTrade: parseFloat(document.getElementById('riskPerTrade').value) / 100,
        maxLoss: -parseFloat(document.getElementById('maxLoss').value),
        minHoldMinutes: parseInt(document.getElementById('minHoldMinutes').value),
        takerFeePct: parseFloat(document.getElementById('takerFeePct').value) / 100,
        trailSlActivation: parseFloat(document.getElementById('trailSlActivation').value),
        trailSlDistance: parseFloat(document.getElementById('trailSlDistance').value),
        trailTpActivation: parseFloat(document.getElementById('trailTpActivation').value) / 100,
        trailTpExtension: parseFloat(document.getElementById('trailTpExtension').value),
        trailTpMaxExt: parseInt(document.getElementById('trailTpMaxExt').value),
        volFilterEnabled: document.getElementById('volFilterEnabled').value === 'true',
        volMinPct: parseFloat(document.getElementById('volMinPct').value) / 100,
        volMaxPct: parseFloat(document.getElementById('volMaxPct').value) / 100,
    };
}

function toggleAtrMode() {
    const mode = document.getElementById('atrMode').value;
    document.getElementById('fixedAtrRow').style.display = mode === 'fixed' ? 'grid' : 'none';
    document.getElementById('atrDynamicHint').style.display = mode === 'dynamic' ? 'block' : 'none';
}

async function runBacktest() {
    const btn = document.getElementById('runBtn');
    const panel = document.getElementById('resultsPanel');
    const params = getParams();

    btn.disabled = true;
    btn.classList.add('loading');

    // Show progress
    panel.innerHTML = `
        <div class="progress-bar-container">
            <div class="progress-header">
                <span id="progressLabel">Starting backtest...</span>
                <span id="progressPct">0%</span>
            </div>
            <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
            <div class="progress-log" id="progressLog"></div>
        </div>
    `;

    try {
        const res = await fetch(`${API}/api/backtest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });

        if (!res.ok) {
            throw new Error(`Server error: ${res.status}`);
        }

        // Stream response
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let report = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const msg = JSON.parse(line);
                    if (msg.type === 'progress') {
                        updateProgress(msg);
                    } else if (msg.type === 'result') {
                        report = msg.data;
                    } else if (msg.type === 'error') {
                        throw new Error(msg.message);
                    }
                } catch (e) {
                    if (e.message.startsWith('Server') || e.message.startsWith('Backtest')) throw e;
                }
            }
        }

        if (report) {
            _lastReport = report;
            _lastParams = params;
            renderResults(report, params);
            // Refresh history table
            loadReportHistory();
        } else {
            panel.innerHTML = '<div class="results-placeholder"><div class="icon">⚠️</div><h3>No results</h3><p>Backtest returned no data.</p></div>';
        }

    } catch (err) {
        panel.innerHTML = `<div class="results-placeholder"><div class="icon">❌</div><h3>Error</h3><p>${err.message}</p></div>`;
    }

    btn.disabled = false;
    btn.classList.remove('loading');
}

function updateProgress(msg) {
    const fill = document.getElementById('progressFill');
    const label = document.getElementById('progressLabel');
    const pct = document.getElementById('progressPct');
    const log = document.getElementById('progressLog');

    if (fill) fill.style.width = `${msg.pct}%`;
    if (pct) pct.textContent = `${msg.pct}%`;
    if (label) label.textContent = msg.label || 'Processing...';
    if (log && msg.log) {
        log.innerHTML += `${msg.log}\n`;
        log.scrollTop = log.scrollHeight;
    }
}

function renderResults(r, params) {
    const panel = document.getElementById('resultsPanel');
    const s = r.summary;
    const pnlClass = s.total_pnl >= 0 ? 'green' : 'red';
    const pnlSign = s.total_pnl >= 0 ? '+' : '';

    let html = `
        <!-- Actions -->
        <div class="results-actions">
            <button class="pdf-btn" onclick="generatePDF()">Download PDF Report</button>
        </div>

        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="stat-card">
                <div class="label">Total Return</div>
                <div class="value ${pnlClass}">${pnlSign}${s.total_return_pct}%</div>
                <div class="sub">${pnlSign}$${s.total_pnl.toLocaleString()}</div>
            </div>
            <div class="stat-card">
                <div class="label">Total Trades</div>
                <div class="value blue">${s.total_trades}</div>
                <div class="sub">${s.coins_tested} coins</div>
            </div>
            <div class="stat-card">
                <div class="label">Win Rate</div>
                <div class="value">${s.win_rate_pct}%</div>
                <div class="sub">${s.wins}W / ${s.losses}L</div>
            </div>
            <div class="stat-card">
                <div class="label">Profit Factor</div>
                <div class="value ${s.profit_factor >= 1 ? 'green' : 'red'}">${s.profit_factor}</div>
                <div class="sub">Avg W: $${s.avg_win} / L: $${s.avg_loss}</div>
            </div>
        </div>

        <div class="summary-grid">
            <div class="stat-card">
                <div class="label">Final Equity</div>
                <div class="value">$${s.final_equity.toLocaleString()}</div>
                <div class="sub">from $${r.config.initial_balance.toLocaleString()}</div>
            </div>
            <div class="stat-card">
                <div class="label">Max Drawdown</div>
                <div class="value red">${s.max_drawdown_pct}%</div>
                <div class="sub">$${s.max_drawdown.toLocaleString()}</div>
            </div>
            <div class="stat-card">
                <div class="label">Best Trade</div>
                <div class="value green">+$${s.best_trade}</div>
            </div>
            <div class="stat-card">
                <div class="label">Worst Trade</div>
                <div class="value red">$${s.worst_trade}</div>
            </div>
        </div>
    `;

    // By Regime table
    html += `<div class="tables-grid">
        <div class="result-card">
            <h3>📊 By Regime</h3>
            <table class="result-table">
                <tr><th>Regime</th><th class="num">Trades</th><th class="num">Win Rate</th><th class="num">P&L</th></tr>`;
    for (const [regime, stats] of Object.entries(r.by_regime)) {
        const cls = stats.pnl >= 0 ? 'green' : 'red';
        html += `<tr>
            <td>${regime}</td>
            <td class="num">${stats.trades}</td>
            <td class="num">${stats.win_rate}%</td>
            <td class="num ${cls}">$${stats.pnl >= 0 ? '+' : ''}${stats.pnl.toFixed(2)}</td>
        </tr>`;
    }
    html += `</table></div>`;

    // By Leverage table
    html += `<div class="result-card">
            <h3>⚡ By Leverage</h3>
            <table class="result-table">
                <tr><th>Leverage</th><th class="num">Trades</th><th class="num">Win Rate</th><th class="num">P&L</th></tr>`;
    for (const [lev, stats] of Object.entries(r.by_leverage)) {
        const cls = stats.pnl >= 0 ? 'green' : 'red';
        html += `<tr>
            <td>${lev}x</td>
            <td class="num">${stats.trades}</td>
            <td class="num">${stats.win_rate}%</td>
            <td class="num ${cls}">$${stats.pnl >= 0 ? '+' : ''}${stats.pnl.toFixed(2)}</td>
        </tr>`;
    }
    html += `</table></div></div>`;

    // Exit Reasons
    html += `<div class="result-card">
        <h3>🚪 By Exit Reason</h3>
        <table class="result-table">
            <tr><th>Reason</th><th class="num">Count</th><th class="num">P&L</th></tr>`;
    for (const [reason, stats] of Object.entries(r.by_exit_reason)) {
        const cls = stats.pnl >= 0 ? 'green' : 'red';
        html += `<tr>
            <td>${reason}</td>
            <td class="num">${stats.count}</td>
            <td class="num ${cls}">$${stats.pnl >= 0 ? '+' : ''}${stats.pnl.toFixed(2)}</td>
        </tr>`;
    }
    html += `</table></div>`;

    // Top & Bottom coins
    html += `<div class="tables-grid">`;
    html += buildCoinTable('🏆 Top 10 Coins', r.top_10_coins);
    html += buildCoinTable('📉 Bottom 10 Coins', r.bottom_10_coins);
    html += `</div>`;

    panel.innerHTML = html;
}

function buildCoinTable(title, coins) {
    let html = `<div class="result-card">
        <h3>${title}</h3>
        <table class="result-table">
            <tr><th>Symbol</th><th class="num">Trades</th><th class="num">Win Rate</th><th class="num">P&L</th></tr>`;
    for (const [sym, stats] of Object.entries(coins)) {
        const cls = stats.pnl >= 0 ? 'green' : 'red';
        html += `<tr>
            <td>${sym}</td>
            <td class="num">${stats.trades}</td>
            <td class="num">${stats.win_rate}%</td>
            <td class="num ${cls}">$${stats.pnl >= 0 ? '+' : ''}${stats.pnl.toFixed(2)}</td>
        </tr>`;
    }
    html += `</table></div>`;
    return html;
}

// ─── PDF Generation ──────────────────────────────────────────────────────────

function generatePDF(reportData, paramsData) {
    const r = reportData || _lastReport;
    const p = paramsData || _lastParams;
    if (!r) return alert('No report data available');

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF('p', 'mm', 'a4');
    const W = doc.internal.pageSize.getWidth();
    const H = doc.internal.pageSize.getHeight();
    const margin = 15;
    let y = 0;

    // ─── Header ───
    doc.setFillColor(30, 41, 59); // slate-800
    doc.rect(0, 0, W, 38, 'F');
    // Accent gradient bar
    doc.setFillColor(59, 130, 246); // blue-500
    doc.rect(0, 38, W, 3, 'F');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(22);
    doc.setTextColor(255, 255, 255);
    doc.text('SENTINEL', margin, 18);

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(148, 163, 184);
    doc.text('Backtest Performance Report', margin, 26);

    const now = new Date();
    doc.setFontSize(9);
    doc.setTextColor(148, 163, 184);
    doc.text(now.toLocaleString(), W - margin, 18, { align: 'right' });
    doc.text('AI Trading Command Center', W - margin, 26, { align: 'right' });

    y = 48;

    // ─── Summary Box ───
    const s = r.summary;
    doc.setFillColor(248, 250, 252);
    doc.roundedRect(margin, y, W - 2 * margin, 36, 3, 3, 'F');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8);
    doc.setTextColor(148, 163, 184);

    const cols = [margin + 5, margin + 45, margin + 85, margin + 125];
    const labels = ['TOTAL RETURN', 'WIN RATE', 'PROFIT FACTOR', 'MAX DRAWDOWN'];
    const vals = [
        `${s.total_pnl >= 0 ? '+' : ''}${s.total_return_pct}%`,
        `${s.win_rate_pct}%`,
        `${s.profit_factor}`,
        `${s.max_drawdown_pct}%`,
    ];
    const valColors = [
        s.total_pnl >= 0 ? [22, 163, 74] : [239, 68, 68],
        [26, 35, 50],
        s.profit_factor >= 1 ? [22, 163, 74] : [239, 68, 68],
        [239, 68, 68],
    ];

    for (let i = 0; i < 4; i++) {
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(7);
        doc.setTextColor(148, 163, 184);
        doc.text(labels[i], cols[i], y + 10);

        doc.setFont('helvetica', 'bold');
        doc.setFontSize(16);
        doc.setTextColor(...valColors[i]);
        doc.text(vals[i], cols[i], y + 20);
    }

    // Second row in summary box
    const labels2 = ['TOTAL TRADES', 'FINAL EQUITY', 'BEST TRADE', 'WORST TRADE'];
    const vals2 = [
        `${s.total_trades}`,
        `$${s.final_equity?.toLocaleString() || 'N/A'}`,
        `+$${s.best_trade}`,
        `$${s.worst_trade}`,
    ];

    for (let i = 0; i < 4; i++) {
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(7);
        doc.setTextColor(148, 163, 184);
        doc.text(labels2[i], cols[i], y + 27);

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);
        doc.setTextColor(26, 35, 50);
        doc.text(vals2[i], cols[i], y + 33);
    }

    y += 44;

    // ─── Input Parameters ───
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(26, 35, 50);
    doc.text('Input Parameters', margin, y);
    y += 2;

    const paramRows = [];
    if (p) {
        paramRows.push(
            ['Initial Balance', `$${p.initialBalance?.toLocaleString()}`, 'Capital / Trade', `$${p.capitalPerTrade}`],
            ['Coins Tested', `${p.coinCount}`, 'HMM States', `${p.hmmStates}`],
            ['Timeframes', `${p.tfExecution} / ${p.tfPrimary} / ${p.tfMacro}`, 'Lookback', `${p.hmmLookback} candles`],
            ['Leverage (H/M/L)', `${p.levHigh}x / ${p.levModerate}x / ${p.levLow}x`, 'Covariance', p.hmmCovariance],
            ['ATR SL / TP', `${p.atrSL}x / ${p.atrTP}x`, 'Risk/Trade', `${(p.riskPerTrade * 100).toFixed(1)}%`],
            ['Trail SL Act/Dist', `${p.trailSlActivation}x / ${p.trailSlDistance}x`, 'Trail TP Trigger', `${(p.trailTpActivation * 100).toFixed(0)}%`],
            ['TP Extend / Max', `${p.trailTpExtension}x / ${p.trailTpMaxExt}`, 'Vol Filter', p.volFilterEnabled ? 'ON' : 'OFF'],
        );
        if (p.volFilterEnabled) {
            paramRows.push(['Vol Range', `${(p.volMinPct * 100).toFixed(1)}% – ${(p.volMaxPct * 100).toFixed(1)}%`, '', '']);
        }
    }

    doc.autoTable({
        startY: y,
        head: [['Parameter', 'Value', 'Parameter', 'Value']],
        body: paramRows,
        margin: { left: margin, right: margin },
        theme: 'plain',
        styles: {
            fontSize: 8,
            cellPadding: 2.5,
            textColor: [26, 35, 50],
            lineColor: [226, 232, 240],
            lineWidth: 0.2,
        },
        headStyles: {
            fillColor: [241, 245, 249],
            textColor: [100, 116, 139],
            fontStyle: 'bold',
            fontSize: 7,
        },
        alternateRowStyles: { fillColor: [248, 250, 252] },
    });

    y = doc.lastAutoTable.finalY + 8;

    // ─── By Regime ───
    if (y > H - 60) { doc.addPage(); y = 20; }
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(26, 35, 50);
    doc.text('Performance by Regime', margin, y);
    y += 2;

    const regimeRows = Object.entries(r.by_regime).map(([regime, st]) => [
        regime, `${st.trades}`, `${st.win_rate}%`, `$${st.pnl >= 0 ? '+' : ''}${st.pnl.toFixed(2)}`
    ]);

    doc.autoTable({
        startY: y,
        head: [['Regime', 'Trades', 'Win Rate', 'P&L']],
        body: regimeRows,
        margin: { left: margin, right: margin },
        theme: 'striped',
        styles: { fontSize: 8, cellPadding: 3, textColor: [26, 35, 50] },
        headStyles: { fillColor: [59, 130, 246], textColor: [255, 255, 255], fontStyle: 'bold', fontSize: 8 },
        alternateRowStyles: { fillColor: [248, 250, 252] },
        columnStyles: { 1: { halign: 'right' }, 2: { halign: 'right' }, 3: { halign: 'right' } },
    });
    y = doc.lastAutoTable.finalY + 8;

    // ─── By Leverage ───
    if (y > H - 60) { doc.addPage(); y = 20; }
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(26, 35, 50);
    doc.text('Performance by Leverage', margin, y);
    y += 2;

    const levRows = Object.entries(r.by_leverage).map(([lev, st]) => [
        `${lev}x`, `${st.trades}`, `${st.win_rate}%`, `$${st.pnl >= 0 ? '+' : ''}${st.pnl.toFixed(2)}`
    ]);

    doc.autoTable({
        startY: y,
        head: [['Leverage', 'Trades', 'Win Rate', 'P&L']],
        body: levRows,
        margin: { left: margin, right: margin },
        theme: 'striped',
        styles: { fontSize: 8, cellPadding: 3, textColor: [26, 35, 50] },
        headStyles: { fillColor: [139, 92, 246], textColor: [255, 255, 255], fontStyle: 'bold', fontSize: 8 },
        alternateRowStyles: { fillColor: [248, 250, 252] },
        columnStyles: { 1: { halign: 'right' }, 2: { halign: 'right' }, 3: { halign: 'right' } },
    });
    y = doc.lastAutoTable.finalY + 8;

    // ─── By Exit Reason ───
    if (y > H - 60) { doc.addPage(); y = 20; }
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(26, 35, 50);
    doc.text('Exit Reason Breakdown', margin, y);
    y += 2;

    const exitRows = Object.entries(r.by_exit_reason).map(([reason, st]) => [
        reason, `${st.count}`, `$${st.pnl >= 0 ? '+' : ''}${st.pnl.toFixed(2)}`
    ]);

    doc.autoTable({
        startY: y,
        head: [['Exit Reason', 'Count', 'P&L']],
        body: exitRows,
        margin: { left: margin, right: margin },
        theme: 'striped',
        styles: { fontSize: 8, cellPadding: 3, textColor: [26, 35, 50] },
        headStyles: { fillColor: [245, 158, 11], textColor: [255, 255, 255], fontStyle: 'bold', fontSize: 8 },
        alternateRowStyles: { fillColor: [248, 250, 252] },
        columnStyles: { 1: { halign: 'right' }, 2: { halign: 'right' } },
    });
    y = doc.lastAutoTable.finalY + 8;

    // ─── Top & Bottom Coins ───
    const renderCoinTable = (title, coins, color) => {
        if (y > H - 60) { doc.addPage(); y = 20; }
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(11);
        doc.setTextColor(26, 35, 50);
        doc.text(title, margin, y);
        y += 2;

        const rows = Object.entries(coins).map(([sym, st]) => [
            sym, `${st.trades}`, `${st.win_rate}%`, `$${st.pnl >= 0 ? '+' : ''}${st.pnl.toFixed(2)}`
        ]);

        doc.autoTable({
            startY: y,
            head: [['Symbol', 'Trades', 'Win Rate', 'P&L']],
            body: rows,
            margin: { left: margin, right: margin },
            theme: 'striped',
            styles: { fontSize: 8, cellPadding: 3, textColor: [26, 35, 50] },
            headStyles: { fillColor: color, textColor: [255, 255, 255], fontStyle: 'bold', fontSize: 8 },
            alternateRowStyles: { fillColor: [248, 250, 252] },
            columnStyles: { 1: { halign: 'right' }, 2: { halign: 'right' }, 3: { halign: 'right' } },
        });
        y = doc.lastAutoTable.finalY + 8;
    };

    renderCoinTable('Top 10 Coins', r.top_10_coins, [22, 163, 74]);
    renderCoinTable('Bottom 10 Coins', r.bottom_10_coins, [239, 68, 68]);

    // ─── Footer on all pages ───
    const totalPages = doc.getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
        doc.setPage(i);
        doc.setFillColor(241, 245, 249);
        doc.rect(0, H - 12, W, 12, 'F');

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(7);
        doc.setTextColor(148, 163, 184);
        doc.text('Generated by SENTINEL Trading Engine', margin, H - 5);
        doc.text(`Page ${i} of ${totalPages}`, W - margin, H - 5, { align: 'right' });
    }

    // Save
    const ts = now.toISOString().slice(0, 19).replace(/[T:]/g, '-');
    doc.save(`SENTINEL_Backtest_Report_${ts}.pdf`);
}

// ─── Report History ──────────────────────────────────────────────────────────

async function loadReportHistory() {
    const container = document.getElementById('historyContent');
    try {
        const res = await fetch(`${API}/api/backtest-reports`);
        const reports = await res.json();

        if (!reports.length) {
            container.innerHTML = '<div class="history-empty">No backtest reports yet. Run a backtest to see history here.</div>';
            return;
        }

        let html = `<table class="history-table">
            <thead><tr>
                <th>Date & Time</th>
                <th class="num">Return</th>
                <th class="num">P&L</th>
                <th class="num">Win Rate</th>
                <th class="num">Trades</th>
                <th class="num">Coins</th>
                <th class="num">PF</th>
                <th class="num">Max DD</th>
                <th>Actions</th>
            </tr></thead><tbody>`;

        for (const r of reports) {
            const date = new Date(r.timestamp);
            const dateStr = date.toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' });
            const retClass = r.total_return >= 0 ? 'positive' : 'negative';
            const retSign = r.total_return >= 0 ? '+' : '';

            html += `<tr>
                <td>${dateStr}</td>
                <td class="num"><span class="badge ${retClass}">${retSign}${r.total_return}%</span></td>
                <td class="num" style="color: ${r.total_pnl >= 0 ? '#16A34A' : '#EF4444'}; font-weight: 700;">${retSign}$${r.total_pnl?.toFixed(2) || '0.00'}</td>
                <td class="num">${r.win_rate}%</td>
                <td class="num">${r.total_trades}</td>
                <td class="num">${r.coins_tested}</td>
                <td class="num">${r.profit_factor}</td>
                <td class="num" style="color: #EF4444;">${r.max_drawdown}%</td>
                <td>
                    <div class="history-actions">
                        <button class="history-btn view" onclick="viewReport('${r.id}')">View</button>
                        <button class="history-btn pdf" onclick="downloadSavedPDF('${r.id}')">PDF</button>
                        <button class="history-btn delete" onclick="deleteReport('${r.id}')">Del</button>
                    </div>
                </td>
            </tr>`;
        }

        html += `</tbody></table>`;
        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = '<div class="history-empty">Failed to load report history.</div>';
    }
}

async function viewReport(id) {
    try {
        const res = await fetch(`${API}/api/backtest-reports/${id}`);
        const data = await res.json();
        _lastReport = data.report;
        _lastParams = data.params;
        renderResults(data.report, data.params);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (e) {
        alert('Failed to load report');
    }
}

async function downloadSavedPDF(id) {
    try {
        const res = await fetch(`${API}/api/backtest-reports/${id}`);
        const data = await res.json();
        generatePDF(data.report, data.params);
    } catch (e) {
        alert('Failed to generate PDF');
    }
}

async function deleteReport(id) {
    if (!confirm('Delete this backtest report?')) return;
    try {
        await fetch(`${API}/api/backtest-reports/${id}`, { method: 'DELETE' });
        loadReportHistory();
    } catch (e) {
        alert('Failed to delete report');
    }
}

// ─── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadReportHistory();
});
