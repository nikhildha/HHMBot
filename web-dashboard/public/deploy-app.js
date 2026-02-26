// ═══════════════════════════════════════════════════════════════════════════════
//  SENTINEL — Deploy Control App
// ═══════════════════════════════════════════════════════════════════════════════

const API_BASE = window.location.hostname === 'localhost'
    ? `http://localhost:${window.location.port || 3001}`
    : '';
const socket = io(API_BASE);

let paperEnabled = true;
let liveEnabled = false;

// ═══════════════════════════════════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    fetchStatus();
    setInterval(fetchStatus, 30000);
});

socket.on('connect', () => {
    document.getElementById('statusPill').className = 'status-pill online';
    document.getElementById('statusText').textContent = 'ONLINE';
});

socket.on('disconnect', () => {
    document.getElementById('statusPill').className = 'status-pill offline';
    document.getElementById('statusText').textContent = 'OFFLINE';
});

// ═══════════════════════════════════════════════════════════════════════════════
//  FETCH STATUS
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/deploy/status`);
        const data = await res.json();

        paperEnabled = data.paper_enabled !== false;  // default true
        liveEnabled = data.live_enabled === true;
        updateModeUI();

        // API Key status
        const apiEl = document.getElementById('apiKeyStatus');
        if (data.api_key_configured) {
            apiEl.textContent = 'Configured';
            apiEl.className = 'status-value ok';
        } else {
            apiEl.textContent = 'Not Configured';
            apiEl.className = 'status-value error';
        }

        // Network
        const netEl = document.getElementById('networkStatus');
        netEl.textContent = data.testnet ? 'Testnet' : 'Mainnet';
        netEl.className = 'status-value ' + (data.testnet ? 'pending' : 'ok');

        // Trade counts
        document.getElementById('paperTradeCount').textContent = data.paper_active ?? '—';
        document.getElementById('liveTradeCount').textContent = data.live_active ?? '—';

        // Update checklist
        updateChecklist(data);

        document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
    } catch (e) {
        console.error('Failed to fetch deploy status:', e);
    }
}

function updateModeUI() {
    const paperCard = document.getElementById('paperModeCard');
    const liveCard = document.getElementById('liveModeCard');
    const paperBox = document.getElementById('paperCheckBox');
    const liveBox = document.getElementById('liveCheckBox');
    const banner = document.getElementById('modeBanner');

    // Paper card
    paperCard.classList.toggle('active', paperEnabled);
    paperBox.textContent = paperEnabled ? '✓' : '';

    // Live card
    liveCard.classList.toggle('active', liveEnabled);
    liveBox.textContent = liveEnabled ? '✓' : '';

    // Banner
    if (paperEnabled && liveEnabled) {
        banner.className = 'mode-banner live';
        banner.textContent = 'DUAL MODE — Paper simulation + Live execution on CoinDCX Futures USDT';
    } else if (liveEnabled) {
        banner.className = 'mode-banner live';
        banner.textContent = 'LIVE TRADE MODE — Real orders on CoinDCX Futures USDT';
    } else if (paperEnabled) {
        banner.className = 'mode-banner paper';
        banner.textContent = 'PAPER TRADE MODE — No real orders are placed';
    } else {
        banner.className = 'mode-banner paper';
        banner.textContent = 'NO ACTIVE MODE — Enable Paper or Live trading to begin';
    }
}

function updateChecklist(data) {
    setCheck('checkApi', data.api_key_configured);
    const balOk = data.balance >= 50;
    setCheck('checkBalance', balOk);
    setCheck('checkMargin', true);
    setCheck('checkNetwork', !data.testnet);
}

function setCheck(id, ok) {
    const el = document.getElementById(id);
    el.textContent = ok ? '✓' : '✗';
    el.className = 'check-icon ' + (ok ? 'pass' : 'fail');
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MODE TOGGLES (dual checkboxes)
// ═══════════════════════════════════════════════════════════════════════════════

function togglePaperMode() {
    const newVal = !paperEnabled;
    setModes(newVal, liveEnabled);
}

function toggleLiveMode() {
    if (!liveEnabled) {
        // Show confirm dialog before enabling live
        document.getElementById('liveConfirmModal').style.display = 'flex';
    } else {
        // Disable live
        setModes(paperEnabled, false);
    }
}

function cancelLiveToggle() {
    document.getElementById('liveConfirmModal').style.display = 'none';
}

async function confirmLiveToggle() {
    document.getElementById('liveConfirmModal').style.display = 'none';
    await setModes(paperEnabled, true);
}

async function setModes(paper, live) {
    try {
        addLog(`Setting modes: Paper=${paper ? 'ON' : 'OFF'}, Live=${live ? 'ON' : 'OFF'}`);
        const res = await fetch(`${API_BASE}/api/deploy/mode`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paper, live }),
        });
        const data = await res.json();
        if (data.success) {
            paperEnabled = paper;
            liveEnabled = live;
            updateModeUI();
            addLog(`Mode updated: Paper=${paper ? 'ON' : 'OFF'}, Live=${live ? 'ON' : 'OFF'}`);
        } else {
            addLog(`ERROR: ${data.error || 'Failed to set modes'}`);
        }
    } catch (e) {
        addLog(`ERROR: ${e.message}`);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TEST CONNECTION
// ═══════════════════════════════════════════════════════════════════════════════

async function testConnection() {
    const connEl = document.getElementById('connStatus');
    connEl.textContent = 'Testing...';
    connEl.className = 'status-value pending';
    addLog('Testing CoinDCX API connection...');

    try {
        const res = await fetch(`${API_BASE}/api/deploy/test-connection`, { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            connEl.textContent = 'Connected';
            connEl.className = 'status-value ok';
            addLog(`Connection OK — Server time: ${data.server_time || 'N/A'}`);

            if (data.balance !== undefined) {
                document.getElementById('usdtBalance').textContent = `$${parseFloat(data.balance).toFixed(2)}`;
                document.getElementById('usdtBalance').className = 'status-value ok';
            }
        } else {
            connEl.textContent = 'Failed';
            connEl.className = 'status-value error';
            addLog(`Connection FAILED: ${data.error || 'Unknown error'}`);
        }
    } catch (e) {
        connEl.textContent = 'Error';
        connEl.className = 'status-value error';
        addLog(`Connection ERROR: ${e.message}`);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  REFRESH BALANCE
// ═══════════════════════════════════════════════════════════════════════════════

async function refreshBalance() {
    addLog('Refreshing account balance...');
    try {
        const res = await fetch(`${API_BASE}/api/deploy/balance`);
        const data = await res.json();
        if (data.success) {
            document.getElementById('usdtBalance').textContent = `$${parseFloat(data.balance).toFixed(2)}`;
            document.getElementById('usdtBalance').className = 'status-value ok';
            document.getElementById('openPositions').textContent = data.open_positions ?? '—';
            addLog(`Balance: $${parseFloat(data.balance).toFixed(2)}, Open positions: ${data.open_positions ?? 0}`);
        } else {
            addLog(`Balance refresh failed: ${data.error || 'Unknown'}`);
        }
    } catch (e) {
        addLog(`Balance ERROR: ${e.message}`);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  KILL SWITCH
// ═══════════════════════════════════════════════════════════════════════════════

async function killSwitch() {
    if (!confirm('KILL SWITCH: This will close ALL live positions and cancel all open orders on CoinDCX Futures. Continue?')) {
        return;
    }

    const btn = document.getElementById('killBtn');
    btn.disabled = true;
    btn.textContent = 'Closing positions...';
    addLog('KILL SWITCH ACTIVATED — Closing all positions...');

    try {
        const res = await fetch(`${API_BASE}/api/deploy/kill-switch`, { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            addLog(`Kill switch complete: ${data.message || 'All positions closed'}`);
        } else {
            addLog(`Kill switch error: ${data.error || 'Unknown'}`);
        }
    } catch (e) {
        addLog(`Kill switch ERROR: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'KILL SWITCH — Close All Live Positions';
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  ACTIVITY LOG
// ═══════════════════════════════════════════════════════════════════════════════

function addLog(msg) {
    const log = document.getElementById('deployLog');
    const ts = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = 'entry';
    entry.innerHTML = `<span class="ts">[${ts}]</span> ${msg}`;
    log.prepend(entry);

    while (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}
