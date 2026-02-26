/**
 * Project Regime-Master — Premium Dashboard API Server
 * Express + Socket.IO backend that reads Python bot state files
 * and pushes real-time updates to the web dashboard.
 */
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const chokidar = require('chokidar');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

const PORT = 3001;
const DATA_DIR = path.join(__dirname, '..', 'data');

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ─── Helper: Safe JSON read ──────────────────────────────────────────────────
function readJSON(filename) {
    const filepath = path.join(DATA_DIR, filename);
    try {
        if (!fs.existsSync(filepath)) return null;
        return JSON.parse(fs.readFileSync(filepath, 'utf8'));
    } catch (e) {
        return null;
    }
}

function readCSV(filename) {
    const filepath = path.join(DATA_DIR, filename);
    try {
        if (!fs.existsSync(filepath)) return [];
        const content = fs.readFileSync(filepath, 'utf8').trim();
        if (!content) return [];
        const lines = content.split('\n');
        if (lines.length < 2) return [];
        const headers = lines[0].split(',');
        return lines.slice(1).map(line => {
            const values = line.split(',');
            const obj = {};
            headers.forEach((h, i) => obj[h.trim()] = values[i]?.trim() || '');
            return obj;
        });
    } catch (e) {
        return [];
    }
}

// ─── REST Endpoints ──────────────────────────────────────────────────────────
app.get('/api/state', (req, res) => {
    res.json(readJSON('bot_state.json') || { regime: 'OFFLINE', confidence: 0 });
});

app.get('/api/multi-state', (req, res) => {
    res.json(readJSON('multi_bot_state.json') || { coins_scanned: 0, active_positions: {} });
});

app.get('/api/scanner', (req, res) => {
    res.json(readJSON('scanner_state.json') || { coins: [] });
});

app.get('/api/trades', (req, res) => {
    res.json(readCSV('trade_log.csv'));
});

app.get('/api/all', (req, res) => {
    res.json({
        state: readJSON('bot_state.json') || { regime: 'OFFLINE', confidence: 0 },
        multi: readJSON('multi_bot_state.json') || { coins_scanned: 0, active_positions: {} },
        scanner: readJSON('scanner_state.json') || { coins: [] },
        trades: readCSV('trade_log.csv'),
        tradebook: readJSON('tradebook.json') || { trades: [], summary: {} },
    });
});

app.get('/api/tradebook', (req, res) => {
    const mode = req.query.mode;
    const filename = mode === 'live' ? 'tradebook_live.json' : 'tradebook.json';
    res.json(readJSON(filename) || { trades: [], summary: {} });
});

// ─── CoinDCX Live Positions API ──────────────────────────────────────────────
app.get('/api/coindcx/positions', (req, res) => {
    const { execSync } = require('child_process');
    const scriptPath = path.join(__dirname, 'fetch_positions.py');
    try {
        const result = execSync(`cd "${path.join(__dirname, '..')}" && python3 "${scriptPath}"`, {
            timeout: 15000,
            encoding: 'utf8',
        });
        const data = JSON.parse(result.trim().split('\n').pop());
        res.json(data);
    } catch (e) {
        console.error('[CoinDCX API] Error:', e.message);
        res.json({ success: false, error: e.message, positions: [], wallet_balance: 0, count: 0 });
    }
});

// ─── Command Endpoint ────────────────────────────────────────────────────────
app.post('/api/command', (req, res) => {
    const { command } = req.body;
    if (!['KILL', 'RESET'].includes(command)) {
        return res.status(400).json({ error: 'Invalid command' });
    }
    const cmdFile = path.join(DATA_DIR, 'commands.json');
    fs.writeFileSync(cmdFile, JSON.stringify({
        command,
        timestamp: new Date().toISOString(),
    }));
    io.emit('command', { command });
    res.json({ success: true, command });
});

// ─── Delete Single Trade ─────────────────────────────────────────────────────
app.delete('/api/tradebook/trade/:tradeId', (req, res) => {
    const { tradeId } = req.params;
    const tbFile = path.join(DATA_DIR, 'tradebook.json');
    try {
        const book = JSON.parse(fs.readFileSync(tbFile, 'utf8'));
        const idx = book.trades.findIndex(t => t.trade_id === tradeId);
        if (idx === -1) return res.status(404).json({ error: 'Trade not found' });
        book.trades.splice(idx, 1);
        fs.writeFileSync(tbFile, JSON.stringify(book, null, 2));
        io.emit('tradebook-update', book);
        res.json({ success: true, deleted: tradeId });
    } catch (e) {
        res.status(500).json({ error: 'Failed to delete trade' });
    }
});

// ─── Close Trade(s) — set status to CLOSED with current price ────────────────
app.post('/api/tradebook/close', (req, res) => {
    const { trade_ids } = req.body;   // Array of trade IDs to close
    if (!trade_ids || !Array.isArray(trade_ids) || trade_ids.length === 0) {
        return res.status(400).json({ error: 'trade_ids array required' });
    }
    const tbFile = path.join(DATA_DIR, 'tradebook.json');
    try {
        const book = JSON.parse(fs.readFileSync(tbFile, 'utf8'));
        const closedIds = [];
        for (const tid of trade_ids) {
            const trade = book.trades.find(t => t.trade_id === tid && t.status === 'ACTIVE');
            if (!trade) continue;

            const exitPrice = trade.current_price || trade.entry_price;
            const entry = trade.entry_price;
            const qty = trade.quantity;
            const lev = trade.leverage;
            const capital = trade.capital || 100;

            let rawPnl;
            if (trade.position === 'LONG') {
                rawPnl = (exitPrice - entry) * qty;
            } else {
                rawPnl = (entry - exitPrice) * qty;
            }

            // Commission (taker fee both legs — 0.05% per leg)
            const commission = parseFloat(((entry * qty + exitPrice * qty) * 0.0005).toFixed(4));
            const leveragedPnl = parseFloat((rawPnl * lev - commission).toFixed(4));
            const pnlPct = capital > 0 ? parseFloat((leveragedPnl / capital * 100).toFixed(2)) : 0;

            trade.exit_price = exitPrice;
            trade.exit_timestamp = new Date().toISOString();
            trade.status = 'CLOSED';
            trade.exit_reason = 'MANUAL';
            trade.commission = commission;
            trade.realized_pnl = leveragedPnl;
            trade.realized_pnl_pct = pnlPct;
            trade.unrealized_pnl = 0;
            trade.unrealized_pnl_pct = 0;
            closedIds.push(tid);
        }

        fs.writeFileSync(tbFile, JSON.stringify(book, null, 2));
        io.emit('tradebook-update', book);

        // Also remove from multi_bot_state active_positions
        const multiFile = path.join(DATA_DIR, 'multi_bot_state.json');
        if (fs.existsSync(multiFile)) {
            try {
                const multi = JSON.parse(fs.readFileSync(multiFile, 'utf8'));
                const closedSymbols = book.trades
                    .filter(t => closedIds.includes(t.trade_id))
                    .map(t => t.symbol);
                for (const sym of closedSymbols) {
                    delete multi.active_positions?.[sym];
                }
                fs.writeFileSync(multiFile, JSON.stringify(multi, null, 2));
                io.emit('multi-update', multi);
            } catch (e) { /* silent */ }
        }

        res.json({ success: true, closed: closedIds.length, trade_ids: closedIds });
    } catch (e) {
        res.status(500).json({ error: 'Failed to close trades' });
    }
});

// ─── Delete All Trades ───────────────────────────────────────────────────────
app.delete('/api/tradebook/all', (req, res) => {
    try {
        // Clear tradebook
        const tbFile = path.join(DATA_DIR, 'tradebook.json');
        const emptyBook = { trades: [], summary: {} };
        fs.writeFileSync(tbFile, JSON.stringify(emptyBook, null, 2));
        io.emit('tradebook-update', emptyBook);

        // Also clear active positions from multi_bot_state
        const multiFile = path.join(DATA_DIR, 'multi_bot_state.json');
        if (fs.existsSync(multiFile)) {
            try {
                const multi = JSON.parse(fs.readFileSync(multiFile, 'utf8'));
                multi.active_positions = {};
                multi.deployed_count = 0;
                multi.total_trades = 0;
                multi.timestamp = new Date().toISOString();
                fs.writeFileSync(multiFile, JSON.stringify(multi, null, 2));
                io.emit('multi-update', multi);
            } catch (e) { /* silent */ }
        }

        // Clear trade log CSV (keep header only)
        const csvFile = path.join(DATA_DIR, 'trade_log.csv');
        if (fs.existsSync(csvFile)) {
            try {
                const content = fs.readFileSync(csvFile, 'utf8');
                const header = content.split('\n')[0];
                if (header) {
                    fs.writeFileSync(csvFile, header + '\n');
                    io.emit('trades-update', []);
                }
            } catch (e) { /* silent */ }
        }

        res.json({ success: true, message: 'All trades and positions deleted' });
    } catch (e) {
        res.status(500).json({ error: 'Failed to delete trades' });
    }
});

// ─── Socket.IO ───────────────────────────────────────────────────────────────

// Helper: read last N lines from a text file
function readLastLines(filename, maxLines = 100) {
    const filepath = path.join(DATA_DIR, filename);
    try {
        if (!fs.existsSync(filepath)) return [];
        const content = fs.readFileSync(filepath, 'utf8');
        const lines = content.trim().split('\n');
        return lines.slice(-maxLines);
    } catch (e) {
        return [];
    }
}

// Track log file size for incremental reads
let lastLogSize = 0;

io.on('connection', (socket) => {
    console.log(`[Dashboard] Client connected: ${socket.id}`);

    // Send initial state
    socket.emit('full-update', {
        state: readJSON('bot_state.json'),
        multi: readJSON('multi_bot_state.json'),
        scanner: readJSON('scanner_state.json'),
        trades: readCSV('trade_log.csv'),
        tradebook: readJSON('tradebook.json'),
    });

    // Send last 100 log lines on connect
    const logLines = readLastLines('bot.log', 100);
    if (logLines.length > 0) {
        socket.emit('log-init', logLines);
    }

    // Manual cycle trigger from dashboard
    socket.on('trigger-cycle', () => {
        console.log(`[Dashboard] Manual cycle trigger from ${socket.id}`);
        const triggerFile = path.join(DATA_DIR, 'force_cycle.trigger');
        fs.writeFileSync(triggerFile, Date.now().toString());
        socket.emit('trigger-ack', { status: 'ok', message: 'Cycle trigger sent to bot' });
    });

    socket.on('disconnect', () => {
        console.log(`[Dashboard] Client disconnected: ${socket.id}`);
    });
});

// ─── Log File Endpoint ───────────────────────────────────────────────────────
app.get('/api/logs', (req, res) => {
    const lines = parseInt(req.query.lines) || 100;
    res.json({ lines: readLastLines('bot.log', lines) });
});

// ─── Execution Log History ───────────────────────────────────────────────────
const EXEC_LOG_FILE = path.join(DATA_DIR, 'execution_log_history.json');
const MAX_EXEC_LOG_ENTRIES = 200;
let lastLoggedCycle = -1;

function appendExecLogEntry(multi) {
    if (!multi || multi.cycle == null || multi.cycle === lastLoggedCycle) return;
    lastLoggedCycle = multi.cycle;

    let history = [];
    try {
        if (fs.existsSync(EXEC_LOG_FILE)) {
            history = JSON.parse(fs.readFileSync(EXEC_LOG_FILE, 'utf-8'));
        }
    } catch (e) { history = []; }

    const tradebook = readJSON('tradebook.json');
    const closedCount = (tradebook?.trades || []).filter(t => t.status === 'CLOSED').length;

    history.unshift({
        timestamp: multi.timestamp || new Date().toISOString(),
        cycle: multi.cycle,
        coins_scanned: multi.coins_scanned ?? 0,
        eligible: multi.eligible_count ?? 0,
        deployed: multi.deployed_count ?? 0,
        total_trades: multi.total_trades ?? 0,
        closed_trades: closedCount,
        exec_time: multi.cycle_execution_time_seconds ?? 0,
    });

    if (history.length > MAX_EXEC_LOG_ENTRIES) {
        history = history.slice(0, MAX_EXEC_LOG_ENTRIES);
    }

    try {
        fs.writeFileSync(EXEC_LOG_FILE, JSON.stringify(history, null, 2));
    } catch (e) {
        console.error('[ExecLog] Failed to save:', e.message);
    }
}

app.get('/api/execution-log', (req, res) => {
    try {
        if (!fs.existsSync(EXEC_LOG_FILE)) return res.json([]);
        const history = JSON.parse(fs.readFileSync(EXEC_LOG_FILE, 'utf-8'));
        res.json(history);
    } catch (e) {
        res.json([]);
    }
});

// ─── File Watcher → Push updates ─────────────────────────────────────────────
if (fs.existsSync(DATA_DIR)) {
    const watcher = chokidar.watch(DATA_DIR, {
        persistent: true,
        ignoreInitial: true,
        awaitWriteFinish: { stabilityThreshold: 500, pollInterval: 100 },
    });

    watcher.on('change', (filepath) => {
        const basename = path.basename(filepath);
        console.log(`[Watcher] File changed: ${basename}`);

        if (basename === 'bot_state.json') {
            io.emit('state-update', readJSON('bot_state.json'));
        } else if (basename === 'multi_bot_state.json') {
            const multi = readJSON('multi_bot_state.json');
            io.emit('multi-update', multi);
            // Persist execution log entry
            appendExecLogEntry(multi);
        } else if (basename === 'scanner_state.json') {
            io.emit('scanner-update', readJSON('scanner_state.json'));
        } else if (basename === 'trade_log.csv') {
            io.emit('trades-update', readCSV('trade_log.csv'));
        } else if (basename === 'tradebook.json') {
            const tbData = readJSON('tradebook.json');
            io.emit('tradebook-update', tbData);
            // Detect new trades and send entry alerts
            checkForNewTrades(tbData);
        } else if (basename === 'bot.log') {
            // Stream new log lines incrementally
            try {
                const logPath = path.join(DATA_DIR, 'bot.log');
                const stat = fs.statSync(logPath);
                if (stat.size > lastLogSize) {
                    const buf = Buffer.alloc(stat.size - lastLogSize);
                    const fd = fs.openSync(logPath, 'r');
                    fs.readSync(fd, buf, 0, buf.length, lastLogSize);
                    fs.closeSync(fd);
                    const newLines = buf.toString('utf8').trim().split('\n').filter(l => l);
                    if (newLines.length > 0) {
                        io.emit('log-lines', newLines);
                    }
                }
                lastLogSize = stat.size;
            } catch (e) { /* silent */ }
        }
    });
}

// ─── 10-Second Live Price Ticker ─────────────────────────────────────────────
// Fetches current prices from Binance public API every 10 seconds
// and broadcasts to all connected clients via Socket.IO.

let trackedSymbols = ['BTCUSDT', 'ETHUSDT'];

function refreshTrackedSymbols() {
    const symbols = new Set(['BTCUSDT', 'ETHUSDT']);
    const multi = readJSON('multi_bot_state.json');
    const scanner = readJSON('scanner_state.json');

    if (multi?.active_positions) {
        Object.keys(multi.active_positions).forEach(s => symbols.add(s));
    }
    if (multi?.coin_states) {
        Object.keys(multi.coin_states).forEach(s => symbols.add(s));
    }
    if (scanner?.coins) {
        scanner.coins.forEach(c => { if (c.symbol) symbols.add(c.symbol); });
    }
    const tradebook = readJSON('tradebook.json');
    if (tradebook?.trades) {
        tradebook.trades
            .filter(t => t.status === 'ACTIVE')
            .forEach(t => { if (t.symbol) symbols.add(t.symbol); });
    }
    trackedSymbols = Array.from(symbols);
}

async function fetchLivePrices() {
    if (trackedSymbols.length === 0) return {};
    try {
        // Use Binance public API (no auth needed)
        const https = require('https');
        const symbols = JSON.stringify(trackedSymbols);
        const url = `https://api.binance.com/api/v3/ticker/price?symbols=${encodeURIComponent(symbols)}`;

        return new Promise((resolve) => {
            https.get(url, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    try {
                        const tickers = JSON.parse(data);
                        const prices = {};
                        tickers.forEach(t => { prices[t.symbol] = parseFloat(t.price); });
                        resolve(prices);
                    } catch (e) { resolve({}); }
                });
            }).on('error', () => resolve({}));
        });
    } catch (e) {
        return {};
    }
}

// Refresh symbol list every 30 seconds
refreshTrackedSymbols();
setInterval(refreshTrackedSymbols, 30000);

// ─── Server-Side SL/TP/MAX_LOSS Enforcement ─────────────────────────────────
// Runs every 10 seconds with live prices to enforce stop-loss, take-profit,
// and max-loss thresholds. This prevents trades from blowing past limits between
// the 5-minute main bot cycles.

function readConfigValues() {
    try {
        const configPath = path.join(__dirname, '..', 'config.py');
        const content = fs.readFileSync(configPath, 'utf8');
        const getVal = (name, fallback) => {
            const m = content.match(new RegExp(`^${name}\\s*=\\s*(.+)`, 'm'));
            if (!m) return fallback;
            let v = m[1].trim().split('#')[0].trim(); // Strip comments
            if (v === 'True') return true;
            if (v === 'False') return false;
            return parseFloat(v) || fallback;
        };
        return {
            MAX_LOSS_PER_TRADE_PCT: getVal('MAX_LOSS_PER_TRADE_PCT', -30),
            TAKER_FEE: getVal('TAKER_FEE', 0.0005),
            PAPER_TRADE: getVal('PAPER_TRADE', true),
            TRAILING_SL_ENABLED: getVal('TRAILING_SL_ENABLED', true),
            TRAILING_SL_ACTIVATION_ATR: getVal('TRAILING_SL_ACTIVATION_ATR', 1.0),
            TRAILING_SL_DISTANCE_ATR: getVal('TRAILING_SL_DISTANCE_ATR', 1.0),
            TRAILING_TP_ENABLED: getVal('TRAILING_TP_ENABLED', true),
            TRAILING_TP_ACTIVATION_PCT: getVal('TRAILING_TP_ACTIVATION_PCT', 0.75),
            TRAILING_TP_EXTENSION_ATR: getVal('TRAILING_TP_EXTENSION_ATR', 1.5),
            TRAILING_TP_MAX_EXTENSIONS: getVal('TRAILING_TP_MAX_EXTENSIONS', 3),
        };
    } catch (e) {
        return {
            MAX_LOSS_PER_TRADE_PCT: -30, TAKER_FEE: 0.0005, PAPER_TRADE: true,
            TRAILING_SL_ENABLED: true, TRAILING_SL_ACTIVATION_ATR: 1.0,
            TRAILING_SL_DISTANCE_ATR: 1.0, TRAILING_TP_ENABLED: true,
            TRAILING_TP_ACTIVATION_PCT: 0.75, TRAILING_TP_EXTENSION_ATR: 1.5,
            TRAILING_TP_MAX_EXTENSIONS: 3,
        };
    }
}

function sendTelegramAlert(trade, reason, pnl, pnlPct) {
    try {
        const envPath = path.join(__dirname, '..', '.env');
        const envContent = fs.readFileSync(envPath, 'utf8');
        const getEnv = (key) => {
            const m = envContent.match(new RegExp(`^${key}=(.*)`, 'm'));
            return m ? m[1].trim() : '';
        };
        const token = getEnv('TELEGRAM_BOT_TOKEN');
        const chatId = getEnv('TELEGRAM_CHAT_ID');
        if (!token || !chatId || getEnv('TELEGRAM_ENABLED') !== 'true') return;
        if (getEnv('TELEGRAM_NOTIFY_TRADES') === 'false') return;

        const reasonMap = {
            'STOP_LOSS': '🛑 Stop Loss', 'TRAILING_SL': '🛑 Trailing SL',
            'TAKE_PROFIT': '🎯 Take Profit', 'TRAILING_TP': '🎯 Trailing TP',
            'MAX_LOSS': '🚨 MAX LOSS GUARD',
        };
        const emoji = pnl >= 0 ? '✅' : '❌';
        const sign = pnl >= 0 ? '+' : '';

        const text = [
            `${emoji} <b>TRADE CLOSED</b>`,
            '━━━━━━━━━━━━━━━━━━',
            `📊 <b>${trade.symbol}</b> | ${trade.position}`,
            `📍 ${reasonMap[reason] || reason}`,
            `📈 Entry: <code>${trade.entry_price.toFixed(6)}</code>`,
            `📉 Exit: <code>${trade.exit_price.toFixed(6)}</code>`,
            `💰 P&L: <b>${sign}$${pnl.toFixed(2)}</b> (${sign}${pnlPct.toFixed(2)}%)`,
            `⏱ Duration: <b>${Math.round(trade.duration_minutes || 0)}m</b>`,
            `🕐 ${new Date().toISOString().slice(11, 19)} UTC`,
        ].join('\n');

        const https = require('https');
        const data = JSON.stringify({ chat_id: chatId, text, parse_mode: 'HTML' });
        const url = new URL(`https://api.telegram.org/bot${token}/sendMessage`);
        const req = https.request(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
        }, () => { }); // fire and forget
        req.on('error', () => { });
        req.write(data);
        req.end();
    } catch (e) {
        // Non-critical: don't let Telegram errors break SL engine
    }
}

// ─── Entry Alert: track known trade IDs & send alerts for new ones ───────────
let knownTradeIds = new Set();
// Initialize from current tradebook on startup
try {
    const tb = readJSON('tradebook.json');
    if (tb && tb.trades) {
        tb.trades.forEach(t => knownTradeIds.add(t.trade_id));
    }
    console.log(`[Telegram] Initialized with ${knownTradeIds.size} known trade IDs`);
} catch (e) { /* ok */ }

let entryAlertTimer = null;
let pendingNewTrades = [];

function flushEntryAlerts() {
    if (pendingNewTrades.length === 0) return;
    const trades = [...pendingNewTrades];
    pendingNewTrades = [];
    entryAlertTimer = null;

    try {
        const envPath = path.join(__dirname, '..', '.env');
        const envContent = fs.readFileSync(envPath, 'utf8');
        const getEnv = (key) => {
            const m = envContent.match(new RegExp(`^${key}=(.*)`, 'm'));
            return m ? m[1].trim() : '';
        };
        const token = getEnv('TELEGRAM_BOT_TOKEN');
        const chatId = getEnv('TELEGRAM_CHAT_ID');
        if (!token || !chatId || getEnv('TELEGRAM_ENABLED') !== 'true') return;
        if (getEnv('TELEGRAM_NOTIFY_TRADES') === 'false') return;

        const count = trades.length;
        const lines = [
            `📦 <b>${count} NEW TRADE${count > 1 ? 'S' : ''} DEPLOYED</b>`,
            '━━━━━━━━━━━━━━━━━━',
        ];
        trades.forEach(t => {
            const emoji = t.position === 'LONG' ? '🟢' : '🔴';
            const conf = t.confidence != null ? (t.confidence * 100).toFixed(0) + '%' : '?';
            const ep = t.entry_price ? Number(t.entry_price).toFixed(6) : '?';
            const sl = t.stop_loss ? Number(t.stop_loss).toFixed(6) : '—';
            const tp = t.take_profit ? Number(t.take_profit).toFixed(6) : '—';
            lines.push(
                `${emoji} <b>${t.symbol}</b> ${t.position} ${t.leverage || 1}× | ${t.regime || '?'} ${conf}`,
                `   📈 <code>${ep}</code>  🛑 <code>${sl}</code>  🎯 <code>${tp}</code>`
            );
        });
        lines.push(``, `💵 Capital: $100 each  |  🕐 ${new Date().toISOString().slice(11, 19)} UTC`);

        const text = lines.join('\n');
        const https = require('https');
        const data = JSON.stringify({ chat_id: chatId, text, parse_mode: 'HTML' });
        const url = new URL(`https://api.telegram.org/bot${token}/sendMessage`);
        const req = https.request(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
        }, (res) => {
            let body = '';
            res.on('data', d => body += d);
            res.on('end', () => {
                try {
                    const r = JSON.parse(body);
                    if (r.ok) console.log(`[Telegram] ✅ Entry alert sent for ${count} trade(s)`);
                    else console.log(`[Telegram] ❌ Entry alert failed: ${r.description}`);
                } catch (e) { /* ignore */ }
            });
        });
        req.on('error', (e) => console.log(`[Telegram] ❌ Entry alert error: ${e.message}`));
        req.write(data);
        req.end();
    } catch (e) {
        console.log(`[Telegram] Entry alert error: ${e.message}`);
    }
}

function checkForNewTrades(book) {
    if (!book || !book.trades) return;
    const newTrades = book.trades.filter(t => t.status === 'ACTIVE' && !knownTradeIds.has(t.trade_id));
    newTrades.forEach(t => {
        knownTradeIds.add(t.trade_id);
        pendingNewTrades.push(t);
        console.log(`[Telegram] New trade detected: ${t.symbol} ${t.position} ${t.leverage}×`);
    });
    // Also add any closed trades we haven't seen
    book.trades.forEach(t => knownTradeIds.add(t.trade_id));

    if (newTrades.length > 0 && !entryAlertTimer) {
        // Wait 5 seconds for more trades to accumulate, then flush
        entryAlertTimer = setTimeout(flushEntryAlerts, 5000);
    }
}

function updateTradebookWithPrices(prices) {
    const tbFile = path.join(DATA_DIR, 'tradebook.json');
    let book;
    try {
        book = JSON.parse(fs.readFileSync(tbFile, 'utf8'));
    } catch (e) { return; }

    if (!book?.trades?.length) return;

    const cfg = readConfigValues();
    let changed = false;
    const now = new Date().toISOString();

    for (const trade of book.trades) {
        if (trade.status !== 'ACTIVE') continue;

        const current = prices[trade.symbol];
        if (!current) continue;

        const entry = trade.entry_price;
        const qty = trade.quantity;
        const lev = trade.leverage;
        const capital = trade.capital || 100;
        const isLong = trade.position === 'LONG';

        // Calculate P&L
        const rawPnl = isLong ? (current - entry) * qty : (entry - current) * qty;
        const entryNotional = entry * qty;
        const exitNotional = current * qty;
        const estCommission = (entryNotional + exitNotional) * cfg.TAKER_FEE;
        const leveragedPnl = Math.round((rawPnl * lev - estCommission) * 10000) / 10000;
        const pnlPct = capital ? Math.round((leveragedPnl / capital) * 10000) / 100 : 0;

        // Track max favorable / adverse
        if (leveragedPnl > (trade.max_favorable || 0)) trade.max_favorable = leveragedPnl;
        if (leveragedPnl < (trade.max_adverse || 0)) trade.max_adverse = leveragedPnl;

        // Duration
        const entryTime = new Date(trade.entry_timestamp);
        const duration = (Date.now() - entryTime.getTime()) / 60000;

        trade.current_price = current;
        trade.unrealized_pnl = leveragedPnl;
        trade.unrealized_pnl_pct = pnlPct;
        trade.duration_minutes = Math.round(duration * 10) / 10;

        // ── Trailing SL Logic ─────────────────────────────────────────
        const atr = trade.atr_at_entry || 0;

        // Initialize missing trailing fields
        if (trade.trailing_sl === undefined) trade.trailing_sl = trade.stop_loss;
        if (trade.trailing_tp === undefined) trade.trailing_tp = trade.take_profit;
        if (trade.peak_price === undefined) trade.peak_price = entry;
        if (trade.trailing_active === undefined) trade.trailing_active = false;
        if (trade.tp_extensions === undefined) trade.tp_extensions = 0;

        if (cfg.TRAILING_SL_ENABLED && atr > 0) {
            // Update peak price
            if (isLong) {
                if (current > trade.peak_price) trade.peak_price = current;
            } else {
                if (current < trade.peak_price) trade.peak_price = current;
            }

            // Check activation
            const activationDist = atr * cfg.TRAILING_SL_ACTIVATION_ATR;
            const favorableMove = isLong ? (current - entry) : (entry - current);
            if (favorableMove >= activationDist) trade.trailing_active = true;

            // Trail the SL
            if (trade.trailing_active) {
                const trailDist = atr * cfg.TRAILING_SL_DISTANCE_ATR;
                if (isLong) {
                    const newSl = Math.round((trade.peak_price - trailDist) * 1e6) / 1e6;
                    if (newSl > trade.trailing_sl) trade.trailing_sl = newSl;
                } else {
                    const newSl = Math.round((trade.peak_price + trailDist) * 1e6) / 1e6;
                    if (newSl < trade.trailing_sl) trade.trailing_sl = newSl;
                }
            }
        }

        // ── Trailing TP Logic ──────────────────────────────────────────
        if (cfg.TRAILING_TP_ENABLED && atr > 0 && trade.tp_extensions < cfg.TRAILING_TP_MAX_EXTENSIONS) {
            const tpDist = isLong
                ? (trade.trailing_tp - entry)
                : (entry - trade.trailing_tp);
            const progress = tpDist > 0
                ? (isLong ? (current - entry) : (entry - current)) / tpDist
                : 0;

            if (progress >= cfg.TRAILING_TP_ACTIVATION_PCT) {
                const extAmount = atr * cfg.TRAILING_TP_EXTENSION_ATR;
                trade.trailing_tp = isLong
                    ? Math.round((trade.trailing_tp + extAmount) * 1e6) / 1e6
                    : Math.round((trade.trailing_tp - extAmount) * 1e6) / 1e6;
                trade.tp_extensions++;
                console.log(`[SL-Engine] 📈 Trailing TP extended for ${trade.trade_id}: new TP=${trade.trailing_tp}`);
            }
        }

        // ── HARD MAX LOSS GUARD — fires FIRST ─────────────────────────
        if (pnlPct <= cfg.MAX_LOSS_PER_TRADE_PCT) {
            console.log(`[SL-Engine] 🛑 MAX LOSS hit on ${trade.symbol} (${pnlPct}% <= ${cfg.MAX_LOSS_PER_TRADE_PCT}%) — auto-closing ${trade.trade_id}`);
            trade.status = 'CLOSED';
            trade.exit_price = current;
            trade.exit_timestamp = now;
            trade.exit_reason = 'MAX_LOSS';
            trade.realized_pnl = leveragedPnl;
            trade.realized_pnl_pct = pnlPct;
            trade.unrealized_pnl = 0;
            trade.unrealized_pnl_pct = 0;
            sendTelegramAlert(trade, 'MAX_LOSS', leveragedPnl, pnlPct);
            changed = true;
            continue;
        }

        // ── SL / TP hit checks ────────────────────────────────────────
        const effectiveSl = trade.trailing_sl ?? trade.stop_loss;
        const effectiveTp = trade.trailing_tp ?? trade.take_profit;

        if (isLong) {
            if (current <= effectiveSl) {
                const reason = trade.trailing_active ? 'TRAILING_SL' : 'STOP_LOSS';
                console.log(`[SL-Engine] 🔻 ${reason} hit on ${trade.symbol} — closing ${trade.trade_id}`);
                trade.status = 'CLOSED';
                trade.exit_price = effectiveSl;
                const closePnl = ((effectiveSl - entry) * qty * lev - estCommission);
                trade.realized_pnl = Math.round(closePnl * 10000) / 10000;
                trade.realized_pnl_pct = capital ? Math.round((closePnl / capital) * 10000) / 100 : 0;
                trade.exit_timestamp = now;
                trade.exit_reason = reason;
                trade.unrealized_pnl = 0;
                trade.unrealized_pnl_pct = 0;
                sendTelegramAlert(trade, reason, trade.realized_pnl, trade.realized_pnl_pct);
                changed = true;
                continue;
            }
            if (current >= effectiveTp) {
                const reason = trade.tp_extensions > 0 ? 'TRAILING_TP' : 'TAKE_PROFIT';
                console.log(`[SL-Engine] 🎯 ${reason} hit on ${trade.symbol} — closing ${trade.trade_id}`);
                trade.status = 'CLOSED';
                trade.exit_price = effectiveTp;
                const closePnl = ((effectiveTp - entry) * qty * lev - estCommission);
                trade.realized_pnl = Math.round(closePnl * 10000) / 10000;
                trade.realized_pnl_pct = capital ? Math.round((closePnl / capital) * 10000) / 100 : 0;
                trade.exit_timestamp = now;
                trade.exit_reason = reason;
                trade.unrealized_pnl = 0;
                trade.unrealized_pnl_pct = 0;
                sendTelegramAlert(trade, reason, trade.realized_pnl, trade.realized_pnl_pct);
                changed = true;
                continue;
            }
        } else {
            if (current >= effectiveSl) {
                const reason = trade.trailing_active ? 'TRAILING_SL' : 'STOP_LOSS';
                console.log(`[SL-Engine] 🔻 ${reason} hit on ${trade.symbol} — closing ${trade.trade_id}`);
                trade.status = 'CLOSED';
                trade.exit_price = effectiveSl;
                const closePnl = ((entry - effectiveSl) * qty * lev - estCommission);
                trade.realized_pnl = Math.round(closePnl * 10000) / 10000;
                trade.realized_pnl_pct = capital ? Math.round((closePnl / capital) * 10000) / 100 : 0;
                trade.exit_timestamp = now;
                trade.exit_reason = reason;
                trade.unrealized_pnl = 0;
                trade.unrealized_pnl_pct = 0;
                sendTelegramAlert(trade, reason, trade.realized_pnl, trade.realized_pnl_pct);
                changed = true;
                continue;
            }
            if (current <= effectiveTp) {
                const reason = trade.tp_extensions > 0 ? 'TRAILING_TP' : 'TAKE_PROFIT';
                console.log(`[SL-Engine] 🎯 ${reason} hit on ${trade.symbol} — closing ${trade.trade_id}`);
                trade.status = 'CLOSED';
                trade.exit_price = effectiveTp;
                const closePnl = ((entry - effectiveTp) * qty * lev - estCommission);
                trade.realized_pnl = Math.round(closePnl * 10000) / 10000;
                trade.realized_pnl_pct = capital ? Math.round((closePnl / capital) * 10000) / 100 : 0;
                trade.exit_timestamp = now;
                trade.exit_reason = reason;
                trade.unrealized_pnl = 0;
                trade.unrealized_pnl_pct = 0;
                sendTelegramAlert(trade, reason, trade.realized_pnl, trade.realized_pnl_pct);
                changed = true;
                continue;
            }
        }

        changed = true; // P&L values updated
    }

    if (changed) {
        // Recompute summary
        const closed = book.trades.filter(t => t.status === 'CLOSED');
        const active = book.trades.filter(t => t.status === 'ACTIVE');
        const wins = closed.filter(t => (t.realized_pnl || 0) > 0);
        const totalRealized = closed.reduce((s, t) => s + (t.realized_pnl || 0), 0);
        const totalUnrealized = active.reduce((s, t) => s + (t.unrealized_pnl || 0), 0);

        const MAX_CAPITAL = 1500;
        const deployedCapital = active.length * 100;
        const losses = closed.filter(t => (t.realized_pnl || 0) < 0);
        const closedPnls = closed.map(t => t.realized_pnl || 0);
        const bestTrade = closedPnls.length > 0 ? Math.max(...closedPnls) : 0;
        const worstTrade = closedPnls.length > 0 ? Math.min(...closedPnls) : 0;

        book.summary = {
            total_trades: book.trades.length,
            active_trades: active.length,
            closed_trades: closed.length,
            wins: wins.length,
            losses: losses.length,
            win_rate_pct: closed.length > 0 ? Math.round((wins.length / closed.length) * 1000) / 10 : 0,
            total_realized_pnl: Math.round(totalRealized * 10000) / 10000,
            total_realized_pnl_pct: MAX_CAPITAL > 0 ? Math.round((totalRealized / MAX_CAPITAL) * 10000) / 100 : 0,
            total_unrealized_pnl: Math.round(totalUnrealized * 10000) / 10000,
            total_unrealized_pnl_pct: deployedCapital > 0 ? Math.round((totalUnrealized / deployedCapital) * 10000) / 100 : 0,
            cumulative_pnl: Math.round((totalRealized + totalUnrealized) * 10000) / 10000,
            cumulative_pnl_pct: MAX_CAPITAL > 0 ? Math.round(((totalRealized + totalUnrealized) / MAX_CAPITAL) * 10000) / 100 : 0,
            best_trade: Math.round(bestTrade * 10000) / 10000,
            worst_trade: Math.round(worstTrade * 10000) / 10000,
            net_pnl: Math.round((totalRealized + totalUnrealized) * 100) / 100,
            last_updated: new Date().toISOString(),
        };

        try {
            fs.writeFileSync(tbFile, JSON.stringify(book, null, 2));
        } catch (e) {
            console.error('[SL-Engine] Failed to save tradebook:', e.message);
        }
        io.emit('tradebook-update', book);
    }
}

// Broadcast live prices every 10 seconds
setInterval(async () => {
    if (io.engine?.clientsCount === 0) return; // Skip if no clients
    const prices = await fetchLivePrices();
    if (Object.keys(prices).length > 0) {
        io.emit('price-tick', { prices, timestamp: new Date().toISOString() });
        // Enforce SL/TP/MAX_LOSS with fresh prices every tick
        try {
            updateTradebookWithPrices(prices);
        } catch (e) {
            console.error('[SL-Engine] Update error:', e.message);
        }
    }
}, 10000);

// REST endpoint for on-demand price fetch
app.get('/api/prices', async (req, res) => {
    const prices = await fetchLivePrices();
    res.json({ prices, timestamp: new Date().toISOString() });
});

// ─── Klines API (proxies Binance for chart data) ────────────────────────────
app.get('/api/klines', async (req, res) => {
    const { symbol, interval = '1h', limit = 300 } = req.query;
    if (!symbol) return res.status(400).json({ error: 'symbol required' });

    try {
        const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
        const response = await fetch(url);
        const data = await response.json();

        const candles = data.map(k => ({
            time: Math.floor(k[0] / 1000),
            open: parseFloat(k[1]),
            high: parseFloat(k[2]),
            low: parseFloat(k[3]),
            close: parseFloat(k[4]),
            volume: parseFloat(k[5]),
        }));

        res.json({ symbol, interval, candles });
    } catch (err) {
        console.error('[Klines] Error:', err.message);
        res.status(500).json({ error: 'Failed to fetch klines' });
    }
});

// ─── Backtest API ───────────────────────────────────────────────────────────
app.post('/api/backtest', (req, res) => {
    const params = req.body;
    res.setHeader('Content-Type', 'application/x-ndjson');
    res.setHeader('Transfer-Encoding', 'chunked');

    // Write params to temp file for the Python script
    const paramPath = path.join(DATA_DIR, '_backtest_params.json');
    fs.writeFileSync(paramPath, JSON.stringify(params, null, 2));

    // Delete stale report so we never serve old results
    const reportPath = path.join(DATA_DIR, 'backtest_report.json');
    try { fs.unlinkSync(reportPath); } catch (e) { /* no old report, fine */ }

    // Spawn Python backtest with streaming
    const projectRoot = path.join(__dirname, '..');
    // Use .venv python if available (local dev), otherwise system python3
    const venvPython = path.join(projectRoot, '.venv', 'bin', 'python3');
    const pythonPath = fs.existsSync(venvPython) ? venvPython : 'python3';
    const scriptPath = path.join(projectRoot, 'backtest.py');

    const child = require('child_process').spawn(pythonPath, [scriptPath, '--server', paramPath], {
        cwd: projectRoot,
        env: { ...process.env },
    });

    let stderrBuf = '';
    child.stdout.on('data', (data) => {
        const text = data.toString();
        // Forward each JSON line as progress
        const lines = text.split('\n');
        for (const line of lines) {
            if (line.startsWith('{')) {
                try { res.write(line + '\n'); } catch (e) { }
            }
        }
    });

    child.stderr.on('data', (data) => {
        stderrBuf += data.toString();
    });

    child.on('close', (code) => {
        if (code === 0 && fs.existsSync(reportPath)) {
            try {
                const report = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));
                res.write(JSON.stringify({ type: 'result', data: report }) + '\n');

                // Auto-save report to history
                try {
                    const reportsDir = path.join(DATA_DIR, 'backtest_reports');
                    if (!fs.existsSync(reportsDir)) fs.mkdirSync(reportsDir, { recursive: true });
                    const ts = new Date().toISOString().replace(/[:.]/g, '-');
                    const savedReport = { id: ts, timestamp: new Date().toISOString(), params, report };
                    fs.writeFileSync(path.join(reportsDir, `report_${ts}.json`), JSON.stringify(savedReport, null, 2));
                    console.log(`[Backtest] Report saved: report_${ts}.json`);
                } catch (saveErr) {
                    console.error('[Backtest] Failed to save report:', saveErr.message);
                }
            } catch (e) {
                res.write(JSON.stringify({ type: 'error', message: 'Failed to parse report' }) + '\n');
            }
        } else {
            const errMsg = stderrBuf.trim().split('\n').pop() || `Backtest process exited with code ${code}`;
            res.write(JSON.stringify({ type: 'error', message: errMsg }) + '\n');
        }
        res.end();
    });

    child.on('error', (err) => {
        res.write(JSON.stringify({ type: 'error', message: err.message }) + '\n');
        res.end();
    });
});

// ─── Backtest Report History ─────────────────────────────────────────────────
const REPORTS_DIR = path.join(DATA_DIR, 'backtest_reports');

app.get('/api/backtest-reports', (req, res) => {
    try {
        if (!fs.existsSync(REPORTS_DIR)) return res.json([]);
        const files = fs.readdirSync(REPORTS_DIR)
            .filter(f => f.endsWith('.json'))
            .sort().reverse();
        const summaries = files.map(f => {
            try {
                const data = JSON.parse(fs.readFileSync(path.join(REPORTS_DIR, f), 'utf-8'));
                const s = data.report?.summary || {};
                return {
                    id: data.id,
                    timestamp: data.timestamp,
                    total_return: s.total_return_pct || 0,
                    total_pnl: s.total_pnl || 0,
                    win_rate: s.win_rate_pct || 0,
                    total_trades: s.total_trades || 0,
                    coins_tested: s.coins_tested || 0,
                    profit_factor: s.profit_factor || 0,
                    max_drawdown: s.max_drawdown_pct || 0,
                };
            } catch (e) { return null; }
        }).filter(Boolean);
        res.json(summaries);
    } catch (e) {
        res.status(500).json({ error: 'Failed to list reports' });
    }
});

app.get('/api/backtest-reports/:id', (req, res) => {
    try {
        const filePath = path.join(REPORTS_DIR, `report_${req.params.id}.json`);
        if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'Report not found' });
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        res.json(data);
    } catch (e) {
        res.status(500).json({ error: 'Failed to read report' });
    }
});

app.delete('/api/backtest-reports/:id', (req, res) => {
    try {
        const filePath = path.join(REPORTS_DIR, `report_${req.params.id}.json`);
        if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'Report not found' });
        fs.unlinkSync(filePath);
        res.json({ success: true });
    } catch (e) {
        res.status(500).json({ error: 'Failed to delete report' });
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
//  DEPLOY CONTROL API
// ═══════════════════════════════════════════════════════════════════════════════

const ENV_FILE = path.join(__dirname, '..', '.env');
const { execSync } = require('child_process');

function readEnv() {
    const env = {};
    try {
        const content = fs.readFileSync(ENV_FILE, 'utf8');
        content.split('\n').forEach(line => {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith('#')) return;
            const [key, ...rest] = trimmed.split('=');
            env[key.trim()] = rest.join('=').trim();
        });
    } catch (e) { /* no .env file — fall through to process.env */ }
    // Fall back to process.env for any missing keys (Railway sets env vars directly)
    const importantKeys = ['PAPER_TRADE', 'LIVE_TRADE', 'TESTNET', 'COINDCX_API_KEY', 'COINDCX_API_SECRET',
        'BINANCE_API_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'TELEGRAM_ENABLED',
        'TELEGRAM_NOTIFY_TRADES', 'TELEGRAM_NOTIFY_ALERTS', 'TELEGRAM_NOTIFY_SUMMARY'];
    for (const key of importantKeys) {
        if (!env[key] && process.env[key]) env[key] = process.env[key];
    }
    return env;
}

function writeEnv(updates) {
    // Always update process.env so spawned child processes inherit the change
    for (const [key, value] of Object.entries(updates)) {
        process.env[key] = value;
    }
    // Also persist to .env file (works locally; on Railway this is in-container only)
    try {
        let content = fs.existsSync(ENV_FILE) ? fs.readFileSync(ENV_FILE, 'utf8') : '';
        for (const [key, value] of Object.entries(updates)) {
            const regex = new RegExp(`^${key}=.*$`, 'm');
            if (regex.test(content)) {
                content = content.replace(regex, `${key}=${value}`);
            } else {
                content += `\n${key}=${value}`;
            }
        }
        fs.writeFileSync(ENV_FILE, content);
    } catch (e) { console.error('Failed to write .env:', e); }
}

// GET /api/deploy/status
app.get('/api/deploy/status', (req, res) => {
    const env = readEnv();
    const paperEnabled = (env.PAPER_TRADE || 'true').toLowerCase() === 'true';
    const liveEnabled = (env.LIVE_TRADE || 'false').toLowerCase() === 'true';
    const isTestnet = (env.TESTNET || 'true').toLowerCase() === 'true';
    const hasKey = !!(env.COINDCX_API_KEY && env.COINDCX_API_KEY.length > 10);

    // Count active trades
    let paperActive = 0, liveActive = 0;
    const paperBook = readJSON('tradebook.json');
    const liveBook = readJSON('tradebook_live.json');
    if (paperBook?.trades) paperActive = paperBook.trades.filter(t => t.status === 'ACTIVE').length;
    if (liveBook?.trades) liveActive = liveBook.trades.filter(t => t.status === 'ACTIVE').length;

    res.json({
        paper_enabled: paperEnabled,
        live_enabled: liveEnabled,
        testnet: isTestnet,
        api_key_configured: hasKey,
        balance: 0,
        paper_active: paperActive,
        live_active: liveActive,
    });
});

// POST /api/deploy/mode
app.post('/api/deploy/mode', (req, res) => {
    const { paper, live } = req.body;
    const updates = {};
    if (typeof paper === 'boolean') updates.PAPER_TRADE = paper ? 'true' : 'false';
    if (typeof live === 'boolean') updates.LIVE_TRADE = live ? 'true' : 'false';
    if (Object.keys(updates).length === 0) {
        return res.status(400).json({ error: 'At least one of paper or live must be specified' });
    }
    writeEnv(updates);
    res.json({ success: true, paper, live });
});

// POST /api/deploy/test-connection
app.post('/api/deploy/test-connection', (req, res) => {
    try {
        const pythonScript = `
import sys, json
sys.path.insert(0, '${path.join(__dirname, '..').replace(/'/g, "\\'")}') 
import config
import coindcx_client as cdx
try:
    balance = cdx.get_usdt_balance()
    instruments = cdx.get_active_instruments()
    price = cdx.get_current_price('B-BTC_USDT')
    print(json.dumps({"success": True, "server_time": "CoinDCX API OK", "balance": balance, "instruments": len(instruments), "btc_price": price}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
`;
        const result = execSync(`python3 -c '${pythonScript.replace(/'/g, "'\"'\"'")}'`, {
            timeout: 15000, encoding: 'utf8',
            cwd: path.join(__dirname, '..'),
        });
        const data = JSON.parse(result.trim());
        res.json(data);
    } catch (e) {
        res.json({ success: false, error: e.message || 'Connection test failed' });
    }
});

// GET /api/deploy/balance
app.get('/api/deploy/balance', (req, res) => {
    try {
        const pythonScript = `
import sys, json
sys.path.insert(0, '${path.join(__dirname, '..').replace(/'/g, "\\'")}') 
import config
import coindcx_client as cdx
try:
    balance = cdx.get_usdt_balance()
    positions = cdx.list_positions()
    open_pos = len([p for p in positions if float(p.get('active_pos', 0)) != 0])
    print(json.dumps({"success": True, "balance": balance, "open_positions": open_pos}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
`;
        const result = execSync(`python3 -c '${pythonScript.replace(/'/g, "'\"'\"'")}'`, {
            timeout: 15000, encoding: 'utf8',
            cwd: path.join(__dirname, '..'),
        });
        const data = JSON.parse(result.trim());
        res.json(data);
    } catch (e) {
        res.json({ success: false, error: e.message || 'Failed to get balance' });
    }
});

// POST /api/deploy/kill-switch
app.post('/api/deploy/kill-switch', (req, res) => {
    try {
        const pythonScript = `
import sys, json
sys.path.insert(0, '${path.join(__dirname, '..').replace(/'/g, "\\'")}') 
import config
import coindcx_client as cdx
try:
    positions = cdx.list_positions()
    closed = 0
    for pos in positions:
        active = float(pos.get('active_pos', 0))
        if active == 0:
            continue
        cdx.exit_position(pos['id'])
        closed += 1
    cdx.cancel_all_open_orders()
    print(json.dumps({"success": True, "message": f"Closed {closed} position(s) on CoinDCX"}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
`;
        const result = execSync(`python3 -c '${pythonScript.replace(/'/g, "'\"'\"'")}'`, {
            timeout: 30000, encoding: 'utf8',
            cwd: path.join(__dirname, '..'),
        });
        const data = JSON.parse(result.trim());
        res.json(data);
    } catch (e) {
        res.json({ success: false, error: e.message || 'Kill switch failed' });
    }
});

// ─── Engine Process Manager ─────────────────────────────────────────────────
const ENGINE_STATE_FILE = path.join(DATA_DIR, 'engine_state.json');
let botProcess = null;  // child_process reference

function getEngineState() {
    // If we have a live process, report running; otherwise check if it crashed
    if (botProcess && !botProcess.killed) {
        return { status: 'running', pid: botProcess.pid, started_at: botProcess._startedAt || null };
    }
    try {
        if (fs.existsSync(ENGINE_STATE_FILE)) {
            return JSON.parse(fs.readFileSync(ENGINE_STATE_FILE, 'utf8'));
        }
    } catch (e) { }
    return { status: 'stopped', pid: null };
}

function setEngineState(state) {
    fs.writeFileSync(ENGINE_STATE_FILE, JSON.stringify(state, null, 2));
    io.emit('engine_state', state);
}

function startEngine() {
    if (botProcess && !botProcess.killed) {
        return { success: false, error: 'Engine is already running', state: getEngineState() };
    }

    const projectRoot = path.join(__dirname, '..');
    const venvPython = path.join(projectRoot, '.venv', 'bin', 'python3');
    const pythonPath = fs.existsSync(venvPython) ? venvPython : 'python3';
    const scriptPath = path.join(projectRoot, 'main.py');

    console.log(`[Engine] 🚀 Starting bot engine: ${pythonPath} ${scriptPath}`);

    botProcess = require('child_process').spawn(pythonPath, ['-u', scriptPath], {
        cwd: projectRoot,
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
        stdio: ['ignore', 'pipe', 'pipe'],
    });

    botProcess._startedAt = new Date().toISOString();

    // Pipe stdout/stderr to bot.log for the file watcher to pick up
    const logFile = path.join(DATA_DIR, 'bot.log');
    const logStream = fs.createWriteStream(logFile, { flags: 'a' });

    botProcess.stdout.on('data', (data) => {
        logStream.write(data);
        // Also emit to connected clients for real-time logs
        const lines = data.toString().split('\n').filter(l => l.trim());
        lines.forEach(l => io.emit('log-line', l));
    });

    botProcess.stderr.on('data', (data) => {
        logStream.write(data);
        const lines = data.toString().split('\n').filter(l => l.trim());
        lines.forEach(l => io.emit('log-line', `[STDERR] ${l}`));
    });

    botProcess.on('close', (code) => {
        console.log(`[Engine] ⚠️ Bot process exited with code ${code}`);
        const state = {
            status: 'stopped',
            pid: null,
            stopped_at: new Date().toISOString(),
            exit_code: code,
            reason: code === 0 ? 'normal' : 'crashed',
        };
        setEngineState(state);
        botProcess = null;
        logStream.end();
    });

    const state = {
        status: 'running',
        pid: botProcess.pid,
        started_at: botProcess._startedAt,
    };
    setEngineState(state);
    return { success: true, state };
}

function stopEngine() {
    if (!botProcess || botProcess.killed) {
        return { success: false, error: 'Engine is not running', state: getEngineState() };
    }

    console.log(`[Engine] ⏹ Stopping bot engine (PID ${botProcess.pid})...`);
    botProcess.kill('SIGTERM');

    // Force kill after 5 seconds if it doesn't stop
    setTimeout(() => {
        if (botProcess && !botProcess.killed) {
            console.log('[Engine] ⚠️ Force killing bot process...');
            botProcess.kill('SIGKILL');
        }
    }, 5000);

    const state = {
        status: 'stopped',
        pid: null,
        stopped_at: new Date().toISOString(),
        reason: 'user',
    };
    setEngineState(state);
    return { success: true, state };
}

app.get('/api/engine/state', (req, res) => {
    res.json(getEngineState());
});

app.post('/api/engine/toggle', (req, res) => {
    const current = getEngineState();
    if (current.status === 'running') {
        const result = stopEngine();
        console.log('[Engine] 🔄 Engine STOPPED via dashboard');
        res.json(result);
    } else {
        const result = startEngine();
        console.log('[Engine] 🔄 Engine STARTED via dashboard');
        res.json(result);
    }
});

// ─── Config API (reads config.py for settings page) ──────────────────────────
app.get('/api/config', (req, res) => {
    try {
        const configPath = path.join(__dirname, '..', 'config.py');
        const content = fs.readFileSync(configPath, 'utf8');

        const extract = (name, fallback = '') => {
            // Match: NAME = value  or  NAME = "value"
            const m = content.match(new RegExp(`^${name}\\s*=\\s*(.+)`, 'm'));
            if (!m) return fallback;
            let val = m[1].split('#')[0].trim(); // strip inline comments
            // Remove quotes
            if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
                val = val.slice(1, -1);
            }
            return val;
        };

        const num = (name, fallback = 0) => {
            const v = extract(name);
            const n = parseFloat(v);
            return isNaN(n) ? fallback : n;
        };

        const bool = (name, fallback = false) => {
            const v = extract(name).toLowerCase();
            if (v === 'true') return true;
            if (v === 'false') return false;
            return fallback;
        };

        const config = {
            // Trading
            PAPER_MAX_CAPITAL: num('PAPER_MAX_CAPITAL', 1500),
            PRIMARY_SYMBOL: extract('PRIMARY_SYMBOL', 'BTCUSDT'),

            // Timeframes
            TIMEFRAME_EXECUTION: extract('TIMEFRAME_EXECUTION', '5m'),
            TIMEFRAME_CONFIRMATION: extract('TIMEFRAME_CONFIRMATION', '1h'),
            TIMEFRAME_MACRO: extract('TIMEFRAME_MACRO', '4h'),

            // HMM
            HMM_N_STATES: num('HMM_N_STATES', 4),
            HMM_COVARIANCE: extract('HMM_COVARIANCE', 'diag'),
            HMM_ITERATIONS: num('HMM_ITERATIONS', 100),
            HMM_LOOKBACK: num('HMM_LOOKBACK', 500),
            HMM_RETRAIN_HOURS: num('HMM_RETRAIN_HOURS', 24),

            // Leverage
            LEVERAGE_HIGH: num('LEVERAGE_HIGH', 50),
            LEVERAGE_MODERATE: num('LEVERAGE_MODERATE', 25),
            LEVERAGE_LOW: num('LEVERAGE_LOW', 15),
            LEVERAGE_NONE: num('LEVERAGE_NONE', 1),

            // Confidence
            CONFIDENCE_HIGH: num('CONFIDENCE_HIGH', 0.99),
            CONFIDENCE_MEDIUM: num('CONFIDENCE_MEDIUM', 0.96),
            CONFIDENCE_LOW: num('CONFIDENCE_LOW', 0.92),

            // Risk
            RISK_PER_TRADE: num('RISK_PER_TRADE', 0.02),
            KILL_SWITCH_DRAWDOWN: num('KILL_SWITCH_DRAWDOWN', 0.10),
            MAX_LOSS_PER_TRADE_PCT: num('MAX_LOSS_PER_TRADE_PCT', -30),
            MIN_HOLD_MINUTES: num('MIN_HOLD_MINUTES', 30),
            DEFAULT_QUANTITY: num('DEFAULT_QUANTITY', 0.002),
            MARGIN_TYPE: extract('MARGIN_TYPE', 'ISOLATED'),

            // SL/TP
            ATR_SL_MULTIPLIER: num('ATR_SL_MULTIPLIER', 1.5),
            ATR_TP_MULTIPLIER: num('ATR_TP_MULTIPLIER', 3.0),
            SLIPPAGE_BUFFER: num('SLIPPAGE_BUFFER', 0.0005),

            // Trailing
            TRAILING_SL_ENABLED: bool('TRAILING_SL_ENABLED', true),
            TRAILING_SL_ACTIVATION_ATR: num('TRAILING_SL_ACTIVATION_ATR', 1.0),
            TRAILING_SL_DISTANCE_ATR: num('TRAILING_SL_DISTANCE_ATR', 1.0),
            TRAILING_TP_ENABLED: bool('TRAILING_TP_ENABLED', true),
            TRAILING_TP_ACTIVATION_PCT: num('TRAILING_TP_ACTIVATION_PCT', 0.75),
            TRAILING_TP_EXTENSION_ATR: num('TRAILING_TP_EXTENSION_ATR', 1.5),
            TRAILING_TP_MAX_EXTENSIONS: num('TRAILING_TP_MAX_EXTENSIONS', 3),

            // Volatility
            VOL_FILTER_ENABLED: bool('VOL_FILTER_ENABLED', true),
            VOL_MIN_ATR_PCT: num('VOL_MIN_ATR_PCT', 0.005),
            VOL_MAX_ATR_PCT: num('VOL_MAX_ATR_PCT', 0.04),

            // Fees
            TAKER_FEE: num('TAKER_FEE', 0.0005),
            MAKER_FEE: num('MAKER_FEE', 0.0002),

            // Sideways
            BB_LENGTH: num('BB_LENGTH', 20),
            BB_STD: num('BB_STD', 2.0),
            RSI_LENGTH: num('RSI_LENGTH', 14),
            RSI_OVERSOLD: num('RSI_OVERSOLD', 35),
            RSI_OVERBOUGHT: num('RSI_OVERBOUGHT', 65),
            SIDEWAYS_POSITION_REDUCTION: num('SIDEWAYS_POSITION_REDUCTION', 0.30),

            // Bot Loop
            LOOP_INTERVAL_SECONDS: num('LOOP_INTERVAL_SECONDS', 60),
            ANALYSIS_INTERVAL_SECONDS: num('ANALYSIS_INTERVAL_SECONDS', 900),
            ERROR_RETRY_SECONDS: num('ERROR_RETRY_SECONDS', 60),

            // Multi-Coin
            MAX_CONCURRENT_POSITIONS: num('MAX_CONCURRENT_POSITIONS', 15),
            TOP_COINS_LIMIT: num('TOP_COINS_LIMIT', 50),
            CAPITAL_PER_COIN_PCT: num('CAPITAL_PER_COIN_PCT', 0.03),
            SCAN_INTERVAL_CYCLES: num('SCAN_INTERVAL_CYCLES', 4),
            MULTI_COIN_MODE: bool('MULTI_COIN_MODE', true),
        };

        res.json(config);
    } catch (e) {
        res.status(500).json({ error: 'Failed to read config: ' + e.message });
    }
});


// ─── Telegram API ────────────────────────────────────────────────────────────

app.get('/api/telegram/status', (req, res) => {
    try {
        const envPath = path.join(__dirname, '..', '.env');
        const envContent = fs.readFileSync(envPath, 'utf8');
        const getEnv = (key, fallback = '') => {
            const m = envContent.match(new RegExp(`^${key}=(.*)`, 'm'));
            return m ? m[1].trim() : fallback;
        };
        res.json({
            enabled: getEnv('TELEGRAM_ENABLED', 'false') === 'true',
            bot_token: getEnv('TELEGRAM_BOT_TOKEN') ? '••••' + getEnv('TELEGRAM_BOT_TOKEN').slice(-6) : '',
            chat_id: getEnv('TELEGRAM_CHAT_ID'),
            notify_trades: getEnv('TELEGRAM_NOTIFY_TRADES', 'true') === 'true',
            notify_alerts: getEnv('TELEGRAM_NOTIFY_ALERTS', 'true') === 'true',
            notify_summary: getEnv('TELEGRAM_NOTIFY_SUMMARY', 'true') === 'true',
            has_token: !!getEnv('TELEGRAM_BOT_TOKEN'),
        });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.post('/api/telegram/config', (req, res) => {
    try {
        const { chat_id, enabled, notify_trades, notify_alerts, notify_summary } = req.body;
        const envPath = path.join(__dirname, '..', '.env');
        let content = fs.readFileSync(envPath, 'utf8');

        const setEnv = (key, val) => {
            const regex = new RegExp(`^${key}=.*`, 'm');
            if (regex.test(content)) {
                content = content.replace(regex, `${key}=${val}`);
            } else {
                content += `\n${key}=${val}`;
            }
        };

        if (chat_id !== undefined) setEnv('TELEGRAM_CHAT_ID', chat_id);
        if (enabled !== undefined) setEnv('TELEGRAM_ENABLED', enabled ? 'true' : 'false');
        if (notify_trades !== undefined) setEnv('TELEGRAM_NOTIFY_TRADES', notify_trades ? 'true' : 'false');
        if (notify_alerts !== undefined) setEnv('TELEGRAM_NOTIFY_ALERTS', notify_alerts ? 'true' : 'false');
        if (notify_summary !== undefined) setEnv('TELEGRAM_NOTIFY_SUMMARY', notify_summary ? 'true' : 'false');

        fs.writeFileSync(envPath, content);
        res.json({ success: true });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.post('/api/telegram/test', async (req, res) => {
    try {
        const envPath = path.join(__dirname, '..', '.env');
        const envContent = fs.readFileSync(envPath, 'utf8');
        const getEnv = (key) => {
            const m = envContent.match(new RegExp(`^${key}=(.*)`, 'm'));
            return m ? m[1].trim() : '';
        };
        const token = getEnv('TELEGRAM_BOT_TOKEN');
        const chatId = req.body.chat_id || getEnv('TELEGRAM_CHAT_ID');

        if (!token) return res.status(400).json({ error: 'Bot token not configured' });
        if (!chatId) return res.status(400).json({ error: 'Chat ID not set' });

        const https = require('https');
        const data = JSON.stringify({
            chat_id: chatId,
            text: '✅ <b>SENTINEL Bot Connected!</b>\n\nYour trading bot notifications are now active.\n\n📊 Trade alerts • 🛑 Risk warnings • 📈 Daily summaries',
            parse_mode: 'HTML',
        });

        const url = new URL(`https://api.telegram.org/bot${token}/sendMessage`);
        const result = await new Promise((resolve, reject) => {
            const req = https.request(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) } }, (resp) => {
                let body = '';
                resp.on('data', d => body += d);
                resp.on('end', () => resolve(JSON.parse(body)));
            });
            req.on('error', reject);
            req.write(data);
            req.end();
        });

        if (result.ok) {
            res.json({ success: true, message: 'Test message sent!' });
        } else {
            res.status(400).json({ error: result.description || 'Failed to send' });
        }
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.get('/api/telegram/detect-chat-id', async (req, res) => {
    try {
        const envPath = path.join(__dirname, '..', '.env');
        const envContent = fs.readFileSync(envPath, 'utf8');
        const m = envContent.match(/^TELEGRAM_BOT_TOKEN=(.*)$/m);
        const token = m ? m[1].trim() : '';

        if (!token) return res.status(400).json({ error: 'Bot token not configured' });

        const https = require('https');
        const url = new URL(`https://api.telegram.org/bot${token}/getUpdates`);
        const result = await new Promise((resolve, reject) => {
            https.get(url, (resp) => {
                let body = '';
                resp.on('data', d => body += d);
                resp.on('end', () => resolve(JSON.parse(body)));
            }).on('error', reject);
        });

        if (result.ok && result.result.length > 0) {
            const chatIds = [...new Set(result.result.map(u => {
                const msg = u.message || u.channel_post;
                return msg ? { id: msg.chat.id, type: msg.chat.type, name: msg.chat.first_name || msg.chat.title || '' } : null;
            }).filter(Boolean))];
            res.json({ success: true, chats: chatIds });
        } else {
            res.json({ success: false, chats: [], message: 'No messages found. Send /start to your bot first.' });
        }
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

// ─── Config Preset API ───────────────────────────────────────────────────────

app.post('/api/config/preset', (req, res) => {
    try {
        const { preset } = req.body;
        const configPath = path.join(__dirname, '..', 'config.py');
        let content = fs.readFileSync(configPath, 'utf8');

        const setVal = (name, val) => {
            const regex = new RegExp(`^(${name}\\s*=\\s*).*`, 'm');
            if (regex.test(content)) {
                const fmtVal = typeof val === 'string' ? `"${val}"` : typeof val === 'boolean' ? (val ? 'True' : 'False') : val;
                content = content.replace(regex, `$1${fmtVal}`);
            }
        };

        const presets = {
            conservative: {
                RISK_PER_TRADE: 0.01, LEVERAGE_HIGH: 20, LEVERAGE_MODERATE: 15, LEVERAGE_LOW: 10,
                MAX_LOSS_PER_TRADE_PCT: -20, CONFIDENCE_HIGH: 0.99, CONFIDENCE_MEDIUM: 0.97, CONFIDENCE_LOW: 0.95,
                VOL_MIN_ATR_PCT: 0.008, VOL_MAX_ATR_PCT: 0.03, TRAILING_SL_ENABLED: true, TRAILING_TP_ENABLED: true,
            },
            balanced: {
                RISK_PER_TRADE: 0.02, LEVERAGE_HIGH: 35, LEVERAGE_MODERATE: 20, LEVERAGE_LOW: 10,
                MAX_LOSS_PER_TRADE_PCT: -30, CONFIDENCE_HIGH: 0.99, CONFIDENCE_MEDIUM: 0.96, CONFIDENCE_LOW: 0.92,
                VOL_MIN_ATR_PCT: 0.005, VOL_MAX_ATR_PCT: 0.04, TRAILING_SL_ENABLED: true, TRAILING_TP_ENABLED: true,
            },
            aggressive: {
                RISK_PER_TRADE: 0.03, LEVERAGE_HIGH: 35, LEVERAGE_MODERATE: 25, LEVERAGE_LOW: 15,
                MAX_LOSS_PER_TRADE_PCT: -30, CONFIDENCE_HIGH: 0.99, CONFIDENCE_MEDIUM: 0.96, CONFIDENCE_LOW: 0.92,
                VOL_MIN_ATR_PCT: 0.003, VOL_MAX_ATR_PCT: 0.06, TRAILING_SL_ENABLED: true, TRAILING_TP_ENABLED: true,
            },
        };

        const values = presets[preset];
        if (!values) return res.status(400).json({ error: 'Unknown preset: ' + preset });

        for (const [key, val] of Object.entries(values)) {
            setVal(key, val);
        }

        fs.writeFileSync(configPath, content);
        res.json({ success: true, preset, applied: Object.keys(values) });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});


app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/tradebook', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'tradebook.html'));
});

app.get('/charts', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'charts.html'));
});

app.get('/backtest', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'backtest.html'));
});

app.get('/deploy', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'deploy.html'));
});

app.get('/settings', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'settings.html'));
});

// ─── Start server ────────────────────────────────────────────────────────────
server.listen(PORT, () => {
    console.log(`\n  🤖 Regime-Master Dashboard API running on http://localhost:${PORT}`);
    console.log(`  📂 Watching data directory: ${DATA_DIR}`);
    console.log(`  🔌 WebSocket ready for real-time updates`);
    console.log(`  ⚡ Live price ticker: 10-second updates\n`);
});

