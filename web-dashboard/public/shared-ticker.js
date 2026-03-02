/**
 * Shared Ticker Tape Logic
 * Used across all pages (Dashboard, Intelligence, Backtest, etc.)
 */

// Global constants for Regime mapping (available to other scripts if loaded before them)
window.REGIME_MAP = {
    'BULLISH': { emoji: '🟢', class: 'bull', color: '#22C55E' },
    'BEARISH': { emoji: '🔴', class: 'bear', color: '#EF4444' },
    'SIDEWAYS/CHOP': { emoji: '🟡', class: 'chop', color: '#F59E0B' },
    'CRASH/PANIC': { emoji: '💀', class: 'crash', color: '#DC2626' },
    'WAITING': { emoji: '⏳', class: 'chop', color: '#F59E0B' },
    'SCANNING': { emoji: '🔍', class: 'chop', color: '#3B82F6' },
    'OFFLINE': { emoji: '⚫', class: 'chop', color: '#8E9BB3' },
};

window.getRegimeInfo = function(regimeName) {
    return window.REGIME_MAP[regimeName] || window.REGIME_MAP['OFFLINE'];
};

window.formatPrice = function(price) {
    const p = parseFloat(price);
    if (isNaN(p)) return '$0';
    if (p >= 1000) return '$' + p.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (p >= 1) return '$' + p.toFixed(4);
    return '$' + p.toFixed(6);
};

window.updateTicker = function(coins) {
    const track = document.getElementById('tickerTrack');
    if (!track) return; // Ticker element not found on this page

    const items = [...coins, ...coins]; // Duplicate for seamless loop
    
    // Safety check for empty data
    if (items.length === 0) {
        track.innerHTML = '<span class="ticker-item">Waiting for data...</span>';
        return;
    }

    track.innerHTML = items.map(coin => {
        const regime = coin.regime_name || 'SIDEWAYS/CHOP';
        const info = window.getRegimeInfo(regime);
        const symbol = (coin.symbol || '').replace('USDT', '');
        const badgeClass = `badge-${info.class}`;

        return `<span class="ticker-item">
      <span class="symbol">${symbol}</span>
      <span class="price">${window.formatPrice(coin.price)}</span>
      <span class="regime-badge ${badgeClass}">${regime.split('/')[0]}</span>
    </span>`;
    }).join('');
};

// Auto-initialize socket connection for ticker if not already handled
document.addEventListener('DOMContentLoaded', () => {
    // If socket is already defined (e.g. by app.js), reuse it, otherwise create new connection
    // Note: pages like tradebook.html already import socket.io
    
    // Check if we have the ticker container before doing anything
    if (!document.getElementById('tickerTrack')) {
        console.log('No tickerTrack found, skipping ticker update.');
        return;
    }

    // Connect if global socket not present
    let socket = window.socket;
    if (!socket && typeof io !== 'undefined') {
        socket = io();
        window.socket = socket; // Share it
    }

    if (socket) {
        socket.on('multi_state', (data) => {
            if (data && data.coin_states) {
                // Convert object to array if needed (backend sends dict sometimes?)
                // Usually coin_states is a dict { "BTCUSDT": {...}, ... }
                // We need array for map
                let coinsArray = [];
                if (Array.isArray(data.coin_states)) {
                    coinsArray = data.coin_states;
                } else if (typeof data.coin_states === 'object') {
                    coinsArray = Object.values(data.coin_states);
                }
                
                window.updateTicker(coinsArray);
            }
        });
        
        // Also listen for connect to request initial data if API exists?
        // Usually the server pushes 'multi_state' periodically.
    } else {
        console.warn('Socket.io not found. Ticker will not update.');
    }
});
