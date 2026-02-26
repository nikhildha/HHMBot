"""
Helper script for SENTINEL dashboard server.
Fetches live positions and wallet balance from CoinDCX.
Uses real-time ticker prices (not stale mark_price from positions API).
Output: JSON to stdout.
"""
import sys, json, os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import coindcx_client as cdx

try:
    positions = cdx.list_positions()
    wallet = cdx.get_usdt_balance()

    # Fetch REAL-TIME prices from CoinDCX ticker (positions API has stale mark_price)
    live_prices = cdx.get_current_prices()  # {pair: {ls: last_price, ...}}

    active = []
    for p in positions:
        qty = float(p.get('active_pos', 0))
        if qty == 0:
            continue
        pair = p.get('pair', '')
        symbol = cdx.from_coindcx_pair(pair)
        side = 'SHORT' if qty < 0 else 'LONG'
        entry_price = float(p.get('avg_price', 0))
        leverage = int(float(p.get('leverage', 1)))
        locked_margin = float(p.get('locked_margin', 0))
        liq_price = float(p.get('liquidation_price', 0))
        sl_trigger = float(p.get('stop_loss_trigger', 0))
        tp_trigger = float(p.get('take_profit_trigger', 0))
        pos_id = p.get('id', '')
        updated_at = p.get('updated_at', '')

        # Use LIVE ticker price, fallback to stale mark_price
        stale_mark = float(p.get('mark_price', 0))
        ticker_info = live_prices.get(pair, {})
        live_mark = float(ticker_info.get('ls', 0)) if ticker_info else 0
        mark_price = live_mark if live_mark > 0 else stale_mark

        # Compute unrealized PnL from entry price and live mark price
        abs_qty = abs(qty)
        if side == 'LONG':
            unrealized_pnl = (mark_price - entry_price) * abs_qty
        else:
            unrealized_pnl = (entry_price - mark_price) * abs_qty

        # PnL % based on locked margin (margin = capital deployed)
        pnl_pct = round(unrealized_pnl / locked_margin * 100, 2) if locked_margin > 0 else 0

        active.append({
            'symbol': symbol,
            'pair': pair,
            'side': side,
            'position': side,
            'quantity': abs_qty,
            'leverage': leverage,
            'entry_price': entry_price,
            'mark_price': round(mark_price, 6),
            'current_price': round(mark_price, 6),
            'locked_margin': round(locked_margin, 4),
            'pnl': round(unrealized_pnl, 4),
            'pnl_pct': pnl_pct,
            'liquidation_price': liq_price,
            'stop_loss': sl_trigger,
            'take_profit': tp_trigger,
            'position_id': pos_id,
            'updated_at': updated_at,
            'status': 'ACTIVE',
            'mode': 'LIVE',
        })

    print(json.dumps({
        'success': True,
        'positions': active,
        'wallet_balance': wallet,
        'count': len(active)
    }))
except Exception as e:
    print(json.dumps({
        'success': False,
        'error': str(e),
        'positions': [],
        'wallet_balance': 0,
        'count': 0
    }))
