"""
Helper script: Close a single CoinDCX futures position by symbol.
Called by the SENTINEL dashboard server when a live trade is manually
closed from the UI or when the SL/TP engine fires on a live trade.

Usage:  python3 close_position.py BTCUSDT
Output: JSON to stdout — { success, symbol, pair, position_id, error? }
"""
import sys
import json
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import coindcx_client as cdx

symbol = sys.argv[1].strip() if len(sys.argv) > 1 else ''

if not symbol:
    print(json.dumps({'success': False, 'error': 'No symbol provided'}))
    sys.exit(1)

try:
    pair = cdx.to_coindcx_pair(symbol)
    positions = cdx.list_positions()

    closed_id = None
    for p in positions:
        if p.get('pair') == pair and float(p.get('active_pos', 0)) != 0:
            cdx.exit_position(p['id'])
            closed_id = p['id']
            break

    if closed_id:
        print(json.dumps({'success': True, 'symbol': symbol, 'pair': pair, 'position_id': closed_id}))
    else:
        # Position may already be closed on the exchange side — not an error
        print(json.dumps({'success': True, 'symbol': symbol, 'pair': pair,
                          'position_id': None, 'note': 'No active position found (may already be closed)'}))

except Exception as e:
    print(json.dumps({'success': False, 'symbol': symbol, 'error': str(e)}))
    sys.exit(1)
