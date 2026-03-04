import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import * as fs from 'fs';
import * as path from 'path';

export const dynamic = 'force-dynamic';

// Sentinelbot reads from its own data/ folder
const DATA_DIR = path.resolve(process.cwd(), '..', '..', 'data');

function readJSON(filename: string, fallback: any = {}) {
    try {
        const filepath = path.join(DATA_DIR, filename);
        if (fs.existsSync(filepath)) {
            return JSON.parse(fs.readFileSync(filepath, 'utf-8'));
        }
    } catch { /* silent */ }
    return fallback;
}

/**
 * Tradebook API — reads from local data/tradebook.json
 * GET /api/trades?status=active&coin=BTC&page=1&limit=50
 */
export async function GET(request: NextRequest) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const { searchParams } = new URL(request.url);
        const statusFilter = searchParams.get('status');
        const coinFilter = searchParams.get('coin');
        const page = parseInt(searchParams.get('page') || '1');
        const limit = Math.min(parseInt(searchParams.get('limit') || '50'), 100);

        const tradebook = readJSON('tradebook.json', { trades: [] });
        let trades: any[] = tradebook.trades || [];

        // Apply filters
        if (statusFilter) {
            trades = trades.filter((t: any) => (t.status || '').toUpperCase() === statusFilter.toUpperCase());
        }
        if (coinFilter) {
            const cf = coinFilter.toUpperCase();
            trades = trades.filter((t: any) => (t.symbol || t.coin || '').toUpperCase().includes(cf));
        }

        // Sort by entry time descending
        trades.sort((a: any, b: any) => {
            const ta = a.entry_time || a.entryTime || a.timestamp || '';
            const tb = b.entry_time || b.entryTime || b.timestamp || '';
            return tb.localeCompare(ta);
        });

        const total = trades.length;
        const skip = (page - 1) * limit;
        const paged = trades.slice(skip, skip + limit);

        return NextResponse.json({
            trades: paged.map((t: any) => ({
                id: t.trade_id || t.id || `T-${Math.random().toString(36).slice(2, 8)}`,
                coin: (t.symbol || t.coin || '').replace('USDT', ''),
                symbol: t.symbol || t.coin || '',
                position: (t.side || t.position || '').toLowerCase(),
                side: t.side || t.position || '',
                regime: t.regime || '',
                confidence: t.confidence || 0,
                leverage: t.leverage || 1,
                capital: t.capital || t.position_size || 0,
                entryPrice: t.entry_price || t.entryPrice || 0,
                currentPrice: t.current_price || t.currentPrice || null,
                exitPrice: t.exit_price || t.exitPrice || null,
                stopLoss: t.stop_loss || t.stopLoss || 0,
                takeProfit: t.take_profit || t.takeProfit || 0,
                slType: t.sl_type || t.slType || 'ATR',
                status: (t.status || '').toLowerCase(),
                activePnl: t.unrealized_pnl || t.active_pnl || t.activePnl || 0,
                activePnlPercent: t.unrealized_pnl_pct || t.activePnlPercent || 0,
                totalPnl: t.pnl || t.total_pnl || t.totalPnl || 0,
                totalPnlPercent: t.pnl_pct || t.totalPnlPercent || 0,
                exitPercent: t.exit_percent || null,
                entryTime: t.entry_time || t.entryTime || t.timestamp || new Date().toISOString(),
                exitTime: t.exit_time || t.exitTime || null,
                botName: 'Sentinel Marshal',
                exchange: t.exchange || 'binance_testnet',
                mode: t.mode || 'paper',
            })),
            pagination: {
                page, limit, total,
                totalPages: Math.ceil(total / limit),
                hasMore: skip + limit < total,
            },
        });
    } catch (error: any) {
        console.error('Tradebook GET error:', error);
        return NextResponse.json({ error: 'Failed to fetch trades', detail: String(error) }, { status: 500 });
    }
}

export async function DELETE(request: Request) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }
        // For now, trades from JSON are read-only
        return NextResponse.json({ error: 'Trade deletion from engine tradebook not supported yet' }, { status: 400 });
    } catch (error: any) {
        console.error('Tradebook DELETE error:', error);
        return NextResponse.json({ error: 'Failed to delete trade' }, { status: 500 });
    }
}
