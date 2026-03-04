import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import * as fs from 'fs';
import * as path from 'path';

export const dynamic = 'force-dynamic';

// Sentinelbot reads directly from its own data/ folder
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

export async function GET() {
    try {
        // Get session to filter trades by user
        const session = await getServerSession(authOptions);
        const userId = (session?.user as any)?.id;
        const isAdmin = (session?.user as any)?.role === 'admin';

        const multi = readJSON('multi_bot_state.json', {
            coin_states: {},
            last_analysis_time: null,
            analysis_interval_seconds: 300,
            deployed_count: 0,
        });

        const tradebook = readJSON('tradebook.json', { trades: [], stats: {} });
        const engineState = readJSON('engine_state.json', { status: 'stopped' });

        // Build the response shape that the dashboard expects
        const coinStates = multi.coin_states || {};
        const allTrades = tradebook.trades || [];

        // Filter trades by user: admin sees all, regular users see only their trades
        // Safety: if no userId resolved, return empty trades (don't leak data)
        let trades: any[];
        if (isAdmin) {
            trades = allTrades;
        } else if (userId) {
            trades = allTrades.filter((t: any) => t.user_id === userId);
        } else {
            trades = []; // No session or userId → empty
        }

        const activeTrades = trades.filter((t: any) => (t.status || '').toUpperCase() === 'ACTIVE');

        return NextResponse.json({
            state: {
                regime: multi.macro_regime || 'WAITING',
                confidence: 0,
                symbol: 'BTCUSDT',
                timestamp: multi.last_analysis_time || null,
            },
            multi: {
                ...multi,
                coins_scanned: Object.keys(coinStates).length,
                eligible_count: Object.values(coinStates).filter((c: any) => (c.action || '').includes('ELIGIBLE')).length,
                deployed_count: multi.deployed_count || 0,
                total_trades: trades.length,
                active_positions: Object.fromEntries(
                    activeTrades.map((t: any) => [t.symbol, t])
                ),
                coin_states: coinStates,
                cycle: multi.cycle || 0,
                timestamp: multi.last_analysis_time || null,
            },
            scanner: { coins: Object.keys(coinStates) },
            tradebook: {
                trades,
                summary: tradebook.stats || {},
            },
            engine: engineState,
        });
    } catch (err) {
        return NextResponse.json({
            state: { regime: 'WAITING', confidence: 0, symbol: 'BTCUSDT', timestamp: null },
            multi: { coins_scanned: 0, eligible_count: 0, deployed_count: 0, total_trades: 0, active_positions: {}, coin_states: {}, cycle: 0, timestamp: null },
            scanner: { coins: [] },
            tradebook: { trades: [], summary: {} },
            error: String(err),
        });
    }
}

