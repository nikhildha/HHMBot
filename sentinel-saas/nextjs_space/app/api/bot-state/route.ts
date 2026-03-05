import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import { prisma } from '@/lib/prisma';
import { syncEngineTrades, getUserTrades } from '@/lib/sync-engine-trades';
import * as fs from 'fs';
import * as path from 'path';

export const dynamic = 'force-dynamic';

// ─── Engine API URL (Railway internal) or local file fallback ────────────────
const ENGINE_API_URL = process.env.ENGINE_API_URL; // e.g. http://sentinelbot-engine.railway.internal:3001

// Sentinelbot reads directly from its own data/ folder (local dev)
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

async function fetchEngineData() {
    if (!ENGINE_API_URL) return null;
    try {
        const res = await fetch(`${ENGINE_API_URL}/api/all`, {
            cache: 'no-store',
            signal: AbortSignal.timeout(8000),
        });
        if (res.ok) return await res.json();
    } catch (err) {
        console.error('[bot-state] Engine API fetch failed:', err);
    }
    return null;
}

export async function GET() {
    try {
        // Get session to filter trades by user
        const session = await getServerSession(authOptions);
        const userId = (session?.user as any)?.id;
        const isAdmin = (session?.user as any)?.role === 'admin';

        // Try fetching from engine API first (production), fall back to local files
        const engineData = await fetchEngineData();

        let multi: any, engineTradebook: any, engineState: any;

        if (engineData) {
            // Production: data from engine Express API
            multi = engineData.multi || {};
            engineTradebook = engineData.tradebook || { trades: [], summary: {} };
            engineState = engineData.engine || { status: 'running' };
        } else {
            // Local dev: read from filesystem
            multi = readJSON('multi_bot_state.json', {
                coin_states: {},
                last_analysis_time: null,
                analysis_interval_seconds: 300,
                deployed_count: 0,
            });
            engineTradebook = readJSON('tradebook.json', { trades: [], stats: {} });
            engineState = readJSON('engine_state.json', { status: 'stopped' });
        }

        // Build the engine state part of the response (shared — not per-user)
        const coinStates = multi.coin_states || {};
        const engineTradesRaw = engineTradebook.trades || [];

        // ─── Per-User Trade Isolation ────────────────────────────────
        let trades: any[] = [];

        if (session && userId) {
            // Find user's active bot to sync engine trades against
            const userBot = await prisma.bot.findFirst({
                where: { userId },
                orderBy: { updatedAt: 'desc' },
            });

            // Only sync engine trades for ADMIN users.
            // Regular users only read their own Prisma trades — they never
            // inherit trades from the shared engine tradebook.
            if (isAdmin && userBot && engineTradesRaw.length > 0) {
                // Sync engine trades into Prisma for admin's bot
                // Only syncs trades whose entry_time >= bot.startedAt
                try {
                    await syncEngineTrades(
                        engineTradesRaw,
                        userBot.id,
                        userBot.startedAt || userBot.createdAt
                    );
                } catch (err) {
                    console.error('[bot-state] Trade sync failed:', err);
                }
            }

            // Read this user's trades from Prisma (isolated)
            try {
                trades = await getUserTrades(userId);
            } catch (err) {
                console.error('[bot-state] getUserTrades failed:', err);
                trades = [];
            }
        }

        const activeTrades = trades.filter((t: any) => (t.status || '').toUpperCase() === 'ACTIVE');

        return NextResponse.json({
            state: {
                regime: multi.macro_regime || coinStates?.BTCUSDT?.regime || 'WAITING',
                confidence: coinStates?.BTCUSDT?.confidence || 0,
                symbol: 'BTCUSDT',
                btc_price: coinStates?.BTCUSDT?.price || null,
                timestamp: multi.last_analysis_time || multi.timestamp || null,
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
                timestamp: multi.last_analysis_time || multi.timestamp || null,
            },
            scanner: { coins: Object.keys(coinStates) },
            tradebook: {
                trades,
                summary: engineTradebook.stats || engineTradebook.summary || {},
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
