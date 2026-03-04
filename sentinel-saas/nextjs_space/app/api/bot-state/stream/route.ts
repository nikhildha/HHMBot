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

/**
 * SSE endpoint for real-time bot state
 * Reads from local data files every 3s and pushes via Server-Sent Events
 */
export async function GET() {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
        return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const encoder = new TextEncoder();
    let closed = false;

    const stream = new ReadableStream({
        async start(controller) {
            const send = (data: any) => {
                if (closed) return;
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
            };

            send({ type: 'connected', timestamp: new Date().toISOString() });

            const poll = async () => {
                if (closed) return;
                try {
                    const multi = readJSON('multi_bot_state.json', { coin_states: {} });
                    const tradebook = readJSON('tradebook.json', { trades: [] });
                    const coinStates = multi.coin_states || {};
                    const trades = tradebook.trades || [];

                    send({
                        type: 'state',
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
                            active_positions: {},
                            coin_states: coinStates,
                            cycle: multi.cycle || 0,
                        },
                        tradebook: { trades, summary: tradebook.stats || {} },
                        timestamp: new Date().toISOString(),
                    });
                } catch {
                    send({ type: 'error', message: 'Data read error', timestamp: new Date().toISOString() });
                }
            };

            await poll();
            const interval = setInterval(poll, 3000);

            setTimeout(() => {
                closed = true;
                clearInterval(interval);
                try { controller.close(); } catch { }
            }, 5 * 60 * 1000);
        },
        cancel() {
            closed = true;
        },
    });

    return new Response(stream, {
        headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache, no-transform',
            Connection: 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    });
}
