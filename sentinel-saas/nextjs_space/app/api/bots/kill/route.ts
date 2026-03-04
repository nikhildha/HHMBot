import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';

export const dynamic = 'force-dynamic';

const ORCHESTRATOR_URL = process.env.ORCHESTRATOR_URL || 'http://localhost:5000';

/**
 * 8.8 — Bot Kill Switch
 * POST /api/bots/kill  { botId }
 * Stops bot via orchestrator + force-closes all active trades
 */

export async function POST(request: Request) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const { botId } = await request.json();
        if (!botId) {
            return NextResponse.json({ error: 'botId required' }, { status: 400 });
        }

        // Verify ownership
        const bot = await prisma.bot.findFirst({
            where: { id: botId, userId: session.user.id },
        });
        if (!bot) {
            return NextResponse.json({ error: 'Bot not found' }, { status: 404 });
        }

        // 1. Stop bot via orchestrator (best effort)
        try {
            await fetch(`${ORCHESTRATOR_URL}/bots/${botId}/stop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: AbortSignal.timeout(5000),
            });
        } catch {
            // Orchestrator might be offline — continue with DB cleanup
        }

        // 2. Close all active trades for this bot
        const activeTrades = await prisma.trade.findMany({
            where: { botId, status: 'active' },
        });

        const closeTime = new Date();
        for (const trade of activeTrades) {
            await prisma.trade.update({
                where: { id: trade.id },
                data: {
                    status: 'closed',
                    exitReason: 'KILL_SWITCH',
                    exitPrice: trade.currentPrice || trade.entryPrice,
                    exitTime: closeTime,
                    totalPnl: trade.activePnl,
                    totalPnlPercent: trade.activePnlPercent,
                },
            });
        }

        // 3. Update bot status
        await prisma.bot.update({
            where: { id: botId },
            data: {
                status: 'stopped',
                isActive: false,
                stoppedAt: closeTime,
            },
        });

        // 4. Update bot state
        await prisma.botState.upsert({
            where: { botId },
            update: { engineStatus: 'killed', errorMessage: 'Kill switch activated' },
            create: { botId, engineStatus: 'killed', errorMessage: 'Kill switch activated' },
        });

        return NextResponse.json({
            success: true,
            message: `Bot stopped, ${activeTrades.length} trade(s) closed`,
            tradesClosed: activeTrades.length,
        });
    } catch (error: any) {
        console.error('Kill switch error:', error);
        return NextResponse.json({ error: 'Kill switch failed' }, { status: 500 });
    }
}
