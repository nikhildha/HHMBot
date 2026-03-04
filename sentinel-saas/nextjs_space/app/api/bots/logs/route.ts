import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';

export const dynamic = 'force-dynamic';

/**
 * 8.9 — Execution Log / Cycle History
 * GET /api/bots/logs?botId=x&limit=20
 * Returns recent engine cycle history + state info
 */

export async function GET(request: NextRequest) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const { searchParams } = new URL(request.url);
        const botId = searchParams.get('botId');
        const limit = Math.min(parseInt(searchParams.get('limit') || '20'), 50);

        if (!botId) {
            return NextResponse.json({ error: 'botId required' }, { status: 400 });
        }

        // Verify ownership
        const bot = await prisma.bot.findFirst({
            where: { id: botId, userId: session.user.id },
            include: { state: true },
        });

        if (!bot) {
            return NextResponse.json({ error: 'Bot not found' }, { status: 404 });
        }

        // Get recent trades as execution history proxy
        const recentTrades = await prisma.trade.findMany({
            where: { botId },
            orderBy: { createdAt: 'desc' },
            take: limit,
            select: {
                id: true,
                coin: true,
                position: true,
                regime: true,
                confidence: true,
                leverage: true,
                status: true,
                exitReason: true,
                activePnl: true,
                totalPnl: true,
                entryTime: true,
                exitTime: true,
                createdAt: true,
            },
        });

        return NextResponse.json({
            bot: {
                id: bot.id,
                name: bot.name,
                status: bot.status,
                isActive: bot.isActive,
                startedAt: bot.startedAt?.toISOString() ?? null,
                stoppedAt: bot.stoppedAt?.toISOString() ?? null,
            },
            engine: bot.state ? {
                engineStatus: bot.state.engineStatus,
                lastCycleAt: bot.state.lastCycleAt?.toISOString() ?? null,
                cycleCount: bot.state.cycleCount,
                cycleDurationMs: bot.state.cycleDurationMs,
                errorMessage: bot.state.errorMessage,
                errorAt: bot.state.errorAt?.toISOString() ?? null,
                coinStates: bot.state.coinStates,
            } : null,
            recentActivity: recentTrades.map(t => ({
                ...t,
                entryTime: t.entryTime.toISOString(),
                exitTime: t.exitTime?.toISOString() ?? null,
                createdAt: t.createdAt.toISOString(),
            })),
        });
    } catch (error: any) {
        console.error('Bot logs error:', error);
        return NextResponse.json({ error: 'Failed to fetch logs' }, { status: 500 });
    }
}
