import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';

export const dynamic = 'force-dynamic';

/**
 * 8.3 — BotConfig CRUD
 * GET  /api/bots/config?botId=x  — Fetch config
 * PUT  /api/bots/config          — Update config
 */

export async function GET(request: NextRequest) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const botId = new URL(request.url).searchParams.get('botId');
        if (!botId) {
            return NextResponse.json({ error: 'botId required' }, { status: 400 });
        }

        // Verify ownership
        const bot = await prisma.bot.findFirst({
            where: { id: botId, userId: session.user.id },
            include: { config: true, state: true },
        });

        if (!bot) {
            return NextResponse.json({ error: 'Bot not found' }, { status: 404 });
        }

        return NextResponse.json({
            bot: {
                id: bot.id,
                name: bot.name,
                exchange: bot.exchange,
                status: bot.status,
                isActive: bot.isActive,
            },
            config: bot.config,
            state: bot.state ? {
                engineStatus: bot.state.engineStatus,
                lastCycleAt: bot.state.lastCycleAt?.toISOString() ?? null,
                cycleCount: bot.state.cycleCount,
                cycleDurationMs: bot.state.cycleDurationMs,
                errorMessage: bot.state.errorMessage,
            } : null,
        });
    } catch (error: any) {
        console.error('BotConfig GET error:', error);
        return NextResponse.json({ error: 'Failed to fetch config' }, { status: 500 });
    }
}

export async function PUT(request: Request) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const body = await request.json();
        const { botId, ...configData } = body;

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

        // Allowed fields
        const allowed: Record<string, any> = {};
        const fields = [
            'mode', 'capitalPerTrade', 'maxOpenTrades',
            'slMultiplier', 'tpMultiplier', 'maxLossPct',
            'multiTargetEnabled', 't1Multiplier', 't2Multiplier', 't3Multiplier',
            't1BookPct', 't2BookPct', 'coinList', 'leverageTiers',
        ];
        for (const f of fields) {
            if (configData[f] !== undefined) allowed[f] = configData[f];
        }

        const updated = await prisma.botConfig.upsert({
            where: { botId },
            update: allowed,
            create: { botId, ...allowed },
        });

        return NextResponse.json({ success: true, config: updated });
    } catch (error: any) {
        console.error('BotConfig PUT error:', error);
        return NextResponse.json({ error: 'Failed to update config' }, { status: 500 });
    }
}
