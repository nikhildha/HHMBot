import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';

export const dynamic = 'force-dynamic';

export async function GET() {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user || (session.user as any).role !== 'admin') {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 403 });
        }

        const bots = await prisma.bot.findMany({
            select: {
                id: true,
                name: true,
                isActive: true,
                createdAt: true,
                user: {
                    select: { name: true, email: true },
                },
            },
            orderBy: { createdAt: 'desc' },
        });

        // Get trade counts per bot
        const botsWithCounts = await Promise.all(
            bots.map(async (bot) => {
                const tradeCount = await prisma.trade.count({ where: { botId: bot.id } });
                return { ...bot, _count: { trades: tradeCount } };
            })
        );

        return NextResponse.json(botsWithCounts);
    } catch (error: any) {
        console.error('Admin bots error:', error);
        return NextResponse.json({ error: 'Internal error' }, { status: 500 });
    }
}
