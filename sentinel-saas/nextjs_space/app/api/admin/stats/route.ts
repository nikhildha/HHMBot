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

        const [
            totalUsers,
            totalBots,
            activeBots,
            totalTrades,
            activeTrades,
            activeSubscriptions,
            trades,
        ] = await Promise.all([
            prisma.user.count(),
            prisma.bot.count(),
            prisma.bot.count({ where: { isActive: true } }),
            prisma.trade.count(),
            prisma.trade.count({ where: { status: 'active' } }),
            prisma.subscription.count({ where: { status: 'active' } }),
            prisma.trade.findMany({
                where: { status: 'closed' },
                select: { totalPnl: true },
            }),
        ]);

        const totalPnl = trades.reduce((sum, t) => sum + (t.totalPnl || 0), 0);

        // Revenue estimate: count pro ($999) and ultra ($2499) subscriptions
        const proCount = await prisma.subscription.count({ where: { tier: 'pro', status: 'active' } });
        const ultraCount = await prisma.subscription.count({ where: { tier: 'ultra', status: 'active' } });
        const revenueEstimate = proCount * 999 + ultraCount * 2499;

        return NextResponse.json({
            totalUsers,
            totalBots,
            activeBots,
            totalTrades,
            activeTrades,
            activeSubscriptions,
            totalPnl: Math.round(totalPnl * 100) / 100,
            revenueEstimate,
        });
    } catch (error: any) {
        console.error('Admin stats error:', error);
        return NextResponse.json({ error: 'Internal error' }, { status: 500 });
    }
}
