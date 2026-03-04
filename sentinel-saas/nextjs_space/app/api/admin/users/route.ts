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

        const users = await prisma.user.findMany({
            select: {
                id: true,
                email: true,
                name: true,
                role: true,
                createdAt: true,
                subscription: {
                    select: {
                        tier: true,
                        status: true,
                        coinScans: true,
                    },
                },
            },
            orderBy: { createdAt: 'desc' },
        });

        // Get bot and trade counts per user
        const usersWithCounts = await Promise.all(
            users.map(async (user) => {
                const botCount = await prisma.bot.count({ where: { userId: user.id } });
                const tradeCount = await prisma.trade.count({
                    where: { bot: { userId: user.id } },
                });
                return {
                    ...user,
                    _count: { bots: botCount, trades: tradeCount },
                };
            })
        );

        return NextResponse.json(usersWithCounts);
    } catch (error: any) {
        console.error('Admin users error:', error);
        return NextResponse.json({ error: 'Internal error' }, { status: 500 });
    }
}
