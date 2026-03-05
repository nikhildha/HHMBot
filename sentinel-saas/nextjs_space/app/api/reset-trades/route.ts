import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import { clearUserTrades } from '@/lib/sync-engine-trades';
import { prisma } from '@/lib/prisma';

export const dynamic = 'force-dynamic';

export async function POST() {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const userId = (session.user as any).id;

        // Delete only THIS user's trades from Prisma — no other user affected
        const deletedCount = await clearUserTrades(userId);

        // Move bot.startedAt to NOW so re-sync won't re-import old engine trades
        await prisma.bot.updateMany({
            where: { userId },
            data: { startedAt: new Date() },
        });

        return NextResponse.json({
            success: true,
            message: `Cleared ${deletedCount} trades for your account`,
            deletedCount,
        });
    } catch (error) {
        console.error('[reset-trades] Error:', error);
        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
}
