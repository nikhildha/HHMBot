import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';

export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user || (session.user as any).role !== 'admin') {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 403 });
        }

        const { userId, tier } = await request.json();
        if (!userId || !['free', 'pro', 'ultra'].includes(tier)) {
            return NextResponse.json({ error: 'Invalid request' }, { status: 400 });
        }

        // Check if user exists
        const user = await prisma.user.findUnique({ where: { id: userId } });
        if (!user) {
            return NextResponse.json({ error: 'User not found' }, { status: 404 });
        }

        const tierLimits = { free: { bots: 1, scans: 5 }, pro: { bots: 3, scans: 20 }, ultra: { bots: 10, scans: 50 } };
        const limits = tierLimits[tier as keyof typeof tierLimits];

        // Upsert subscription
        await prisma.subscription.upsert({
            where: { userId },
            update: {
                tier,
                status: 'active',
                coinScans: limits.scans,
            },
            create: {
                userId,
                tier,
                status: 'active',
                coinScans: limits.scans,
            },
        });

        return NextResponse.json({ success: true, tier, userId });
    } catch (error: any) {
        console.error('Subscription change error:', error);
        return NextResponse.json({ error: 'Internal error' }, { status: 500 });
    }
}
