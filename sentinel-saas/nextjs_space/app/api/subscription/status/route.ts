/**
 * Subscription Status API
 * GET /api/subscription/status
 * 
 * Returns the current user's subscription status, tier, and limits.
 * Used by the frontend to gate features and show upgrade prompts.
 */

import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import { checkSubscription, TIER_LIMITS, Tier } from '@/lib/subscription';

export const dynamic = 'force-dynamic';

export async function GET() {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const status = await checkSubscription(session.user.id);
        const limits = TIER_LIMITS[status.tier as Tier];

        return NextResponse.json({
            ...status,
            limits,
        });
    } catch (error: any) {
        console.error('Subscription status error:', error);
        return NextResponse.json({ error: 'Failed to check subscription' }, { status: 500 });
    }
}
