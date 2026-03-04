import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';
import { checkSubscription, TIER_LIMITS, Tier } from '@/lib/subscription';

export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // ─── Subscription & Trial Check ───────────────────────────
    const subStatus = await checkSubscription(session.user.id);
    if (!subStatus.isActive) {
      return NextResponse.json(
        { error: subStatus.message, expired: true },
        { status: 403 }
      );
    }

    const { name, exchange } = await request.json();

    if (!name || !exchange) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    // Check bot count limits for the user's tier
    const user = await prisma.user.findUnique({
      where: { id: session.user.id },
      include: { subscription: true, bots: true },
    });

    const limits = TIER_LIMITS[subStatus.tier];
    const maxBots = limits.maxBots;

    if (user && user.bots.length >= maxBots) {
      return NextResponse.json(
        { error: `Bot limit reached (${maxBots}). Upgrade your plan for more bots.` },
        { status: 403 }
      );
    }

    // Determine coin scan limit from subscription
    const coinScansLimit = user?.subscription?.coinScans || 5; // free trial = 5

    // Create bot with default config
    const bot = await prisma.bot.create({
      data: {
        userId: session.user.id,
        name,
        exchange,
        status: 'stopped',
        isActive: false,
        config: {
          create: {
            mode: 'paper',
            capitalPerTrade: 100,
            maxOpenTrades: 5,
            slMultiplier: 0.8,
            tpMultiplier: 1.0,
            maxLossPct: -15,
            multiTargetEnabled: true,
            t1Multiplier: 0.5,
            t2Multiplier: 1.0,
            t3Multiplier: 1.5,
            t1BookPct: 0.25,
            t2BookPct: 0.50,
            coinList: JSON.stringify([
              'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'DOGEUSDT',
            ].slice(0, coinScansLimit)),
          },
        },
        state: {
          create: {
            engineStatus: 'idle',
          },
        },
      },
      include: {
        config: true,
        state: true,
      },
    });

    return NextResponse.json({ success: true, bot });
  } catch (error: any) {
    console.error('Bot creation error:', error);
    return NextResponse.json({ error: 'Failed to create bot' }, { status: 500 });
  }
}