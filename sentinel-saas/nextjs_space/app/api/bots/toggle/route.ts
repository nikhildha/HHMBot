import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';
import { checkSubscription } from '@/lib/subscription';

export const dynamic = 'force-dynamic';

const ORCHESTRATOR_URL = process.env.ORCHESTRATOR_URL || 'http://localhost:5000';

export async function POST(request: Request) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { botId, isActive } = await request.json();

    if (!botId) {
      return NextResponse.json({ error: 'botId required' }, { status: 400 });
    }

    // ─── Block starting bots if subscription expired (stopping is always OK) ──
    if (isActive) {
      const subStatus = await checkSubscription(session.user.id);
      if (!subStatus.isActive) {
        return NextResponse.json(
          { error: subStatus.message, expired: true },
          { status: 403 }
        );
      }
    }

    // Verify ownership
    const bot = await prisma.bot.findFirst({
      where: { id: botId, userId: session.user.id },
    });

    if (!bot) {
      return NextResponse.json({ error: 'Bot not found' }, { status: 404 });
    }

    // Call the Python orchestrator to start/stop the engine worker
    const orchEndpoint = isActive ? 'start' : 'stop';
    try {
      const orchResponse = await fetch(`${ORCHESTRATOR_URL}/api/bots/${orchEndpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ botId }),
      });

      if (!orchResponse.ok) {
        const err = await orchResponse.json().catch(() => ({}));
        console.error('Orchestrator error:', err);
        // Still update DB status even if orchestrator call fails
      }
    } catch (orchError) {
      console.error('Orchestrator unreachable:', orchError);
      // Don't fail — update DB status anyway (orchestrator might be starting up)
    }

    // Update bot status in database
    await prisma.bot.update({
      where: { id: botId },
      data: {
        isActive,
        status: isActive ? 'running' : 'stopped',
        ...(isActive ? { startedAt: new Date() } : { stoppedAt: new Date() }),
      },
    });

    return NextResponse.json({ success: true, isActive });
  } catch (error: any) {
    console.error('Bot toggle error:', error);
    return NextResponse.json({ error: 'Failed to toggle bot' }, { status: 500 });
  }
}