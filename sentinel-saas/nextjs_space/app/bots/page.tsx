import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';
import { BotsClient } from './bots-client';

export const dynamic = 'force-dynamic';

export default async function BotsPage() {
  const session = await getServerSession(authOptions);

  if (!session?.user) {
    redirect('/login');
  }

  const bots = await prisma.bot.findMany({
    where: { userId: session.user.id },
    include: {
      _count: {
        select: { trades: true },
      },
    },
    orderBy: { createdAt: 'desc' },
  });

  return (
    <BotsClient
      bots={bots.map((bot) => ({
        id: bot.id,
        name: bot.name,
        exchange: bot.exchange,
        status: bot.status,
        isActive: bot?.isActive ?? false,
        startedAt: bot?.startedAt ?? null,
        _count: {
          trades: bot?._count?.trades ?? 0,
        },
      }))}
    />
  );
}