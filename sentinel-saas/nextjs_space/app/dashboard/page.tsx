import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';
import { DashboardClient } from './dashboard-client';

export const dynamic = 'force-dynamic';

export default async function DashboardPage() {
  const session = await getServerSession(authOptions);

  if (!session?.user) {
    redirect('/login');
  }

  try {
    const user = await prisma.user.findUnique({
      where: { id: session.user.id },
      include: {
        subscription: true,
        bots: {
          include: {
            _count: {
              select: { trades: true },
            },
          },
        },
      },
    });

    if (!user) {
      redirect('/login');
    }

    const activeTrades = await prisma.trade.count({
      where: {
        bot: {
          userId: user.id,
        },
        status: 'active',
      },
    });

    const totalTrades = await prisma.trade.count({
      where: {
        bot: {
          userId: user.id,
        },
      },
    });

    const trades = await prisma.trade.findMany({
      where: {
        bot: {
          userId: user.id,
        },
      },
      orderBy: {
        entryTime: 'desc',
      },
      take: 10,
    });

    const totalPnl = trades.reduce((sum, trade) => sum + (trade?.totalPnl ?? 0), 0);
    const activePnl = trades
      .filter((t) => t?.status === 'active')
      .reduce((sum, trade) => sum + (trade?.activePnl ?? 0), 0);

    return (
      <DashboardClient
        user={{
          id: user.id,
          name: user?.name ?? '',
          email: user.email,
          subscription: user?.subscription ?? null,
        }}
        stats={{
          activeBots: user.bots.filter((b) => b?.isActive ?? false).length,
          totalBots: user.bots.length,
          activeTrades,
          totalTrades,
          totalPnl,
          activePnl,
        }}
        bots={user.bots.map((bot) => ({
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
        recentTrades={trades.map((trade) => ({
          id: trade.id,
          coin: trade.coin,
          position: trade.position,
          regime: trade.regime,
          confidence: trade.confidence,
          leverage: trade.leverage,
          capital: trade.capital,
          entryPrice: trade.entryPrice,
          currentPrice: trade?.currentPrice ?? null,
          exitPrice: trade?.exitPrice ?? null,
          stopLoss: trade.stopLoss,
          takeProfit: trade.takeProfit,
          slType: trade.slType,
          status: trade.status,
          activePnl: trade.activePnl,
          activePnlPercent: trade.activePnlPercent,
          totalPnl: trade.totalPnl,
          totalPnlPercent: trade.totalPnlPercent,
          exitPercent: trade?.exitPercent ?? null,
          entryTime: trade.entryTime.toISOString(),
          exitTime: trade?.exitTime?.toISOString?.() ?? null,
        }))}
      />
    );
  } catch (error) {
    console.error('Dashboard error:', error);
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-2 text-[var(--color-danger)]">Error Loading Dashboard</h2>
          <p className="text-[var(--color-text-secondary)]">Please try refreshing the page</p>
        </div>
      </div>
    );
  }
}