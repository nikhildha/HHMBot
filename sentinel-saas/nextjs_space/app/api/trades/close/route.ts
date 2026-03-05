import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import { prisma } from '@/lib/prisma';

export const dynamic = 'force-dynamic';

const ENGINE_API_URL = process.env.ENGINE_API_URL;

export async function POST(request: Request) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const { tradeId, symbol } = await request.json();
        if (!tradeId && !symbol) {
            return NextResponse.json({ error: 'tradeId or symbol required' }, { status: 400 });
        }

        const userId = (session.user as any)?.id;
        const isAdmin = (session.user as any)?.role === 'admin';

        // ─── Find the trade in Prisma ────────────────────────────────────
        let trade = null;

        if (tradeId) {
            // Try direct ID match first
            trade = await prisma.trade.findFirst({
                where: {
                    id: tradeId,
                    status: 'active',
                    bot: isAdmin ? {} : { userId },
                },
                include: { bot: true },
            });

            // Fallback: match by exchangeOrderId (engine trade_id)
            if (!trade) {
                trade = await prisma.trade.findFirst({
                    where: {
                        exchangeOrderId: tradeId,
                        status: 'active',
                        bot: isAdmin ? {} : { userId },
                    },
                    include: { bot: true },
                });
            }

            // Fallback: partial match (trade IDs from frontend may be engine_XXX_botId)
            if (!trade) {
                trade = await prisma.trade.findFirst({
                    where: {
                        id: { contains: tradeId },
                        status: 'active',
                        bot: isAdmin ? {} : { userId },
                    },
                    include: { bot: true },
                });
            }
        }

        if (!trade && symbol) {
            trade = await prisma.trade.findFirst({
                where: {
                    coin: symbol.toUpperCase(),
                    status: 'active',
                    bot: isAdmin ? {} : { userId },
                },
                include: { bot: true },
                orderBy: { entryTime: 'desc' },
            });
        }

        if (!trade) {
            return NextResponse.json({ error: 'No matching active trade found' }, { status: 404 });
        }

        // ─── Calculate PNL at current price ──────────────────────────────
        const currentPrice = trade.currentPrice || trade.entryPrice;
        const entry = trade.entryPrice;
        const capital = trade.capital;
        const lev = trade.leverage;
        const isLong = trade.position === 'long';

        const priceDiff = isLong ? (currentPrice - entry) : (entry - currentPrice);
        const rawPnl = priceDiff / entry * lev * capital;
        const leveragedPnl = Math.round(rawPnl * 10000) / 10000;
        const pnlPct = capital > 0 ? Math.round(leveragedPnl / capital * 100 * 100) / 100 : 0;

        // ─── Update trade in Prisma ──────────────────────────────────────
        await prisma.trade.update({
            where: { id: trade.id },
            data: {
                status: 'closed',
                exitPrice: currentPrice,
                exitTime: new Date(),
                exitReason: 'MANUAL_CLOSE',
                totalPnl: leveragedPnl,
                totalPnlPercent: pnlPct,
                activePnl: 0,
                activePnlPercent: 0,
            },
        });

        // ─── Also try to close on engine (best-effort) ───────────────────
        if (ENGINE_API_URL) {
            try {
                const engineTradeId = trade.exchangeOrderId || trade.id;
                await fetch(`${ENGINE_API_URL}/api/close-trade`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        trade_id: engineTradeId,
                        symbol: trade.coin,
                    }),
                    signal: AbortSignal.timeout(5000),
                });
            } catch {
                // Engine close is best-effort — Prisma is source of truth
            }
        }

        return NextResponse.json({
            success: true,
            closed: [{
                trade_id: trade.exchangeOrderId || trade.id,
                symbol: trade.coin,
                pnl: leveragedPnl,
                pnl_pct: pnlPct,
            }],
        });
    } catch (error: any) {
        console.error('Trade close error:', error);
        return NextResponse.json({ error: 'Failed to close trade' }, { status: 500 });
    }
}
