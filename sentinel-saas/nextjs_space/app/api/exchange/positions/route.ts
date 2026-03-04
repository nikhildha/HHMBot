import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';

export const dynamic = 'force-dynamic';

/**
 * 8.6 — Exchange Adapter (Positions + Balance)
 * GET /api/exchange/positions?exchange=binance
 * Decrypts stored API keys and fetches open positions + balance
 */

export async function GET(request: NextRequest) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const exchange = new URL(request.url).searchParams.get('exchange') || 'binance';

        // Get stored API keys
        const apiKeyRecord = await prisma.exchangeApiKey.findUnique({
            where: {
                userId_exchange: {
                    userId: session.user.id,
                    exchange,
                },
            },
        });

        if (!apiKeyRecord) {
            return NextResponse.json({
                error: 'No API keys configured for this exchange',
                positions: [],
                balance: 0,
            }, { status: 404 });
        }

        // Decrypt keys
        const { decryptApiKeys } = await import('@/lib/encryption');
        if (!apiKeyRecord.encryptionIv) {
            return NextResponse.json({ error: 'Keys corrupted (no IV)', positions: [], balance: 0 }, { status: 500 });
        }
        const keys = decryptApiKeys(apiKeyRecord.apiKey, apiKeyRecord.apiSecret, apiKeyRecord.encryptionIv);

        if (exchange === 'binance') {
            return await fetchBinancePositions(keys.apiKey, keys.apiSecret);
        }

        return NextResponse.json({ error: 'Unsupported exchange', positions: [], balance: 0 });
    } catch (error: any) {
        console.error('Exchange positions error:', error);
        return NextResponse.json({ error: 'Failed to fetch positions', positions: [], balance: 0 }, { status: 500 });
    }
}

async function fetchBinancePositions(apiKey: string, apiSecret: string) {
    try {
        const crypto = await import('crypto');
        const timestamp = Date.now();
        const queryString = `timestamp=${timestamp}`;
        const signature = crypto
            .createHmac('sha256', apiSecret)
            .update(queryString)
            .digest('hex');

        const res = await fetch(
            `https://fapi.binance.com/fapi/v2/account?${queryString}&signature=${signature}`,
            {
                headers: { 'X-MBX-APIKEY': apiKey },
                signal: AbortSignal.timeout(10000),
            }
        );

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            return NextResponse.json({
                error: err.msg || 'Binance API error',
                positions: [],
                balance: 0,
            });
        }

        const data = await res.json();

        // Extract open positions (non-zero)
        const positions = (data.positions || [])
            .filter((p: any) => parseFloat(p.positionAmt) !== 0)
            .map((p: any) => ({
                symbol: p.symbol,
                side: parseFloat(p.positionAmt) > 0 ? 'LONG' : 'SHORT',
                size: Math.abs(parseFloat(p.positionAmt)),
                entryPrice: parseFloat(p.entryPrice),
                markPrice: parseFloat(p.markPrice || '0'),
                unrealizedPnl: parseFloat(p.unrealizedProfit),
                leverage: parseInt(p.leverage),
                marginType: p.marginType,
            }));

        return NextResponse.json({
            exchange: 'binance',
            balance: parseFloat(data.totalWalletBalance || '0'),
            availableBalance: parseFloat(data.availableBalance || '0'),
            unrealizedPnl: parseFloat(data.totalUnrealizedProfit || '0'),
            positions,
            positionCount: positions.length,
        });
    } catch (error: any) {
        return NextResponse.json({
            error: error.message || 'Failed to fetch Binance positions',
            positions: [],
            balance: 0,
        });
    }
}
