import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';

export const dynamic = 'force-dynamic';

/**
 * 8.7 — Exchange API Key Validation
 * POST /api/exchange/validate  { exchange, apiKey, apiSecret }
 * Tests connection to Binance or CoinDCX with provided keys
 */

export async function POST(request: Request) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const { exchange, apiKey, apiSecret } = await request.json();

        if (!exchange || !apiKey || !apiSecret) {
            return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
        }

        if (exchange === 'binance') {
            return await validateBinance(apiKey, apiSecret);
        } else if (exchange === 'coindcx') {
            return await validateCoinDCX(apiKey, apiSecret);
        } else {
            return NextResponse.json({ error: 'Unsupported exchange' }, { status: 400 });
        }
    } catch (error: any) {
        console.error('Exchange validate error:', error);
        return NextResponse.json({ valid: false, error: 'Validation failed' }, { status: 500 });
    }
}

async function validateBinance(apiKey: string, apiSecret: string) {
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

        if (res.ok) {
            const data = await res.json();
            return NextResponse.json({
                valid: true,
                exchange: 'binance',
                balance: parseFloat(data.totalWalletBalance || '0'),
                availableBalance: parseFloat(data.availableBalance || '0'),
            });
        } else {
            const err = await res.json().catch(() => ({}));
            return NextResponse.json({
                valid: false,
                error: err.msg || `Binance API error (${res.status})`,
            });
        }
    } catch (error: any) {
        return NextResponse.json({
            valid: false,
            error: error.message || 'Failed to connect to Binance',
        });
    }
}

async function validateCoinDCX(apiKey: string, apiSecret: string) {
    try {
        const crypto = await import('crypto');
        const body = JSON.stringify({ timestamp: Date.now() });
        const signature = crypto
            .createHmac('sha256', apiSecret)
            .update(body)
            .digest('hex');

        const res = await fetch('https://api.coindcx.com/exchange/v1/users/info', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-AUTH-APIKEY': apiKey,
                'X-AUTH-SIGNATURE': signature,
            },
            body,
            signal: AbortSignal.timeout(10000),
        });

        if (res.ok) {
            const data = await res.json();
            return NextResponse.json({
                valid: true,
                exchange: 'coindcx',
                email: data.email,
            });
        } else {
            return NextResponse.json({
                valid: false,
                error: `CoinDCX API error (${res.status})`,
            });
        }
    } catch (error: any) {
        return NextResponse.json({
            valid: false,
            error: error.message || 'Failed to connect to CoinDCX',
        });
    }
}
