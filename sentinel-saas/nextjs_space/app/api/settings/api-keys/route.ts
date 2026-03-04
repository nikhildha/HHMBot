import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';
import { encryptApiKeys } from '@/lib/encryption';

export const dynamic = 'force-dynamic';

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

    if (!['binance', 'coindcx'].includes(exchange)) {
      return NextResponse.json({ error: 'Invalid exchange' }, { status: 400 });
    }

    // Encrypt API keys before storing
    const encrypted = encryptApiKeys(apiKey, apiSecret);

    await prisma.exchangeApiKey.upsert({
      where: {
        userId_exchange: {
          userId: session.user.id,
          exchange,
        },
      },
      update: {
        apiKey: encrypted.apiKey,
        apiSecret: encrypted.apiSecret,
        encryptionIv: encrypted.encryptionIv,
      },
      create: {
        userId: session.user.id,
        exchange,
        apiKey: encrypted.apiKey,
        apiSecret: encrypted.apiSecret,
        encryptionIv: encrypted.encryptionIv,
      },
    });

    return NextResponse.json({ success: true });
  } catch (error: any) {
    console.error('API key save error:', error);
    return NextResponse.json({ error: 'Failed to save API keys' }, { status: 500 });
  }
}