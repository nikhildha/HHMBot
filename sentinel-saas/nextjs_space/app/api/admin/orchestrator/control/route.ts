import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';

export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const { botId, action } = await request.json();
        if (!botId || !['start', 'stop', 'restart'].includes(action)) {
            return NextResponse.json({ error: 'Invalid request' }, { status: 400 });
        }

        const orchestratorUrl = process.env.ORCHESTRATOR_URL || 'http://localhost:5000';
        const endpoint = `${orchestratorUrl}/bots/${botId}/${action}`;

        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: AbortSignal.timeout(10000),
        });

        if (res.ok) {
            const data = await res.json();
            return NextResponse.json(data);
        }
        return NextResponse.json({ error: 'Orchestrator error' }, { status: res.status });
    } catch (error: any) {
        console.error('Orchestrator control error:', error);
        return NextResponse.json({ error: 'Failed to reach orchestrator' }, { status: 503 });
    }
}
