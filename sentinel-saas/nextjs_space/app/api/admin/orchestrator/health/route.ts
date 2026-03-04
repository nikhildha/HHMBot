import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';

export const dynamic = 'force-dynamic';

export async function GET() {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const orchestratorUrl = process.env.ORCHESTRATOR_URL || 'http://localhost:5000';
        const res = await fetch(`${orchestratorUrl}/health`, { signal: AbortSignal.timeout(3000) });

        if (res.ok) {
            const data = await res.json();
            return NextResponse.json({ online: true, ...data });
        }
        return NextResponse.json({ online: false }, { status: 503 });
    } catch {
        return NextResponse.json({ online: false }, { status: 503 });
    }
}
