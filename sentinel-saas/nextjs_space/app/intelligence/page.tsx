import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import { redirect } from 'next/navigation';
import { IntelligenceClient } from './intelligence-client';

export default async function IntelligencePage() {
    const session = await getServerSession(authOptions);
    if (!session) redirect('/login');

    return <IntelligenceClient />;
}
