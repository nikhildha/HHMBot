import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';
import { AccountClient } from './account-client';

export const dynamic = 'force-dynamic';

export default async function AccountPage() {
  const session = await getServerSession(authOptions);

  if (!session?.user) {
    redirect('/login');
  }

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    include: {
      subscription: true,
    },
  });

  if (!user) {
    redirect('/login');
  }

  return (
    <AccountClient
      user={{
        id: user.id,
        name: user?.name ?? '',
        email: user.email,
        createdAt: user.createdAt.toISOString(),
      }}
      subscription={user?.subscription ? {
        tier: user.subscription.tier,
        status: user.subscription.status,
        coinScans: user.subscription.coinScans,
        trialEndsAt: user.subscription?.trialEndsAt?.toISOString?.() ?? null,
        currentPeriodEnd: user.subscription?.currentPeriodEnd?.toISOString?.() ?? null,
      } : null}
    />
  );
}