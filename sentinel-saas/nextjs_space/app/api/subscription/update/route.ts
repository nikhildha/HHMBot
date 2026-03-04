import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import prisma from '@/lib/prisma';

export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { tier, razorpayPaymentId, razorpayOrderId } = await request.json();

    const coinScans = tier === 'pro' ? 15 : tier === 'ultra' ? 50 : 0;

    const subscription = await prisma.subscription.upsert({
      where: { userId: session.user.id },
      create: {
        userId: session.user.id,
        tier,
        coinScans,
        status: 'active',
        razorpayPaymentId,
        razorpayOrderId,
        currentPeriodEnd: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      },
      update: {
        tier,
        coinScans,
        status: 'active',
        razorpayPaymentId,
        razorpayOrderId,
        currentPeriodEnd: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      },
    });

    // Send notification to admin
    try {
      const user = await prisma.user.findUnique({
        where: { id: session.user.id },
      });

      const appUrl = process.env.NEXTAUTH_URL || '';
      const appName = appUrl ? new URL(appUrl).hostname.split('.')[0] : 'Sentinel';

      const htmlBody = `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #0891B2; border-bottom: 2px solid #0891B2; padding-bottom: 10px;">
            Subscription Update - Sentinel
          </h2>
          <div style="background: #f0f9ff; padding: 20px; border-radius: 8px; margin: 20px 0;">
            <p style="margin: 10px 0;"><strong>User:</strong> ${user?.name ?? 'N/A'}</p>
            <p style="margin: 10px 0;"><strong>Email:</strong> ${user?.email ?? 'N/A'}</p>
            <p style="margin: 10px 0;"><strong>New Tier:</strong> ${tier.toUpperCase()}</p>
            <p style="margin: 10px 0;"><strong>Coin Scans:</strong> ${coinScans}</p>
            ${razorpayPaymentId ? `<p style="margin: 10px 0;"><strong>Payment ID:</strong> ${razorpayPaymentId}</p>` : ''}
          </div>
          <p style="color: #666; font-size: 12px;">
            Updated at: ${new Date().toLocaleString()}
          </p>
        </div>
      `;

      await fetch('https://apps.abacus.ai/api/sendNotificationEmail', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          deployment_token: process.env.ABACUSAI_API_KEY,
          app_id: process.env.WEB_APP_ID,
          notification_id: process.env.NOTIF_ID_SUBSCRIPTION_CHANGE,
          subject: `Subscription Update: ${user?.name ?? 'User'} - ${tier.toUpperCase()}`,
          body: htmlBody,
          is_html: true,
          recipient_email: 'nikhildha@gmail.com',
          sender_email: `noreply@${appUrl ? new URL(appUrl).hostname : 'sentinel.app'}`,
          sender_alias: appName,
        }),
      });
    } catch (emailError) {
      console.error('Failed to send subscription notification:', emailError);
    }

    return NextResponse.json({ success: true, subscription });
  } catch (error: any) {
    console.error('Subscription update error:', error);
    return NextResponse.json({ error: 'Failed to update subscription' }, { status: 500 });
  }
}