/**
 * Razorpay Webhook Handler
 * POST /api/webhooks/razorpay
 * 
 * Handles payment success events from Razorpay to auto-update 
 * user subscriptions after payment.
 * 
 * In production, verify the webhook signature using RAZORPAY_WEBHOOK_SECRET.
 * For now, accepts events and updates subscription accordingly.
 */

import { NextResponse } from 'next/server';
import prisma from '@/lib/prisma';
import crypto from 'crypto';

export const dynamic = 'force-dynamic';

// Razorpay webhook events we care about
const HANDLED_EVENTS = [
    'payment.captured',
    'subscription.activated',
    'subscription.charged',
    'subscription.cancelled',
    'subscription.expired',
];

export async function POST(request: Request) {
    try {
        const rawBody = await request.text();
        const signature = request.headers.get('x-razorpay-signature');

        // ─── Signature Verification ───────────────────────────────
        const webhookSecret = process.env.RAZORPAY_WEBHOOK_SECRET;
        if (webhookSecret && signature) {
            const expectedSig = crypto
                .createHmac('sha256', webhookSecret)
                .update(rawBody)
                .digest('hex');

            if (signature !== expectedSig) {
                console.error('Razorpay webhook signature mismatch');
                return NextResponse.json({ error: 'Invalid signature' }, { status: 401 });
            }
        }

        const payload = JSON.parse(rawBody);
        const event = payload.event;

        if (!HANDLED_EVENTS.includes(event)) {
            // Acknowledge unhandled events
            return NextResponse.json({ received: true, event });
        }

        console.log(`[Razorpay Webhook] Event: ${event}`);

        // ─── Payment Captured ─────────────────────────────────────
        if (event === 'payment.captured') {
            const payment = payload.payload?.payment?.entity;
            if (!payment) {
                return NextResponse.json({ error: 'No payment entity' }, { status: 400 });
            }

            const email = payment.email || payment.notes?.email;
            const planId = payment.notes?.plan_id;  // 'pro' or 'ultra'
            const razorpayPaymentId = payment.id;
            const razorpayOrderId = payment.order_id;

            if (!email) {
                console.error('[Razorpay Webhook] No email in payment:', payment.id);
                return NextResponse.json({ error: 'No email in payment' }, { status: 400 });
            }

            // Find user by email
            const user = await prisma.user.findUnique({
                where: { email },
                include: { subscription: true },
            });

            if (!user) {
                console.error(`[Razorpay Webhook] No user found for email: ${email}`);
                return NextResponse.json({ error: 'User not found' }, { status: 404 });
            }

            // Determine tier from payment amount or plan notes
            let tier = planId || 'pro';
            if (!planId) {
                // Infer from amount (in paise): 99900 = Pro, 249900 = Ultra
                const amountInr = payment.amount / 100;
                tier = amountInr >= 2000 ? 'ultra' : 'pro';
            }

            const coinScans = tier === 'ultra' ? 50 : 15;
            const periodEnd = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);

            // Update subscription
            await prisma.subscription.upsert({
                where: { userId: user.id },
                create: {
                    userId: user.id,
                    tier,
                    coinScans,
                    status: 'active',
                    razorpayPaymentId,
                    razorpayOrderId,
                    currentPeriodEnd: periodEnd,
                },
                update: {
                    tier,
                    coinScans,
                    status: 'active',
                    razorpayPaymentId,
                    razorpayOrderId,
                    currentPeriodEnd: periodEnd,
                },
            });

            console.log(
                `[Razorpay Webhook] Updated subscription for ${email}: tier=${tier}, ends=${periodEnd.toISOString()}`
            );

            // Notify admin via email
            try {
                const appUrl = process.env.NEXTAUTH_URL || '';
                const appName = appUrl ? new URL(appUrl).hostname.split('.')[0] : 'Sentinel';

                await fetch('https://apps.abacus.ai/api/sendNotificationEmail', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        deployment_token: process.env.ABACUSAI_API_KEY,
                        app_id: process.env.WEB_APP_ID,
                        notification_id: process.env.NOTIF_ID_SUBSCRIPTION_CHANGE,
                        subject: `💰 Payment: ${user.name ?? email} upgraded to ${tier.toUpperCase()}`,
                        body: `<div style="font-family: Arial, sans-serif;">
              <h2 style="color: #10b981;">Payment Received</h2>
              <p><b>User:</b> ${user.name} (${email})</p>
              <p><b>Tier:</b> ${tier.toUpperCase()}</p>
              <p><b>Amount:</b> ₹${payment.amount / 100}</p>
              <p><b>Payment ID:</b> ${razorpayPaymentId}</p>
              <p><b>Period End:</b> ${periodEnd.toLocaleDateString()}</p>
            </div>`,
                        is_html: true,
                        recipient_email: process.env.ADMIN_EMAIL || 'admin@sentinel.app',
                        sender_email: `noreply@${appUrl ? new URL(appUrl).hostname : 'sentinel.app'}`,
                        sender_alias: appName,
                    }),
                });
            } catch (emailError) {
                console.error('Failed to send payment notification:', emailError);
            }

            return NextResponse.json({ success: true, tier, userId: user.id });
        }

        // ─── Subscription Cancelled ───────────────────────────────
        if (event === 'subscription.cancelled' || event === 'subscription.expired') {
            const subEntity = payload.payload?.subscription?.entity;
            const email = subEntity?.notes?.email;

            if (email) {
                const user = await prisma.user.findUnique({ where: { email } });
                if (user) {
                    await prisma.subscription.update({
                        where: { userId: user.id },
                        data: {
                            status: event === 'subscription.cancelled' ? 'cancelled' : 'expired',
                        },
                    });
                    console.log(`[Razorpay Webhook] Subscription ${event} for ${email}`);
                }
            }

            return NextResponse.json({ success: true });
        }

        return NextResponse.json({ received: true, event });
    } catch (error: any) {
        console.error('[Razorpay Webhook] Error:', error);
        return NextResponse.json({ error: 'Webhook processing failed' }, { status: 500 });
    }
}
