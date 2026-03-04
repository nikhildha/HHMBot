'use client';

import { useState, useEffect } from 'react';
import { Header } from '@/components/header';
import { User, Crown, Calendar, Check } from 'lucide-react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { format } from 'date-fns';

interface AccountClientProps {
  user: {
    id: string;
    name: string;
    email: string;
    createdAt: string;
  };
  subscription: {
    tier: string;
    status: string;
    coinScans: number;
    trialEndsAt?: string | null;
    currentPeriodEnd?: string | null;
  } | null;
}

export function AccountClient({ user, subscription }: AccountClientProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  const getTierDisplay = (tier: string) => {
    if (tier === 'free') return 'Free Trial';
    if (tier === 'pro') return 'Pro';
    if (tier === 'ultra') return 'Ultra';
    return tier;
  };

  const getStatusColor = (status: string) => {
    if (status === 'active') return 'text-[var(--color-success)]';
    if (status === 'trial') return 'text-[var(--color-warning)]';
    return 'text-[var(--color-text-secondary)]';
  };

  return (
    <div className="min-h-screen">
      <Header />

      <main className="pt-24 pb-12 px-4">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <h1 className="text-3xl font-bold mb-1">
              <span className="text-gradient">Account Details</span>
            </h1>
            <p className="text-[var(--color-text-secondary)]">
              Manage your profile and subscription
            </p>
          </motion.div>

          {/* Profile Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="card-gradient p-8 rounded-xl mb-6"
          >
            <div className="flex items-center space-x-3 mb-6">
              <div className="p-3 bg-[var(--color-primary)]/20 rounded-lg">
                <User className="w-6 h-6 text-[var(--color-primary)]" />
              </div>
              <h2 className="text-xl font-bold text-cyan-400">Profile Information</h2>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-sm text-[var(--color-text-secondary)]">Full Name</label>
                <p className="text-lg font-medium">{user?.name ?? 'N/A'}</p>
              </div>
              <div>
                <label className="text-sm text-[var(--color-text-secondary)]">Email Address</label>
                <p className="text-lg font-medium">{user?.email ?? 'N/A'}</p>
              </div>
              <div>
                <label className="text-sm text-[var(--color-text-secondary)]">Member Since</label>
                <p className="text-lg font-medium">
                  {format(new Date(user?.createdAt ?? new Date()), 'MMMM d, yyyy')}
                </p>
              </div>
            </div>
          </motion.div>

          {/* Subscription Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="card-gradient p-8 rounded-xl"
          >
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-3">
                <div className="p-3 bg-[var(--color-primary)]/20 rounded-lg">
                  <Crown className="w-6 h-6 text-[var(--color-primary)]" />
                </div>
                <h2 className="text-xl font-bold text-cyan-400">Subscription</h2>
              </div>
              <Link
                href="/pricing"
                className="px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary-dark)] transition-colors"
              >
                Upgrade Plan
              </Link>
            </div>

            {subscription ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-[var(--color-surface)] rounded-lg">
                  <div>
                    <p className="text-sm text-[var(--color-text-secondary)]">Current Plan</p>
                    <p className="text-2xl font-bold text-[var(--color-primary)]">
                      {getTierDisplay(subscription.tier)}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-[var(--color-text-secondary)]">Status</p>
                    <p className={`text-lg font-semibold capitalize ${getStatusColor(subscription.status)}`}>
                      {subscription.status}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 bg-[var(--color-surface)] rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <Check className="w-5 h-5 text-[var(--color-success)]" />
                      <p className="text-sm text-[var(--color-text-secondary)]">Coin Scans</p>
                    </div>
                    <p className="text-xl font-bold">
                      {subscription.coinScans === 0 ? 'Unlimited' : subscription.coinScans}
                    </p>
                  </div>

                  {subscription.trialEndsAt && subscription.status === 'trial' && (
                    <div className="p-4 bg-[var(--color-surface)] rounded-lg">
                      <div className="flex items-center space-x-2 mb-2">
                        <Calendar className="w-5 h-5 text-[var(--color-warning)]" />
                        <p className="text-sm text-[var(--color-text-secondary)]">Trial Ends</p>
                      </div>
                      <p className="text-xl font-bold">
                        {format(new Date(subscription.trialEndsAt), 'MMM d, yyyy')}
                      </p>
                    </div>
                  )}

                  {subscription.currentPeriodEnd && subscription.status === 'active' && (
                    <div className="p-4 bg-[var(--color-surface)] rounded-lg">
                      <div className="flex items-center space-x-2 mb-2">
                        <Calendar className="w-5 h-5 text-[var(--color-primary)]" />
                        <p className="text-sm text-[var(--color-text-secondary)]">Renews On</p>
                      </div>
                      <p className="text-xl font-bold">
                        {format(new Date(subscription.currentPeriodEnd), 'MMM d, yyyy')}
                      </p>
                    </div>
                  )}
                </div>

                {subscription.status === 'trial' && (
                  <div className="mt-6 p-4 bg-[var(--color-warning)]/20 border border-[var(--color-warning)] rounded-lg">
                    <p className="text-sm">
                      Your free trial will end on{' '}
                      <strong>
                        {subscription?.trialEndsAt ? format(new Date(subscription.trialEndsAt), 'MMMM d, yyyy') : 'N/A'}
                      </strong>
                      . Upgrade to a paid plan to continue using Sentinel after your trial ends.
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8">
                <p className="text-[var(--color-text-secondary)] mb-4">
                  No active subscription found
                </p>
                <Link
                  href="/pricing"
                  className="inline-block px-6 py-3 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary-dark)] transition-colors"
                >
                  View Plans
                </Link>
              </div>
            )}
          </motion.div>
        </div>
      </main>
    </div>
  );
}