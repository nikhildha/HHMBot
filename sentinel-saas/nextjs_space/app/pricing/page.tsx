'use client';

import Link from 'next/link';
import { Header } from '@/components/header';
import { Check, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

export default function PricingPage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  const plans = [
    {
      name: 'Free Trial',
      price: 'Free',
      period: '14 days',
      coinScans: 'Unlimited',
      description: 'Perfect for testing the platform',
      features: [
        'All bot features',
        'CoinDCX & Binance support',
        'Real-time trade execution',
        'Basic analytics',
        '14-day trial period',
      ],
      cta: 'Start Free Trial',
      href: '/signup',
      popular: false,
    },
    {
      name: 'Pro',
      price: '₹999',
      period: 'per month',
      coinScans: '15 coin scans',
      description: 'For individual traders',
      features: [
        'All Free features',
        '15 simultaneous coin scans',
        'Advanced analytics',
        'Priority support',
        'Custom alerts',
        'Trade history export',
      ],
      cta: 'Upgrade to Pro',
      href: 'https://rzp.io/rzp/ktPoQNJz',
      popular: true,
    },
    {
      name: 'Ultra',
      price: '₹2,499',
      period: 'per month',
      coinScans: '50 coin scans',
      description: 'For professional traders',
      features: [
        'All Pro features',
        '50 simultaneous coin scans',
        'Premium analytics dashboard',
        'Dedicated support',
        'API access',
        'Multi-bot management',
        'Advanced risk management',
      ],
      cta: 'Upgrade to Ultra',
      href: 'https://rzp.io/rzp/ktPoQNJz',
      popular: false,
    },
  ];

  return (
    <div className="min-h-screen">
      <Header />

      <section className="pt-32 pb-20 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-16"
          >
            <h1 className="text-5xl font-bold mb-4">
              Choose Your <span className="text-gradient">Trading Plan</span>
            </h1>
            <p className="text-xl text-[var(--color-text-secondary)] max-w-2xl mx-auto">
              Start with a free trial and upgrade when you're ready to scale
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {plans.map((plan, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`card-gradient rounded-2xl p-8 hover-lift relative ${
                  plan.popular ? 'ring-2 ring-[var(--color-primary)] glow' : ''
                }`}
              >
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 bg-[var(--color-primary)] text-white rounded-full text-sm font-medium flex items-center space-x-1">
                    <Sparkles className="w-4 h-4" />
                    <span>Most Popular</span>
                  </div>
                )}

                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold mb-2">{plan.name}</h3>
                  <div className="mb-2">
                    <span className="text-4xl font-bold text-[var(--color-primary)]">
                      {plan.price}
                    </span>
                    <span className="text-[var(--color-text-secondary)] ml-2">
                      {plan.period}
                    </span>
                  </div>
                  <p className="text-sm text-[var(--color-accent)] font-medium">
                    {plan.coinScans}
                  </p>
                  <p className="text-sm text-[var(--color-text-secondary)] mt-2">
                    {plan.description}
                  </p>
                </div>

                <ul className="space-y-3 mb-8">
                  {plan.features.map((feature, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <Check className="w-5 h-5 text-[var(--color-success)] flex-shrink-0 mt-0.5" />
                      <span className="text-sm">{feature}</span>
                    </li>
                  ))}
                </ul>

                {plan.href.startsWith('http') ? (
                  <a
                    href={plan.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`block w-full py-3 rounded-xl font-semibold text-center transition-colors ${
                      plan.popular
                        ? 'bg-[var(--color-primary)] text-white hover:bg-[var(--color-primary-dark)]'
                        : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-light)]'
                    }`}
                  >
                    {plan.cta}
                  </a>
                ) : (
                  <Link
                    href={plan.href}
                    className={`block w-full py-3 rounded-xl font-semibold text-center transition-colors ${
                      plan.popular
                        ? 'bg-[var(--color-primary)] text-white hover:bg-[var(--color-primary-dark)]'
                        : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-light)]'
                    }`}
                  >
                    {plan.cta}
                  </Link>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <footer className="border-t border-[var(--color-surface-light)] py-8 px-4">
        <div className="max-w-6xl mx-auto text-center text-[var(--color-text-secondary)]">
          <p>&copy; 2026 Sentinel. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}