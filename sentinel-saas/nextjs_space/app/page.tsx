'use client';

import Link from 'next/link';
import { Header } from '@/components/header';
import { Shield, TrendingUp, Lock, Zap, BarChart3, Clock } from 'lucide-react';
import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

export default function LandingPage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  const features = [
    {
      icon: TrendingUp,
      title: 'Automated Trading',
      description: 'Let advanced algorithms execute trades 24/7 based on market conditions and technical analysis.',
    },
    {
      icon: Lock,
      title: 'Secure & Reliable',
      description: 'Bank-grade security with encrypted API connections to protect your trading credentials.',
    },
    {
      icon: Zap,
      title: 'Lightning Fast',
      description: 'Execute trades in milliseconds to capitalize on market opportunities before they disappear.',
    },
    {
      icon: BarChart3,
      title: 'Advanced Analytics',
      description: 'Comprehensive performance metrics and detailed trade history for informed decision-making.',
    },
    {
      icon: Clock,
      title: '24/7 Monitoring',
      description: 'Continuous market surveillance with real-time alerts for critical trading events.',
    },
    {
      icon: Shield,
      title: 'Risk Management',
      description: 'Intelligent stop-loss and take-profit mechanisms to protect your capital.',
    },
  ];

  return (
    <div className="min-h-screen">
      <Header />

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-4 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-[var(--color-primary)]/10 via-transparent to-[var(--color-accent)]/10"></div>
        <div className="max-w-6xl mx-auto text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              Advanced <span className="text-gradient">Crypto Trading</span>
              <br />
              On Autopilot
            </h1>
            <p className="text-xl text-[var(--color-text-secondary)] mb-8 max-w-3xl mx-auto">
              Harness the power of automated trading with Sentinel. Execute strategies on CoinDCX and Binance with precision, speed, and confidence.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/signup"
                className="px-8 py-4 bg-[var(--color-primary)] text-white rounded-xl font-semibold hover:bg-[var(--color-primary-dark)] transition-all glow-hover text-lg"
              >
                Start 14-Day Free Trial
              </Link>
              <Link
                href="/pricing"
                className="px-8 py-4 bg-[var(--color-surface)] text-white rounded-xl font-semibold hover:bg-[var(--color-surface-light)] transition-all text-lg"
              >
                View Pricing
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold mb-4">Why Choose Sentinel?</h2>
            <p className="text-xl text-[var(--color-text-secondary)]">
              Professional-grade trading automation built for traders of all levels
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="card-gradient p-8 rounded-xl glow-hover hover-lift"
              >
                <div className="p-4 bg-[var(--color-primary)]/20 rounded-lg inline-block mb-4">
                  <feature.icon className="w-8 h-8 text-[var(--color-primary)]" />
                </div>
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-[var(--color-text-secondary)]">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="card-gradient p-12 rounded-2xl text-center glow"
          >
            <h2 className="text-4xl font-bold mb-4">Ready to Transform Your Trading?</h2>
            <p className="text-xl text-[var(--color-text-secondary)] mb-8">
              Join traders who are already leveraging automated strategies
            </p>
            <Link
              href="/signup"
              className="inline-block px-8 py-4 bg-[var(--color-primary)] text-white rounded-xl font-semibold hover:bg-[var(--color-primary-dark)] transition-all text-lg"
            >
              Get Started Free
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-[var(--color-surface-light)] py-8 px-4">
        <div className="max-w-6xl mx-auto text-center text-[var(--color-text-secondary)]">
          <p>&copy; 2026 Sentinel. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}