'use client';

import { LucideIcon } from 'lucide-react';
import { motion } from 'framer-motion';
import { useEffect, useState, useRef } from 'react';

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  animated?: boolean;
}

export function StatsCard({
  title,
  value,
  icon: Icon,
  trend,
  trendValue,
  animated = false,
}: StatsCardProps) {
  const [displayValue, setDisplayValue] = useState(animated ? 0 : value);
  const [hasAnimated, setHasAnimated] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  // Keep displayValue in sync with value prop for non-animated cards
  useEffect(() => {
    if (!animated) {
      setDisplayValue(value);
    }
  }, [value, animated]);

  useEffect(() => {
    if (!animated || hasAnimated) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !hasAnimated) {
            const numericValue = typeof value === 'string' ? parseFloat(value) : value;
            if (!isNaN(numericValue)) {
              let start = 0;
              const duration = 1500;
              const increment = numericValue / (duration / 16);
              const timer = setInterval(() => {
                start += increment;
                if (start >= numericValue) {
                  setDisplayValue(numericValue);
                  clearInterval(timer);
                  setHasAnimated(true);
                } else {
                  setDisplayValue(Math.floor(start));
                }
              }, 16);
            }
          }
        });
      },
      { threshold: 0.1 }
    );

    if (cardRef.current) {
      observer.observe(cardRef.current);
    }

    return () => observer.disconnect();
  }, [value, animated, hasAnimated]);

  const getTrendColor = () => {
    if (trend === 'up') return 'text-[var(--color-success)]';
    if (trend === 'down') return 'text-[var(--color-danger)]';
    return 'text-[var(--color-text-secondary)]';
  };

  return (
    <motion.div
      ref={cardRef}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card-gradient rounded-xl p-6 glow-hover hover-lift"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="p-3 bg-[var(--color-primary)]/20 rounded-lg">
          <Icon className="w-6 h-6 text-[var(--color-primary)]" />
        </div>
        {trendValue && (
          <span className={`text-sm font-medium ${getTrendColor()}`}>
            {trendValue}
          </span>
        )}
      </div>
      <h3 className="text-sm text-[var(--color-text-secondary)] mb-2">{title}</h3>
      <p className="text-3xl font-bold">{displayValue}</p>
    </motion.div>
  );
}