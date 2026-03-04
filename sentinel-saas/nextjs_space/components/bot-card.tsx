'use client';

import { Bot, Play, Square, TrendingUp, TrendingDown } from 'lucide-react';
import { motion } from 'framer-motion';

interface BotCardProps {
  bot: {
    id: string;
    name: string;
    exchange: string;
    status: string;
    isActive: boolean;
    startedAt?: Date | null;
    _count?: {
      trades: number;
    };
  };
  onToggle: (botId: string, currentStatus: boolean) => void;
  liveTradeCount?: number;
}

export function BotCard({ bot, onToggle, liveTradeCount }: BotCardProps) {
  const isRunning = bot?.isActive ?? false;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card-gradient rounded-xl p-6 glow-hover hover-lift"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-3 bg-[var(--color-primary)]/20 rounded-lg">
            <Bot className="w-6 h-6 text-[var(--color-primary)]" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">{bot?.name ?? 'Bot'}</h3>
            <p className="text-sm text-[var(--color-text-secondary)]">
              {bot?.exchange ?? 'Unknown'}
            </p>
          </div>
        </div>
        <button
          onClick={() => onToggle(bot?.id ?? '', isRunning)}
          className={`p-2 rounded-lg transition-colors ${isRunning
              ? 'bg-[var(--color-danger)] hover:opacity-80'
              : 'bg-[var(--color-success)] hover:opacity-80'
            }`}
        >
          {isRunning ? (
            <Square className="w-5 h-5 text-white" />
          ) : (
            <Play className="w-5 h-5 text-white" />
          )}
        </button>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-[var(--color-text-secondary)]">Status</span>
          <span
            className={`px-3 py-1 rounded-full text-xs font-medium ${isRunning
                ? 'bg-[var(--color-success)]/20 text-[var(--color-success)]'
                : 'bg-[var(--color-text-secondary)]/20 text-[var(--color-text-secondary)]'
              }`}
          >
            {isRunning ? 'Running' : 'Stopped'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-[var(--color-text-secondary)]">Active Trades</span>
          <span className="font-semibold">{liveTradeCount ?? bot?._count?.trades ?? 0}</span>
        </div>
        {bot?.startedAt && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-[var(--color-text-secondary)]">Started</span>
            <span className="text-sm">
              {new Date(bot.startedAt).toLocaleDateString()}
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
}