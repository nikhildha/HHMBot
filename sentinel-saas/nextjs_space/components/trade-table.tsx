'use client';

import { TrendingUp, TrendingDown, Clock } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface Trade {
  id: string;
  coin: string;
  position: string;
  regime: string;
  confidence: number;
  leverage: number;
  capital: number;
  entryPrice: number;
  currentPrice?: number | null;
  exitPrice?: number | null;
  stopLoss: number;
  takeProfit: number;
  slType: string;
  status: string;
  activePnl: number;
  activePnlPercent: number;
  totalPnl: number;
  totalPnlPercent: number;
  exitPercent?: number | null;
  entryTime: Date | string;
  exitTime?: Date | string | null;
}

interface TradeTableProps {
  trades: Trade[];
}

export function TradeTable({ trades }: TradeTableProps) {
  const formatCurrency = (value: number) => `$${value.toFixed(2)}`;
  const formatPercent = (value: number) => `${value.toFixed(2)}%`;

  const getDuration = (entryTime: Date | string, exitTime?: Date | string | null) => {
    const entry = new Date(entryTime);
    const exit = exitTime ? new Date(exitTime) : new Date();
    return formatDistanceToNow(entry, { addSuffix: false });
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-[var(--color-surface-light)]">
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Coin</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Position</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Regime</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Conf. %</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Lev.</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Capital</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Entry</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">CMP</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">SL / TP</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Status</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Active PNL</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Total PNL</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Duration</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Exit %</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Exit Price</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Entry Time</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">Exit Time</th>
            <th className="text-left py-4 px-4 text-sm font-medium text-[var(--color-text-secondary)]">SL Type</th>
          </tr>
        </thead>
        <tbody>
          {trades?.map?.((trade) => (
            <tr
              key={trade?.id}
              className="border-b border-[var(--color-surface-light)] hover:bg-[var(--color-surface)] transition-colors"
            >
              <td className="py-4 px-4 font-medium">{trade?.coin ?? 'N/A'}</td>
              <td className="py-4 px-4">
                <div className="flex items-center space-x-1">
                  {trade?.position === 'long' ? (
                    <>
                      <TrendingUp className="w-4 h-4 text-[var(--color-success)]" />
                      <span className="text-[var(--color-success)] uppercase text-sm">Long</span>
                    </>
                  ) : (
                    <>
                      <TrendingDown className="w-4 h-4 text-[var(--color-danger)]" />
                      <span className="text-[var(--color-danger)] uppercase text-sm">Short</span>
                    </>
                  )}
                </div>
              </td>
              <td className="py-4 px-4">
                <span
                  className={`px-2 py-1 rounded text-xs uppercase ${
                    trade?.regime === 'bullish'
                      ? 'bg-[var(--color-success)]/20 text-[var(--color-success)]'
                      : 'bg-[var(--color-danger)]/20 text-[var(--color-danger)]'
                  }`}
                >
                  {trade?.regime ?? 'N/A'}
                </span>
              </td>
              <td className="py-4 px-4">{formatPercent(trade?.confidence ?? 0)}</td>
              <td className="py-4 px-4">{trade?.leverage ?? 0}x</td>
              <td className="py-4 px-4">{formatCurrency(trade?.capital ?? 0)}</td>
              <td className="py-4 px-4 text-sm">{formatCurrency(trade?.entryPrice ?? 0)}</td>
              <td className="py-4 px-4 text-sm">
                {trade?.currentPrice ? formatCurrency(trade.currentPrice) : '—'}
              </td>
              <td className="py-4 px-4 text-sm">
                {formatCurrency(trade?.stopLoss ?? 0)} / {formatCurrency(trade?.takeProfit ?? 0)}
              </td>
              <td className="py-4 px-4">
                <span
                  className={`px-2 py-1 rounded text-xs uppercase ${
                    trade?.status === 'active'
                      ? 'bg-[var(--color-success)]/20 text-[var(--color-success)]'
                      : 'bg-[var(--color-text-secondary)]/20 text-[var(--color-text-secondary)]'
                  }`}
                >
                  {trade?.status ?? 'N/A'}
                </span>
              </td>
              <td className="py-4 px-4">
                <div className={trade?.activePnl ?? 0 >= 0 ? 'text-[var(--color-success)]' : 'text-[var(--color-danger)]'}>
                  {formatCurrency(trade?.activePnl ?? 0)}
                  <span className="text-xs ml-1">({formatPercent(trade?.activePnlPercent ?? 0)})</span>
                </div>
              </td>
              <td className="py-4 px-4">
                {trade?.totalPnl ? (
                  <div className={trade.totalPnl >= 0 ? 'text-[var(--color-success)]' : 'text-[var(--color-danger)]'}>
                    {formatCurrency(trade.totalPnl)}
                    <span className="text-xs ml-1">({formatPercent(trade?.totalPnlPercent ?? 0)})</span>
                  </div>
                ) : (
                  '—'
                )}
              </td>
              <td className="py-4 px-4 text-sm">
                <div className="flex items-center space-x-1">
                  <Clock className="w-3 h-3" />
                  <span>{getDuration(trade?.entryTime ?? new Date(), trade?.exitTime)}</span>
                </div>
              </td>
              <td className="py-4 px-4 text-sm">
                {trade?.exitPercent ? formatPercent(trade.exitPercent) : '—'}
              </td>
              <td className="py-4 px-4 text-sm">
                {trade?.exitPrice ? formatCurrency(trade.exitPrice) : '—'}
              </td>
              <td className="py-4 px-4 text-sm">
                {new Date(trade?.entryTime ?? new Date()).toLocaleString()}
              </td>
              <td className="py-4 px-4 text-sm">
                {trade?.exitTime ? new Date(trade.exitTime).toLocaleString() : '—'}
              </td>
              <td className="py-4 px-4">
                <span className="px-2 py-1 rounded text-xs bg-[var(--color-surface-light)]">
                  {trade?.slType === 'trail_x1' ? 'Trail x1' : 'Fixed'}
                </span>
              </td>
            </tr>
          )) ?? []}
        </tbody>
      </table>
      {(!trades || trades.length === 0) && (
        <div className="text-center py-12 text-[var(--color-text-secondary)]">
          No trades found
        </div>
      )}
    </div>
  );
}