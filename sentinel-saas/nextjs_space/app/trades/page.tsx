import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import { authOptions } from '@/lib/auth-options';
import { TradesClient } from './trades-client';
import * as fs from 'fs';
import * as path from 'path';

export const dynamic = 'force-dynamic';

const DATA_DIR = path.resolve(process.cwd(), '..', '..', 'data');

function readJSON(filename: string, fallback: any = {}) {
  try {
    const filepath = path.join(DATA_DIR, filename);
    if (fs.existsSync(filepath)) {
      return JSON.parse(fs.readFileSync(filepath, 'utf-8'));
    }
  } catch { /* silent */ }
  return fallback;
}

export default async function TradesPage() {
  const session = await getServerSession(authOptions);

  if (!session?.user) {
    redirect('/login');
  }

  // Read from local data/tradebook.json (engine writes here)
  const tradebook = readJSON('tradebook.json', { trades: [] });
  const rawTrades: any[] = tradebook.trades || [];

  return (
    <TradesClient
      trades={rawTrades.map((t: any) => ({
        id: t.trade_id || t.id || `T-${Math.random().toString(36).slice(2, 8)}`,
        coin: (t.symbol || t.coin || '').replace('USDT', ''),
        symbol: t.symbol || t.coin || '',
        position: (t.side || t.position || '').toLowerCase(),
        regime: t.regime || '',
        confidence: t.confidence || 0,
        leverage: t.leverage || 1,
        capital: t.capital || t.position_size || 0,
        entryPrice: t.entry_price || t.entryPrice || 0,
        currentPrice: t.current_price || t.currentPrice || null,
        exitPrice: t.exit_price || t.exitPrice || null,
        stopLoss: t.stop_loss || t.stopLoss || 0,
        takeProfit: t.take_profit || t.takeProfit || 0,
        slType: t.sl_type || t.slType || 'Default',
        status: (t.status || '').toLowerCase(),
        mode: t.mode || 'paper',
        activePnl: t.unrealized_pnl || t.active_pnl || t.activePnl || 0,
        activePnlPercent: t.unrealized_pnl_pct || t.activePnlPercent || 0,
        totalPnl: t.realized_pnl || t.pnl || t.total_pnl || t.totalPnl || 0,
        totalPnlPercent: t.realized_pnl_pct || t.pnl_pct || t.totalPnlPercent || 0,
        exitPercent: t.exit_percent || null,
        exitReason: t.exit_reason || t.exitReason || null,
        entryTime: t.entry_time || t.entryTime || t.timestamp || new Date().toISOString(),
        exitTime: t.exit_time || t.exitTime || null,
        botName: 'Sentinel Marshal',
      }))}
    />
  );
}