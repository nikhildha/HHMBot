/**
 * Trade History Migration Script
 * 
 * Migrates HMMBOT tradebook.json (paper) and tradebook_live.json (live)
 * into the PostgreSQL Trade model for a specific bot.
 * 
 * Usage:
 *   BOT_ID=<bot_cuid> npx tsx scripts/migrate-trades.ts
 * 
 * This reads the JSON trade files from the HMMBOT data directory
 * and inserts them into the PostgreSQL database.
 */
import { PrismaClient } from '@prisma/client';
import fs from 'fs';
import path from 'path';

const prisma = new PrismaClient();

const HMMBOT_DIR = path.resolve(__dirname, '..', '..', '..');
const DATA_DIR = path.join(HMMBOT_DIR, 'data');

interface HMMBOTTrade {
    trade_id: string;
    symbol: string;
    position: string;    // LONG / SHORT
    regime: string;
    confidence: number;
    leverage: number;
    capital: number;
    quantity: number;
    entry_price: number;
    current_price?: number;
    exit_price?: number;
    stop_loss: number;
    take_profit: number;
    t1_price?: number;
    t2_price?: number;
    t3_price?: number;
    t1_hit?: boolean;
    t2_hit?: boolean;
    trailing_sl?: number;
    trailing_tp?: number;
    trailing_active?: boolean;
    trail_sl_count?: number;
    capital_protection_active?: boolean;
    original_qty?: number;
    original_capital?: number;
    status: string;      // ACTIVE / CLOSED
    unrealized_pnl?: number;
    unrealized_pnl_pct?: number;
    realized_pnl?: number;
    realized_pnl_pct?: number;
    exit_reason?: string;
    exit_pct?: number;
    mode?: string;
    timestamp: string;
    exit_time?: string;
    tp_extensions?: number;
}

async function migrateTrades() {
    const botId = process.env.BOT_ID;
    if (!botId) {
        console.error('❌ BOT_ID env var required. Set it to the Prisma Bot ID to migrate trades into.');
        process.exit(1);
    }

    // Verify bot exists
    const bot = await prisma.bot.findUnique({ where: { id: botId } });
    if (!bot) {
        console.error(`❌ Bot ${botId} not found in database.`);
        process.exit(1);
    }

    console.log(`📦 Migrating trades for bot: ${bot.name} (${botId})`);

    // Read both tradebooks
    const files = [
        { path: path.join(DATA_DIR, 'tradebook.json'), mode: 'paper' },
        { path: path.join(DATA_DIR, 'tradebook_live.json'), mode: 'live' },
    ];

    let totalMigrated = 0;
    let totalSkipped = 0;

    for (const file of files) {
        if (!fs.existsSync(file.path)) {
            console.log(`⚠️  ${file.path} not found, skipping.`);
            continue;
        }

        const raw = JSON.parse(fs.readFileSync(file.path, 'utf8'));
        const trades: HMMBOTTrade[] = raw.trades || [];
        console.log(`\n📄 ${path.basename(file.path)}: ${trades.length} trades`);

        for (const t of trades) {
            // Skip duplicates by checking exchange order ID or timestamp + symbol
            const existing = await prisma.trade.findFirst({
                where: {
                    botId,
                    coin: t.symbol,
                    entryTime: new Date(t.timestamp),
                },
            });

            if (existing) {
                totalSkipped++;
                continue;
            }

            await prisma.trade.create({
                data: {
                    botId,
                    coin: t.symbol,
                    position: t.position.toLowerCase(),
                    regime: t.regime || 'unknown',
                    confidence: t.confidence || 0,
                    mode: t.mode?.toLowerCase() || file.mode,
                    leverage: t.leverage,
                    capital: t.capital,
                    quantity: t.quantity || 0,
                    entryPrice: t.entry_price,
                    currentPrice: t.current_price,
                    exitPrice: t.exit_price,
                    stopLoss: t.stop_loss,
                    takeProfit: t.take_profit,
                    t1Price: t.t1_price,
                    t2Price: t.t2_price,
                    t3Price: t.t3_price,
                    t1Hit: t.t1_hit || false,
                    t2Hit: t.t2_hit || false,
                    trailingSl: t.trailing_sl,
                    trailingTp: t.trailing_tp,
                    trailingActive: t.trailing_active || false,
                    trailSlCount: t.trail_sl_count || 0,
                    capitalProtectionActive: t.capital_protection_active || false,
                    originalQty: t.original_qty,
                    originalCapital: t.original_capital,
                    status: t.status.toLowerCase(),
                    activePnl: t.unrealized_pnl || 0,
                    activePnlPercent: t.unrealized_pnl_pct || 0,
                    totalPnl: t.realized_pnl || 0,
                    totalPnlPercent: t.realized_pnl_pct || 0,
                    exitReason: t.exit_reason,
                    exitPercent: t.exit_pct,
                    entryTime: new Date(t.timestamp),
                    exitTime: t.exit_time ? new Date(t.exit_time) : null,
                },
            });

            totalMigrated++;
        }
    }

    console.log(`\n✅ Migration complete!`);
    console.log(`   Migrated: ${totalMigrated}`);
    console.log(`   Skipped (duplicates): ${totalSkipped}`);

    await prisma.$disconnect();
}

migrateTrades().catch((e) => {
    console.error('Migration failed:', e);
    prisma.$disconnect();
    process.exit(1);
});
