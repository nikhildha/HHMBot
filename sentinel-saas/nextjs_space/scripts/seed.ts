/**
 * Seed Script — Creates Admin + Test user accounts
 * 
 * Usage: npx tsx scripts/seed.ts
 */
import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Seeding database...\n');

  // ─── Admin Account ──────────────────────────────────────────────
  const adminPassword = await bcrypt.hash('Admin@2026', 12);
  const admin = await prisma.user.upsert({
    where: { email: 'admin@sentinel.app' },
    update: {},
    create: {
      email: 'admin@sentinel.app',
      name: 'Nikhil (Admin)',
      password: adminPassword,
      role: 'admin',
      subscription: {
        create: {
          tier: 'ultra',
          coinScans: 50,
          status: 'active',
          currentPeriodEnd: new Date('2030-12-31'),
        },
      },
    },
  });
  console.log('✅ Admin account created:');
  console.log('   Email:    admin@sentinel.app');
  console.log('   Password: Admin@2026');
  console.log('   Role:     admin');
  console.log('   Plan:     Ultra (50 coin scans)\n');

  // ─── Test User Account ──────────────────────────────────────────
  const testPassword = await bcrypt.hash('Test@1234', 12);
  const testUser = await prisma.user.upsert({
    where: { email: 'testuser@sentinel.app' },
    update: {},
    create: {
      email: 'testuser@sentinel.app',
      name: 'Test Trader',
      password: testPassword,
      role: 'user',
      subscription: {
        create: {
          tier: 'pro',
          coinScans: 15,
          status: 'active',
          currentPeriodEnd: new Date('2026-06-30'),
        },
      },
    },
  });
  console.log('✅ Test user account created:');
  console.log('   Email:    testuser@sentinel.app');
  console.log('   Password: Test@1234');
  console.log('   Role:     user');
  console.log('   Plan:     Pro (15 coin scans)\n');

  // ─── Demo Bot for Test User ─────────────────────────────────────
  const demoBot = await prisma.bot.create({
    data: {
      userId: testUser.id,
      name: 'Demo Paper Bot',
      exchange: 'binance',
      status: 'stopped',
      isActive: false,
      config: {
        create: {
          mode: 'paper',
          capitalPerTrade: 100,
          maxOpenTrades: 5,
          slMultiplier: 0.8,
          tpMultiplier: 1.0,
          maxLossPct: -15,
          multiTargetEnabled: true,
          t1Multiplier: 0.5,
          t2Multiplier: 1.0,
          t3Multiplier: 1.5,
          t1BookPct: 0.25,
          t2BookPct: 0.50,
          coinList: JSON.stringify(['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'SUIUSDT']),
        },
      },
      state: {
        create: {
          engineStatus: 'idle',
        },
      },
    },
  });
  console.log('✅ Demo bot created for test user:');
  console.log('   Name:     Demo Paper Bot');
  console.log('   Exchange: Binance');
  console.log('   Coins:    BTC, ETH, SOL, DOGE, SUI\n');

  // ─── Sample Trades for Test User ────────────────────────────────
  const now = new Date();
  const trades = [
    {
      botId: demoBot.id,
      coin: 'BTCUSDT',
      position: 'long',
      regime: 'bullish',
      confidence: 72,
      mode: 'paper',
      leverage: 10,
      capital: 100,
      quantity: 0.0012,
      entryPrice: 83245.50,
      currentPrice: 84120.30,
      stopLoss: 82500.00,
      takeProfit: 84950.00,
      t1Price: 83700.00,
      t2Price: 84200.00,
      t3Price: 84950.00,
      t1Hit: true,
      status: 'active',
      activePnl: 10.50,
      activePnlPercent: 10.5,
      entryTime: new Date(now.getTime() - 2 * 60 * 60 * 1000),
    },
    {
      botId: demoBot.id,
      coin: 'ETHUSDT',
      position: 'long',
      regime: 'bullish',
      confidence: 65,
      mode: 'paper',
      leverage: 8,
      capital: 100,
      quantity: 0.045,
      entryPrice: 2185.40,
      exitPrice: 2220.10,
      stopLoss: 2150.00,
      takeProfit: 2260.00,
      status: 'closed',
      exitReason: 'T3',
      totalPnl: 12.70,
      totalPnlPercent: 12.7,
      entryTime: new Date(now.getTime() - 24 * 60 * 60 * 1000),
      exitTime: new Date(now.getTime() - 20 * 60 * 60 * 1000),
    },
    {
      botId: demoBot.id,
      coin: 'DOGEUSDT',
      position: 'short',
      regime: 'bearish',
      confidence: 58,
      mode: 'paper',
      leverage: 5,
      capital: 100,
      quantity: 400,
      entryPrice: 0.2480,
      exitPrice: 0.2530,
      stopLoss: 0.2550,
      takeProfit: 0.2380,
      status: 'closed',
      exitReason: 'FIXED_SL',
      totalPnl: -10.08,
      totalPnlPercent: -10.08,
      entryTime: new Date(now.getTime() - 48 * 60 * 60 * 1000),
      exitTime: new Date(now.getTime() - 44 * 60 * 60 * 1000),
    },
  ];

  for (const t of trades) {
    await prisma.trade.create({ data: t });
  }
  console.log('✅ 3 sample trades created (1 active, 2 closed)\n');

  console.log('━'.repeat(50));
  console.log('🎉 Seeding complete!\n');
  console.log('📋 CREDENTIALS:');
  console.log('┌──────────────┬──────────────────────────┬──────────────┐');
  console.log('│ Role         │ Email                    │ Password     │');
  console.log('├──────────────┼──────────────────────────┼──────────────┤');
  console.log('│ 🔑 Admin     │ admin@sentinel.app       │ Admin@2026   │');
  console.log('│ 👤 Test User │ testuser@sentinel.app    │ Test@1234    │');
  console.log('└──────────────┴──────────────────────────┴──────────────┘');
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (e) => {
    console.error('Seed failed:', e);
    await prisma.$disconnect();
    process.exit(1);
  });
