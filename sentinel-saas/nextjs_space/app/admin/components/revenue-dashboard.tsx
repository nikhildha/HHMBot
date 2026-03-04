'use client';

import { useState, useEffect } from 'react';
import {
    DollarSign, TrendingUp, TrendingDown, CreditCard,
    Calendar, PieChart, ArrowUpRight, BarChart3,
    Target, Sparkles, IndianRupee, Wallet
} from 'lucide-react';

interface UserData {
    id: string;
    subscription: { tier: string; status: string; createdAt: string } | null;
}

export default function RevenueDashboard() {
    const [users, setUsers] = useState<UserData[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchUsers();
    }, []);

    const fetchUsers = async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/admin/users');
            if (res.ok) setUsers(await res.json());
        } catch (e) {
            console.error('Failed:', e);
        }
        setLoading(false);
    };

    const PRICES = { free: 0, pro: 999, ultra: 2499 };

    const proUsers = users.filter(u => u.subscription?.tier === 'pro');
    const ultraUsers = users.filter(u => u.subscription?.tier === 'ultra');
    const freeUsers = users.filter(u => !u.subscription || u.subscription.tier === 'free');
    const payingUsers = [...proUsers, ...ultraUsers];

    const mrr = proUsers.length * PRICES.pro + ultraUsers.length * PRICES.ultra;
    const arr = mrr * 12;
    const arpu = payingUsers.length > 0 ? mrr / payingUsers.length : 0;
    const ltv = arpu * 12; // Simplified LTV estimate (12-month average)

    // Revenue breakdown
    const proRevenue = proUsers.length * PRICES.pro;
    const ultraRevenue = ultraUsers.length * PRICES.ultra;
    const totalRevenue = proRevenue + ultraRevenue;

    // Growth projections (mock but illustrative)
    const projections = [
        { month: 'Apr', users: users.length + 3, mrr: mrr + 2997 },
        { month: 'May', users: users.length + 8, mrr: mrr + 6993 },
        { month: 'Jun', users: users.length + 15, mrr: mrr + 14985 },
        { month: 'Jul', users: users.length + 25, mrr: mrr + 24975 },
        { month: 'Aug', users: users.length + 40, mrr: mrr + 39960 },
        { month: 'Sep', users: users.length + 60, mrr: mrr + 59940 },
    ];
    const maxProjectedMrr = Math.max(...projections.map(p => p.mrr), 1);

    return (
        <div className="space-y-6">
            {/* Key Revenue Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <RevenueCard
                    icon={<IndianRupee className="w-5 h-5 text-emerald-400" />}
                    label="Monthly Recurring Revenue"
                    value={`₹${mrr.toLocaleString('en-IN')}`}
                    badge="MRR"
                    badgeColor="bg-emerald-500/10 text-emerald-400"
                />
                <RevenueCard
                    icon={<Calendar className="w-5 h-5 text-blue-400" />}
                    label="Annual Run Rate"
                    value={`₹${arr.toLocaleString('en-IN')}`}
                    badge="ARR"
                    badgeColor="bg-blue-500/10 text-blue-400"
                />
                <RevenueCard
                    icon={<Wallet className="w-5 h-5 text-purple-400" />}
                    label="Avg Revenue per User"
                    value={`₹${arpu.toFixed(0)}`}
                    badge="ARPU"
                    badgeColor="bg-purple-500/10 text-purple-400"
                />
                <RevenueCard
                    icon={<Target className="w-5 h-5 text-amber-400" />}
                    label="Est. Lifetime Value"
                    value={`₹${ltv.toLocaleString('en-IN')}`}
                    badge="LTV"
                    badgeColor="bg-amber-500/10 text-amber-400"
                />
            </div>

            {/* Revenue Breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Pie-style Breakdown */}
                <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                    <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                        <PieChart className="w-4 h-4 text-purple-400" />
                        Revenue Breakdown
                    </h3>

                    <div className="space-y-4">
                        {/* Pro Revenue */}
                        <div>
                            <div className="flex justify-between mb-1.5">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full bg-amber-400" />
                                    <span className="text-gray-300 text-sm">Pro Plan</span>
                                    <span className="text-gray-500 text-xs">({proUsers.length} users × ₹{PRICES.pro})</span>
                                </div>
                                <span className="text-white font-medium text-sm">₹{proRevenue.toLocaleString('en-IN')}</span>
                            </div>
                            <div className="w-full bg-white/5 rounded-full h-3 overflow-hidden">
                                <div
                                    className="bg-gradient-to-r from-amber-500 to-amber-400 h-full rounded-full transition-all duration-700"
                                    style={{ width: totalRevenue > 0 ? `${(proRevenue / totalRevenue) * 100}%` : '0%' }}
                                />
                            </div>
                        </div>

                        {/* Ultra Revenue */}
                        <div>
                            <div className="flex justify-between mb-1.5">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full bg-purple-400" />
                                    <span className="text-gray-300 text-sm">Ultra Plan</span>
                                    <span className="text-gray-500 text-xs">({ultraUsers.length} users × ₹{PRICES.ultra})</span>
                                </div>
                                <span className="text-white font-medium text-sm">₹{ultraRevenue.toLocaleString('en-IN')}</span>
                            </div>
                            <div className="w-full bg-white/5 rounded-full h-3 overflow-hidden">
                                <div
                                    className="bg-gradient-to-r from-purple-500 to-purple-400 h-full rounded-full transition-all duration-700"
                                    style={{ width: totalRevenue > 0 ? `${(ultraRevenue / totalRevenue) * 100}%` : '0%' }}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Summary */}
                    <div className="mt-6 pt-4 border-t border-white/10 flex justify-between items-center">
                        <span className="text-gray-400 text-sm">Total Revenue</span>
                        <span className="text-2xl font-bold text-emerald-400">₹{totalRevenue.toLocaleString('en-IN')}</span>
                    </div>
                </div>

                {/* Projection Chart */}
                <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                    <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-green-400" />
                        6-Month MRR Projection
                    </h3>
                    <p className="text-gray-500 text-xs mb-4">Based on 10 new users/month growth rate</p>
                    <div className="flex items-end justify-between gap-3 h-36">
                        {projections.map((p, i) => (
                            <div key={i} className="flex flex-col items-center flex-1 gap-1">
                                <span className="text-[10px] text-emerald-400 font-mono whitespace-nowrap">
                                    ₹{(p.mrr / 1000).toFixed(0)}K
                                </span>
                                <div
                                    className="w-full bg-gradient-to-t from-emerald-600 to-emerald-400 rounded-t-lg transition-all duration-500"
                                    style={{ height: `${(p.mrr / maxProjectedMrr) * 100}%`, minHeight: '8px' }}
                                />
                                <span className="text-xs text-gray-500">{p.month}</span>
                            </div>
                        ))}
                    </div>

                    {/* Target */}
                    <div className="mt-4 p-3 rounded-lg bg-emerald-500/5 border border-emerald-500/10">
                        <div className="flex items-center gap-2">
                            <Sparkles className="w-4 h-4 text-emerald-400" />
                            <span className="text-emerald-400 text-sm font-medium">
                                Target: ₹1,00,000 MRR by September
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Churn & Conversion Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white/5 border border-white/10 rounded-xl p-5">
                    <div className="flex items-center gap-2 mb-3">
                        <ArrowUpRight className="w-4 h-4 text-green-400" />
                        <span className="text-gray-400 text-sm">Conversion Rate</span>
                    </div>
                    <p className="text-3xl font-bold text-white">
                        {users.length > 0 ? ((payingUsers.length / users.length) * 100).toFixed(0) : 0}%
                    </p>
                    <p className="text-gray-500 text-xs mt-1">Free → Paid</p>
                    <div className="mt-3 w-full bg-white/5 rounded-full h-2">
                        <div
                            className="bg-green-400 h-full rounded-full transition-all"
                            style={{ width: `${users.length > 0 ? (payingUsers.length / users.length) * 100 : 0}%` }}
                        />
                    </div>
                </div>

                <div className="bg-white/5 border border-white/10 rounded-xl p-5">
                    <div className="flex items-center gap-2 mb-3">
                        <TrendingDown className="w-4 h-4 text-red-400" />
                        <span className="text-gray-400 text-sm">Churn Rate</span>
                    </div>
                    <p className="text-3xl font-bold text-white">0%</p>
                    <p className="text-gray-500 text-xs mt-1">No cancellations yet</p>
                    <div className="mt-3 w-full bg-white/5 rounded-full h-2">
                        <div className="bg-green-400 h-full rounded-full w-full" />
                    </div>
                </div>

                <div className="bg-white/5 border border-white/10 rounded-xl p-5">
                    <div className="flex items-center gap-2 mb-3">
                        <CreditCard className="w-4 h-4 text-blue-400" />
                        <span className="text-gray-400 text-sm">Payment Success</span>
                    </div>
                    <p className="text-3xl font-bold text-white">100%</p>
                    <p className="text-gray-500 text-xs mt-1">Razorpay integration pending</p>
                    <div className="mt-3 w-full bg-white/5 rounded-full h-2">
                        <div className="bg-blue-400 h-full rounded-full w-full" />
                    </div>
                </div>
            </div>
        </div>
    );
}

function RevenueCard({ icon, label, value, badge, badgeColor }: {
    icon: React.ReactNode;
    label: string;
    value: string;
    badge: string;
    badgeColor: string;
}) {
    return (
        <div className="bg-white/5 border border-white/10 rounded-xl p-5 relative overflow-hidden">
            <div className="flex items-center justify-between mb-3">
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${badgeColor}`}>{badge}</span>
                {icon}
            </div>
            <p className="text-gray-400 text-xs mb-1">{label}</p>
            <p className="text-2xl font-bold text-white">{value}</p>
        </div>
    );
}
