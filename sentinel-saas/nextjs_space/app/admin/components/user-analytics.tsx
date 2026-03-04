'use client';

import { useState, useEffect } from 'react';
import {
    Users, UserPlus, TrendingUp, Calendar, BarChart3,
    ArrowUp, ArrowDown, Activity, Timer, Star
} from 'lucide-react';

interface UserData {
    id: string;
    email: string;
    name: string;
    role: string;
    createdAt: string;
    subscription: { tier: string; status: string } | null;
    _count: { bots: number; trades: number };
}

export default function UserAnalytics() {
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

    // Calculate analytics
    const now = new Date();
    const today = users.filter(u => {
        const d = new Date(u.createdAt);
        return d.toDateString() === now.toDateString();
    });

    const thisWeek = users.filter(u => {
        const d = new Date(u.createdAt);
        const diff = now.getTime() - d.getTime();
        return diff < 7 * 24 * 60 * 60 * 1000;
    });

    const thisMonth = users.filter(u => {
        const d = new Date(u.createdAt);
        return d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
    });

    const paidUsers = users.filter(u => u.subscription && u.subscription.tier !== 'free');
    const conversionRate = users.length > 0 ? ((paidUsers.length / users.length) * 100).toFixed(1) : '0';

    // Most active users by trade count
    const mostActive = [...users]
        .sort((a, b) => (b._count.trades || 0) - (a._count.trades || 0))
        .slice(0, 5);

    // Signup trend (by day) — last 7 days
    const signupsByDay: { day: string; count: number }[] = [];
    for (let i = 6; i >= 0; i--) {
        const target = new Date(now);
        target.setDate(target.getDate() - i);
        const dayStr = target.toLocaleDateString('en-US', { weekday: 'short' });
        const count = users.filter(u => {
            const d = new Date(u.createdAt);
            return d.toDateString() === target.toDateString();
        }).length;
        signupsByDay.push({ day: dayStr, count });
    }
    const maxSignups = Math.max(...signupsByDay.map(d => d.count), 1);

    // Tier distribution
    const tierDist = [
        { tier: 'Free', count: users.filter(u => !u.subscription || u.subscription.tier === 'free').length, color: 'bg-gray-400' },
        { tier: 'Pro', count: users.filter(u => u.subscription?.tier === 'pro').length, color: 'bg-amber-400' },
        { tier: 'Ultra', count: users.filter(u => u.subscription?.tier === 'ultra').length, color: 'bg-purple-400' },
    ];
    const totalForDist = Math.max(users.length, 1);

    return (
        <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard
                    icon={<Users className="w-5 h-5 text-blue-400" />}
                    label="Total Users"
                    value={users.length}
                    change="+100%"
                    positive
                />
                <MetricCard
                    icon={<UserPlus className="w-5 h-5 text-green-400" />}
                    label="New Today"
                    value={today.length}
                    change={`${thisWeek.length} this week`}
                    positive
                />
                <MetricCard
                    icon={<Star className="w-5 h-5 text-amber-400" />}
                    label="Paid Users"
                    value={paidUsers.length}
                    change={`${conversionRate}% conversion`}
                    positive={parseFloat(conversionRate) > 0}
                />
                <MetricCard
                    icon={<Activity className="w-5 h-5 text-purple-400" />}
                    label="Avg Trades/User"
                    value={(users.reduce((s, u) => s + (u._count.trades || 0), 0) / Math.max(users.length, 1)).toFixed(1)}
                    change="per user"
                    positive
                />
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Signups Chart */}
                <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                    <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                        <Calendar className="w-4 h-4 text-blue-400" />
                        Signups — Last 7 Days
                    </h3>
                    <div className="flex items-end justify-between gap-2 h-32">
                        {signupsByDay.map((d, i) => (
                            <div key={i} className="flex flex-col items-center flex-1 gap-1">
                                <span className="text-xs text-white font-mono">{d.count}</span>
                                <div
                                    className="w-full bg-gradient-to-t from-blue-600 to-blue-400 rounded-t-md transition-all"
                                    style={{ height: `${(d.count / maxSignups) * 100}%`, minHeight: d.count > 0 ? '8px' : '2px' }}
                                />
                                <span className="text-xs text-gray-500">{d.day}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Plan Distribution */}
                <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                    <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-purple-400" />
                        Plan Distribution
                    </h3>
                    <div className="space-y-4">
                        {tierDist.map((t) => (
                            <div key={t.tier}>
                                <div className="flex justify-between items-center mb-1.5">
                                    <span className="text-gray-300 text-sm">{t.tier}</span>
                                    <span className="text-white font-medium text-sm">{t.count} ({((t.count / totalForDist) * 100).toFixed(0)}%)</span>
                                </div>
                                <div className="w-full bg-white/5 rounded-full h-2.5 overflow-hidden">
                                    <div
                                        className={`${t.color} h-full rounded-full transition-all duration-700`}
                                        style={{ width: `${(t.count / totalForDist) * 100}%`, minWidth: t.count > 0 ? '4px' : '0' }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Donut Visual */}
                    <div className="flex items-center justify-center mt-6 gap-6">
                        {tierDist.map((t) => (
                            <div key={t.tier} className="flex items-center gap-2">
                                <div className={`w-3 h-3 rounded-full ${t.color}`} />
                                <span className="text-gray-400 text-xs">{t.tier}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Most Active Users */}
            <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                <div className="px-6 py-4 border-b border-white/10">
                    <h3 className="text-white font-semibold flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-green-400" />
                        Most Active Users
                    </h3>
                </div>
                <div className="divide-y divide-white/5">
                    {mostActive.map((user, i) => (
                        <div key={user.id} className="px-6 py-3 flex items-center justify-between hover:bg-white/[0.03] transition">
                            <div className="flex items-center gap-4">
                                <span className={`w-7 h-7 rounded-full flex items-center justify-center text-sm font-bold ${i === 0 ? 'bg-amber-500/20 text-amber-400'
                                        : i === 1 ? 'bg-gray-400/20 text-gray-300'
                                            : i === 2 ? 'bg-orange-500/20 text-orange-400'
                                                : 'bg-white/5 text-gray-500'
                                    }`}>
                                    {i + 1}
                                </span>
                                <div>
                                    <p className="text-white text-sm font-medium">{user.name || user.email}</p>
                                    <p className="text-gray-500 text-xs">{user.email}</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-6">
                                <div className="text-right">
                                    <p className="text-white text-sm font-mono">{user._count.trades}</p>
                                    <p className="text-gray-500 text-xs">trades</p>
                                </div>
                                <div className="text-right">
                                    <p className="text-white text-sm font-mono">{user._count.bots}</p>
                                    <p className="text-gray-500 text-xs">bots</p>
                                </div>
                                <span className={`px-2 py-0.5 rounded-full text-xs ${user.subscription?.tier === 'ultra' ? 'bg-purple-500/10 text-purple-400'
                                        : user.subscription?.tier === 'pro' ? 'bg-amber-500/10 text-amber-400'
                                            : 'bg-gray-500/10 text-gray-400'
                                    }`}>
                                    {user.subscription?.tier || 'free'}
                                </span>
                            </div>
                        </div>
                    ))}

                    {mostActive.length === 0 && (
                        <div className="text-center py-8 text-gray-500">
                            <Users className="w-8 h-8 mx-auto mb-2 opacity-30" />
                            <p>No user data available</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function MetricCard({ icon, label, value, change, positive }: {
    icon: React.ReactNode;
    label: string;
    value: string | number;
    change: string;
    positive: boolean;
}) {
    return (
        <div className="bg-white/5 border border-white/10 rounded-xl p-5">
            <div className="flex items-center justify-between mb-3">
                <span className="text-gray-400 text-sm">{label}</span>
                {icon}
            </div>
            <p className="text-2xl font-bold text-white">{value}</p>
            <div className="flex items-center gap-1 mt-1">
                {positive ? (
                    <ArrowUp className="w-3 h-3 text-green-400" />
                ) : (
                    <ArrowDown className="w-3 h-3 text-red-400" />
                )}
                <span className={`text-xs ${positive ? 'text-green-400' : 'text-red-400'}`}>{change}</span>
            </div>
        </div>
    );
}
