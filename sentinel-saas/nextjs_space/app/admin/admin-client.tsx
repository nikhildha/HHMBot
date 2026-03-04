'use client';

import { useSession } from 'next-auth/react';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import {
    Users, Bot, BarChart3, Activity, Shield, Settings,
    ArrowLeft, RefreshCw, TrendingUp, TrendingDown,
    AlertTriangle, CheckCircle2, XCircle,
    FileText, CreditCard, Heart, LineChart, DollarSign
} from 'lucide-react';

// Tab components
import EngineControl from './components/engine-control';
import AuditLog from './components/audit-log';
import SubscriptionMgmt from './components/subscription-mgmt';
import SystemHealth from './components/system-health';
import UserAnalytics from './components/user-analytics';
import RevenueDashboard from './components/revenue-dashboard';
import KernelView from './components/kernel-view';

interface UserData {
    id: string;
    email: string;
    name: string;
    role: string;
    createdAt: string;
    subscription: { tier: string; status: string; coinScans: number } | null;
    _count: { bots: number; trades: number };
}

interface SystemStats {
    totalUsers: number;
    activeSubscriptions: number;
    totalBots: number;
    activeBots: number;
    totalTrades: number;
    activeTrades: number;
    totalPnl: number;
    revenueEstimate: number;
}

type TabId = 'overview' | 'users' | 'engine' | 'kernel' | 'subscriptions' | 'analytics' | 'revenue' | 'audit' | 'health';

export default function AdminDashboard() {
    const { data: session } = useSession();
    const [stats, setStats] = useState<SystemStats | null>(null);
    const [users, setUsers] = useState<UserData[]>([]);
    const [activeTab, setActiveTab] = useState<TabId>('overview');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        setLoading(true);
        try {
            const [statsRes, usersRes] = await Promise.all([
                fetch('/api/admin/stats'),
                fetch('/api/admin/users'),
            ]);
            if (statsRes.ok) setStats(await statsRes.json());
            if (usersRes.ok) setUsers(await usersRes.json());
        } catch (e) {
            console.error('Failed to fetch admin data:', e);
        }
        setLoading(false);
    };

    const tabs = [
        { id: 'overview' as TabId, label: 'Overview', icon: BarChart3 },
        { id: 'users' as TabId, label: 'Users', icon: Users },
        { id: 'engine' as TabId, label: 'Engine', icon: Bot },
        { id: 'kernel' as TabId, label: 'Kernel', icon: Activity },
        { id: 'subscriptions' as TabId, label: 'Subs', icon: CreditCard },
        { id: 'analytics' as TabId, label: 'Analytics', icon: LineChart },
        { id: 'revenue' as TabId, label: 'Revenue', icon: DollarSign },
        { id: 'audit' as TabId, label: 'Audit', icon: FileText },
        { id: 'health' as TabId, label: 'Health', icon: Heart },
    ];

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 pt-6 pb-12">
            {/* Header */}
            <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-4">
                    <Link href="/dashboard" className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition">
                        <ArrowLeft className="w-5 h-5 text-gray-400" />
                    </Link>
                    <div>
                        <div className="flex items-center gap-2">
                            <Shield className="w-6 h-6 text-[var(--color-primary)]" />
                            <h1 className="text-3xl font-bold text-white">Admin Panel</h1>
                        </div>
                        <p className="text-gray-400 text-sm mt-1">System management & monitoring</p>
                    </div>
                </div>
                <button
                    onClick={fetchData}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 transition"
                >
                    <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                    Refresh
                </button>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 mb-8 bg-white/5 rounded-xl p-1 overflow-x-auto">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${activeTab === tab.id
                            ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                            : 'text-gray-400 hover:text-white hover:bg-white/5'
                            }`}
                    >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Overview Tab */}
            {activeTab === 'overview' && stats && (
                <div className="space-y-6">
                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <StatCard
                            label="Total Users"
                            value={stats.totalUsers}
                            icon={<Users className="w-5 h-5 text-blue-400" />}
                            color="blue"
                        />
                        <StatCard
                            label="Active Subscriptions"
                            value={stats.activeSubscriptions}
                            icon={<CheckCircle2 className="w-5 h-5 text-green-400" />}
                            color="green"
                        />
                        <StatCard
                            label="Total Bots"
                            value={stats.totalBots}
                            subtitle={`${stats.activeBots} active`}
                            icon={<Bot className="w-5 h-5 text-purple-400" />}
                            color="purple"
                        />
                        <StatCard
                            label="Total Trades"
                            value={stats.totalTrades}
                            subtitle={`${stats.activeTrades} active`}
                            icon={<Activity className="w-5 h-5 text-orange-400" />}
                            color="orange"
                        />
                    </div>

                    {/* Revenue & PNL */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                            <h3 className="text-gray-400 text-sm font-medium mb-2">Platform-Wide PNL</h3>
                            <p className={`text-3xl font-bold ${stats.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {stats.totalPnl >= 0 ? '+' : ''}${stats.totalPnl.toFixed(2)}
                            </p>
                            <div className="flex items-center gap-1 mt-1">
                                {stats.totalPnl >= 0
                                    ? <TrendingUp className="w-4 h-4 text-green-400" />
                                    : <TrendingDown className="w-4 h-4 text-red-400" />}
                                <span className="text-xs text-gray-500">Across all users</span>
                            </div>
                        </div>
                        <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                            <h3 className="text-gray-400 text-sm font-medium mb-2">Estimated Monthly Revenue</h3>
                            <p className="text-3xl font-bold text-emerald-400">
                                ₹{stats.revenueEstimate.toLocaleString('en-IN')}
                            </p>
                            <span className="text-xs text-gray-500">Based on active subscriptions</span>
                        </div>
                    </div>

                    {/* Quick Links */}
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        <QuickLink label="Manage Bots" icon={<Bot className="w-5 h-5" />} onClick={() => setActiveTab('engine')} color="purple" />
                        <QuickLink label="View Analytics" icon={<LineChart className="w-5 h-5" />} onClick={() => setActiveTab('analytics')} color="blue" />
                        <QuickLink label="Revenue Details" icon={<DollarSign className="w-5 h-5" />} onClick={() => setActiveTab('revenue')} color="emerald" />
                    </div>
                </div>
            )}

            {/* Users Tab */}
            {activeTab === 'users' && (
                <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                    <div className="px-6 py-4 border-b border-white/10">
                        <h3 className="text-white font-semibold">All Users ({users.length})</h3>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-white/5">
                                    <th className="text-left px-6 py-3 text-xs font-medium text-gray-400 uppercase">User</th>
                                    <th className="text-left px-6 py-3 text-xs font-medium text-gray-400 uppercase">Role</th>
                                    <th className="text-left px-6 py-3 text-xs font-medium text-gray-400 uppercase">Plan</th>
                                    <th className="text-left px-6 py-3 text-xs font-medium text-gray-400 uppercase">Bots</th>
                                    <th className="text-left px-6 py-3 text-xs font-medium text-gray-400 uppercase">Trades</th>
                                    <th className="text-left px-6 py-3 text-xs font-medium text-gray-400 uppercase">Joined</th>
                                </tr>
                            </thead>
                            <tbody>
                                {users.map((user) => (
                                    <tr key={user.id} className="border-b border-white/5 hover:bg-white/5 transition">
                                        <td className="px-6 py-4">
                                            <div>
                                                <p className="text-white font-medium">{user.name || 'No name'}</p>
                                                <p className="text-gray-400 text-sm">{user.email}</p>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium ${user.role === 'admin'
                                                ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                                                : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                                                }`}>
                                                {user.role === 'admin' && <Shield className="w-3 h-3" />}
                                                {user.role}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${user.subscription?.tier === 'ultra'
                                                ? 'bg-purple-500/10 text-purple-400'
                                                : user.subscription?.tier === 'pro'
                                                    ? 'bg-amber-500/10 text-amber-400'
                                                    : 'bg-gray-500/10 text-gray-400'
                                                }`}>
                                                {user.subscription?.tier || 'none'}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-gray-300">{user._count.bots}</td>
                                        <td className="px-6 py-4 text-gray-300">{user._count.trades}</td>
                                        <td className="px-6 py-4 text-gray-400 text-sm">
                                            {new Date(user.createdAt).toLocaleDateString()}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Delegated Tabs */}
            {activeTab === 'engine' && <EngineControl />}
            {activeTab === 'kernel' && <KernelView />}
            {activeTab === 'subscriptions' && <SubscriptionMgmt />}
            {activeTab === 'analytics' && <UserAnalytics />}
            {activeTab === 'revenue' && <RevenueDashboard />}
            {activeTab === 'audit' && <AuditLog />}
            {activeTab === 'health' && <SystemHealth />}

            {loading && !stats && (
                <div className="flex items-center justify-center py-20">
                    <RefreshCw className="w-8 h-8 text-gray-500 animate-spin" />
                </div>
            )}
        </div>
    );
}

function StatCard({ label, value, subtitle, icon, color }: {
    label: string;
    value: number;
    subtitle?: string;
    icon: React.ReactNode;
    color: string;
}) {
    return (
        <div className="bg-white/5 border border-white/10 rounded-xl p-5">
            <div className="flex items-center justify-between mb-3">
                <span className="text-gray-400 text-sm">{label}</span>
                {icon}
            </div>
            <p className="text-2xl font-bold text-white">{value}</p>
            {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
        </div>
    );
}

function QuickLink({ label, icon, onClick, color }: {
    label: string;
    icon: React.ReactNode;
    onClick: () => void;
    color: string;
}) {
    return (
        <button
            onClick={onClick}
            className={`flex items-center gap-3 p-4 rounded-xl bg-${color}-500/5 border border-${color}-500/10 hover:bg-${color}-500/10 text-${color}-400 transition group`}
        >
            {icon}
            <span className="text-sm font-medium">{label}</span>
        </button>
    );
}
