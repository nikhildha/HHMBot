'use client';

import { useState, useEffect } from 'react';
import {
    FileText, Filter, RefreshCw, Shield, User, Bot, Settings,
    LogIn, LogOut, CreditCard, Zap, AlertTriangle, Info,
    ChevronDown, Search
} from 'lucide-react';

interface AuditEntry {
    id: string;
    timestamp: string;
    actor: string;
    actorRole: string;
    action: string;
    category: 'auth' | 'bot' | 'trade' | 'admin' | 'subscription' | 'system';
    details: string;
    ipAddress?: string;
    severity: 'info' | 'warning' | 'critical';
}

const MOCK_LOGS: AuditEntry[] = [
    { id: '1', timestamp: new Date().toISOString(), actor: 'admin@sentinel.app', actorRole: 'admin', action: 'Admin Login', category: 'auth', details: 'Admin logged in via credentials', severity: 'info' },
    { id: '2', timestamp: new Date(Date.now() - 300000).toISOString(), actor: 'testuser@sentinel.app', actorRole: 'user', action: 'Bot Created', category: 'bot', details: 'Created bot "HMM Alpha" in paper mode', severity: 'info' },
    { id: '3', timestamp: new Date(Date.now() - 600000).toISOString(), actor: 'system', actorRole: 'system', action: 'Database Seeded', category: 'system', details: 'Initial seed script executed — 2 users, 1 bot, 3 trades', severity: 'info' },
    { id: '4', timestamp: new Date(Date.now() - 900000).toISOString(), actor: 'admin@sentinel.app', actorRole: 'admin', action: 'Schema Migration', category: 'system', details: 'prisma db push executed — SQLite dev.db created', severity: 'warning' },
    { id: '5', timestamp: new Date(Date.now() - 1200000).toISOString(), actor: 'testuser@sentinel.app', actorRole: 'user', action: 'Trade Opened', category: 'trade', details: 'LONG on ETHUSDT @ 3245.50 — 2x leverage, conviction 0.78', severity: 'info' },
    { id: '6', timestamp: new Date(Date.now() - 1500000).toISOString(), actor: 'system', actorRole: 'system', action: 'Failed Login Attempt', category: 'auth', details: 'Invalid credentials for unknown@test.com from 192.168.1.1', severity: 'critical' },
    { id: '7', timestamp: new Date(Date.now() - 1800000).toISOString(), actor: 'testuser@sentinel.app', actorRole: 'user', action: 'Subscription Upgraded', category: 'subscription', details: 'Upgraded from Free → Pro plan', severity: 'info' },
    { id: '8', timestamp: new Date(Date.now() - 2400000).toISOString(), actor: 'admin@sentinel.app', actorRole: 'admin', action: 'User Role Changed', category: 'admin', details: 'Changed testuser@sentinel.app role from user to moderator', severity: 'warning' },
];

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
    auth: <LogIn className="w-4 h-4" />,
    bot: <Bot className="w-4 h-4" />,
    trade: <Zap className="w-4 h-4" />,
    admin: <Shield className="w-4 h-4" />,
    subscription: <CreditCard className="w-4 h-4" />,
    system: <Settings className="w-4 h-4" />,
};

const CATEGORY_COLORS: Record<string, string> = {
    auth: 'text-blue-400 bg-blue-500/10',
    bot: 'text-purple-400 bg-purple-500/10',
    trade: 'text-green-400 bg-green-500/10',
    admin: 'text-red-400 bg-red-500/10',
    subscription: 'text-amber-400 bg-amber-500/10',
    system: 'text-gray-400 bg-gray-500/10',
};

const SEVERITY_COLORS: Record<string, string> = {
    info: 'text-blue-400',
    warning: 'text-yellow-400',
    critical: 'text-red-400',
};

export default function AuditLog() {
    const [logs, setLogs] = useState<AuditEntry[]>(MOCK_LOGS);
    const [filter, setFilter] = useState<string>('all');
    const [searchQuery, setSearchQuery] = useState('');

    const filteredLogs = logs.filter(log => {
        const matchesCategory = filter === 'all' || log.category === filter;
        const matchesSearch = searchQuery === '' ||
            log.action.toLowerCase().includes(searchQuery.toLowerCase()) ||
            log.actor.toLowerCase().includes(searchQuery.toLowerCase()) ||
            log.details.toLowerCase().includes(searchQuery.toLowerCase());
        return matchesCategory && matchesSearch;
    });

    const categories = ['all', 'auth', 'bot', 'trade', 'admin', 'subscription', 'system'];

    return (
        <div className="space-y-6">
            {/* Info Banner */}
            <div className="flex items-center gap-3 p-4 rounded-xl bg-blue-500/5 border border-blue-500/10">
                <Info className="w-5 h-5 text-blue-400 shrink-0" />
                <p className="text-gray-400 text-sm">
                    Audit log tracks all administrative actions, login events, and system changes.
                    Logs will be persisted to the database once the audit model is added.
                </p>
            </div>

            {/* Search + Filter Bar */}
            <div className="flex flex-col md:flex-row gap-3">
                <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                    <input
                        type="text"
                        placeholder="Search logs..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 text-sm focus:outline-none focus:border-blue-500/50"
                    />
                </div>
                <div className="flex gap-1.5 flex-wrap">
                    {categories.map((cat) => (
                        <button
                            key={cat}
                            onClick={() => setFilter(cat)}
                            className={`px-3 py-2 rounded-lg text-xs font-medium capitalize transition ${filter === cat
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-white/5 text-gray-400 hover:bg-white/10'
                                }`}
                        >
                            {cat}
                        </button>
                    ))}
                </div>
            </div>

            {/* Log Entries */}
            <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                <div className="px-6 py-3 border-b border-white/10 flex items-center justify-between">
                    <h3 className="text-white font-semibold text-sm">
                        Activity Log ({filteredLogs.length} entries)
                    </h3>
                    <button className="text-gray-400 hover:text-white text-xs transition">
                        Export CSV
                    </button>
                </div>

                <div className="divide-y divide-white/5">
                    {filteredLogs.map((log) => (
                        <div key={log.id} className="px-6 py-4 hover:bg-white/[0.03] transition">
                            <div className="flex items-start gap-4">
                                {/* Category Icon */}
                                <div className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${CATEGORY_COLORS[log.category]}`}>
                                    {CATEGORY_ICONS[log.category]}
                                </div>

                                {/* Content */}
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-1">
                                        <p className="text-white font-medium text-sm">{log.action}</p>
                                        <span className={`text-xs ${SEVERITY_COLORS[log.severity]}`}>
                                            {log.severity === 'critical' && '⚠️'}
                                            {log.severity === 'warning' && '⚡'}
                                        </span>
                                    </div>
                                    <p className="text-gray-400 text-sm">{log.details}</p>
                                    <div className="flex items-center gap-3 mt-2">
                                        <span className="text-gray-500 text-xs flex items-center gap-1">
                                            {log.actorRole === 'admin' ? <Shield className="w-3 h-3" /> : <User className="w-3 h-3" />}
                                            {log.actor}
                                        </span>
                                        <span className="text-gray-600 text-xs">•</span>
                                        <span className="text-gray-500 text-xs">
                                            {new Date(log.timestamp).toLocaleString()}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                {filteredLogs.length === 0 && (
                    <div className="text-center py-12 text-gray-500">
                        <FileText className="w-10 h-10 mx-auto mb-3 opacity-30" />
                        <p>No log entries match your filter</p>
                    </div>
                )}
            </div>
        </div>
    );
}
