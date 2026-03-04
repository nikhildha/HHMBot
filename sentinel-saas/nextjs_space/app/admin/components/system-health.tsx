'use client';

import { useState, useEffect } from 'react';
import {
    Cpu, HardDrive, Clock, Wifi, Database, Server,
    CheckCircle2, XCircle, AlertTriangle, RefreshCw,
    Activity, Gauge, MemoryStick, Globe
} from 'lucide-react';

interface HealthMetric {
    name: string;
    status: 'healthy' | 'degraded' | 'down';
    value: string;
    detail: string;
    icon: React.ReactNode;
}

export default function SystemHealth() {
    const [lastCheck, setLastCheck] = useState(new Date());
    const [checking, setChecking] = useState(false);

    const runHealthCheck = async () => {
        setChecking(true);
        // Simulate health check latency
        await new Promise(r => setTimeout(r, 1500));
        setLastCheck(new Date());
        setChecking(false);
    };

    const services: HealthMetric[] = [
        {
            name: 'Next.js Frontend',
            status: 'healthy',
            value: 'Running',
            detail: 'Port 4000 • v14.x • Response < 50ms',
            icon: <Globe className="w-5 h-5" />,
        },
        {
            name: 'SQLite Database',
            status: 'healthy',
            value: '2 tables active',
            detail: 'dev.db • 48 KB • Local filesystem',
            icon: <Database className="w-5 h-5" />,
        },
        {
            name: 'NextAuth.js',
            status: 'healthy',
            value: 'JWT Active',
            detail: 'Credentials provider • bcrypt hashing',
            icon: <CheckCircle2 className="w-5 h-5" />,
        },
        {
            name: 'Python Orchestrator',
            status: 'down',
            value: 'Offline',
            detail: 'Port 5000 • Not started',
            icon: <Server className="w-5 h-5" />,
        },
        {
            name: 'Encryption Service',
            status: 'healthy',
            value: 'AES-256-GCM',
            detail: 'ENCRYPTION_KEY configured • 32-byte hex',
            icon: <CheckCircle2 className="w-5 h-5" />,
        },
        {
            name: 'WebSocket/SSE Layer',
            status: 'down',
            value: 'Not Implemented',
            detail: 'Pending Phase 3 — real-time data push',
            icon: <Wifi className="w-5 h-5" />,
        },
    ];

    const systemMetrics = [
        { label: 'Uptime', value: '5h 12m', icon: <Clock className="w-4 h-4 text-blue-400" /> },
        { label: 'Memory', value: '124 MB', icon: <MemoryStick className="w-4 h-4 text-purple-400" /> },
        { label: 'CPU', value: '2.3%', icon: <Cpu className="w-4 h-4 text-green-400" /> },
        { label: 'Disk', value: '48 KB DB', icon: <HardDrive className="w-4 h-4 text-amber-400" /> },
    ];

    const healthyCt = services.filter(s => s.status === 'healthy').length;
    const degradedCt = services.filter(s => s.status === 'degraded').length;
    const downCt = services.filter(s => s.status === 'down').length;

    const statusColor = (status: string) => {
        switch (status) {
            case 'healthy': return 'text-green-400 bg-green-500/10 border-green-500/20';
            case 'degraded': return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20';
            case 'down': return 'text-red-400 bg-red-500/10 border-red-500/20';
            default: return 'text-gray-400 bg-gray-500/10 border-gray-500/20';
        }
    };

    const statusIcon = (status: string) => {
        switch (status) {
            case 'healthy': return <CheckCircle2 className="w-4 h-4 text-green-400" />;
            case 'degraded': return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
            case 'down': return <XCircle className="w-4 h-4 text-red-400" />;
            default: return null;
        }
    };

    const overallStatus = downCt > 0 ? 'degraded' : degradedCt > 0 ? 'warning' : 'operational';

    return (
        <div className="space-y-6">
            {/* Overall Status Banner */}
            <div className={`p-5 rounded-xl border ${overallStatus === 'operational'
                    ? 'bg-green-500/5 border-green-500/20'
                    : overallStatus === 'warning'
                        ? 'bg-yellow-500/5 border-yellow-500/20'
                        : 'bg-orange-500/5 border-orange-500/20'
                }`}>
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className={`w-4 h-4 rounded-full ${overallStatus === 'operational' ? 'bg-green-400' : 'bg-orange-400'
                            } animate-pulse`} />
                        <div>
                            <p className="text-white font-semibold">
                                {overallStatus === 'operational' ? 'All Systems Operational' : 'Partial Outage Detected'}
                            </p>
                            <p className="text-gray-400 text-sm mt-0.5">
                                {healthyCt}/{services.length} services healthy • Last check: {lastCheck.toLocaleTimeString()}
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={runHealthCheck}
                        disabled={checking}
                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 text-sm transition disabled:opacity-50"
                    >
                        <RefreshCw className={`w-4 h-4 ${checking ? 'animate-spin' : ''}`} />
                        Run Check
                    </button>
                </div>
            </div>

            {/* System Metrics Row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {systemMetrics.map((m) => (
                    <div key={m.label} className="bg-white/5 border border-white/10 rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-2">
                            {m.icon}
                            <span className="text-gray-400 text-xs">{m.label}</span>
                        </div>
                        <p className="text-xl font-bold text-white">{m.value}</p>
                    </div>
                ))}
            </div>

            {/* Service Status Grid */}
            <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                <div className="px-6 py-4 border-b border-white/10">
                    <h3 className="text-white font-semibold">Service Health</h3>
                </div>
                <div className="divide-y divide-white/5">
                    {services.map((service) => (
                        <div key={service.name} className="px-6 py-4 flex items-center justify-between hover:bg-white/[0.03] transition">
                            <div className="flex items-center gap-4">
                                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${statusColor(service.status)}`}>
                                    {service.icon}
                                </div>
                                <div>
                                    <p className="text-white font-medium">{service.name}</p>
                                    <p className="text-gray-500 text-sm">{service.detail}</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-3">
                                <span className="text-gray-300 text-sm font-mono">{service.value}</span>
                                <span className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${statusColor(service.status)}`}>
                                    {statusIcon(service.status)}
                                    {service.status}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Tech Stack Info */}
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <h3 className="text-white font-semibold mb-4">Technology Stack</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <TechRow label="Frontend" value="Next.js 14 (App Router)" />
                    <TechRow label="Database" value="SQLite (dev) → PostgreSQL (prod)" />
                    <TechRow label="ORM" value="Prisma 5.x" />
                    <TechRow label="Auth" value="NextAuth.js (JWT + bcrypt)" />
                    <TechRow label="Engine" value="Python 3.11 + HMM" />
                    <TechRow label="Encryption" value="AES-256-GCM for API keys" />
                    <TechRow label="Orchestrator" value="Python subprocess manager" />
                    <TechRow label="Deployment" value="Vercel (FE) + Railway (BE)" />
                </div>
            </div>
        </div>
    );
}

function TechRow({ label, value }: { label: string; value: string }) {
    return (
        <div className="flex justify-between items-center p-3 rounded-lg bg-white/5">
            <span className="text-gray-400 text-sm">{label}</span>
            <span className="text-white text-sm font-mono">{value}</span>
        </div>
    );
}
