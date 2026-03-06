'use client';

import { useState, useEffect, useRef } from 'react';
import {
    Bot, Power, PowerOff, RefreshCw, Play, Pause, Trash2,
    Activity, Clock, AlertTriangle, CheckCircle2, XCircle,
    Cpu, Zap, Signal, Terminal, Loader2
} from 'lucide-react';

// ─── Local Engine State ──────────────────────────────────────────────────────

interface EngineState {
    status: 'running' | 'stopped' | 'unknown';
    pid: number | null;
    uptime: string | null;
    logs: string[];
}

// ─── Bot / Orchestrator types (existing) ─────────────────────────────────────

interface BotInfo {
    id: string;
    name: string;
    isActive: boolean;
    mode: string;
    createdAt: string;
    user: { name: string; email: string };
    _count: { trades: number };
}

export default function EngineControl() {
    // Local engine state
    const [engine, setEngine] = useState<EngineState>({
        status: 'unknown', pid: null, uptime: null, logs: [],
    });
    const [engineActionLoading, setEngineActionLoading] = useState(false);
    const logRef = useRef<HTMLDivElement>(null);

    // Existing bot list (orchestrator)
    const [bots, setBots] = useState<BotInfo[]>([]);
    const [loading, setLoading] = useState(true);
    const [actionLoading, setActionLoading] = useState<string | null>(null);
    const [orchestratorOnline, setOrchestratorOnline] = useState(false);

    useEffect(() => {
        fetchEngineStatus();
        fetchBots();
        checkOrchestrator();
    }, []);

    // Auto-poll engine status every 5s when running
    useEffect(() => {
        if (engine.status !== 'running') return;
        const interval = setInterval(fetchEngineStatus, 5000);
        return () => clearInterval(interval);
    }, [engine.status]);

    // Auto-scroll logs
    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight;
        }
    }, [engine.logs]);

    // ─── Local Engine Actions ────────────────────────────────────────────────

    const fetchEngineStatus = async () => {
        try {
            const res = await fetch('/api/admin/engine');
            if (res.ok) {
                const data = await res.json();
                setEngine(data);
            }
        } catch {
            setEngine(prev => ({ ...prev, status: 'unknown' }));
        }
    };

    const controlEngine = async (action: 'start' | 'stop') => {
        setEngineActionLoading(true);
        try {
            const res = await fetch('/api/admin/engine', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action }),
            });
            if (res.ok) {
                // Wait a moment for process to spin up / down, then refresh
                await new Promise(r => setTimeout(r, 1500));
                await fetchEngineStatus();
            }
        } catch (e) {
            console.error(`Failed to ${action} engine:`, e);
        }
        setEngineActionLoading(false);
    };

    // ─── Existing Orchestrator / Bot Actions ─────────────────────────────────

    const fetchBots = async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/admin/bots');
            if (res.ok) setBots(await res.json());
        } catch (e) {
            console.error('Failed to fetch bots:', e);
        }
        setLoading(false);
    };

    const checkOrchestrator = async () => {
        try {
            const res = await fetch('/api/admin/orchestrator/health');
            setOrchestratorOnline(res.ok);
        } catch {
            setOrchestratorOnline(false);
        }
    };

    const controlBot = async (botId: string, action: 'start' | 'stop' | 'restart') => {
        setActionLoading(botId);
        try {
            await fetch('/api/admin/orchestrator/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ botId, action }),
            });
            await fetchBots();
        } catch (e) {
            console.error(`Failed to ${action} bot:`, e);
        }
        setActionLoading(null);
    };

    const activeBots = bots.filter(b => b.isActive);
    const inactiveBots = bots.filter(b => !b.isActive);

    const isRunning = engine.status === 'running';

    return (
        <div className="space-y-6">
            {/* ═══ LOCAL ENGINE CARD ═══════════════════════════════════════════ */}
            <div className={`rounded-2xl border-2 p-6 transition-all ${isRunning
                    ? 'bg-green-500/5 border-green-500/30 shadow-lg shadow-green-500/5'
                    : 'bg-white/5 border-white/10'
                }`}>
                <div className="flex items-center justify-between mb-5">
                    <div className="flex items-center gap-4">
                        <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${isRunning ? 'bg-green-500/15' : 'bg-gray-500/10'
                            }`}>
                            <Cpu className={`w-6 h-6 ${isRunning ? 'text-green-400' : 'text-gray-500'}`} />
                        </div>
                        <div>
                            <div className="flex items-center gap-3">
                                <h2 className="text-xl font-bold text-white">Local Engine</h2>
                                <div className="flex items-center gap-2">
                                    <div className={`w-2.5 h-2.5 rounded-full ${isRunning ? 'bg-green-400 animate-pulse' : 'bg-gray-500'
                                        }`} />
                                    <span className={`text-sm font-medium ${isRunning ? 'text-green-400' : 'text-gray-400'
                                        }`}>
                                        {isRunning ? 'Running' : engine.status === 'unknown' ? 'Checking...' : 'Stopped'}
                                    </span>
                                </div>
                            </div>
                            <p className="text-gray-500 text-sm mt-0.5">
                                {isRunning
                                    ? `PID ${engine.pid} · Uptime: ${engine.uptime || '—'}`
                                    : 'main.py — RegimeMaster Bot'
                                }
                            </p>
                        </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex items-center gap-3">
                        <button
                            onClick={fetchEngineStatus}
                            className="p-2.5 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 transition"
                            title="Refresh status"
                        >
                            <RefreshCw className="w-4 h-4" />
                        </button>

                        {isRunning ? (
                            <button
                                onClick={() => controlEngine('stop')}
                                disabled={engineActionLoading}
                                className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition font-medium disabled:opacity-50"
                            >
                                {engineActionLoading ? (
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                ) : (
                                    <PowerOff className="w-4 h-4" />
                                )}
                                Stop Engine
                            </button>
                        ) : (
                            <button
                                onClick={() => controlEngine('start')}
                                disabled={engineActionLoading || engine.status === 'unknown'}
                                className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500/10 text-green-400 border border-green-500/20 hover:bg-green-500/20 transition font-medium disabled:opacity-50"
                            >
                                {engineActionLoading ? (
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                ) : (
                                    <Play className="w-4 h-4" />
                                )}
                                Start Engine
                            </button>
                        )}
                    </div>
                </div>

                {/* Log Console */}
                {engine.logs.length > 0 && (
                    <div className="mt-4">
                        <div className="flex items-center gap-2 mb-2">
                            <Terminal className="w-3.5 h-3.5 text-gray-500" />
                            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Engine Logs</span>
                        </div>
                        <div
                            ref={logRef}
                            className="bg-black/40 border border-white/5 rounded-xl p-4 max-h-48 overflow-y-auto font-mono text-xs text-gray-400 space-y-0.5"
                        >
                            {engine.logs.map((line, i) => (
                                <div key={i} className={
                                    line.includes('ERROR') || line.includes('⚠️')
                                        ? 'text-red-400'
                                        : line.includes('🚀') || line.includes('✅') || line.includes('🔥')
                                            ? 'text-green-400'
                                            : line.includes('🧠') || line.includes('📊')
                                                ? 'text-blue-400'
                                                : 'text-gray-400'
                                }>
                                    {line}
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* ═══ ORCHESTRATOR STATUS (existing) ═════════════════════════════ */}
            <div className={`flex items-center justify-between p-4 rounded-xl border ${orchestratorOnline
                ? 'bg-green-500/5 border-green-500/20'
                : 'bg-red-500/5 border-red-500/20'
                }`}>
                <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${orchestratorOnline ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                    <div>
                        <p className="text-white font-medium">Python Orchestrator</p>
                        <p className="text-gray-400 text-sm">
                            {orchestratorOnline ? 'Connected on port 5000' : 'Offline — not needed for local dev'}
                        </p>
                    </div>
                </div>
                <button
                    onClick={checkOrchestrator}
                    className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 text-sm transition"
                >
                    <RefreshCw className="w-4 h-4" />
                </button>
            </div>

            {/* Summary Row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MiniStat icon={<Bot className="w-4 h-4 text-blue-400" />} label="Total Bots" value={bots.length} />
                <MiniStat icon={<Play className="w-4 h-4 text-green-400" />} label="Running" value={activeBots.length} />
                <MiniStat icon={<Pause className="w-4 h-4 text-yellow-400" />} label="Stopped" value={inactiveBots.length} />
                <MiniStat icon={<Signal className="w-4 h-4 text-purple-400" />} label="Orchestrator" value={orchestratorOnline ? 'Online' : 'Offline'} />
            </div>

            {/* Bot Cards */}
            {loading ? (
                <div className="flex items-center justify-center py-12">
                    <RefreshCw className="w-6 h-6 text-gray-500 animate-spin" />
                </div>
            ) : bots.length === 0 ? (
                <div className="text-center py-12 text-gray-400">
                    <Bot className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <p>No bots created yet</p>
                </div>
            ) : (
                <div className="space-y-3">
                    {bots.map((bot) => (
                        <div
                            key={bot.id}
                            className="bg-white/5 border border-white/10 rounded-xl p-5 hover:bg-white/[0.07] transition"
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${bot.isActive ? 'bg-green-500/10' : 'bg-gray-500/10'
                                        }`}>
                                        <Bot className={`w-5 h-5 ${bot.isActive ? 'text-green-400' : 'text-gray-500'}`} />
                                    </div>
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <p className="text-white font-medium">{bot.name}</p>
                                            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${bot.isActive
                                                ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                                                : 'bg-gray-500/10 text-gray-400 border border-gray-500/20'
                                                }`}>
                                                {bot.isActive ? 'Active' : 'Stopped'}
                                            </span>
                                            <span className={`px-2 py-0.5 rounded-full text-xs ${bot.mode === 'live' ? 'bg-red-500/10 text-red-400' : 'bg-blue-500/10 text-blue-400'
                                                }`}>
                                                {bot.mode}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-4 mt-1">
                                            <span className="text-gray-400 text-sm">{bot.user.name || bot.user.email}</span>
                                            <span className="text-gray-500 text-xs">•</span>
                                            <span className="text-gray-500 text-xs">{bot._count.trades} trades</span>
                                            <span className="text-gray-500 text-xs">•</span>
                                            <span className="text-gray-500 text-xs">Created {new Date(bot.createdAt).toLocaleDateString()}</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Controls */}
                                <div className="flex items-center gap-2">
                                    {bot.isActive ? (
                                        <>
                                            <button
                                                onClick={() => controlBot(bot.id, 'restart')}
                                                disabled={actionLoading === bot.id || !orchestratorOnline}
                                                className="p-2 rounded-lg bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 transition disabled:opacity-30"
                                                title="Restart"
                                            >
                                                <RefreshCw className={`w-4 h-4 ${actionLoading === bot.id ? 'animate-spin' : ''}`} />
                                            </button>
                                            <button
                                                onClick={() => controlBot(bot.id, 'stop')}
                                                disabled={actionLoading === bot.id || !orchestratorOnline}
                                                className="p-2 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 transition disabled:opacity-30"
                                                title="Stop"
                                            >
                                                <PowerOff className="w-4 h-4" />
                                            </button>
                                        </>
                                    ) : (
                                        <button
                                            onClick={() => controlBot(bot.id, 'start')}
                                            disabled={actionLoading === bot.id || !orchestratorOnline}
                                            className="p-2 rounded-lg bg-green-500/10 text-green-400 hover:bg-green-500/20 transition disabled:opacity-30"
                                            title="Start"
                                        >
                                            <Power className="w-4 h-4" />
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

function MiniStat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number }) {
    return (
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
                {icon}
                <span className="text-gray-400 text-xs">{label}</span>
            </div>
            <p className="text-xl font-bold text-white">{value}</p>
        </div>
    );
}
