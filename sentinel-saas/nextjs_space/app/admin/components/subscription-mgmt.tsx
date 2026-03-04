'use client';

import { useState, useEffect } from 'react';
import {
    CreditCard, Crown, Star, User, ArrowUpRight, ArrowDownRight,
    RefreshCw, Gift, Tag, CheckCircle2, XCircle, Clock, Sparkles
} from 'lucide-react';

interface SubUser {
    id: string;
    email: string;
    name: string;
    subscription: {
        id: string;
        tier: string;
        status: string;
        coinScans: number;
        createdAt: string;
    } | null;
}

const TIERS = [
    { id: 'free', name: 'Free', price: 0, color: 'gray', bots: 1, scans: 5 },
    { id: 'pro', name: 'Pro', price: 999, color: 'amber', bots: 3, scans: 20 },
    { id: 'ultra', name: 'Ultra', price: 2499, color: 'purple', bots: 10, scans: 50 },
];

export default function SubscriptionMgmt() {
    const [users, setUsers] = useState<SubUser[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedUser, setSelectedUser] = useState<SubUser | null>(null);
    const [changingTier, setChangingTier] = useState(false);
    const [promoCode, setPromoCode] = useState('');
    const [promoMessage, setPromoMessage] = useState('');

    useEffect(() => {
        fetchUsers();
    }, []);

    const fetchUsers = async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/admin/users');
            if (res.ok) {
                const data = await res.json();
                setUsers(data);
            }
        } catch (e) {
            console.error('Failed to fetch:', e);
        }
        setLoading(false);
    };

    const changeTier = async (userId: string, newTier: string) => {
        setChangingTier(true);
        try {
            const res = await fetch('/api/admin/subscriptions/change', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ userId, tier: newTier }),
            });
            if (res.ok) {
                await fetchUsers();
                setSelectedUser(null);
            }
        } catch (e) {
            console.error('Failed to change tier:', e);
        }
        setChangingTier(false);
    };

    const applyPromo = () => {
        if (!promoCode.trim()) return;
        // Mock promo validation
        const validCodes: Record<string, string> = {
            'LAUNCH2026': '30% off Pro for 3 months',
            'BETATESTER': 'Free Pro for 1 month',
            'SENTINEL50': '50% off first month',
        };
        if (validCodes[promoCode.toUpperCase()]) {
            setPromoMessage(`✅ Valid! ${validCodes[promoCode.toUpperCase()]}`);
        } else {
            setPromoMessage('❌ Invalid promo code');
        }
        setTimeout(() => setPromoMessage(''), 4000);
    };

    // Count subscriptions by tier
    const tierCounts = TIERS.map(t => ({
        ...t,
        count: users.filter(u => (u.subscription?.tier || 'free') === t.id).length,
    }));

    const totalMRR = users.reduce((sum, u) => {
        const tier = TIERS.find(t => t.id === (u.subscription?.tier || 'free'));
        return sum + (tier?.price || 0);
    }, 0);

    return (
        <div className="space-y-6">
            {/* Tier Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {tierCounts.map((tier) => (
                    <div
                        key={tier.id}
                        className="bg-white/5 border border-white/10 rounded-xl p-5 relative overflow-hidden"
                    >
                        <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-2">
                                {tier.id === 'ultra' && <Crown className="w-5 h-5 text-purple-400" />}
                                {tier.id === 'pro' && <Star className="w-5 h-5 text-amber-400" />}
                                {tier.id === 'free' && <User className="w-5 h-5 text-gray-400" />}
                                <h3 className="text-white font-semibold">{tier.name}</h3>
                            </div>
                            <span className={`text-sm font-mono ${tier.id === 'ultra' ? 'text-purple-400' : tier.id === 'pro' ? 'text-amber-400' : 'text-gray-400'
                                }`}>
                                {tier.price > 0 ? `₹${tier.price}/mo` : 'Free'}
                            </span>
                        </div>
                        <p className="text-3xl font-bold text-white mb-1">{tier.count}</p>
                        <p className="text-gray-500 text-sm">subscribers</p>
                        <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
                            <span>{tier.bots} bots</span>
                            <span>{tier.scans} scans</span>
                        </div>
                        {/* Gradient accent */}
                        <div className={`absolute top-0 right-0 w-20 h-20 rounded-full blur-3xl opacity-10 ${tier.id === 'ultra' ? 'bg-purple-500' : tier.id === 'pro' ? 'bg-amber-500' : 'bg-gray-500'
                            }`} />
                    </div>
                ))}
            </div>

            {/* MRR Banner */}
            <div className="bg-gradient-to-r from-emerald-500/10 to-blue-500/10 border border-emerald-500/20 rounded-xl p-6">
                <div className="flex items-center justify-between">
                    <div>
                        <p className="text-gray-400 text-sm">Total Monthly Recurring Revenue</p>
                        <p className="text-4xl font-bold text-emerald-400 mt-1">₹{totalMRR.toLocaleString('en-IN')}</p>
                        <p className="text-gray-500 text-xs mt-1">Based on {users.filter(u => u.subscription && u.subscription.tier !== 'free').length} paying subscribers</p>
                    </div>
                    <Sparkles className="w-10 h-10 text-emerald-400/30" />
                </div>
            </div>

            {/* User Subscription Table */}
            <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
                    <h3 className="text-white font-semibold">Manage Subscriptions</h3>
                    <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2">
                            <input
                                type="text"
                                placeholder="Promo code..."
                                value={promoCode}
                                onChange={(e) => setPromoCode(e.target.value)}
                                className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm placeholder-gray-500 focus:outline-none focus:border-blue-500/50 w-36"
                            />
                            <button
                                onClick={applyPromo}
                                className="px-3 py-1.5 rounded-lg bg-blue-500/10 text-blue-400 text-sm hover:bg-blue-500/20 transition"
                            >
                                <Tag className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </div>

                {promoMessage && (
                    <div className="px-6 py-2 bg-white/5 border-b border-white/10 text-sm">
                        {promoMessage}
                    </div>
                )}

                {loading ? (
                    <div className="flex items-center justify-center py-12">
                        <RefreshCw className="w-6 h-6 text-gray-500 animate-spin" />
                    </div>
                ) : (
                    <div className="divide-y divide-white/5">
                        {users.map((user) => {
                            const currentTier = TIERS.find(t => t.id === (user.subscription?.tier || 'free'));
                            return (
                                <div key={user.id} className="px-6 py-4 flex items-center justify-between hover:bg-white/[0.03] transition">
                                    <div className="flex items-center gap-4">
                                        <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center text-white font-bold text-sm">
                                            {(user.name || user.email)[0].toUpperCase()}
                                        </div>
                                        <div>
                                            <p className="text-white font-medium">{user.name || 'Unnamed'}</p>
                                            <p className="text-gray-400 text-sm">{user.email}</p>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-4">
                                        {/* Current Plan */}
                                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${currentTier?.id === 'ultra'
                                                ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20'
                                                : currentTier?.id === 'pro'
                                                    ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                                                    : 'bg-gray-500/10 text-gray-400 border border-gray-500/20'
                                            }`}>
                                            {currentTier?.name || 'Free'}
                                        </span>

                                        {/* Upgrade/Downgrade Buttons */}
                                        <div className="flex gap-1">
                                            {TIERS.filter(t => t.id !== (user.subscription?.tier || 'free')).map((tier) => {
                                                const isUpgrade = (TIERS.findIndex(t => t.id === tier.id)) > (TIERS.findIndex(t => t.id === (user.subscription?.tier || 'free')));
                                                return (
                                                    <button
                                                        key={tier.id}
                                                        onClick={() => changeTier(user.id, tier.id)}
                                                        disabled={changingTier}
                                                        className={`px-2.5 py-1 rounded-lg text-xs font-medium transition flex items-center gap-1 ${isUpgrade
                                                                ? 'bg-green-500/10 text-green-400 hover:bg-green-500/20'
                                                                : 'bg-orange-500/10 text-orange-400 hover:bg-orange-500/20'
                                                            } disabled:opacity-30`}
                                                        title={isUpgrade ? `Upgrade to ${tier.name}` : `Downgrade to ${tier.name}`}
                                                    >
                                                        {isUpgrade ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                                                        {tier.name}
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        </div>
    );
}
