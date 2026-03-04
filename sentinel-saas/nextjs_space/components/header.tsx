'use client';

import Link from 'next/link';
import { useSession, signOut } from 'next-auth/react';
import { useState, useEffect } from 'react';
import { Shield, Menu, X, LogOut, User, Settings, LayoutDashboard } from 'lucide-react';
import { ThemeSwitcher } from './theme-switcher';

export function Header() {
  const { data: session, status } = useSession() || {};
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  const handleSignOut = async () => {
    await signOut({ callbackUrl: '/' });
  };

  return (
    <header className="fixed top-[38px] left-0 right-0 z-50 bg-[var(--color-surface)]/80 backdrop-blur-md border-b border-[var(--color-surface-light)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link href={session ? '/dashboard' : '/'} className="flex items-center space-x-2 hover:opacity-80 transition-opacity">
            <Shield className="w-8 h-8 text-[var(--color-primary)]" />
            <span className="text-2xl font-bold text-gradient">Sentinel</span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-6">
            {session ? (
              <>
                <Link href="/dashboard" className="text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Dashboard
                </Link>
                <Link href="/bots" className="text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Bots
                </Link>
                <Link href="/trades" className="text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Trade Book
                </Link>
                <Link href="/intelligence" className="text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Intelligence
                </Link>
                <Link href="/account" className="text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Account
                </Link>
                {(session.user as any)?.role === 'admin' && (
                  <Link href="/admin" className="text-amber-400 hover:text-amber-300 transition-colors">
                    Admin
                  </Link>
                )}
                <button
                  onClick={handleSignOut}
                  className="flex items-center space-x-1 px-4 py-2 bg-[var(--color-danger)] text-white rounded-lg hover:opacity-90 transition-opacity"
                >
                  <LogOut className="w-4 h-4" />
                  <span>Logout</span>
                </button>
              </>
            ) : (
              <>
                <Link href="/pricing" className="text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Pricing
                </Link>
                <Link href="/login" className="text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Login
                </Link>
                <Link href="/signup" className="px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary-dark)] transition-colors">
                  Sign Up
                </Link>
              </>
            )}
          </nav>

          {/* Mobile menu button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden text-[var(--color-text)]">
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 space-y-3">
            {session ? (
              <>
                <Link href="/dashboard" className="block text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Dashboard
                </Link>
                <Link href="/bots" className="block text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Bots
                </Link>
                <Link href="/trades" className="block text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Trade Book
                </Link>
                <Link href="/intelligence" className="block text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Intelligence
                </Link>
                <Link href="/account" className="block text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Account
                </Link>
                <button
                  onClick={handleSignOut}
                  className="w-full text-left px-4 py-2 bg-[var(--color-danger)] text-white rounded-lg hover:opacity-90 transition-opacity"
                >
                  Logout
                </button>
              </>
            ) : (
              <>
                <Link href="/pricing" className="block text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Pricing
                </Link>
                <Link href="/login" className="block text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors">
                  Login
                </Link>
                <Link href="/signup" className="block px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary-dark)] transition-colors text-center">
                  Sign Up
                </Link>
              </>
            )}
          </div>
        )}
      </div>
    </header>
  );
}