'use client';

import { Palette } from 'lucide-react';
import { useTheme } from './providers/theme-provider';
import { themes, ThemeKey } from '@/lib/themes';
import { useState, useEffect } from 'react';

export function ThemeSwitcher() {
  const { currentTheme, setTheme } = useTheme();
  const [isOpen, setIsOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 rounded-lg transition-colors"
        aria-label="Change theme"
      >
        <Palette className="w-5 h-5 text-[var(--color-primary)]" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-48 bg-[var(--color-surface)] rounded-lg shadow-lg p-2 space-y-1 z-50">
          {Object.entries(themes).map(([key, theme]) => (
            <button
              key={key}
              onClick={() => {
                setTheme(key as ThemeKey);
                setIsOpen(false);
              }}
              className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${currentTheme === key
                  ? 'bg-[var(--color-primary)] text-white'
                  : 'hover:bg-[var(--color-surface-light)]'
                }`}
            >
              <div className="flex items-center space-x-2">
                <div
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: theme.colors.primary }}
                />
                <span>{theme.name}</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}