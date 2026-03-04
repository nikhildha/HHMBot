'use client';

import { createContext, useContext, useState, useEffect } from 'react';
import { themes, ThemeKey } from '@/lib/themes';

interface ThemeContextType {
  currentTheme: ThemeKey;
  setTheme: (theme: ThemeKey) => void;
  themeColors: typeof themes[ThemeKey]['colors'];
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [currentTheme, setCurrentTheme] = useState<ThemeKey>('ocean');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const savedTheme = localStorage.getItem('sentinel-theme') as ThemeKey;
    if (savedTheme && themes[savedTheme]) {
      setCurrentTheme(savedTheme);
    }
  }, []);

  const setTheme = (theme: ThemeKey) => {
    setCurrentTheme(theme);
    localStorage.setItem('sentinel-theme', theme);
  };

  if (!mounted) {
    return null;
  }

  const themeColors = themes[currentTheme].colors;

  return (
    <ThemeContext.Provider value={{ currentTheme, setTheme, themeColors }}>
      <style jsx global>{`
        :root {
          --color-primary: ${themeColors.primary};
          --color-primary-dark: ${themeColors.primaryDark};
          --color-primary-light: ${themeColors.primaryLight};
          --color-accent: ${themeColors.accent};
          --color-background: ${themeColors.background};
          --color-surface: ${themeColors.surface};
          --color-surface-light: ${themeColors.surfaceLight};
          --color-text: ${themeColors.text};
          --color-text-secondary: ${themeColors.textSecondary};
          --color-success: ${themeColors.success};
          --color-danger: ${themeColors.danger};
          --color-warning: ${themeColors.warning};
        }
      `}</style>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
}