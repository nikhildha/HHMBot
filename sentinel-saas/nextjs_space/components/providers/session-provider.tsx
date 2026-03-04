'use client';

import { SessionProvider as NextAuthSessionProvider } from 'next-auth/react';
import { useState, useEffect } from 'react';

interface Props {
  children: React.ReactNode;
}

export function SessionProvider({ children }: Props) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return <NextAuthSessionProvider>{children}</NextAuthSessionProvider>;
}