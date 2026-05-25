import { computed, ref } from 'vue';

// Theme preference: 'light' | 'dark' | 'system'. Persisted to localStorage and
// applied to <html data-theme>. Falls back to system preference gracefully.

export type ThemePref = 'light' | 'dark' | 'system';
const STORAGE_KEY = 'tsb.theme';

function readStored(): ThemePref {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === 'light' || v === 'dark' || v === 'system') return v;
  } catch {
    /* ignore */
  }
  return 'system';
}

function systemDark(): boolean {
  return typeof window !== 'undefined'
    && typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

const pref = ref<ThemePref>(readStored());

/** Resolved theme actually applied ('light' | 'dark'). */
export const resolvedTheme = computed<'light' | 'dark'>(() =>
  pref.value === 'system' ? (systemDark() ? 'dark' : 'light') : pref.value
);

function apply() {
  if (typeof document === 'undefined') return;
  document.documentElement.setAttribute('data-theme', resolvedTheme.value);
}

export function setTheme(next: ThemePref) {
  pref.value = next;
  try {
    localStorage.setItem(STORAGE_KEY, next);
  } catch {
    /* ignore */
  }
  apply();
}

/** Cycle light → dark → system for a single toggle button. */
export function cycleTheme() {
  const order: ThemePref[] = ['light', 'dark', 'system'];
  const idx = order.indexOf(pref.value);
  setTheme(order[(idx + 1) % order.length]);
}

let initialized = false;
export function useTheme() {
  if (!initialized && typeof window !== 'undefined') {
    initialized = true;
    apply();
    if (typeof window.matchMedia === 'function') {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => {
        if (pref.value === 'system') apply();
      });
    }
  }
  return { pref, resolvedTheme, setTheme, cycleTheme };
}
