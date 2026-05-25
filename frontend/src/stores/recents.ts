import { reactive } from 'vue';

// Lightweight, localStorage-backed registry of artifacts the user has created.
// The backend exposes only get-by-id endpoints (no collections), so list/overview
// pages are driven by what this session has produced. Each entry stores enough to
// render a list row and deep-link into the dedicated detail page.

export type RecentKind = 'dataset' | 'shard' | 'track' | 'run' | 'report';

export interface RecentItem {
  kind: RecentKind;
  id: string;
  title: string;
  subtitle?: string;
  href: string;
  createdAt: string;
}

const STORAGE_KEY = 'tsb.recents.v1';
const MAX = 100;

function load(): RecentItem[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as RecentItem[]) : [];
  } catch {
    return [];
  }
}

const state = reactive({ items: load() as RecentItem[] });

function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.items));
  } catch {
    /* storage may be unavailable; keep in-memory copy */
  }
}

/** Record (or refresh) an artifact. De-dupes by kind+id and moves it to the front. */
export function recordRecent(item: Omit<RecentItem, 'createdAt'> & { createdAt?: string }) {
  if (!item.id) return;
  const existing = state.items.find((x) => x.kind === item.kind && x.id === item.id);
  const createdAt = existing?.createdAt ?? item.createdAt ?? new Date().toISOString();
  const next: RecentItem = { ...item, createdAt };
  state.items = [next, ...state.items.filter((x) => !(x.kind === item.kind && x.id === item.id))].slice(0, MAX);
  persist();
}

export function listRecents(kind?: RecentKind | RecentKind[]): RecentItem[] {
  if (!kind) return state.items;
  const kinds = Array.isArray(kind) ? kind : [kind];
  return state.items.filter((x) => kinds.includes(x.kind));
}

export function countRecents(kind?: RecentKind | RecentKind[]): number {
  return listRecents(kind).length;
}

export function clearRecents() {
  state.items = [];
  persist();
}
