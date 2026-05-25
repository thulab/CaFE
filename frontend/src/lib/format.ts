// Display formatting helpers shared across views. Pure functions, no deps.

/** Format a metric/number compactly with sensible precision. */
export function formatNumber(value: unknown, digits = 4): string {
  if (value === null || value === undefined || value === '') return '—';
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (n === 0) return '0';
  const abs = Math.abs(n);
  if (abs >= 1e6 || abs < 1e-4) return n.toExponential(2);
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

/** Integer with thousands separators. */
export function formatInt(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—';
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString() : String(value);
}

/** Absolute timestamp, locale-aware. */
export function formatDateTime(value?: string | null): string {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  });
}

/** Human "time ago" string. */
export function timeAgo(value?: string | null): string {
  if (!value) return '';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return '';
  const secs = Math.round((Date.now() - d.getTime()) / 1000);
  if (secs < 0) return 'just now';
  const units: Array<[number, string]> = [
    [60, 'second'], [60, 'minute'], [24, 'hour'], [7, 'day'], [4.345, 'week'], [12, 'month'], [Infinity, 'year']
  ];
  let n = secs;
  for (let i = 0; i < units.length; i += 1) {
    const [step, label] = units[i];
    if (n < step) {
      const r = Math.max(1, Math.floor(n));
      return `${r} ${label}${r === 1 ? '' : 's'} ago`;
    }
    n /= step;
  }
  return formatDateTime(value);
}

/** Compact short id for display (keeps prefix + tail). */
export function shortId(id?: string | null, head = 8): string {
  if (!id) return '—';
  if (id.length <= head + 4) return id;
  return `${id.slice(0, head)}…${id.slice(-4)}`;
}

/** Percentage 0-100 from a 0..1 or completed/total pair. */
export function percent(done: number, total: number): number {
  if (!total || total <= 0) return 0;
  return Math.max(0, Math.min(100, Math.round((done / total) * 100)));
}

/** Title-case a snake/kebab status string for display. */
export function humanize(value?: string | null): string {
  if (!value) return '';
  return value.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
