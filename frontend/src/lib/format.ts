// Display formatting helpers shared across views. Pure functions, no deps.

/** Format a metric/number compactly with sensible precision. */
export function formatNumber(value: unknown, digits = 4, locale?: string): string {
  if (value === null || value === undefined || value === '') return '—';
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (n === 0) return '0';
  const abs = Math.abs(n);
  if (abs >= 1e6 || abs < 1e-4) return n.toExponential(2);
  if (Number.isInteger(n)) return n.toLocaleString(locale);
  return n.toLocaleString(locale, { maximumFractionDigits: digits });
}

/** Integer with thousands separators. */
export function formatInt(value: unknown, locale?: string): string {
  if (value === null || value === undefined || value === '') return '—';
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString(locale) : String(value);
}

/** Absolute timestamp, locale-aware. */
export function formatDateTime(value?: string | null, locale?: string): string {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString(locale, {
    year: 'numeric', month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  });
}

/** Human relative time string. */
export function timeAgo(value?: string | null, locale?: string): string {
  if (!value) return '';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return '';
  const seconds = Math.round((d.getTime() - Date.now()) / 1000);
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });
  const abs = Math.abs(seconds);
  if (abs < 5) return rtf.format(0, 'second');
  const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ['second', 60],
    ['minute', 60],
    ['hour', 24],
    ['day', 7],
    ['week', 4.345],
    ['month', 12],
    ['year', Infinity],
  ];
  let valueInUnit = seconds;
  for (const [unit, step] of units) {
    if (Math.abs(valueInUnit) < step) {
      return rtf.format(Math.round(valueInUnit), unit);
    }
    valueInUnit /= step;
  }
  return formatDateTime(value, locale);
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

/** Title-case a snake/kebab status string for display fallback. */
export function humanize(value?: string | null): string {
  if (!value) return '';
  return value.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
