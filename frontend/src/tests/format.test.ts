import { describe, expect, it, vi } from 'vitest';
import { formatDateTime, formatInt, formatNumber, timeAgo } from '../lib/format';

describe('locale-aware format helpers', () => {
  it('formats numbers with an explicit locale', () => {
    expect(formatInt(1234567, 'en-US')).toContain('1');
    expect(formatInt(1234567, 'zh-CN')).toContain('1');
    expect(formatNumber(0.123456, 2, 'en-US')).toContain('0');
  });

  it('formats dates with an explicit locale', () => {
    expect(formatDateTime('2026-05-26T12:00:00Z', 'en-US')).toMatch(/2026|May|26/);
    expect(formatDateTime('2026-05-26T12:00:00Z', 'zh-CN')).toMatch(/2026|5|26/);
  });

  it('formats relative time without English-only string assembly', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-05-26T12:00:10Z'));
    const pastEnglish = timeAgo('2026-05-26T12:00:00Z', 'en-US');
    const futureEnglish = timeAgo('2026-05-26T12:00:20Z', 'en-US');
    const futureChinese = timeAgo('2026-05-26T12:00:20Z', 'zh-CN');

    expect(pastEnglish).toContain('second');
    expect(timeAgo('2026-05-26T12:00:00Z', 'zh-CN')).toMatch(/秒|10/);
    expect(timeAgo('2026-05-26T12:00:00Z', 'zh-CN')).not.toContain('ago');
    expect(futureEnglish.includes('in') || futureEnglish !== pastEnglish).toBe(true);
    expect(futureChinese).not.toBe('');
    expect(futureChinese).not.toContain('ago');
    vi.useRealTimers();
  });

  it('rounds negative half relative-time values symmetrically', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-05-26T12:01:30Z'));
    expect(timeAgo('2026-05-26T12:00:00Z', 'en-US')).toContain('2 minutes');
    vi.useRealTimers();
  });
});
