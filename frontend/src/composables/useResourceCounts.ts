import { reactive } from 'vue';
import { listDatasetManifests, listShards } from '../api/datasets';
import { listReports } from '../api/results';
import { listRuns } from '../api/runs';
import { listTracks } from '../api/tracks';
import { authState } from '../stores/auth';

// 侧边栏角标 + 首页卡片都靠这里：只为拿 total 字段，支持分页的端点用 limit=1。
// 单飞读取失败不要把其他计数也拉下水——拆个体 catch，错的归零、对的归正。
export interface ResourceCounts {
  datasets: number;
  shards: number;
  tracks: number;
  runs: number;
  reports: number;
}

const state = reactive<{ counts: ResourceCounts; loaded: boolean }>({
  counts: { datasets: 0, shards: 0, tracks: 0, runs: 0, reports: 0 },
  loaded: false
});

let inflight: Promise<void> | null = null;

async function totalOrZero(p: Promise<{ total: number }>): Promise<number> {
  try {
    return (await p).total;
  } catch {
    return 0;
  }
}

export async function refreshResourceCounts(): Promise<void> {
  if (!authState.user) {
    state.counts = { datasets: 0, shards: 0, tracks: 0, runs: 0, reports: 0 };
    return;
  }
  if (inflight) return inflight;
  inflight = (async () => {
    const [datasets, shards, tracks, runs, reports] = await Promise.all([
      totalOrZero(listDatasetManifests({ limit: 1 })),
      totalOrZero(listShards({ limit: 1 })),
      totalOrZero(listTracks()),
      totalOrZero(listRuns({ limit: 1 })),
      totalOrZero(listReports({ limit: 1 }))
    ]);
    state.counts = { datasets, shards, tracks, runs, reports };
    state.loaded = true;
  })().finally(() => {
    inflight = null;
  });
  return inflight;
}

/** Subscribe to counts. Triggers a one-shot fetch on first call; callers may `refresh()` to re-pull. */
export function useResourceCounts() {
  if (!state.loaded && !inflight) void refreshResourceCounts();
  return { state, refresh: refreshResourceCounts };
}
