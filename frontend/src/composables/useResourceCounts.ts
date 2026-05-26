import { reactive } from 'vue';
import { listDatasetManifests, listShards } from '../api/datasets';
import { listReports } from '../api/results';
import { listRuns } from '../api/runs';

// 侧边栏角标 + 首页卡片都靠这里：只为拿 total 字段，所以 limit=1 即可，不必把整张表拉下来。
// 单飞读取失败不要把另外 3 个也拉下水——拆个体 catch，错的归零、对的归正。
export interface ResourceCounts {
  datasets: number;
  shards: number;
  runs: number;
  reports: number;
}

const state = reactive<{ counts: ResourceCounts; loaded: boolean }>({
  counts: { datasets: 0, shards: 0, runs: 0, reports: 0 },
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
  if (inflight) return inflight;
  inflight = (async () => {
    const [datasets, shards, runs, reports] = await Promise.all([
      totalOrZero(listDatasetManifests({ limit: 1 })),
      totalOrZero(listShards({ limit: 1 })),
      totalOrZero(listRuns({ limit: 1 })),
      totalOrZero(listReports({ limit: 1 }))
    ]);
    state.counts = { datasets, shards, runs, reports };
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
