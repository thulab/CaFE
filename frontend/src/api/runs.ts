import { apiRequest } from './client';
import type { RunProgressDTO } from './types';

export function createRun(payload: { track_id: string; model_ids: string[] }) {
  return apiRequest<{ benchmarking_run_id: string; status: string; created_at?: string }>('/benchmarking-runs', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function getRunProgress(runId: string): Promise<RunProgressDTO> {
  return apiRequest<RunProgressDTO>(`/benchmarking-runs/${runId}/progress`);
}

export function cancelRun(runId: string) {
  return apiRequest<{ benchmarking_run_id: string; status: string }>(`/benchmarking-runs/${runId}/cancel`, {
    method: 'POST'
  });
}
