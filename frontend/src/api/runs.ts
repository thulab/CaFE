import { apiRequest } from './client';
import { type ListParams, buildListQuery } from './shared';
import type { BenchmarkingRunSummaryDTO, FailedSamplesDTO, ListResponse, RerunFailedSamplesDTO, RunProgressDTO } from './types';

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

export function getFailedSamples(runId: string): Promise<FailedSamplesDTO> {
  return apiRequest<FailedSamplesDTO>(`/benchmarking-runs/${runId}/failed-samples`);
}

export function rerunFailedSamples(runId: string): Promise<RerunFailedSamplesDTO> {
  return apiRequest<RerunFailedSamplesDTO>(`/benchmarking-runs/${runId}/failed-samples/rerun`, {
    method: 'POST'
  });
}

export function listRuns(
  params: ListParams & { trackId?: string } = {}
): Promise<ListResponse<BenchmarkingRunSummaryDTO>> {
  const { trackId, includeArchived, ...rest } = params;
  return apiRequest<ListResponse<BenchmarkingRunSummaryDTO>>(
    `/benchmarking-runs${buildListQuery({ ...rest, track_id: trackId, include_archived: includeArchived || undefined })}`
  );
}
