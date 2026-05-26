import { apiRequest } from './client';
import { type ListParams, buildListQuery } from './shared';
import type { ListResponse, RankingDTO, ReportDTO, ReportSummaryDTO, SampleForecastDTO } from './types';

export interface LeaderboardTopEntry {
  rank: number;
  model_id: string;
  /** Backend 已在 /ranking-lists 内嵌解析后的名字，匿名访问也能拿到。 */
  model_name?: string;
  metric_value: number;
}

export interface LeaderboardItem {
  ranking_list_id: string;
  track_id: string;
  track_name: string;
  track_type: string;
  primary_metric_id: string;
  default_policy: string;
  updated_at: string;
  model_count: number;
  run_count: number;
  top: LeaderboardTopEntry[];
}

export interface LeaderboardListDTO {
  items: LeaderboardItem[];
}

export function getRanking(trackId: string, metric: string, policy: string): Promise<RankingDTO> {
  return apiRequest<RankingDTO>(`/tracks/${trackId}/ranking?metric=${encodeURIComponent(metric)}&policy=${encodeURIComponent(policy)}`);
}

export function getReport(reportId: string): Promise<ReportDTO> {
  return apiRequest<ReportDTO>(`/reports/${reportId}`);
}

export function getSampleForecast(sampleId: string, runId: string): Promise<SampleForecastDTO> {
  return apiRequest<SampleForecastDTO>(`/samples/${sampleId}/forecast?run_id=${encodeURIComponent(runId)}`);
}

export function listRankingLists(): Promise<LeaderboardListDTO> {
  return apiRequest<LeaderboardListDTO>('/ranking-lists');
}

export function listReports(
  params: ListParams & { benchmarkingRunId?: string; trackId?: string } = {}
): Promise<ListResponse<ReportSummaryDTO>> {
  const { benchmarkingRunId, trackId, ...rest } = params;
  return apiRequest<ListResponse<ReportSummaryDTO>>(
    `/reports${buildListQuery({ ...rest, benchmarking_run_id: benchmarkingRunId, track_id: trackId })}`
  );
}
