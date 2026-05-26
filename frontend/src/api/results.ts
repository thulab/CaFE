import { apiRequest } from './client';
import type { RankingDTO, ReportDTO, SampleForecastDTO } from './types';

export interface LeaderboardTopEntry {
  rank: number;
  model_id: string;
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
