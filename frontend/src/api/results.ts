import { apiRequest } from './client';
import type { RankingDTO, ReportDTO, SampleForecastDTO } from './types';

export function getRanking(trackId: string, metric: string, policy: string): Promise<RankingDTO> {
  return apiRequest<RankingDTO>(`/tracks/${trackId}/ranking?metric=${encodeURIComponent(metric)}&policy=${encodeURIComponent(policy)}`);
}

export function getReport(reportId: string): Promise<ReportDTO> {
  return apiRequest<ReportDTO>(`/reports/${reportId}`);
}

export function getSampleForecast(sampleId: string, runId: string): Promise<SampleForecastDTO> {
  return apiRequest<SampleForecastDTO>(`/samples/${sampleId}/forecast?run_id=${encodeURIComponent(runId)}`);
}
