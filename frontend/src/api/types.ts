export interface UploadPreviewDTO {
  upload_id: string;
  source_uri: string;
  filename?: string;
  file_format?: string;
  detected_delimiter?: string;
  device?: string | null;
  devices?: string[];
  columns: Array<{ name: string; inferred_type?: string; nullable?: boolean; sample_values?: string[] }>;
  preview_rows: Array<Record<string, string>>;
}

export interface DatasetManifestCreateDTO {
  name: string;
  domain: string;
  source_uri: string;
  file_format: 'csv' | 'tsfile' | string;
  time_column: string;
}

export interface DatasetLoadJobCreateDTO {
  dataset_manifest_id: string;
  split_config: { context_length: number; horizon: number; stride?: number; target_columns: string[]; covariate_columns?: string[]; max_samples?: number };
  seed?: number;
}

export interface DatasetLoadJobDTO {
  load_job_id: string;
  status: string;
  output_shard_id?: string;
  error_code?: string | null;
  error_message?: string | null;
}

export interface DatasetManifestDTO {
  dataset_manifest_id: string;
  name: string;
  domain: string;
  source_uri: string;
  file_format: string;
  time_column: string;
  frequency?: string | null;
  timezone?: string | null;
  status: string;
  archived_at?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface DatasetLoadJobDetailDTO extends DatasetLoadJobDTO {
  dataset_manifest_id: string;
  current_step?: string | null;
  validation_summary?: Record<string, unknown>;
  split_config: Record<string, unknown>;
  seed?: number;
  error_code?: string | null;
  error_message?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface ShardDTO {
  shard_id: string;
  name?: string | null;
  dataset_name?: string | null;
  shard_type?: string;
  capability_type?: string | null;
  dataset_manifest_id: string;
  load_job_id?: string | null;
  capability_block_id?: string | null;
  source_uri: string;
  storage_uri?: string | null;
  time_range_start?: string | null;
  time_range_end?: string | null;
  row_count: number;
  target_columns: string[];
  target_dim: number;
  covariate_columns?: string[];
  covariate_dim?: number;
  frequency?: string | null;
  context_length: number;
  horizon: number;
  stride: number;
  sample_count: number;
  generation_config?: Record<string, unknown>;
  status: string;
  archived_at?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface SyntheticCapabilityDTO {
  capability_id: string;
  label: string;
  description: string;
  label_i18n?: Record<string, string>;
  description_i18n?: Record<string, string>;
  task_type: string;
  target_dim_mode: 'fixed_1' | 'multi' | 'covariate' | string;
  covariate_columns: string[];
  default_params?: Record<string, number | string>;
  limits?: Record<string, { min?: number; max?: number }>;
}

export interface SyntheticShardGenerateDTO {
  name: string;
  capabilities: string[];
  context_length: number;
  horizon: number;
  sample_count: number;
  intensity: number;
  difficulty?: number;
  season_length: number;
  target_dim: number;
  seed: number;
  frequency: string;
}

export interface SyntheticShardGenerateResponseDTO {
  items: ShardDTO[];
  shard_ids: string[];
}

export interface SampleIndexDTO {
  sample_id: string;
  sample_index?: number;
  context_start?: number;
  context_end?: number;
  horizon_start?: number;
  horizon_end?: number;
  target_columns?: string[];
  context_length?: number;
  horizon?: number;
}

export interface ShardSamplesDTO {
  items: SampleIndexDTO[];
  total?: number;
  limit?: number;
  offset?: number;
}

export interface SamplePreviewDTO extends SampleWindowMeta {
  sample_id: string;
  shard_id?: string;
  target_column_names?: string[];
  covariate_column_names?: string[];
  history_timestamps?: string[];
  future_timestamps?: string[];
  target_history: number[][];
  target_future: number[][];
  history_cov?: number[][];
  future_cov?: number[][];
}

export interface RunProgressDTO {
  benchmarking_run_id: string;
  status: string;
  activity_status?: string;
  progress: Record<string, number>;
  units: Array<Record<string, unknown>>;
  tasks: Array<Record<string, unknown>>;
  recent_events: Array<{ created_at: string; message?: string; event_type?: string; level?: string }>;
  report_id?: string;
  ranking_list_id?: string;
  archived_at?: string | null;
}

export interface FailedSampleDTO {
  forecast_artifact_id: string;
  sample_id: string;
  sample_index?: number | null;
  model_id: string;
  model_name?: string | null;
  unit_id: string;
  task_id: string;
  capability_block_id?: string | null;
  capability_block_name?: string | null;
  shard_id: string;
  error_code?: string | null;
  error_message?: string | null;
  unit_status?: string | null;
  task_status?: string | null;
}

export interface FailedSampleSummaryDTO {
  error_code?: string | null;
  error_message?: string | null;
  count: number;
  model_count: number;
  capability_count: number;
  sample_count: number;
}

export interface FailedSamplesDTO {
  items: FailedSampleDTO[];
  total: number;
  limit: number;
  offset: number;
  summary: FailedSampleSummaryDTO[];
}

export interface RerunFailedSamplesDTO {
  rerun_job_id: string;
  benchmarking_run_id: string;
  status: string;
  activity_status: string;
  total_samples: number;
  processed_samples: number;
  succeeded_samples: number;
  failed_samples: number;
  error_code?: string | null;
  error_message?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface ActiveRerunFailedSamplesDTO {
  active: RerunFailedSamplesDTO | null;
}

export interface RankingDTO {
  track_id?: string;
  ranking_list_id?: string | null;
  metric?: string;
  policy?: string;
  public_visible?: boolean;
  items: Array<{ model_id: string; rank: number; metric_value: number }>;
}

export type TrackModelEvaluationStatus = 'evaluated' | 'not_evaluated' | 'run_failed';

export interface TrackModelStatusDTO {
  model_id: string;
  model_name?: string | null;
  evaluation_status: TrackModelEvaluationStatus;
  run_id?: string | null;
  report_id?: string | null;
  unit_id?: string | null;
  unit_status?: string | null;
  failed_sample_count?: number | null;
}

export interface TrackResultsDTO {
  track_id: string;
  metric?: string | null;
  model_statuses: TrackModelStatusDTO[];
  model_metrics: Array<Record<string, unknown>>;
  capability_blocks?: CapabilityBlockReportDTO[];
  capability_metrics?: CapabilityMetricReportDTO[];
  sample_forecast_links: SampleForecastLinkDTO[];
  sample_forecast_links_total?: number;
  sample_forecast_links_limit?: number;
  sample_forecast_links_offset?: number;
  sample_forecast_links_capability_block_id?: string | null;
  sample_forecast_links_metric?: string | null;
  sample_forecast_links_model_id?: string | null;
  sample_forecast_links_sort?: SampleForecastSort;
}

export interface TrackDTO {
  track_id: string;
  name: string;
  track_type?: string;
  description?: string | null;
  primary_metric_id: string;
  default_ranking_policy?: string;
  benchmark_version?: string;
  data_version?: string;
  status: string;
  archived_at?: string | null;
  capability_block_ids?: string[];
  shard_ids?: string[];
  shard_count?: number;
  sample_count?: number;
  created_at?: string;
  updated_at?: string;
}

export interface SampleWindowMeta {
  sample_index?: number | null;
  context_start?: number | null;
  context_end?: number | null;
  horizon_start?: number | null;
  horizon_end?: number | null;
  history_start_at?: string | null;
  history_end_at?: string | null;
  forecast_start_at?: string | null;
  forecast_end_at?: string | null;
}

export interface SampleForecastLinkDTO extends SampleWindowMeta {
  sample_id: string;
  run_id: string;
  forecast_artifact_id?: string | null;
  forecast_artifact_ids?: string[];
  model_count?: number | null;
  capability_block_id?: string | null;
  capability_block_name?: string | null;
  capability_type?: string | null;
  capability_label?: string | null;
  metric_id?: string | null;
  metric_value?: number | null;
  metric_model_id?: string | null;
}

export type SampleForecastSort = 'sample_index' | 'metric_desc' | 'metric_asc';

export interface CapabilityBlockReportDTO {
  capability_block_id: string;
  name: string;
  block_type: string;
  capability_type: string;
  capability_label?: string | null;
  task_type?: string | null;
  target_dim?: number | null;
  covariate_dim?: number | null;
  shard_count?: number | null;
  sample_count?: number | null;
  aggregation_policy?: string | null;
  generation_config?: Record<string, unknown>;
  shard_ids?: string[];
}

export interface CapabilityMetricReportDTO {
  task_id: string;
  unit_id: string;
  model_id: string;
  model_name?: string | null;
  capability_block_id: string;
  status: string;
  sample_count?: number | null;
  processed_sample_count?: number | null;
  failed_sample_count?: number | null;
  error_code?: string | null;
  error_message?: string | null;
  metrics: Record<string, unknown>;
}

export interface ReportDTO {
  report_id: string;
  track_id?: string;
  model_metrics: Array<Record<string, unknown>>;
  task_summaries: Array<Record<string, unknown>>;
  capability_blocks?: CapabilityBlockReportDTO[];
  capability_metrics?: CapabilityMetricReportDTO[];
  sample_forecast_links: SampleForecastLinkDTO[];
  sample_forecast_links_total?: number;
  sample_forecast_links_limit?: number;
  sample_forecast_links_offset?: number;
  sample_forecast_links_capability_block_id?: string | null;
  sample_forecast_links_metric?: string | null;
  sample_forecast_links_model_id?: string | null;
  sample_forecast_links_sort?: SampleForecastSort;
}

export interface ListResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface BenchmarkingRunSummaryDTO {
  benchmarking_run_id: string;
  track_id: string;
  model_ids: string[];
  status: string;
  activity_status?: string;
  archived_at?: string | null;
  model_count: number;
  task_count: number;
  sample_count: number;
  report_id?: string | null;
  ranking_list_id?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  created_at?: string;
  updated_at?: string;
}

export type ResourceType = 'dataset_manifest' | 'shard' | 'track' | 'benchmarking_run';

export interface DeletionImpactDTO {
  resource_type: ResourceType;
  resource_id: string;
  archive_available: boolean;
  purge_available: boolean;
  cascade_required: boolean;
  affected: Record<string, number>;
  warnings: string[];
}

export interface ReportSummaryDTO {
  report_id: string;
  report_type: string;
  benchmarking_run_id: string;
  track_id: string;
  status: string;
  storage_uri?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface SampleForecastDTO extends SampleWindowMeta {
  sample_id: string;
  benchmarking_run_id?: string;
  shard_id?: string;
  capability_block_id?: string | null;
  target_column_names?: string[];
  covariate_column_names?: string[];
  target_history: number[][];
  target_future: number[][];
  history_cov?: number[][];
  future_cov?: number[][];
  models: Array<{
    model_id: string;
    model_name?: string;
    status: string;
    forecast?: number[][] | null;
    metrics: Record<string, number>;
    error_message?: string;
  }>;
  links?: {
    run?: string | null;
    report?: string | null;
    ranking?: string | null;
    track?: string | null;
  };
}
