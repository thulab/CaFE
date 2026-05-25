export interface UploadPreviewDTO {
  upload_id: string;
  source_uri: string;
  filename?: string;
  detected_delimiter?: string;
  columns: Array<{ name: string; inferred_type?: string; nullable?: boolean; sample_values?: string[] }>;
  preview_rows: Array<Record<string, string>>;
}

export interface DatasetManifestCreateDTO {
  name: string;
  domain: string;
  source_uri: string;
  file_format: 'csv';
  time_column: string;
  value_columns: string[];
}

export interface DatasetLoadJobCreateDTO {
  dataset_manifest_id: string;
  split_config: { context_length: number; horizon: number; stride?: number; target_columns: string[]; max_samples?: number };
  seed?: number;
}

export interface DatasetLoadJobDTO {
  load_job_id: string;
  status: string;
  output_shard_id?: string;
}

export interface DatasetManifestDTO {
  dataset_manifest_id: string;
  name: string;
  domain: string;
  source_uri: string;
  file_format: string;
  time_column: string;
  value_columns: string[];
  frequency?: string | null;
  timezone?: string | null;
  status: string;
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
  shard_type?: string;
  dataset_manifest_id: string;
  load_job_id?: string | null;
  capability_block_id?: string | null;
  source_uri: string;
  storage_uri?: string | null;
  time_range_start?: string | null;
  time_range_end?: string | null;
  row_count: number;
  target_columns: string[];
  value_columns: string[];
  target_dim: number;
  frequency?: string | null;
  context_length: number;
  horizon: number;
  stride: number;
  sample_count: number;
  status: string;
  created_at?: string;
  updated_at?: string;
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

export interface RunProgressDTO {
  benchmarking_run_id: string;
  status: string;
  progress: Record<string, number>;
  units: Array<Record<string, unknown>>;
  tasks: Array<Record<string, unknown>>;
  recent_events: Array<{ created_at: string; message?: string; event_type?: string; level?: string }>;
  report_id?: string;
  ranking_list_id?: string;
}

export interface RankingDTO {
  items: Array<{ model_id: string; rank: number; metric_value: number }>;
}

export interface ReportDTO {
  report_id: string;
  model_metrics: Array<Record<string, unknown>>;
  task_summaries: Array<Record<string, unknown>>;
  sample_forecast_links: Array<{ sample_id: string; run_id: string }>;
}

export interface SampleForecastDTO {
  sample_id: string;
  target_history: number[][];
  target_future: number[][];
  models: Array<{
    model_id: string;
    model_name?: string;
    status: string;
    forecast?: number[][] | null;
    metrics: Record<string, number>;
    error_message?: string;
  }>;
}
