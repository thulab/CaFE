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
  target_columns: string[];
}

export interface DatasetLoadJobCreateDTO {
  dataset_manifest_id: string;
  split_config: { context_length: number; horizon: number; stride?: number };
  seed?: number;
}

export interface DatasetLoadJobDTO {
  load_job_id: string;
  status: string;
  output_shard_id?: string;
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
  models: Array<{ model_id: string; model_name?: string; status: string; forecast?: number[][]; metrics: Record<string, number> }>;
}
