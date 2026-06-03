import { apiRequest, type ApiRequestOptions } from './client';

export interface ModelDTO {
  model_id: string;
  name: string;
  adapter_type: string;
  forecast_limits?: {
    max_target_count?: number | null;
    [key: string]: unknown;
  } | null;
  loaded?: boolean | null;
  loading?: boolean | null;
  service_state?: string | null;
}

export function listModels(options: ApiRequestOptions = {}) {
  return apiRequest<{ items: ModelDTO[] }>('/models', {}, options);
}

export function loadModel(modelId: string, options: ApiRequestOptions = {}) {
  return apiRequest<ModelDTO>(`/models/${encodeURIComponent(modelId)}/load`, { method: 'POST' }, options);
}
