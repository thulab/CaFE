import { apiRequest, type ApiRequestOptions } from './client';

export interface ModelDTO {
  model_id: string;
  name: string;
  adapter_type: string;
}

export function listModels(options: ApiRequestOptions = {}) {
  return apiRequest<{ items: ModelDTO[] }>('/models', {}, options);
}
