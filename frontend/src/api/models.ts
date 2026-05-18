import { apiRequest } from './client';

export interface ModelDTO {
  model_id: string;
  name: string;
  adapter_type: string;
}

export function listModels() {
  return apiRequest<{ items: ModelDTO[] }>('/models');
}
