import { apiRequest } from './client';
import type { SyntheticCapabilityDTO, SyntheticShardGenerateDTO, SyntheticShardGenerateResponseDTO } from './types';

export function listSyntheticCapabilities(): Promise<{ items: SyntheticCapabilityDTO[] }> {
  return apiRequest<{ items: SyntheticCapabilityDTO[] }>('/synthetic/capabilities');
}

export function createSyntheticShards(payload: SyntheticShardGenerateDTO): Promise<SyntheticShardGenerateResponseDTO> {
  return apiRequest<SyntheticShardGenerateResponseDTO>('/synthetic/shards', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}
