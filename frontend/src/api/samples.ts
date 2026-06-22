import { apiRequest } from './client';
import type { SamplePreviewDTO } from './types';

export function getSamplePreview(sampleId: string): Promise<SamplePreviewDTO> {
  return apiRequest<SamplePreviewDTO>(`/samples/${encodeURIComponent(sampleId)}/preview`);
}
