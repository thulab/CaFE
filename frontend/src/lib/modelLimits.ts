import type { ModelDTO } from '../api/models';

type ForecastLimits = NonNullable<ModelDTO['forecast_limits']>;

export function modelSupportsTargetDim(model: ModelDTO, targetDim: number): boolean {
  if (targetDim <= 1) return true;
  const maxTargetCount = readMaxTargetCount(model.forecast_limits);
  if (maxTargetCount === undefined) return false;
  if (maxTargetCount === null) return true;
  return maxTargetCount >= targetDim;
}

export function modelMaxTargetCount(model: ModelDTO): number | null | undefined {
  return readMaxTargetCount(model.forecast_limits);
}

function readMaxTargetCount(limits?: ForecastLimits | null): number | null | undefined {
  if (!limits || !Object.prototype.hasOwnProperty.call(limits, 'max_target_count')) {
    return undefined;
  }
  const raw = limits.max_target_count;
  if (raw == null) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}
