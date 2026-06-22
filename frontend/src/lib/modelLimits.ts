import type { ModelDTO } from '../api/models';

type ForecastLimits = NonNullable<ModelDTO['forecast_limits']>;

export function modelSupportsTargetDim(model: ModelDTO, targetDim: number): boolean {
  if (targetDim <= 1) return true;
  const maxTargetCount = readMaxTargetCount(model.forecast_limits);
  if (maxTargetCount === undefined) return false;
  if (maxTargetCount === null) return true;
  return maxTargetCount >= targetDim;
}

export function modelSupportsCovariateDim(model: ModelDTO, covariateDim: number): boolean {
  if (covariateDim <= 0) return true;
  const maxCovariateCount = modelMaxCovariateCount(model);
  return maxCovariateCount >= covariateDim;
}

export function modelSupportsWindow(model: ModelDTO, minContextLength: number, maxContextLength: number, maxHorizon: number, maxCovariateHorizon = 0): boolean {
  const limits = model.forecast_limits;
  const minInputLength = readNumericLimit(limits, 'min_input_length');
  if (minInputLength !== undefined && minContextLength < minInputLength) return false;
  const maxInputLength = readNumericLimit(limits, 'max_input_length');
  if (maxInputLength !== undefined && maxContextLength > maxInputLength) return false;
  const maxOutputLength = readNumericLimit(limits, 'max_output_length');
  if (maxOutputLength !== undefined && maxHorizon > maxOutputLength) return false;
  const maxFutureCovsLength = readNumericLimit(limits, 'max_future_covs_length');
  if (maxFutureCovsLength !== undefined && maxCovariateHorizon > maxFutureCovsLength) return false;
  return true;
}

export function modelMaxTargetCount(model: ModelDTO): number | null | undefined {
  return readMaxTargetCount(model.forecast_limits);
}

export function modelMaxCovariateCount(model: ModelDTO): number {
  return readNumericLimit(model.forecast_limits, 'max_covariate_count') ?? 0;
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

function readNumericLimit(limits: ForecastLimits | null | undefined, key: string): number | undefined {
  if (!limits || !Object.prototype.hasOwnProperty.call(limits, key)) return undefined;
  const raw = limits[key];
  if (raw == null) return undefined;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}
