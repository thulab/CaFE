import { describe, expect, it } from 'vitest';
import { modelSupportsWindow } from '../lib/modelLimits';

describe('model limit helpers', () => {
  it('checks optional input and output window limits', () => {
    const limited = {
      model_id: 'timer',
      name: 'Timer',
      adapter_type: 'timer_service',
      forecast_limits: { min_input_length: 16, max_input_length: 128, max_output_length: 32 },
    };
    const unconstrained = {
      model_id: 'local',
      name: 'Local',
      adapter_type: 'stub',
      forecast_limits: {},
    };

    expect(modelSupportsWindow(limited, 16, 60, 16)).toBe(true);
    expect(modelSupportsWindow(limited, 8, 60, 16)).toBe(false);
    expect(modelSupportsWindow(limited, 16, 256, 16)).toBe(false);
    expect(modelSupportsWindow(limited, 16, 60, 64)).toBe(false);
    expect(modelSupportsWindow(unconstrained, 8, 4096, 4096)).toBe(true);
  });
});
