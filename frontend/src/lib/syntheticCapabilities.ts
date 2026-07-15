type Translate = (key: string) => string;

const KNOWN_CAPABILITY_IDS = new Set([
  'trend',
  'multi_seasonal',
  'regime_switching',
  'time_varying_seasonality',
  'nonlinear_persistence',
  'predictable_intermittency',
  'common_factor',
  'hierarchical_coherence',
  'covariate_response',
]);

export function syntheticCapabilityLabel(capabilityId: string | null | undefined, fallback: string | null | undefined, t: Translate): string {
  return syntheticCapabilityText(capabilityId, fallback, t, 'label');
}

export function syntheticCapabilityDescription(capabilityId: string | null | undefined, fallback: string | null | undefined, t: Translate): string {
  return syntheticCapabilityText(capabilityId, fallback, t, 'description');
}

export function fallbackCapabilityLabel(capabilityId: string | null | undefined): string {
  const id = (capabilityId || '').trim();
  if (!id) return '';
  return id.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function syntheticCapabilityText(capabilityId: string | null | undefined, fallback: string | null | undefined, t: Translate, field: 'label' | 'description'): string {
  const id = (capabilityId || '').trim();
  if (id && KNOWN_CAPABILITY_IDS.has(id)) {
    return t(`synthetic.capabilities.${id}.${field}`);
  }
  return (fallback || '').trim() || fallbackCapabilityLabel(id);
}
