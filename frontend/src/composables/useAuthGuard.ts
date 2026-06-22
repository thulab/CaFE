import { authState, has } from '../stores/auth';

export type Tier = 'public' | 'authed' | 'perm';

export interface GuardOutcome {
  /** If non-null, a hash path the caller should redirect to instead of rendering. */
  redirectTo: string | null;
}

/**
 * Evaluate route-level access based on the front-end mirror of backend Tier.
 *
 *   Tier 0 (public): always allowed.
 *   Tier 1 (authed): requires a logged-in user; otherwise -> #/login?next=<currentHash>.
 *   Tier 2 (perm):   requires logged-in AND the user has `perm`; otherwise -> #/login or #/forbidden.
 *
 * `currentHash` is the bare hash (without leading '#'), used to build ?next= for return-after-login.
 */
export function evaluateGuard(tier: Tier, perm: string | null, currentHash: string): GuardOutcome {
  if (tier === 'public') return { redirectTo: null };

  if (!authState.user) {
    const next = encodeURIComponent('#' + currentHash);
    return { redirectTo: `/login?next=${next}` };
  }

  if (tier === 'perm') {
    if (perm && !has(perm)) {
      return { redirectTo: `/forbidden?need=${encodeURIComponent(perm)}` };
    }
  }

  return { redirectTo: null };
}
