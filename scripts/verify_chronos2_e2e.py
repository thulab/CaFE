from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test.integration.test_verify_chronos2_e2e import run_real_chronos2_e2e


if __name__ == "__main__":
    run_real_chronos2_e2e()
