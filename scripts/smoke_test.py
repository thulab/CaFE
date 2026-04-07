from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test.integration.test_smoke_flow import run_smoke_flow


if __name__ == "__main__":
    run_smoke_flow()
