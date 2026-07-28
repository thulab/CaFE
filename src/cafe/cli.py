from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from cafe.analysis import runner as analysis
from cafe.calibration import runner as calibration
from cafe.generation import runner as generation
from cafe.inference import runner as inference
from cafe import pipeline
from cafe.validation import runner as validation


COMMANDS: dict[str, Callable[[], int]] = {
    "calibrate": calibration.main,
    "generate": generation.main,
    "validate": validation.main,
    "infer": inference.main,
    "analyze": analysis.main,
    "run": pipeline.main,
}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        command = sys.argv[1]
        sys.argv = [f"cafe {command}", *sys.argv[2:]]
        return COMMANDS[command]()
    parser = argparse.ArgumentParser(
        prog="cafe",
        description="Capability-Focused Extension benchmark",
    )
    parser.add_argument("command", choices=tuple(COMMANDS))
    args = parser.parse_args()
    return COMMANDS[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
