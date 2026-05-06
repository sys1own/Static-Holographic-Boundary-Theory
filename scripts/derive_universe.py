from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.core.derivation_api import DEFAULT_PRECISION, UniverseFactory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive branch-fixed universe constants from the (26, 8, 312) ledger.")
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION, help="Decimal precision used for the derivation ledger.")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    print(UniverseFactory.generate_ledger(kind="universe", precision=max(args.precision, DEFAULT_PRECISION)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
