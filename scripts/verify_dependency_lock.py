from __future__ import annotations

"""Verify that the current interpreter matches the frozen transitive dependency lock."""

import importlib.metadata as metadata
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_PATH = ROOT_DIR / "requirements.lock"


def load_locked_requirements(lock_path: Path = DEFAULT_LOCK_PATH) -> dict[str, str]:
    locked: dict[str, str] = {}
    for raw_line in Path(lock_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if separator != "==" or not name or not version:
            raise ValueError(f"Invalid lock entry: {raw_line!r}")
        locked[name.lower().replace("_", "-")] = version
    return locked


def verify_dependency_lock(lock_path: Path = DEFAULT_LOCK_PATH) -> tuple[bool, tuple[str, ...]]:
    locked = load_locked_requirements(lock_path)
    drift: list[str] = []
    for name, expected_version in sorted(locked.items()):
        try:
            observed_version = metadata.version(name)
        except metadata.PackageNotFoundError:
            drift.append(f"{name}=={expected_version} is missing")
            continue
        if observed_version != expected_version:
            drift.append(f"{name} expected {expected_version} but observed {observed_version}")
    return not drift, tuple(drift)


def main() -> int:
    verified, drift = verify_dependency_lock()
    if not verified:
        print("DEPENDENCY LOCK FAILURE: the current interpreter drifts from requirements.lock.", file=sys.stderr)
        for entry in drift:
            print(f"- {entry}", file=sys.stderr)
        return 1
    print("Dependency lock PASSED: current interpreter matches requirements.lock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
