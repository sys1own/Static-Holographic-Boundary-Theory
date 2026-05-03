from __future__ import annotations

"""Cryptographic lock for the published SHBT benchmark configuration."""

import hashlib
import sys
from pathlib import Path


BENCHMARK_CONFIG_RELATIVE_PATH = Path("pub/config/benchmark_v1.yaml")
BENCHMARK_CONFIG_PATH = Path(__file__).parent / "config" / "benchmark_v1.yaml"
EXPECTED_SHA256 = "737667c8d0a2925f09ae89e40a68f7d26d2177df383f4a220e7d9c2c6b55dbf4"
READ_CHUNK_SIZE_BYTES = 1024 * 1024


def compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_benchmark_integrity(
    file_path: Path = BENCHMARK_CONFIG_PATH,
    *,
    expected_sha256: str = EXPECTED_SHA256,
) -> tuple[bool, str]:
    actual_sha256 = compute_sha256(Path(file_path))
    return actual_sha256 == expected_sha256, actual_sha256


def main() -> int:
    benchmark_path = BENCHMARK_CONFIG_PATH
    benchmark_label = BENCHMARK_CONFIG_RELATIVE_PATH.as_posix()

    if not benchmark_path.is_file():
        print(
            "CRITICAL INTEGRITY FAILURE: locked benchmark configuration is missing.",
            file=sys.stderr,
        )
        print(f"Target file                    : {benchmark_label}", file=sys.stderr)
        return 1

    integrity_verified, actual_sha256 = verify_benchmark_integrity(benchmark_path)
    if not integrity_verified:
        print(
            "CRITICAL INTEGRITY FAILURE: benchmark configuration drift detected.",
            file=sys.stderr,
        )
        print(f"Target file                    : {benchmark_label}", file=sys.stderr)
        print(f"Expected SHA-256               : {EXPECTED_SHA256}", file=sys.stderr)
        print(f"Observed SHA-256               : {actual_sha256}", file=sys.stderr)
        print(
            "Silent drift is not allowed: dependency updates or accidental edits have altered the 8-observable flavor residues.",
            file=sys.stderr,
        )
        return 1

    print(
        "Cryptographic Audit PASSED: benchmark_v1.yaml is bit-for-bit identical to the published (26, 8, 312) branch."
    )
    print(f"Target file                    : {benchmark_label}")
    print(f"SHA-256                        : {actual_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
