from __future__ import annotations

"""Cryptographic lock for the published SHBT config and physics-profile assets."""

from dataclasses import dataclass
import hashlib
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT_DIR / "config" / "physics_profile_hashes.json"
READ_CHUNK_SIZE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ConfigHashAudit:
    relative_path: str
    expected_sha256: str
    actual_sha256: str | None
    file_exists: bool

    @property
    def matches(self) -> bool:
        return self.file_exists and self.actual_sha256 == self.expected_sha256


def compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, str]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Integrity manifest must be a mapping of relative paths to SHA-256 digests.")
    manifest: dict[str, str] = {}
    for relative_path, expected_sha256 in payload.items():
        if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
            raise TypeError("Integrity manifest keys and values must be strings.")
        manifest[relative_path] = expected_sha256
    return manifest


def verify_config_integrity(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    root_dir: Path = ROOT_DIR,
) -> tuple[bool, tuple[ConfigHashAudit, ...]]:
    manifest = load_manifest(manifest_path)
    audits: list[ConfigHashAudit] = []
    for relative_path, expected_sha256 in sorted(manifest.items()):
        target_path = Path(root_dir) / relative_path
        if not target_path.is_file():
            audits.append(
                ConfigHashAudit(
                    relative_path=relative_path,
                    expected_sha256=expected_sha256,
                    actual_sha256=None,
                    file_exists=False,
                )
            )
            continue
        audits.append(
            ConfigHashAudit(
                relative_path=relative_path,
                expected_sha256=expected_sha256,
                actual_sha256=compute_sha256(target_path),
                file_exists=True,
            )
        )
    audit_tuple = tuple(audits)
    return all(audit.matches for audit in audit_tuple), audit_tuple


def main() -> int:
    try:
        integrity_verified, audits = verify_config_integrity()
    except Exception as exc:
        print(f"CRITICAL INTEGRITY FAILURE: unable to load config integrity manifest: {exc}", file=sys.stderr)
        return 1

    if not integrity_verified:
        print("CRITICAL INTEGRITY FAILURE: configuration drift detected.", file=sys.stderr)
        for audit in audits:
            status = "PASS" if audit.matches else "FAIL"
            actual_sha256 = audit.actual_sha256 if audit.actual_sha256 is not None else "<missing>"
            stream = sys.stdout if audit.matches else sys.stderr
            print(f"[{status}] {audit.relative_path}", file=stream)
            print(f"  Expected SHA-256 : {audit.expected_sha256}", file=stream)
            print(f"  Observed SHA-256 : {actual_sha256}", file=stream)
        return 1

    print("Cryptographic Audit PASSED: physics-profile assets and benchmark config are bit-for-bit locked.")
    for audit in audits:
        print(f"[PASS] {audit.relative_path}")
        print(f"  SHA-256          : {audit.actual_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
