from __future__ import annotations

"""Cryptographic lock for the published SHBT config and physics-profile assets."""

import argparse
from collections.abc import Sequence
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


def build_manifest_hashes(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    root_dir: Path = ROOT_DIR,
) -> dict[str, str]:
    manifest = load_manifest(manifest_path)
    rebuilt_manifest: dict[str, str] = {}
    missing_files: list[str] = []

    for relative_path in sorted(manifest):
        target_path = Path(root_dir) / relative_path
        if not target_path.is_file():
            missing_files.append(relative_path)
            continue
        rebuilt_manifest[relative_path] = compute_sha256(target_path)

    if missing_files:
        missing_display = ", ".join(missing_files)
        raise FileNotFoundError(f"Manifest update requires existing files, but these are missing: {missing_display}")

    return rebuilt_manifest


def update_config_integrity_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    root_dir: Path = ROOT_DIR,
) -> dict[str, str]:
    rebuilt_manifest = build_manifest_hashes(manifest_path=manifest_path, root_dir=root_dir)
    Path(manifest_path).write_text(json.dumps(rebuilt_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rebuilt_manifest


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify or refresh the SHBT config integrity manifest.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Refresh the manifest with the current SHA-256 digests for all tracked config assets.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Manifest JSON path to verify or update.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=ROOT_DIR,
        help="Repository root containing the tracked config files.",
    )
    return parser.parse_args(tuple(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.update:
        try:
            updated_manifest = update_config_integrity_manifest(
                manifest_path=args.manifest_path,
                root_dir=args.root_dir,
            )
        except Exception as exc:
            print(f"CRITICAL INTEGRITY FAILURE: unable to update config integrity manifest: {exc}", file=sys.stderr)
            return 1

        print(f"Updated config integrity manifest: {args.manifest_path}")
        for relative_path, actual_sha256 in updated_manifest.items():
            print(f"[UPDATED] {relative_path}")
            print(f"  SHA-256          : {actual_sha256}")
        return 0

    try:
        integrity_verified, audits = verify_config_integrity(
            manifest_path=args.manifest_path,
            root_dir=args.root_dir,
        )
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
