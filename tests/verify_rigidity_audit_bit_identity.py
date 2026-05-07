from __future__ import annotations

"""Cross-platform SHA-256 verifier for the rigidity audit JSON artifact."""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys


READ_CHUNK_SIZE_BYTES = 1024 * 1024
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "map_rigidity_landscape.py"


def compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_script_module():
    spec = importlib.util.spec_from_file_location("map_rigidity_landscape", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load rigidity landscape script from {SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generate_rigidity_json(output_dir: Path) -> Path:
    module = _load_script_module()
    output_path = Path(output_dir) / "rigidity_moat.json"
    scan = module.build_centered_rigidity_landscape_scan()
    return module.write_rigidity_landscape_json(scan, output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--hash-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = generate_rigidity_json(output_dir)
    artifact_hash = compute_sha256(json_path)
    hash_path = Path(args.hash_path) if args.hash_path is not None else output_dir / "rigidity_moat.sha256"
    hash_path.write_text(f"{artifact_hash}\n", encoding="utf-8")

    print("Rigidity Audit bit-identity artifact generated.")
    print(f"JSON artifact                  : {json_path.as_posix()}")
    print(f"SHA-256                        : {artifact_hash}")
    print(f"Hash record                    : {hash_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
