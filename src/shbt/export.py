from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from shbt.constants import BENCHMARK_MANIFEST_FILENAME


_CANONICAL_JSON_FILENAMES = frozenset({".zarray", ".zattrs", ".zgroup"})
_CANONICAL_TEXT_SUFFIXES = frozenset({".md", ".tex", ".txt", ".yaml", ".yml"})


def _sha256_hexdigest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_text_bytes(payload: bytes) -> bytes:
    return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _canonical_json_bytes(payload: bytes) -> bytes:
    normalized_payload = json.loads(_normalize_text_bytes(payload).decode("utf-8"))
    return (json.dumps(normalized_payload, indent=2, sort_keys=True, default=_normalize_json_value) + "\n").encode(
        "utf-8"
    )


def _canonical_csv_bytes(payload: bytes) -> bytes:
    normalized_text = _normalize_text_bytes(payload).decode("utf-8")
    source = io.StringIO(normalized_text)
    destination = io.StringIO()
    writer = csv.writer(destination, lineterminator="\n")
    for row in csv.reader(source):
        writer.writerow(row)
    return destination.getvalue().encode("utf-8")


def _canonical_zarr_bytes(output_path: Path) -> tuple[bytes, int, int]:
    manifest_bytes = bytearray()
    member_count = 0
    size_bytes = 0

    for child in sorted(output_path.rglob("*"), key=lambda path: path.relative_to(output_path).as_posix()):
        if child.is_dir():
            continue
        member_count += 1
        raw_bytes = child.read_bytes()
        size_bytes += len(raw_bytes)
        if child.name in _CANONICAL_JSON_FILENAMES:
            canonical_bytes = _canonical_json_bytes(raw_bytes)
        else:
            canonical_bytes = raw_bytes
        relative_path = child.relative_to(output_path).as_posix().encode("utf-8")
        manifest_bytes.extend(relative_path)
        manifest_bytes.extend(b"\0")
        manifest_bytes.extend(str(len(canonical_bytes)).encode("ascii"))
        manifest_bytes.extend(b"\0")
        manifest_bytes.extend(canonical_bytes)
        manifest_bytes.extend(b"\0")

    return bytes(manifest_bytes), member_count, size_bytes


def _build_canonical_manifest_entry(output_path: Path, *, root_dir: Path) -> dict[str, Any]:
    relative_path = output_path.relative_to(root_dir).as_posix()
    if output_path.is_dir():
        if output_path.suffix != ".zarr":
            raise ValueError(f"Unsupported manifest directory artifact: {output_path}")
        canonical_bytes, member_count, size_bytes = _canonical_zarr_bytes(output_path)
        return {
            "path": relative_path,
            "kind": "zarr",
            "cross_arch_stable": True,
            "hash_basis": "canonical_zarr",
            "canonical_size_bytes": int(len(canonical_bytes)),
            "member_count": int(member_count),
            "size_bytes": int(size_bytes),
            "sha256": _sha256_hexdigest(canonical_bytes),
        }

    suffix = output_path.suffix.lower()
    raw_bytes = output_path.read_bytes()
    if suffix == ".json":
        canonical_bytes = _canonical_json_bytes(raw_bytes)
        return {
            "path": relative_path,
            "kind": "json",
            "cross_arch_stable": True,
            "hash_basis": "canonical_json",
            "canonical_size_bytes": int(len(canonical_bytes)),
            "sha256": _sha256_hexdigest(canonical_bytes),
        }
    if suffix == ".csv":
        canonical_bytes = _canonical_csv_bytes(raw_bytes)
        return {
            "path": relative_path,
            "kind": "csv",
            "cross_arch_stable": True,
            "hash_basis": "canonical_csv",
            "canonical_size_bytes": int(len(canonical_bytes)),
            "sha256": _sha256_hexdigest(canonical_bytes),
        }
    if suffix in _CANONICAL_TEXT_SUFFIXES:
        canonical_bytes = _normalize_text_bytes(raw_bytes)
        return {
            "path": relative_path,
            "kind": "text",
            "cross_arch_stable": True,
            "hash_basis": "canonical_text",
            "canonical_size_bytes": int(len(canonical_bytes)),
            "sha256": _sha256_hexdigest(canonical_bytes),
        }
    return {
        "path": relative_path,
        "kind": "binary" if suffix else "artifact",
        "cross_arch_stable": False,
        "hash_basis": "listed_only",
    }


def build_canonical_benchmark_manifest(
    output_dir: Path,
    *,
    benchmark_tuple: Sequence[int] | None = None,
    exclude_names: Iterable[str] = (),
) -> dict[str, Any]:
    resolved_output_dir = Path(output_dir)
    if not resolved_output_dir.is_dir():
        raise FileNotFoundError(f"Benchmark output directory not found: {resolved_output_dir}")

    excluded = {BENCHMARK_MANIFEST_FILENAME, *(str(name) for name in exclude_names)}
    artifact_entries = [
        _build_canonical_manifest_entry(candidate, root_dir=resolved_output_dir)
        for candidate in sorted(resolved_output_dir.iterdir(), key=lambda path: path.name)
        if candidate.name not in excluded and (candidate.is_file() or candidate.suffix == ".zarr")
    ]

    stable_entries = [entry for entry in artifact_entries if entry["cross_arch_stable"]]
    aggregate_digest = hashlib.sha256()
    for entry in stable_entries:
        aggregate_digest.update(entry["path"].encode("utf-8"))
        aggregate_digest.update(b"\0")
        aggregate_digest.update(str(entry["sha256"]).encode("ascii"))
        aggregate_digest.update(b"\0")

    payload: dict[str, Any] = {
        "artifact": "Canonical Cross-Architecture Benchmark Manifest",
        "manifest_version": 1,
        "normalization_profile": "canonical-cross-arch-v1",
        "artifact_count": int(len(artifact_entries)),
        "cross_arch_artifact_count": int(len(stable_entries)),
        "aggregate_sha256": aggregate_digest.hexdigest(),
        "artifacts": artifact_entries,
    }
    if benchmark_tuple is not None:
        payload["benchmark_tuple"] = [int(value) for value in benchmark_tuple]
    return payload


def write_canonical_benchmark_manifest(
    output_dir: Path,
    *,
    benchmark_tuple: Sequence[int] | None = None,
    filename: str = BENCHMARK_MANIFEST_FILENAME,
    exclude_names: Iterable[str] = (),
) -> Path:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_canonical_benchmark_manifest(
        resolved_output_dir,
        benchmark_tuple=benchmark_tuple,
        exclude_names={filename, *exclude_names},
    )
    return write_json_artifact(resolved_output_dir / filename, payload)


def _normalize_json_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), sort_keys=True)
    if is_dataclass(value):
        return json.dumps(asdict(value), default=_normalize_json_value, sort_keys=True)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), default=_normalize_json_value, sort_keys=True)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return json.dumps(list(value), default=_normalize_json_value)
    return str(value)


def _normalize_zarr_attr_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    return _normalize_json_value(value)


def _coerce_tensor_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        resolved_array = value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        resolved_array = np.asarray(value)
    else:
        return None

    if resolved_array.ndim < 2:
        return None
    if resolved_array.dtype != object:
        return np.ascontiguousarray(resolved_array)

    flattened = tuple(resolved_array.reshape(-1))
    if not flattened:
        return np.ascontiguousarray(resolved_array.astype(float))
    if all(isinstance(item, Decimal) for item in flattened):
        return np.ascontiguousarray(np.asarray(value, dtype=str))
    if all(isinstance(item, (int, float, np.integer, np.floating, bool, np.bool_)) for item in flattened):
        return np.ascontiguousarray(np.asarray(value, dtype=float))
    if all(isinstance(item, complex) for item in flattened):
        return np.ascontiguousarray(np.asarray(value, dtype=np.complex128))
    if all(isinstance(item, str) for item in flattened):
        return np.ascontiguousarray(np.asarray(value, dtype=str))
    return None


def _normalize_dataset_component(component: str) -> str:
    normalized = component.replace("\\", "/").replace(" ", "_")
    normalized = normalized.replace("..", ".")
    normalized = normalized.strip("/.")
    return normalized or "tensor"


def collect_multidimensional_tensors(payload: Any, *, prefix: str | None = None) -> dict[str, np.ndarray]:
    tensor = _coerce_tensor_array(payload)
    if tensor is not None:
        dataset_name = _normalize_dataset_component(prefix or "tensor")
        return {dataset_name: tensor}

    if is_dataclass(payload):
        payload = asdict(payload)

    extracted: dict[str, np.ndarray] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            extracted.update(collect_multidimensional_tensors(value, prefix=child_prefix))
        return extracted

    if hasattr(payload, "__dict__") and not isinstance(payload, type):
        for key, value in vars(payload).items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            extracted.update(collect_multidimensional_tensors(value, prefix=child_prefix))
        return extracted

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            extracted.update(collect_multidimensional_tensors(value, prefix=child_prefix))
    return extracted


def _zarr_chunk_key(shape: tuple[int, ...]) -> str:
    dimensions = max(len(shape), 1)
    return ".".join("0" for _ in range(dimensions))


def _ensure_zarr_group(path: Path, *, attrs: Mapping[str, Any] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if attrs is not None:
        (path / ".zattrs").write_text(
            json.dumps(dict(attrs), indent=2, sort_keys=True, default=_normalize_zarr_attr_value) + "\n",
            encoding="utf-8",
        )
    elif not (path / ".zattrs").exists():
        (path / ".zattrs").write_text("{}\n", encoding="utf-8")


def _write_zarr_dataset(root_path: Path, dataset_name: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and not np.little_endian):
        array = array.byteswap().newbyteorder("<")

    components = tuple(_normalize_dataset_component(component) for component in dataset_name.split("."))
    group_path = root_path
    for component in components[:-1]:
        group_path = group_path / component
        _ensure_zarr_group(group_path)

    dataset_path = group_path / components[-1]
    dataset_path.mkdir(parents=True, exist_ok=True)
    zarray_payload = {
        "chunks": [int(length) for length in (array.shape or (1,))],
        "compressor": None,
        "dtype": array.dtype.str,
        "fill_value": None,
        "filters": None,
        "order": "C",
        "shape": [int(length) for length in array.shape],
        "zarr_format": 2,
    }
    (dataset_path / ".zarray").write_text(json.dumps(zarray_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dataset_path / ".zattrs").write_text("{}\n", encoding="utf-8")
    (dataset_path / _zarr_chunk_key(array.shape)).write_bytes(array.tobytes(order="C"))


def write_zarr_artifact(
    output_path: Path,
    tensors: Mapping[str, Any],
    *,
    attrs: Mapping[str, Any] | None = None,
) -> Path:
    resolved_output_path = Path(output_path)
    if resolved_output_path.suffix != ".zarr":
        resolved_output_path = resolved_output_path.with_suffix(".zarr")
    if resolved_output_path.exists():
        if resolved_output_path.is_dir():
            shutil.rmtree(resolved_output_path)
        else:
            resolved_output_path.unlink()

    normalized_tensors = {str(name): np.ascontiguousarray(np.asarray(value)) for name, value in tensors.items()}
    root_attrs = {
        "datasets": tuple(sorted(normalized_tensors)),
        "tensor_format": "zarr",
    }
    if attrs is not None:
        root_attrs.update(dict(attrs))

    _ensure_zarr_group(resolved_output_path, attrs=root_attrs)
    for dataset_name, value in normalized_tensors.items():
        _write_zarr_dataset(resolved_output_path, dataset_name, value)
    return resolved_output_path


def read_zarr_artifact(output_path: Path) -> dict[str, np.ndarray]:
    resolved_output_path = Path(output_path)
    artifacts: dict[str, np.ndarray] = {}
    for zarray_path in sorted(resolved_output_path.rglob(".zarray")):
        metadata = json.loads(zarray_path.read_text(encoding="utf-8"))
        dataset_path = zarray_path.parent
        shape = tuple(int(length) for length in metadata.get("shape", ()))
        dtype = np.dtype(metadata["dtype"])
        raw = (dataset_path / _zarr_chunk_key(shape)).read_bytes()
        array = np.frombuffer(raw, dtype=dtype)
        if shape:
            array = array.reshape(shape, order=metadata.get("order", "C"))
        else:
            array = array.reshape(())
        dataset_key = ".".join(dataset_path.relative_to(resolved_output_path).parts)
        artifacts[dataset_key] = array.copy()
    return artifacts


def write_tensor_artifact(
    output_path: Path,
    payload: Any,
    *,
    attrs: Mapping[str, Any] | None = None,
) -> Path | None:
    tensors = collect_multidimensional_tensors(payload)
    if not tensors:
        return None
    return write_zarr_artifact(Path(output_path), tensors, attrs=attrs)


def write_json_artifact(output_path: Path, payload: Any) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_normalize_json_value) + "\n",
        encoding="utf-8",
    )
    return resolved_output_path


def write_csv_artifact(output_path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fieldnames = tuple(str(fieldname) for fieldname in fieldnames)
    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: _normalize_csv_value(row.get(fieldname)) for fieldname in resolved_fieldnames})
    return resolved_output_path


def export_matrix_spectrum_csv(output_path: Path, spectrum_audit: Mapping[str, Any]) -> Path:
    indices = np.asarray(spectrum_audit.get("indices", ()), dtype=int)
    singular_values = np.asarray(spectrum_audit.get("singular_values", ()), dtype=float)
    rank = int(spectrum_audit.get("rank", 0))
    sigma_min = float(spectrum_audit.get("sigma_min", float("nan")))
    rows = (
        {
            "index": int(index),
            "singular_value": float(singular_value),
            "rank": rank,
            "sigma_min": sigma_min,
        }
        for index, singular_value in zip(indices, singular_values, strict=True)
    )
    return write_csv_artifact(Path(output_path), ("index", "singular_value", "rank", "sigma_min"), rows)


def export_dm_fingerprint_figure(
    *,
    output_path: Path,
    dm_mass_gev: float,
    rhn_scale_gev: float,
    gauge_sigma_cm2: float,
    beta_squared: float,
) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = (r"$m_{\rm DM}$ [GeV]", r"$M_N$ [GeV]", r"$\sigma_g$ [cm$^2$]", r"$\beta^2$")
    values = np.asarray((dm_mass_gev, rhn_scale_gev, gauge_sigma_cm2, beta_squared), dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    try:
        ax.bar(labels, values, color=("#1d4ed8", "#0f766e", "#7c3aed", "#b45309"), width=0.62)
        ax.set_yscale("log")
        ax.set_ylabel("value")
        ax.set_title("Dark-sector fingerprint summary")
        ax.grid(True, axis="y", which="both", alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        fig.savefig(resolved_output_path, dpi=200)
    finally:
        plt.close(fig)

    return resolved_output_path


def export_transport_covariance_diagnostics(output_path: Path, transport_covariance: Any) -> Path:
    resolved_output_path = Path(output_path)
    tensor_artifact = write_tensor_artifact(
        resolved_output_path.with_suffix(".zarr"),
        transport_covariance,
        attrs={"source_artifact": resolved_output_path.name},
    )
    if tensor_artifact is None:
        return write_json_artifact(resolved_output_path, transport_covariance)

    if is_dataclass(transport_covariance):
        payload = asdict(transport_covariance)
    elif isinstance(transport_covariance, Mapping):
        payload = dict(transport_covariance)
    else:
        payload = {"payload": transport_covariance}
    payload["tensor_artifact"] = tensor_artifact.as_posix()
    return write_json_artifact(resolved_output_path, payload)


__all__ = [
    "build_canonical_benchmark_manifest",
    "collect_multidimensional_tensors",
    "export_dm_fingerprint_figure",
    "export_matrix_spectrum_csv",
    "export_transport_covariance_diagnostics",
    "read_zarr_artifact",
    "write_canonical_benchmark_manifest",
    "write_tensor_artifact",
    "write_csv_artifact",
    "write_json_artifact",
    "write_zarr_artifact",
]
