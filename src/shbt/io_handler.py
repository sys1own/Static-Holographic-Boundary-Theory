from __future__ import annotations

"""Structured tensor transport I/O for bit-perfect cross-sector exchange.

Historically the transport layer emitted text-oriented logs that were useful
for inspection but lossy for prime-indexed lattice exchange. This module
replaces that pattern with a minimal, dependency-free Zarr directory-store
writer/reader so audit tensors can move between local and clustered runs
without decimal re-serialization or rounding drift.

The implementation intentionally targets the small Zarr v2 subset required by
the SHBT audit stack:

- chunked NumPy tensors written as raw binary payloads,
- JSON metadata for provenance and replay,
- SHA-256 digests for bit-perfect integrity checks.

HDF5 remains a recognized backend label for configuration purposes, but the
portable built-in implementation provided here is the Zarr transport backend.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr


DEFAULT_TENSOR_FORMAT = "zarr"
DEFAULT_DATASET_NAME = "transport_tensor"
DEFAULT_CHUNK_TARGET_BYTES = 8 * 1024 * 1024
SUPPORTED_TENSOR_FORMATS = frozenset({"zarr", "hdf5"})
ZARR_FORMAT_VERSION = 2


class TransportStorageError(RuntimeError):
    """Raised when structured tensor storage fails."""


class TensorIntegrityError(TransportStorageError):
    """Raised when a stored tensor no longer matches its recorded digest."""


@dataclass(frozen=True)
class TransportTensorArtifact:
    name: str
    path: Path
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...]
    sha256: str
    attributes: dict[str, Any]


@dataclass(frozen=True)
class TransportTensorBundle:
    root: Path
    tensor_format: str
    group_attributes: dict[str, Any]
    artifacts: tuple[TransportTensorArtifact, ...]


@dataclass(frozen=True)
class LoadedTransportTensorBundle:
    root: Path
    tensor_format: str
    group_attributes: dict[str, Any]
    tensors: dict[str, np.ndarray]
    dataset_attributes: dict[str, dict[str, Any]]


def _safe_dataset_name(name: str) -> str:
    candidate = Path(name).name
    if candidate != name or candidate in {"", ".", ".."}:
        raise ValueError("dataset names must be simple path segments.")
    return candidate


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _dtype_metadata(dtype: np.dtype[Any]) -> str | list[Any]:
    return dtype_to_descr(dtype)


def _tupleize_dtype_metadata(value: Any) -> Any:
    if isinstance(value, list):
        return [tuple(_tupleize_dtype_metadata(item)) if isinstance(item, list) else _tupleize_dtype_metadata(item) for item in value]
    return value


def _dtype_from_metadata(value: Any) -> np.dtype[Any]:
    normalized_value = _tupleize_dtype_metadata(value)
    return np.dtype(descr_to_dtype(normalized_value))


def _coerce_array(tensor: Any) -> np.ndarray:
    array = np.asarray(tensor)
    if array.dtype.hasobject:
        raise TypeError("object-dtype tensors cannot be stored in bit-perfect structured transport.")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def _normalize_chunk_shape(shape: tuple[int, ...], chunk_shape: Sequence[int] | None, *, itemsize: int) -> tuple[int, ...]:
    if chunk_shape is not None:
        resolved_chunk_shape = tuple(int(axis) for axis in chunk_shape)
        if len(resolved_chunk_shape) != len(shape):
            raise ValueError("chunk_shape must match the tensor dimensionality.")
        if any(axis < 1 for axis in resolved_chunk_shape):
            raise ValueError("chunk_shape entries must be positive integers.")
        return tuple(min(axis, extent) for axis, extent in zip(resolved_chunk_shape, shape, strict=True))

    if not shape:
        return ()

    resolved_chunk_shape = list(shape)
    while math.prod(resolved_chunk_shape) * int(itemsize) > DEFAULT_CHUNK_TARGET_BYTES and any(axis > 1 for axis in resolved_chunk_shape):
        axis_index = max(range(len(resolved_chunk_shape)), key=resolved_chunk_shape.__getitem__)
        resolved_chunk_shape[axis_index] = max(1, math.ceil(resolved_chunk_shape[axis_index] / 2))
    return tuple(resolved_chunk_shape)


def _iter_chunk_regions(shape: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], tuple[slice, ...]], ...]:
    if not shape:
        return (((), ()),)

    chunk_counts = tuple(math.ceil(extent / chunk) for extent, chunk in zip(shape, chunks, strict=True))
    regions: list[tuple[tuple[int, ...], tuple[slice, ...]]] = []
    for chunk_index in product(*(range(count) for count in chunk_counts)):
        slices = tuple(
            slice(index * chunk, min((index + 1) * chunk, extent))
            for index, chunk, extent in zip(chunk_index, chunks, shape, strict=True)
        )
        regions.append((chunk_index, slices))
    return tuple(regions)


def _chunk_key(chunk_index: tuple[int, ...], *, ndim: int) -> str:
    if ndim == 0:
        return "0"
    return ".".join(str(index) for index in chunk_index)


def _chunk_shape_from_slices(slices: tuple[slice, ...]) -> tuple[int, ...]:
    if not slices:
        return ()
    return tuple(int(chunk_slice.stop) - int(chunk_slice.start) for chunk_slice in slices)


def _sha256_digest(array: np.ndarray) -> str:
    return sha256(array.tobytes(order="C")).hexdigest()


def _prepare_destination(destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing structured tensor store {destination}.")
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_zarr_dataset(
    dataset_dir: Path,
    tensor: np.ndarray,
    *,
    dataset_attrs: Mapping[str, Any] | None,
    chunk_shape: Sequence[int] | None,
) -> TransportTensorArtifact:
    dataset_dir.mkdir(parents=True, exist_ok=True)

    resolved_chunks = _normalize_chunk_shape(tuple(int(axis) for axis in tensor.shape), chunk_shape, itemsize=tensor.dtype.itemsize)
    zarray_payload = {
        "zarr_format": ZARR_FORMAT_VERSION,
        "shape": list(tensor.shape),
        "chunks": list(resolved_chunks),
        "dtype": _dtype_metadata(tensor.dtype),
        "compressor": None,
        "fill_value": None,
        "filters": None,
        "order": "C",
        "dimension_separator": ".",
    }
    _write_json(dataset_dir / ".zarray", zarray_payload)

    tensor_digest = _sha256_digest(tensor)
    resolved_dataset_attrs = {
        "bit_perfect_transport": True,
        "bit_perfect_sha256": tensor_digest,
        "dtype": tensor.dtype.str,
        "shape": list(int(axis) for axis in tensor.shape),
        "chunks": list(int(axis) for axis in resolved_chunks),
        "nbytes": int(tensor.nbytes),
    }
    if dataset_attrs is not None:
        resolved_dataset_attrs.update(_json_safe(dataset_attrs))
    _write_json(dataset_dir / ".zattrs", resolved_dataset_attrs)

    for chunk_index, slices in _iter_chunk_regions(tuple(int(axis) for axis in tensor.shape), resolved_chunks):
        chunk_path = dataset_dir / _chunk_key(chunk_index, ndim=tensor.ndim)
        chunk = np.asarray(tensor[slices], dtype=tensor.dtype)
        chunk_path.write_bytes(chunk.tobytes(order="C"))

    return TransportTensorArtifact(
        name=dataset_dir.name,
        path=dataset_dir,
        shape=tuple(int(axis) for axis in tensor.shape),
        dtype=tensor.dtype.str,
        chunks=resolved_chunks,
        sha256=tensor_digest,
        attributes=dict(resolved_dataset_attrs),
    )


def write_transport_tensors(
    destination: Path | str,
    tensors: Mapping[str, Any],
    *,
    group_attributes: Mapping[str, Any] | None = None,
    dataset_attributes: Mapping[str, Mapping[str, Any]] | None = None,
    chunk_shapes: Mapping[str, Sequence[int]] | None = None,
    tensor_format: str = DEFAULT_TENSOR_FORMAT,
    overwrite: bool = False,
) -> TransportTensorBundle:
    resolved_format = str(tensor_format).lower()
    if resolved_format not in SUPPORTED_TENSOR_FORMATS:
        raise ValueError(f"Unsupported tensor format {tensor_format!r}; expected one of {sorted(SUPPORTED_TENSOR_FORMATS)}.")
    if resolved_format != "zarr":
        raise TransportStorageError(
            "The portable transport layer currently ships with a native Zarr backend only; "
            "configure tensor_format='zarr' for dependency-free operation."
        )
    if not tensors:
        raise ValueError("tensors must contain at least one named tensor.")

    resolved_destination = Path(destination)
    _prepare_destination(resolved_destination, overwrite=overwrite)

    _write_json(resolved_destination / ".zgroup", {"zarr_format": ZARR_FORMAT_VERSION})
    resolved_group_attributes = {
        "tensor_format": resolved_format,
        "storage_layout": "directory_store",
        "bit_perfect_transport": True,
        "integrity_hash": "sha256",
        "transport_storage_version": 1,
    }
    if group_attributes is not None:
        resolved_group_attributes.update(_json_safe(group_attributes))
    _write_json(resolved_destination / ".zattrs", resolved_group_attributes)

    artifacts: list[TransportTensorArtifact] = []
    for dataset_name in sorted(tensors):
        resolved_dataset_name = _safe_dataset_name(dataset_name)
        tensor = _coerce_array(tensors[dataset_name])
        artifacts.append(
            _write_zarr_dataset(
                resolved_destination / resolved_dataset_name,
                tensor,
                dataset_attrs=None if dataset_attributes is None else dataset_attributes.get(resolved_dataset_name),
                chunk_shape=None if chunk_shapes is None else chunk_shapes.get(resolved_dataset_name),
            )
        )

    return TransportTensorBundle(
        root=resolved_destination,
        tensor_format=resolved_format,
        group_attributes=dict(resolved_group_attributes),
        artifacts=tuple(artifacts),
    )


def write_transport_tensor(
    destination: Path | str,
    tensor: Any,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    group_attributes: Mapping[str, Any] | None = None,
    dataset_attributes: Mapping[str, Any] | None = None,
    chunk_shape: Sequence[int] | None = None,
    tensor_format: str = DEFAULT_TENSOR_FORMAT,
    overwrite: bool = False,
) -> TransportTensorArtifact:
    bundle = write_transport_tensors(
        destination,
        {dataset_name: tensor},
        group_attributes=group_attributes,
        dataset_attributes=None if dataset_attributes is None else {dataset_name: dataset_attributes},
        chunk_shapes=None if chunk_shape is None else {dataset_name: chunk_shape},
        tensor_format=tensor_format,
        overwrite=overwrite,
    )
    return bundle.artifacts[0]


def _load_zarr_dataset(dataset_dir: Path, *, verify_checksums: bool) -> tuple[np.ndarray, dict[str, Any]]:
    zarray_payload = _read_json(dataset_dir / ".zarray")
    dataset_attributes = dict(_read_json(dataset_dir / ".zattrs")) if (dataset_dir / ".zattrs").is_file() else {}

    tensor_shape = tuple(int(axis) for axis in zarray_payload["shape"])
    chunk_shape = tuple(int(axis) for axis in zarray_payload["chunks"])
    tensor_dtype = _dtype_from_metadata(zarray_payload["dtype"])
    order = str(zarray_payload.get("order", "C"))

    tensor = np.empty(tensor_shape, dtype=tensor_dtype)
    for chunk_index, slices in _iter_chunk_regions(tensor_shape, chunk_shape):
        chunk_path = dataset_dir / _chunk_key(chunk_index, ndim=tensor.ndim)
        if not chunk_path.is_file():
            raise TransportStorageError(f"Missing chunk {chunk_path.name!r} in structured tensor store {dataset_dir}.")
        raw_chunk = chunk_path.read_bytes()
        expected_chunk_shape = _chunk_shape_from_slices(slices)
        expected_item_count = math.prod(expected_chunk_shape) if expected_chunk_shape else 1
        expected_size = int(expected_item_count) * int(tensor_dtype.itemsize)
        if len(raw_chunk) != expected_size:
            raise TransportStorageError(
                f"Chunk {chunk_path.name!r} in {dataset_dir} has {len(raw_chunk)} bytes; expected {expected_size}."
            )
        chunk = np.frombuffer(raw_chunk, dtype=tensor_dtype).reshape(expected_chunk_shape, order=order)
        tensor[slices] = chunk

    if verify_checksums:
        expected_digest = dataset_attributes.get("bit_perfect_sha256")
        if isinstance(expected_digest, str):
            observed_digest = _sha256_digest(np.ascontiguousarray(tensor))
            if observed_digest != expected_digest:
                raise TensorIntegrityError(
                    f"Tensor {dataset_dir.name!r} failed SHA-256 verification: expected {expected_digest}, observed {observed_digest}."
                )

    return tensor, dataset_attributes


def read_transport_tensors(
    source: Path | str,
    *,
    verify_checksums: bool = True,
) -> LoadedTransportTensorBundle:
    resolved_source = Path(source)
    if not (resolved_source / ".zgroup").is_file():
        raise FileNotFoundError(f"Structured tensor store {resolved_source} does not contain a .zgroup marker.")

    group_attributes = dict(_read_json(resolved_source / ".zattrs")) if (resolved_source / ".zattrs").is_file() else {}

    tensors: dict[str, np.ndarray] = {}
    dataset_attributes: dict[str, dict[str, Any]] = {}
    for child in sorted(resolved_source.iterdir()):
        if not child.is_dir() or not (child / ".zarray").is_file():
            continue
        tensor, child_attributes = _load_zarr_dataset(child, verify_checksums=verify_checksums)
        tensors[child.name] = tensor
        dataset_attributes[child.name] = child_attributes

    if not tensors:
        raise TransportStorageError(f"Structured tensor store {resolved_source} does not contain any datasets.")

    return LoadedTransportTensorBundle(
        root=resolved_source,
        tensor_format=str(group_attributes.get("tensor_format", DEFAULT_TENSOR_FORMAT)),
        group_attributes=group_attributes,
        tensors=tensors,
        dataset_attributes=dataset_attributes,
    )


def read_transport_tensor(
    source: Path | str,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    verify_checksums: bool = True,
) -> np.ndarray:
    bundle = read_transport_tensors(source, verify_checksums=verify_checksums)
    if dataset_name not in bundle.tensors:
        raise KeyError(f"Dataset {dataset_name!r} is not present in structured tensor store {source}.")
    return bundle.tensors[dataset_name]


__all__ = [
    "DEFAULT_CHUNK_TARGET_BYTES",
    "DEFAULT_DATASET_NAME",
    "DEFAULT_TENSOR_FORMAT",
    "LoadedTransportTensorBundle",
    "SUPPORTED_TENSOR_FORMATS",
    "TensorIntegrityError",
    "TransportStorageError",
    "TransportTensorArtifact",
    "TransportTensorBundle",
    "read_transport_tensor",
    "read_transport_tensors",
    "write_transport_tensor",
    "write_transport_tensors",
]
