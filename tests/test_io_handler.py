from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shbt.config_loader import ConfigLoader
from shbt.io_handler import TensorIntegrityError, read_transport_tensor, read_transport_tensors, write_transport_tensor


def test_write_transport_tensor_round_trips_bit_perfectly(tmp_path: Path) -> None:
    tensor = (np.arange(3 * 5 * 7, dtype=np.float64).reshape(3, 5, 7) / 11.0).astype(np.float64)

    artifact = write_transport_tensor(
        tmp_path / "prime_lattice.zarr",
        tensor,
        dataset_name="prime_lattice",
        group_attributes={"audit": "full-universe", "transport_stage": "cross-sector"},
        dataset_attributes={"prime_indexed_lattice": True, "sector": "transport"},
        chunk_shape=(2, 3, 4),
    )

    payload = read_transport_tensors(tmp_path / "prime_lattice.zarr")

    assert artifact.name == "prime_lattice"
    assert artifact.chunks == (2, 3, 4)
    assert np.array_equal(payload.tensors["prime_lattice"], tensor)
    assert payload.group_attributes["bit_perfect_transport"] is True
    assert payload.dataset_attributes["prime_lattice"]["prime_indexed_lattice"] is True
    assert payload.dataset_attributes["prime_lattice"]["bit_perfect_sha256"] == artifact.sha256


def test_read_transport_tensor_detects_corrupted_chunk_bytes(tmp_path: Path) -> None:
    tensor = np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    write_transport_tensor(tmp_path / "transport.zarr", tensor, dataset_name="transport", chunk_shape=(1, 3, 5))

    first_chunk = tmp_path / "transport.zarr" / "transport" / "0.0.0"
    raw = bytearray(first_chunk.read_bytes())
    raw[0] ^= 0x01
    first_chunk.write_bytes(bytes(raw))

    with pytest.raises(TensorIntegrityError):
        read_transport_tensor(tmp_path / "transport.zarr", dataset_name="transport")


def test_hpc_scaling_profile_exposes_cluster_ready_uniqueness_scan_defaults() -> None:
    compute_config = ConfigLoader().load_compute_cluster_config("compute/hpc_scaling.yaml")

    assert compute_config["cluster"]["default_backend"] == "dask"
    assert compute_config["cluster"]["failover_backend"] == "mpi"
    assert compute_config["workloads"]["uniqueness_scans"]["target_resolution"]["parent_levels"] == 4096
    assert compute_config["dask"]["adaptive"]["maximum_workers"] == 256
    assert compute_config["storage"]["tensor_format"] == "zarr"
    assert compute_config["storage"]["checksum"] == "sha256"
