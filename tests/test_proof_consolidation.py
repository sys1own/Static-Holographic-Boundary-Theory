from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_proofs_directory_is_consolidated() -> None:
    assert (ROOT_DIR / "proofs").is_dir()
    assert not (ROOT_DIR / "proof").exists()
    assert (ROOT_DIR / "proofs" / "distinction_logic.lean").is_file()
    assert (ROOT_DIR / "proofs" / "anomaly_cancellation.lean").is_file()


def test_anomaly_cancellation_lean_skeleton_declares_gko_theorem() -> None:
    content = (ROOT_DIR / "proofs" / "anomaly_cancellation.lean").read_text(encoding="utf-8")

    assert "namespace SHBT" in content
    assert "def GKOOrthogonal" in content
    assert "axiom anomalyCancellation_of_GKOOrthogonality" in content
    assert "theorem gkoOrthogonality26D" in content
