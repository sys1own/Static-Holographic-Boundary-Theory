from __future__ import annotations

import pytest

import shbt.audit.baryon_asymmetry as baryon_asymmetry
import shbt.audit.proton_stability_audit as proton_stability_audit
from shbt.core.qcd_kernel import build_qcd_residue_audit


def test_qcd_kernel_locks_the_benchmark_branch() -> None:
    audit = build_qcd_residue_audit(parent_level=312, lepton_level=26, quark_level=8)

    assert audit.color_charge.color_residue == 0
    assert audit.color_charge.singlet_locked is True
    assert audit.flux_tube.framing_residue == 0
    assert audit.flux_tube.confinement_locked is True
    assert audit.baryon_asymmetry_channel_locked is True
    assert audit.proton_stability_channel_locked is True


def test_qcd_kernel_reopens_flux_tube_residue_off_shell() -> None:
    audit = build_qcd_residue_audit(parent_level=312, lepton_level=27, quark_level=8)

    assert audit.flux_tube.framing_residue != 0
    assert audit.flux_tube.confinement_locked is False
    assert audit.baryon_asymmetry_channel_locked is False
    assert audit.proton_stability_channel_locked is False


def test_baryon_asymmetry_audit_consults_qcd_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    original = baryon_asymmetry.build_qcd_residue_audit

    def _tracked(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(baryon_asymmetry, "build_qcd_residue_audit", _tracked)

    audit = baryon_asymmetry.build_topological_baryogenesis_audit()

    assert calls["count"] == 1
    assert audit.sakharov.cp_violation_locked is True
    assert audit.sakharov.out_of_equilibrium_locked is True


def test_proton_stability_audit_consults_qcd_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    original = proton_stability_audit.build_qcd_residue_audit

    def _tracked(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(proton_stability_audit, "build_qcd_residue_audit", _tracked)

    audit = proton_stability_audit.build_proton_stability_audit()

    assert calls["count"] == len(audit.cells)
    assert audit.unique_protector == (26, 8, 312)
