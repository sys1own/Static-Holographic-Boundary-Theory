from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest

from shbt import constants as constants_module
import shbt.core.bootstrap as bootstrap_module
from shbt.core.bootstrap import BootstrapSearch, apply_runtime_constants_patch, build_zero_anchor_bootstrap


def test_zero_anchor_bootstrap_search_finds_unique_stable_eigenvalue() -> None:
    search = BootstrapSearch()

    assert search.branch == (26, 8, 312)
    assert search.generation_count == 15
    assert search.unique is True
    assert search.non_singular is True
    assert search.sample_count >= 4097
    assert search.stable_eigenvalue == pytest.approx(0.9887710512663789, rel=0.0, abs=1.0e-12)


def test_bootstrap_search_uses_discovered_stable_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    stable_kernel = bootstrap_module.SymmetryCandidateEvaluation(
        branch=(26, 8, 312),
        generation_count=15,
        stability_residue=Fraction(0, 1),
        low_resolution_transport_drift=0.0,
    )
    calls: list[tuple[int, int]] = []

    def fake_discover_stable_kernel_from_vacuum(*, gauge_level: int, generation_count: int):
        calls.append((gauge_level, generation_count))
        return stable_kernel

    monkeypatch.setattr(bootstrap_module, "_discover_stable_kernel_from_vacuum", fake_discover_stable_kernel_from_vacuum)

    search = BootstrapSearch()

    assert calls == [(8, 15)]
    assert search.stable_kernel == stable_kernel
    assert search.topological_closure is True
    assert search.hits_precision_floor is True


def test_zero_anchor_bootstrap_labels_charge_and_mass_only_after_stability() -> None:
    runtime = build_zero_anchor_bootstrap()

    assert runtime.labels_materialize_after_stability is True
    assert runtime.charge_observables["fine_structure_alpha_inverse"] == pytest.approx(
        runtime.emergent_constants.codata_fine_structure_alpha_inverse,
        rel=0.0,
        abs=1.0e-15,
    )


def test_zero_anchor_bootstrap_exposes_runtime_constants_patch() -> None:
    runtime = build_zero_anchor_bootstrap()
    patch = runtime.runtime_constants_patch

    assert patch["GEOMETRIC_KAPPA"] == pytest.approx(runtime.stable_eigenvalue, rel=0.0, abs=1.0e-15)
    assert patch["KAPPA_D5"] == pytest.approx(runtime.stable_eigenvalue, rel=0.0, abs=1.0e-15)
    assert patch["PLANCK2018_H0_KM_S_MPC"] == pytest.approx(runtime.hubble_km_s_mpc, rel=0.0, abs=1.0e-15)
    assert patch["CODATA_FINE_STRUCTURE_ALPHA_INVERSE"] == pytest.approx(
        runtime.charge_observables["codata_fine_structure_alpha_inverse"],
        rel=0.0,
        abs=1.0e-15,
    )
    assert patch["PDG_PROTON_TO_ELECTRON_MASS_RATIO"] == pytest.approx(
        runtime.proton_electron_mass_ratio,
        rel=0.0,
        abs=1.0e-15,
    )
    assert patch["ALPHA_INV_BENCHMARK"] == pytest.approx(
        runtime.charge_observables["surface_alpha_inverse"],
        rel=0.0,
        abs=1.0e-15,
    )


def test_apply_runtime_constants_patch_updates_namespace() -> None:
    runtime = build_zero_anchor_bootstrap()
    namespace: dict[str, object] = {"sentinel": "ok"}

    applied_runtime = apply_runtime_constants_patch(namespace, bootstrap=runtime)

    assert applied_runtime is runtime
    assert namespace["sentinel"] == "ok"
    assert namespace["PLANCK2018_H0_SI"] == pytest.approx(
        runtime.hubble_km_s_mpc * 1.0e3 / runtime.emergent_constants.mpc_in_meters,
        rel=0.0,
        abs=1.0e-15,
    )
    assert namespace["HOLOGRAPHIC_BITS"] == pytest.approx(
        runtime.emergent_constants.holographic_bits,
        rel=0.0,
        abs=1.0e-15,
    )
    assert namespace["PLANCK_MASS_GEV"] == pytest.approx(
        runtime.emergent_constants.planck_mass_ev * 1.0e-9,
        rel=0.0,
        abs=1.0e-15,
    )
    assert runtime.charge_observables["surface_alpha_inverse"] == pytest.approx(2340.0 / 17.0, rel=0.0, abs=1.0e-15)
    assert runtime.mass_observables["proton_electron_mass_ratio"] == pytest.approx(1835.248988001927, rel=0.0, abs=1.0e-9)
    assert runtime.labeled_residues["fine_structure_alpha_inverse"].label == "Charge"
    assert runtime.labeled_residues["proton_electron_mass_ratio"].label == "Mass"


def test_zero_anchor_runtime_uses_branch_selected_by_search(monkeypatch: pytest.MonkeyPatch) -> None:
    build_zero_anchor_bootstrap.cache_clear()
    original_search = bootstrap_module.BootstrapSearch

    def fake_bootstrap_search(**_: object) -> SimpleNamespace:
        return SimpleNamespace(branch=(26, 8, 312), generation_count=15, stable_eigenvalue=0.9887710512663789)

    monkeypatch.setattr(bootstrap_module, "BootstrapSearch", fake_bootstrap_search)
    try:
        runtime = build_zero_anchor_bootstrap(
            lepton_level=25,
            quark_level=8,
            parent_level=300,
            generation_count=15,
        )
    finally:
        monkeypatch.setattr(bootstrap_module, "BootstrapSearch", original_search)
        build_zero_anchor_bootstrap.cache_clear()

    assert runtime.kernel.branch == (26, 8, 312)
    assert runtime.stable_kernel is None


def test_constants_boot_from_zero_anchor_runtime() -> None:
    runtime = build_zero_anchor_bootstrap()

    assert constants_module.ZERO_ANCHOR_BOOTSTRAP.stable_eigenvalue == pytest.approx(runtime.stable_eigenvalue, rel=0.0, abs=1.0e-15)
    assert constants_module.KAPPA_D5 == pytest.approx(runtime.stable_eigenvalue, rel=0.0, abs=1.0e-15)
    assert constants_module.PLANCK2018_H0_KM_S_MPC == pytest.approx(
        runtime.emergent_constants.planck2018_h0_km_s_mpc,
        rel=0.0,
        abs=1.0e-15,
    )
    assert constants_module.CODATA_FINE_STRUCTURE_ALPHA_INVERSE == pytest.approx(
        runtime.emergent_constants.codata_fine_structure_alpha_inverse,
        rel=0.0,
        abs=1.0e-15,
    )
