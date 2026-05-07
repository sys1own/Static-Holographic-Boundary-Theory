from __future__ import annotations

import pytest

from shbt import constants as constants_module
from shbt.core.bootstrap import BootstrapSearch, build_zero_anchor_bootstrap


def test_zero_anchor_bootstrap_search_finds_unique_stable_eigenvalue() -> None:
    search = BootstrapSearch()

    assert search.branch == (26, 8, 312)
    assert search.generation_count == 15
    assert search.unique is True
    assert search.non_singular is True
    assert search.sample_count >= 4097
    assert search.stable_eigenvalue == pytest.approx(0.9887710512663789, rel=0.0, abs=1.0e-12)


def test_zero_anchor_bootstrap_labels_charge_and_mass_only_after_stability() -> None:
    runtime = build_zero_anchor_bootstrap()

    assert runtime.labels_materialize_after_stability is True
    assert runtime.charge_observables["fine_structure_alpha_inverse"] == pytest.approx(
        runtime.emergent_constants.codata_fine_structure_alpha_inverse,
        rel=0.0,
        abs=1.0e-15,
    )
    assert runtime.charge_observables["surface_alpha_inverse"] == pytest.approx(2340.0 / 17.0, rel=0.0, abs=1.0e-15)
    assert runtime.mass_observables["proton_electron_mass_ratio"] == pytest.approx(1835.248988001927, rel=0.0, abs=1.0e-9)
    assert runtime.labeled_residues["fine_structure_alpha_inverse"].label == "Charge"
    assert runtime.labeled_residues["proton_electron_mass_ratio"].label == "Mass"


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
