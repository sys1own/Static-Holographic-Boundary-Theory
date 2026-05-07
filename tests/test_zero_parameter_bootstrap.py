from __future__ import annotations

import pytest

from shbt import constants as constants_module
from shbt.main import (
    CODATA_FINE_STRUCTURE_ALPHA_INVERSE,
    DEFAULT_TOPOLOGICAL_VACUUM,
    KAPPA_D5,
    PDG_PROTON_TO_ELECTRON_MASS_RATIO,
    PLANCK2018_H0_KM_S_MPC,
    ZERO_PARAMETER_EIGENVALUE_SEARCH,
    ZERO_PARAMETER_RUNTIME_BOOTSTRAP,
)


@pytest.mark.parametrize("classification", ["Geometric Emergence", "Topological Extraction"])
def test_benchmark_tier_metadata_accepts_geometry_first_labels_for_all_tiers(classification: str) -> None:
    classifications: dict[str, str] = {}

    for tier in constants_module.STRICT_BENCHMARK_TIER_DEFINITIONS:
        for constant in tier.constants:
            for metadata_path in constant.legacy_metadata_paths:
                classifications[metadata_path] = classification

    constants_module._validate_benchmark_tier_metadata(
        constants_module.STRICT_BENCHMARK_TIER_DEFINITIONS,
        classifications,
    )


def test_zero_parameter_bootstrap_search_finds_unique_stable_eigenvalue() -> None:
    search = ZERO_PARAMETER_EIGENVALUE_SEARCH

    assert search.branch == (
        constants_module.LEPTON_LEVEL,
        constants_module.QUARK_LEVEL,
        constants_module.PARENT_LEVEL,
    )
    assert search.generation_count == constants_module.G_SM
    assert search.unique is True
    assert search.sample_count >= 4097
    assert search.stable_eigenvalue == pytest.approx(0.9887710512663789, rel=0.0, abs=1.0e-12)
    assert search.stability_gap > 0.0
    assert search.runner_up_eigenvalue != pytest.approx(search.stable_eigenvalue, rel=0.0, abs=1.0e-15)


def test_zero_parameter_bootstrap_populates_runtime_h0_charges_and_mass_ratio() -> None:
    runtime = ZERO_PARAMETER_RUNTIME_BOOTSTRAP

    assert runtime.kernel.branch == (26, 8, 312)
    assert runtime.emergent_constants.geometric_kappa == pytest.approx(runtime.stable_eigenvalue, rel=0.0, abs=1.0e-15)
    assert runtime.emergent_constants.planck2018_h0_km_s_mpc == pytest.approx(67.36010188502927, rel=0.0, abs=1.0e-12)
    assert runtime.charge_observables["codata_fine_structure_alpha_inverse"] == pytest.approx(
        137.0360005617913,
        rel=0.0,
        abs=1.0e-12,
    )
    assert runtime.proton_electron_mass_ratio == pytest.approx(1835.248988001927, rel=0.0, abs=1.0e-9)
    assert DEFAULT_TOPOLOGICAL_VACUUM.kappa_geometric == pytest.approx(KAPPA_D5, rel=0.0, abs=1.0e-15)
    assert DEFAULT_TOPOLOGICAL_VACUUM.bit_count == pytest.approx(
        runtime.emergent_constants.holographic_bits,
        rel=1.0e-15,
    )
    assert PLANCK2018_H0_KM_S_MPC == pytest.approx(runtime.hubble_km_s_mpc, rel=0.0, abs=1.0e-15)
    assert CODATA_FINE_STRUCTURE_ALPHA_INVERSE == pytest.approx(
        runtime.charge_observables["codata_fine_structure_alpha_inverse"],
        rel=0.0,
        abs=1.0e-15,
    )
    assert PDG_PROTON_TO_ELECTRON_MASS_RATIO == pytest.approx(
        runtime.proton_electron_mass_ratio,
        rel=0.0,
        abs=1.0e-15,
    )
