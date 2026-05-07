from __future__ import annotations

import pytest

from shbt.constants import LEPTON_LEVEL, MZ_SCALE_GEV, PARENT_LEVEL, PLANCK2018_ALPHA_EM_INV_MZ, QUARK_LEVEL
from shbt.core.derivation_api import TopologicalVacuum
from shbt.core.renormalization_map import (
    BENCHMARK_BOUNDARY_BRANCH,
    BENCHMARK_UV_SCALE_GEV,
    holographic_smearing_fraction,
    running_alpha_function,
    uv_to_ir_rendering_function,
    verify_rendered_alpha_at_z_boson,
)
from shbt.main import THEORETICAL_MATCHING_UNCERTAINTY_FRACTION


def test_uv_to_ir_rendering_smears_branch_residue_between_uv_and_mz() -> None:
    uv_render = uv_to_ir_rendering_function(BENCHMARK_UV_SCALE_GEV)
    mz_render = uv_to_ir_rendering_function(MZ_SCALE_GEV)

    assert uv_render.vacuum.branch == BENCHMARK_BOUNDARY_BRANCH == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
    assert uv_render.smearing_fraction == pytest.approx(0.0)
    assert mz_render.smearing_fraction == pytest.approx(1.0)
    assert uv_render.rendered_alpha_inverse == pytest.approx(uv_render.alpha_uv_inverse, rel=1e-15)
    assert mz_render.rendered_alpha_inverse != pytest.approx(mz_render.alpha_uv_inverse, rel=1e-12)
    assert mz_render.rendered_alpha == pytest.approx(mz_render.alpha_uv + mz_render.delta_alpha_residue, rel=1e-15)
    assert mz_render.holographic_smearing_residue_inverse < 0.0


def test_running_alpha_function_matches_rendered_alpha() -> None:
    rendering = uv_to_ir_rendering_function(MZ_SCALE_GEV)

    assert running_alpha_function(MZ_SCALE_GEV) == pytest.approx(rendering.rendered_alpha, rel=1e-15)
    assert holographic_smearing_fraction((BENCHMARK_UV_SCALE_GEV * MZ_SCALE_GEV) ** 0.5) == pytest.approx(0.5, rel=1e-3)


def test_z_boson_rendering_aligns_with_standard_model_expectation() -> None:
    verification = verify_rendered_alpha_at_z_boson()

    assert verification.rendering.energy_gev == pytest.approx(MZ_SCALE_GEV)
    assert verification.target_alpha_inverse == pytest.approx(PLANCK2018_ALPHA_EM_INV_MZ)
    assert verification.improves_over_raw is True
    assert verification.aligns_with_standard_model is True
    assert verification.relative_error_inverse <= THEORETICAL_MATCHING_UNCERTAINTY_FRACTION
    assert verification.pull <= 1.0


def test_renormalization_map_rejects_nonbenchmark_branch() -> None:
    with pytest.raises(ValueError, match=r"anomaly-free \(26, 8, 312\) boundary manifold"):
        uv_to_ir_rendering_function(MZ_SCALE_GEV, vacuum=TopologicalVacuum(27, 8, 312))
