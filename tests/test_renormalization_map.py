from __future__ import annotations

import pytest

from shbt.constants import CODATA_FINE_STRUCTURE_ALPHA_INVERSE, LEPTON_LEVEL, MZ_SCALE_GEV, PARENT_LEVEL, PLANCK2018_ALPHA_EM_INV_MZ, QUARK_LEVEL
from shbt.core.derivation_api import TopologicalVacuum
from shbt.core.renormalization_map import (
    BENCHMARK_BOUNDARY_BRANCH,
    BENCHMARK_UV_SCALE_GEV,
    BULK_DIMENSION,
    CODATA_ALIGNMENT_ABS_TOLERANCE,
    NonInvertibleTransport,
    cooling_script,
    holographic_smearing_fraction,
    noninvertible_transport,
    running_alpha_function,
    uv_to_ir_rendering_function,
    verify_noninvertible_transport_at_z_boson,
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


def test_noninvertible_transport_cools_boundary_information_from_26d_to_4d() -> None:
    uv_transport = noninvertible_transport(BENCHMARK_UV_SCALE_GEV)
    mz_transport = NonInvertibleTransport(MZ_SCALE_GEV)

    assert uv_transport.vacuum.branch == BENCHMARK_BOUNDARY_BRANCH == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
    assert uv_transport.boundary_dimension == LEPTON_LEVEL == 26
    assert uv_transport.bulk_dimension == BULK_DIMENSION == 4
    assert uv_transport.discarded_dimensions == LEPTON_LEVEL - BULK_DIMENSION == 22
    assert uv_transport.smearing_fraction == pytest.approx(0.0)
    assert mz_transport.smearing_fraction == pytest.approx(1.0)
    assert uv_transport.information_loss_inverse == pytest.approx(0.0)
    assert mz_transport.information_loss_inverse == pytest.approx(mz_transport.information_loss_budget_inverse, rel=1e-15)
    assert mz_transport.projected_alpha_inverse < mz_transport.alpha_uv_inverse
    assert mz_transport.projected_alpha == pytest.approx(1.0 / mz_transport.projected_alpha_inverse, rel=1e-15)


def test_cooling_script_smears_information_monotonically_across_bulk_scales() -> None:
    trajectory = cooling_script(sample_count=11)

    assert trajectory.branch == BENCHMARK_BOUNDARY_BRANCH
    assert len(trajectory.points) == 11
    assert trajectory.points[0].transport.energy_gev == pytest.approx(BENCHMARK_UV_SCALE_GEV)
    assert trajectory.points[-1].transport.energy_gev == pytest.approx(MZ_SCALE_GEV)
    assert trajectory.monotonic_information_loss
    assert trajectory.monotonic_alpha_inverse_cooling
    assert trajectory.points[-1].transport.projected_alpha_inverse == pytest.approx(
        noninvertible_transport(MZ_SCALE_GEV).projected_alpha_inverse,
        rel=1e-15,
    )


def test_noninvertible_transport_aligns_with_codata_at_z_boson_without_targeting_it() -> None:
    verification = verify_noninvertible_transport_at_z_boson()

    assert verification.transport.energy_gev == pytest.approx(MZ_SCALE_GEV)
    assert verification.codata_alpha_inverse == pytest.approx(CODATA_FINE_STRUCTURE_ALPHA_INVERSE)
    assert verification.target_used_in_transport is False
    assert verification.absolute_error_inverse <= CODATA_ALIGNMENT_ABS_TOLERANCE
    assert verification.aligns_with_codata is True
