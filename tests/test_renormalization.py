from __future__ import annotations

import pytest

from shbt.constants import CODATA_FINE_STRUCTURE_ALPHA_INVERSE
from shbt.core.renormalization import ScaleDependentTransport, derive_running_couplings, simulate_boundary_cooling


def test_renormalization_audit_derives_alpha_shift_from_boundary_cooling() -> None:
    audit = derive_running_couplings()

    assert isinstance(audit.transport, ScaleDependentTransport)
    assert audit.alpha_uv_inverse == pytest.approx(137.64705882352942, rel=0.0, abs=1.0e-12)
    assert audit.alpha_ir_inverse == pytest.approx(137.03628702174976, rel=0.0, abs=1.0e-12)
    assert audit.alpha_shift_inverse == pytest.approx(0.61077180177966, rel=0.0, abs=1.0e-12)
    assert audit.discarded_dimensions == 22
    assert audit.transport_is_noninvertible is True
    assert audit.cooling_is_monotonic is True
    assert audit.aligns_with_experiment is True
    assert audit.verification.target_alpha_inverse == pytest.approx(CODATA_FINE_STRUCTURE_ALPHA_INVERSE)


def test_renormalization_audit_matches_noninvertible_ir_projection() -> None:
    audit = derive_running_couplings()

    assert audit.ir_point.smeared_bit_density_inverse == pytest.approx(audit.smeared_bit_density_inverse, rel=0.0, abs=1.0e-15)
    assert audit.noninvertible_ir_transport.projected_alpha_inverse == pytest.approx(
        audit.alpha_ir_inverse,
        rel=0.0,
        abs=1.0e-12,
    )
    assert audit.noninvertible_ir_transport.information_loss_inverse == pytest.approx(
        audit.alpha_shift_inverse,
        rel=0.0,
        abs=1.0e-12,
    )


def test_simulate_boundary_cooling_exposes_uv_to_ir_trajectory() -> None:
    trajectory = simulate_boundary_cooling(sample_count=7)

    assert len(trajectory.points) == 7
    assert trajectory.points[0].cooling_fraction == pytest.approx(0.0)
    assert trajectory.points[-1].cooling_fraction == pytest.approx(1.0)
    assert trajectory.monotonic_information_loss is True
    assert trajectory.monotonic_alpha_inverse_cooling is True
