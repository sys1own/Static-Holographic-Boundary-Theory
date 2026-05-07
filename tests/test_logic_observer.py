from __future__ import annotations

from decimal import Decimal

import numpy as np

from shbt.logic.observer import (
    AgentLatticeCoordinates,
    InternalObserver,
    Observer,
    shift_holographic_projection,
)


def test_self_valuation_sigma_tracks_frame_dependency_and_local_entropy() -> None:
    center_observer = InternalObserver()
    near_horizon_observer = InternalObserver(
        observer_radius_m=center_observer.global_horizon_radius_m * Decimal("0.9")
    )

    center_audit = center_observer.self_valuate()
    near_horizon_audit = near_horizon_observer.self_valuate()

    assert center_audit.local_entropy_bits > near_horizon_audit.local_entropy_bits
    assert near_horizon_audit.hidden_entropy_bits > center_audit.hidden_entropy_bits
    assert near_horizon_audit.frame_shift > center_audit.frame_shift
    assert near_horizon_audit.sigma > center_audit.sigma
    assert center_audit.internal_observer_consistent
    assert near_horizon_audit.internal_observer_consistent


def test_coordinate_transformation_shifts_holographic_projection_from_agent_coordinates() -> None:
    agent_coordinates = AgentLatticeCoordinates(
        lepton_coordinate=Decimal("0.25"),
        quark_coordinate=Decimal("0.125"),
        support_coordinate=Decimal("0.5"),
    )
    observer = InternalObserver(agent_coordinates=agent_coordinates)
    self_valuation = observer.self_valuate()

    projection = shift_holographic_projection(
        self_valuation=self_valuation,
        agent_coordinates=agent_coordinates,
        geometry=observer.actualize_bulk_geometry(),
    )

    assert projection.jacobian_matrix.shape == (4, 4)
    assert projection.projection_shifted is True
    assert projection.positive_definite is True
    assert projection.coordinate_shift == (
        self_valuation.sigma * agent_coordinates.lepton_coordinate,
        self_valuation.sigma * agent_coordinates.quark_coordinate,
        self_valuation.sigma * agent_coordinates.support_coordinate,
    )
    assert not np.allclose(
        projection.base_geometry.spacetime_metric.components,
        projection.shifted_spacetime_metric.components,
    )


def test_localized_entropy_gradient_manifests_as_gravitational_acceleration() -> None:
    center_observer = InternalObserver()
    near_horizon_observer = InternalObserver(
        observer_radius_m=center_observer.global_horizon_radius_m * Decimal("0.9")
    )

    center_ui = center_observer.derive_general_relativity_ui()
    near_horizon_ui = near_horizon_observer.derive_general_relativity_ui()

    assert near_horizon_ui.localized_entropy_gradient_per_m > center_ui.localized_entropy_gradient_per_m
    assert near_horizon_ui.gravitational_acceleration_m_per_s2 > center_ui.gravitational_acceleration_m_per_s2
    assert near_horizon_ui.observer_frame_curvature > 0
    assert near_horizon_ui.equivalence_principle_verified is True
    assert near_horizon_ui.general_relativity_is_ui is True
    assert near_horizon_ui.statement.startswith("General Relativity is the UI")


def test_observer_class_exposes_coordinate_defined_axiom_ix_frame() -> None:
    coordinates = AgentLatticeCoordinates(
        lepton_coordinate=Decimal("0.2"),
        quark_coordinate=Decimal("0.1"),
        support_coordinate=Decimal("0.4"),
    )
    observer = Observer(coordinates=coordinates)

    audit = observer.self_valuate()

    assert observer.coordinates == coordinates
    assert audit.agent_coordinates == coordinates
    assert audit.axiom_ix_satisfied is True
    assert audit.boundary_weight > 0
    assert audit.bulk_weight >= 0
    assert audit.statement.startswith("Axiom IX")


def test_frame_dependent_alpha_drifts_under_information_smearing() -> None:
    center_observer = Observer()
    near_horizon_observer = Observer(
        observer_radius_m=center_observer.global_horizon_radius_m * Decimal("0.9")
    )

    center_alpha = center_observer.derive_frame_dependent_alpha()
    near_horizon_alpha = near_horizon_observer.derive_frame_dependent_alpha()

    assert center_alpha.benchmark_recovered is True
    assert center_alpha.apparent_drift_fraction == Decimal("0")
    assert near_horizon_alpha.alpha_drift_detected is True
    assert near_horizon_alpha.bulk_smearing_fraction > center_alpha.bulk_smearing_fraction
    assert near_horizon_alpha.sigma_smearing_factor > center_alpha.sigma_smearing_factor
    assert near_horizon_alpha.apparent_drift_fraction > center_alpha.apparent_drift_fraction
    assert near_horizon_alpha.apparent_alpha_inverse < near_horizon_alpha.benchmark_alpha_inverse
    assert near_horizon_alpha.statement.startswith("The fine-structure constant appears")


def test_observer_frame_audit_closes_static_and_dynamic_loop() -> None:
    reference_observer = Observer()
    observer = Observer(
        observer_radius_m=reference_observer.global_horizon_radius_m * Decimal("0.9")
    )

    frame = observer.derive_observer_frame()

    assert frame.self_valuation.axiom_ix_satisfied is True
    assert frame.general_relativity_ui.general_relativity_is_ui is True
    assert frame.alpha_drift.alpha_drift_detected is True
    assert frame.observer_frame_consistent is True
    assert frame.statement.startswith("Axiom IX closes the loop")
