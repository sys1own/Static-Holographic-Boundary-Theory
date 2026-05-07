from __future__ import annotations

from decimal import Decimal

from shbt.core.entropy_engine import (
    build_entropy_feedback_audit,
    build_markov_collar,
    reframe_time_as_update_index,
    simulate_entanglement_entropy_growth,
)


def test_markov_collar_factorizes_observer_from_external_boundary() -> None:
    collar = build_markov_collar()

    assert collar.factorization_verified is True
    assert collar.factorized_joint_state == collar.loading_density
    assert sum(collar.collar_weights, Decimal("0")) == Decimal("1")
    assert sum(collar.observer_marginal, Decimal("0")) == Decimal("1")
    assert sum(collar.boundary_marginal, Decimal("0")) == Decimal("1")
    assert collar.observer_boundary_mutual_information_bits > 0
    assert collar.conditional_mutual_information_bits == Decimal("0")


def test_entropy_feedback_audit_grows_entanglement_monotonically_with_resolution() -> None:
    audit = simulate_entanglement_entropy_growth(entropy_budget_bits=Decimal("9"))

    assert len(audit.steps) == 9
    assert audit.monotonic_entanglement_growth is True
    assert audit.transport_resolution_closed is True
    assert audit.steps[0].entropy_delta_fraction > 0
    assert audit.steps[-1].cumulative_entanglement_fraction == Decimal("1")
    assert audit.final_entanglement_entropy_bits == Decimal("9")
    assert audit.steps[1].feedback_multiplier > audit.steps[0].feedback_multiplier


def test_arrow_of_time_is_the_transport_update_index() -> None:
    audit = build_entropy_feedback_audit(entropy_budget_bits=Decimal("1"))

    assert audit.arrow_of_time_reframed is True
    assert [step.update_index for step in audit.steps] == list(range(1, len(audit.steps) + 1))
    assert [step.arrow_of_time_index for step in audit.steps] == [Decimal(index) for index in range(1, len(audit.steps) + 1)]
    assert [step.collar_coordinate for step in audit.steps] == list(audit.markov_collar.collar_sequence)
    assert reframe_time_as_update_index(4) == Decimal("4")
    assert audit.statement == "Time is the sequential delta update index of holographic transport resolution."
