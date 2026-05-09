from __future__ import annotations

from fractions import Fraction

from shbt.analysis.sensitivity import assert_unique_axiom_ix_survivor, audit_alternative_dimension_kernels, audit_kernel_sensitivity
from shbt.core.derivation_api import UniverseFactory


def _dimension_index_logic_relation(resolve, axioms):
    del axioms
    return int(resolve("dimension_index")) + int(resolve("quark_level"))


def test_recursive_ontic_cascade_propagates_logic_relation_into_alpha_surface() -> None:
    baseline = UniverseFactory.evaluate_ontic_cascade()
    detuned = UniverseFactory.evaluate_ontic_cascade(logic_relation=_dimension_index_logic_relation)
    derivation = UniverseFactory.derive_alpha_surface(logic_relation=_dimension_index_logic_relation)

    assert baseline.alpha_inverse_fraction == Fraction(2340, 17)
    assert detuned.visible_support == 21
    assert detuned.alpha_inverse_fraction == Fraction(1560, 7)
    assert derivation.alpha_inverse_fraction == detuned.alpha_inverse_fraction
    assert detuned.trace.index("dimension_index") < detuned.trace.index("visible_support")
    assert detuned.trace.index("visible_support") < detuned.trace.index("alpha_inverse_fraction")


def test_sensitivity_audit_flags_24d_kernel_as_finite_off_shell_divergence() -> None:
    audit = audit_alternative_dimension_kernels([24, 26], quark_level=8, parent_level=312)

    off_shell = next(point for point in audit.evaluated_points if point.kernel == (24, 8, 312))
    benchmark = next(point for point in audit.evaluated_points if point.kernel == (26, 8, 312))

    assert off_shell.alternative_dimension is True
    assert off_shell.prime_indexed is False
    assert off_shell.axiom_ix_closure is False
    assert off_shell.non_singular_divergence is True
    assert off_shell.closure_tensor_amplitude > 0
    assert off_shell.alpha_shift_fraction == Fraction(585, 68)
    assert benchmark.axiom_ix_closure is True
    assert benchmark.non_singular_divergence is False


def test_only_benchmark_kernel_satisfies_axiom_ix_across_prime_and_nonprime_alternatives() -> None:
    audit = audit_kernel_sensitivity(
        kernels=[
            (22, 8, 312),
            (24, 8, 312),
            (26, 8, 312),
            (28, 8, 312),
        ]
    )

    prime_indexed_alternative = next(point for point in audit.evaluated_points if point.kernel == (22, 8, 312))

    assert audit.axiom_ix_survivors == ((26, 8, 312),)
    assert audit.unique_axiom_ix_survivor == (26, 8, 312)
    assert assert_unique_axiom_ix_survivor(audit) == (26, 8, 312)
    assert prime_indexed_alternative.prime_indexed is True
    assert prime_indexed_alternative.alternative_dimension is True
    assert prime_indexed_alternative.axiom_ix_closure is False
    assert prime_indexed_alternative.non_singular_divergence is True
