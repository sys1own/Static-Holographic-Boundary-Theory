from __future__ import annotations

"""Recursive ontic cascade from branch axioms to derived observables.

The SHBT benchmark treats observables as mandatory residues of a finite set of
topological axioms. This module exposes that dependency graph as a recursive
resolver so that changing a fundamental logic relation ``R`` automatically
propagates through downstream observables such as ``alpha_surf^-1``.

For the sensitivity audits, the same cascade also computes the normalized
closure predicates used to phrase Axiom IX: minimal Diophantine closure,
framing closure, positive GKO completion, and vanishing closure tensor.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from inspect import Parameter, signature
from typing import Any

from shbt.constants import G_SM, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.noether_bridge import bulk_closure_tensor, framing_defect
from shbt.main import verify_gko_orthogonality


DEFAULT_PRECISION = 50
_GUARD_DIGITS = 12

LogicResolver = Callable[[str], Any]
LogicRelation = Callable[..., int | Fraction | Decimal]


@dataclass(frozen=True)
class OnticAxioms:
    lepton_level: int = LEPTON_LEVEL
    quark_level: int = QUARK_LEVEL
    parent_level: int = PARENT_LEVEL
    generation_count: int = G_SM

    @property
    def branch(self) -> tuple[int, int, int]:
        return (int(self.lepton_level), int(self.quark_level), int(self.parent_level))

    @property
    def kernel(self) -> tuple[int, int, int]:
        return self.branch

    @property
    def boundary_dimension(self) -> int:
        return int(self.lepton_level)


@dataclass(frozen=True)
class AxiomIXAudit:
    minimal_parent_level: int
    diophantine_pass: bool
    lepton_branching_index: Fraction
    quark_branching_index: Fraction
    framing_gap: Fraction
    framing_pass: bool
    gko_c_dark_residue: Decimal
    gko_pass: bool
    closure_tensor_amplitude: Decimal
    failure_modes: tuple[str, ...]

    @property
    def topological_closure(self) -> bool:
        return bool(
            self.diophantine_pass
            and self.framing_pass
            and self.gko_pass
            and self.closure_tensor_amplitude == 0
        )

    @property
    def non_singular_divergence(self) -> bool:
        return bool(
            not self.topological_closure
            and self.gko_c_dark_residue.is_finite()
            and self.closure_tensor_amplitude.is_finite()
        )


@dataclass(frozen=True)
class OnticCascade:
    axioms: OnticAxioms
    logic_relation_name: str
    dimension_index: Fraction
    prime_indexed: bool
    visible_support: int | Fraction
    level_density_ratio: Fraction
    alpha_inverse_fraction: Fraction
    alpha_inverse_decimal: Decimal
    axiom_ix: AxiomIXAudit
    trace: tuple[str, ...]

    @property
    def kernel(self) -> tuple[int, int, int]:
        return self.axioms.kernel


def _fraction_to_decimal(value: Fraction, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return +decimal_value


def _coerce_fraction(value: int | Fraction | Decimal) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        return Fraction(value)
    return Fraction(int(value), 1)


def _normalize_support(value: Fraction) -> int | Fraction:
    return int(value) if value.denominator == 1 else value


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value in {2, 3}:
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    factor = 5
    while factor * factor <= value:
        if value % factor == 0 or value % (factor + 2) == 0:
            return False
        factor += 6
    return True


def default_logic_relation(resolve: LogicResolver, axioms: OnticAxioms) -> int:
    del axioms
    return int(resolve("lepton_level")) + int(resolve("quark_level"))


DEFAULT_LOGIC_RELATION = default_logic_relation


def _call_logic_relation(relation: LogicRelation, resolve: LogicResolver, axioms: OnticAxioms) -> int | Fraction | Decimal:
    try:
        parameters = tuple(signature(relation).parameters.values())
    except (TypeError, ValueError):
        return relation(resolve, axioms)

    positional_parameters = [
        parameter
        for parameter in parameters
        if parameter.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if any(parameter.kind is Parameter.VAR_POSITIONAL for parameter in parameters) or len(positional_parameters) >= 2:
        return relation(resolve, axioms)
    if len(positional_parameters) == 1:
        return relation(axioms)
    return relation()


def _logic_relation_name(relation: LogicRelation) -> str:
    name = getattr(relation, "__name__", "")
    return name if isinstance(name, str) and name else relation.__class__.__name__


def evaluate_ontic_cascade(
    axioms: OnticAxioms | None = None,
    *,
    logic_relation: LogicRelation | None = None,
    precision: int = DEFAULT_PRECISION,
) -> OnticCascade:
    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    resolved_axioms = OnticAxioms() if axioms is None else axioms
    if resolved_axioms.lepton_level <= 0 or resolved_axioms.quark_level <= 0:
        raise ValueError("Ontic cascade requires positive visible levels.")
    if resolved_axioms.parent_level <= 0:
        raise ValueError("Ontic cascade requires a positive parent level.")

    relation = DEFAULT_LOGIC_RELATION if logic_relation is None else logic_relation
    cache: dict[str, Any] = {}
    trace: list[str] = []

    def resolve(name: str) -> Any:
        if name in cache:
            return cache[name]

        if name == "lepton_level":
            value: Any = int(resolved_axioms.lepton_level)
        elif name == "quark_level":
            value = int(resolved_axioms.quark_level)
        elif name == "parent_level":
            value = int(resolved_axioms.parent_level)
        elif name == "generation_count":
            value = int(resolved_axioms.generation_count)
        elif name == "dimension_index":
            value = Fraction(int(resolve("lepton_level")), 2)
        elif name == "prime_indexed":
            dimension_index = _coerce_fraction(resolve("dimension_index"))
            value = bool(dimension_index.denominator == 1 and _is_prime(dimension_index.numerator))
        elif name == "visible_support":
            support = _coerce_fraction(_call_logic_relation(relation, resolve, resolved_axioms))
            if support <= 0:
                raise ValueError("Logic relation R must return a positive visible support.")
            value = _normalize_support(support)
        elif name == "level_density_ratio":
            value = Fraction(int(resolve("parent_level")), 1) / _coerce_fraction(resolve("visible_support"))
        elif name == "alpha_inverse_fraction":
            value = Fraction(int(resolve("generation_count")), 1) * _coerce_fraction(resolve("level_density_ratio"))
        elif name == "alpha_inverse_decimal":
            value = _fraction_to_decimal(_coerce_fraction(resolve("alpha_inverse_fraction")), precision=resolved_precision)
        elif name == "minimal_parent_level":
            value = math.lcm(2 * int(resolve("lepton_level")), 3 * int(resolve("quark_level")))
        elif name == "diophantine_pass":
            value = bool(int(resolve("parent_level")) == int(resolve("minimal_parent_level")))
        elif name == "lepton_branching_index":
            value = Fraction(int(resolve("parent_level")), 2 * int(resolve("lepton_level")))
        elif name == "quark_branching_index":
            value = Fraction(int(resolve("parent_level")), 3 * int(resolve("quark_level")))
        elif name == "framing_gap":
            value = framing_defect(
                int(resolve("parent_level")),
                int(resolve("lepton_level")),
                int(resolve("quark_level")),
            ).delta_fr
        elif name == "framing_pass":
            value = bool(_coerce_fraction(resolve("framing_gap")) == 0)
        elif name == "gko_c_dark_residue":
            audit = verify_gko_orthogonality(
                parent_level=int(resolve("parent_level")),
                lepton_level=int(resolve("lepton_level")),
                quark_level=int(resolve("quark_level")),
            )
            value = Decimal(str(audit.c_dark_residue))
        elif name == "gko_pass":
            value = bool(Decimal(resolve("gko_c_dark_residue")) > 0)
        elif name == "closure_tensor_amplitude":
            value = bulk_closure_tensor(_coerce_fraction(resolve("framing_gap")), Decimal("1")).amplitude
        else:
            raise KeyError(f"Unknown ontic cascade node: {name}")

        cache[name] = value
        trace.append(name)
        return value

    dimension_index = _coerce_fraction(resolve("dimension_index"))
    prime_indexed = bool(resolve("prime_indexed"))
    visible_support = resolve("visible_support")
    level_density_ratio = _coerce_fraction(resolve("level_density_ratio"))
    alpha_inverse_fraction = _coerce_fraction(resolve("alpha_inverse_fraction"))
    alpha_inverse_decimal = Decimal(resolve("alpha_inverse_decimal"))

    diophantine_pass = bool(resolve("diophantine_pass"))
    framing_gap = _coerce_fraction(resolve("framing_gap"))
    framing_pass = bool(resolve("framing_pass"))
    gko_c_dark_residue = Decimal(resolve("gko_c_dark_residue"))
    gko_pass = bool(resolve("gko_pass"))
    closure_tensor_amplitude = Decimal(resolve("closure_tensor_amplitude"))

    failure_modes: list[str] = []
    if not diophantine_pass:
        failure_modes.append("Diophantine closure failed")
    if not framing_pass:
        failure_modes.append("Framing closure failed")
    if not gko_pass:
        failure_modes.append("GKO completion failed")
    if closure_tensor_amplitude != 0:
        failure_modes.append("Closure tensor is non-vanishing")

    axiom_ix = AxiomIXAudit(
        minimal_parent_level=int(resolve("minimal_parent_level")),
        diophantine_pass=diophantine_pass,
        lepton_branching_index=_coerce_fraction(resolve("lepton_branching_index")),
        quark_branching_index=_coerce_fraction(resolve("quark_branching_index")),
        framing_gap=framing_gap,
        framing_pass=framing_pass,
        gko_c_dark_residue=gko_c_dark_residue,
        gko_pass=gko_pass,
        closure_tensor_amplitude=closure_tensor_amplitude,
        failure_modes=tuple(failure_modes),
    )
    return OnticCascade(
        axioms=resolved_axioms,
        logic_relation_name=_logic_relation_name(relation),
        dimension_index=dimension_index,
        prime_indexed=prime_indexed,
        visible_support=visible_support,
        level_density_ratio=level_density_ratio,
        alpha_inverse_fraction=alpha_inverse_fraction,
        alpha_inverse_decimal=alpha_inverse_decimal,
        axiom_ix=axiom_ix,
        trace=tuple(trace),
    )


__all__ = [
    "AxiomIXAudit",
    "DEFAULT_LOGIC_RELATION",
    "DEFAULT_PRECISION",
    "LogicRelation",
    "LogicResolver",
    "OnticAxioms",
    "OnticCascade",
    "default_logic_relation",
    "evaluate_ontic_cascade",
]
