from __future__ import annotations

"""Pointer-state selection for SHBT bulk decoherence.

This module turns holographic bulk wavefunctions into classical candidates by
filtering them through a topological checksum. In this narrow bookkeeping,
``c_vis`` denotes the signed visible parity load carried by the candidate bulk
state, not the positive visible central charge used elsewhere in the
manuscript. Classical observables can crystallize only when the boundary parity
sector remains neutral,

    c_vis + c_dark = 0.

The selector therefore treats parity neutrality as the decoherence criterion
that picks stable pointer states out of the holographic code.
"""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Sequence, TypeVar

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.holographic_error_stabilizer import TopologicalChecksum, TopologicalChecksumCode


DEFAULT_PRECISION = 200
_GUARD_DIGITS = 16
ObservableT = TypeVar("ObservableT")


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


@dataclass(frozen=True)
class BulkWavefunction:
    label: str
    amplitude: Decimal
    c_vis: Decimal
    c_dark: Decimal
    observable_family: str = "classical_object"
    branch_coordinates: tuple[int, int, int] = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))

    @property
    def parity_neutrality_residual(self) -> Decimal:
        return self.c_vis + self.c_dark


@dataclass(frozen=True)
class PointerState:
    wavefunction: BulkWavefunction
    checksum: TopologicalChecksum
    pointer_weight: Decimal

    @property
    def classical_observable_ready(self) -> bool:
        return self.pointer_weight > 0 and self.checksum.passed

    @property
    def crystallizes(self) -> bool:
        return self.classical_observable_ready


class PointerStateDecoherenceError(RuntimeError):
    """Raised when no parity-neutral pointer state can crystallize."""


def build_mass_pointer_state(
    *,
    label: str,
    c_dark_fraction: Decimal | Fraction | float | int | str,
    amplitude: Decimal | Fraction | float | int | str = Decimal("1"),
    precision: int = DEFAULT_PRECISION,
) -> BulkWavefunction:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        c_dark = _decimal(c_dark_fraction)
        return BulkWavefunction(
            label=label,
            amplitude=_decimal(amplitude),
            c_vis=-c_dark,
            c_dark=c_dark,
            observable_family="mass_scale",
        )


class PointerStateSelector:
    """Select parity-neutral pointer states from holographic bulk candidates."""

    def __init__(
        self,
        *,
        precision: int = DEFAULT_PRECISION,
        syndrome_tolerance: Decimal | Fraction | float | int | str = Decimal("0"),
    ) -> None:
        self.precision = max(int(precision), DEFAULT_PRECISION)
        self.syndrome_tolerance = _decimal(syndrome_tolerance)
        if self.syndrome_tolerance < 0:
            raise ValueError("syndrome_tolerance must be non-negative.")

    def build_topological_checksum(self, wavefunction: BulkWavefunction) -> TopologicalChecksum:
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            residual = wavefunction.parity_neutrality_residual
        return TopologicalChecksumCode(
            law="parity",
            stabilizer_name="Pointer-State Parity Selector",
            protected_quantity="Boundary parity neutrality",
            checksum_equation="|c_vis + c_dark|",
            boundary_integer=Decimal("0"),
            bulk_projection=residual,
            syndrome_tolerance=self.syndrome_tolerance,
            interpretation=(
                "Only parity-neutral bulk states can decohere into classical observables. "
                "If c_vis + c_dark is non-zero, the holographic code has not canceled its boundary load "
                "and the candidate remains quantum-delocalized rather than crystallized."
            ),
        ).verify()

    def evaluate_wavefunction(self, wavefunction: BulkWavefunction) -> PointerState:
        checksum = self.build_topological_checksum(wavefunction)
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            pointer_weight = abs(wavefunction.amplitude)
        return PointerState(
            wavefunction=wavefunction,
            checksum=checksum,
            pointer_weight=pointer_weight,
        )

    def filter_bulk_wavefunctions(self, wavefunctions: Sequence[BulkWavefunction]) -> tuple[PointerState, ...]:
        return tuple(
            candidate
            for candidate in (self.evaluate_wavefunction(wavefunction) for wavefunction in wavefunctions)
            if candidate.crystallizes
        )

    def select_pointer_state(self, wavefunctions: Sequence[BulkWavefunction]) -> PointerState:
        candidates = self.filter_bulk_wavefunctions(wavefunctions)
        if not candidates:
            raise PointerStateDecoherenceError(
                "No pointer state satisfies the boundary parity-neutrality checksum c_vis + c_dark = 0."
            )
        return min(candidates, key=lambda candidate: (-candidate.pointer_weight, candidate.wavefunction.label))

    def crystallize_classical_observable(
        self,
        wavefunction: BulkWavefunction,
        observable: ObservableT,
    ) -> ObservableT:
        self.select_pointer_state((wavefunction,))
        return observable


__all__ = [
    "BulkWavefunction",
    "DEFAULT_PRECISION",
    "PointerState",
    "PointerStateDecoherenceError",
    "PointerStateSelector",
    "build_mass_pointer_state",
]
