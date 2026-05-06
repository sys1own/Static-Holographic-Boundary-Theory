from __future__ import annotations

"""Boundary-to-bulk correspondence engine for SHBT entrainment.

This module replaces the narrower “bulk decoherence” framing with a stronger
boundary-to-bulk correspondence statement: a classical bulk observable is not
produced by losing information, but by entraining a boundary-neutral checksum
sector into a stable bulk image. The operative lock remains

    c_vis + c_dark = 0,

but it is interpreted here as a correspondence condition that binds the visible
and completion loads into an admissible bulk pointer state.
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

    @property
    def boundary_to_bulk_entrainment_residual(self) -> Decimal:
        return self.parity_neutrality_residual


BoundaryBulkWavefunction = BulkWavefunction


@dataclass(frozen=True)
class BoundaryBulkEntrainment:
    checksum: TopologicalChecksum
    boundary_visible_load: Decimal
    completion_load: Decimal
    entrainment_residual: Decimal
    entrainment_strength: Decimal

    @property
    def correspondence_locked(self) -> bool:
        return self.checksum.passed and self.entrainment_residual == 0

    @property
    def bulk_ready(self) -> bool:
        return self.correspondence_locked and self.entrainment_strength > 0


@dataclass(frozen=True)
class PointerState:
    wavefunction: BulkWavefunction
    checksum: TopologicalChecksum
    entrainment: BoundaryBulkEntrainment
    pointer_weight: Decimal

    @property
    def classical_observable_ready(self) -> bool:
        return self.pointer_weight > 0 and self.entrainment.bulk_ready

    @property
    def crystallizes(self) -> bool:
        return self.classical_observable_ready

    @property
    def boundary_to_bulk_correspondence_locked(self) -> bool:
        return self.entrainment.correspondence_locked


CorrespondenceState = PointerState


class BoundaryBulkCorrespondenceError(RuntimeError):
    """Raised when no parity-neutral correspondence state can entrain the bulk."""


PointerStateDecoherenceError = BoundaryBulkCorrespondenceError


def build_mass_correspondence_state(
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


build_mass_pointer_state = build_mass_correspondence_state


class BoundaryBulkCorrespondenceSelector:
    """Select boundary-neutral states that entrain a stable bulk image."""

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
            residual = wavefunction.boundary_to_bulk_entrainment_residual
        return TopologicalChecksumCode(
            law="parity",
            stabilizer_name="Boundary-to-Bulk Correspondence Selector",
            protected_quantity="Boundary parity neutrality",
            checksum_equation="|c_vis + c_dark|",
            boundary_integer=Decimal("0"),
            bulk_projection=residual,
            syndrome_tolerance=self.syndrome_tolerance,
            interpretation=(
                "Only parity-neutral boundary sectors can entrain a classical bulk observable. "
                "If c_vis + c_dark is non-zero, the boundary load has not closed and the bulk image remains "
                "unentrained rather than crystallized."
            ),
        ).verify()

    def build_boundary_to_bulk_entrainment(
        self,
        wavefunction: BulkWavefunction,
        *,
        checksum: TopologicalChecksum | None = None,
    ) -> BoundaryBulkEntrainment:
        resolved_checksum = checksum if checksum is not None else self.build_topological_checksum(wavefunction)
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            boundary_visible_load = abs(wavefunction.c_vis)
            completion_load = abs(wavefunction.c_dark)
            entrainment_strength = min(boundary_visible_load, completion_load) if resolved_checksum.passed else Decimal("0")
        return BoundaryBulkEntrainment(
            checksum=resolved_checksum,
            boundary_visible_load=boundary_visible_load,
            completion_load=completion_load,
            entrainment_residual=wavefunction.boundary_to_bulk_entrainment_residual,
            entrainment_strength=entrainment_strength,
        )

    def evaluate_wavefunction(self, wavefunction: BulkWavefunction) -> PointerState:
        checksum = self.build_topological_checksum(wavefunction)
        entrainment = self.build_boundary_to_bulk_entrainment(wavefunction, checksum=checksum)
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            pointer_weight = abs(wavefunction.amplitude) * max(entrainment.entrainment_strength, Decimal("1"))
        return PointerState(
            wavefunction=wavefunction,
            checksum=checksum,
            entrainment=entrainment,
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
            raise BoundaryBulkCorrespondenceError(
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


CorrespondenceSelector = BoundaryBulkCorrespondenceSelector
PointerStateSelector = BoundaryBulkCorrespondenceSelector


__all__ = [
    "BoundaryBulkCorrespondenceError",
    "BoundaryBulkCorrespondenceSelector",
    "BoundaryBulkEntrainment",
    "BoundaryBulkWavefunction",
    "BulkWavefunction",
    "CorrespondenceSelector",
    "CorrespondenceState",
    "DEFAULT_PRECISION",
    "PointerState",
    "PointerStateDecoherenceError",
    "PointerStateSelector",
    "build_mass_correspondence_state",
    "build_mass_pointer_state",
]
