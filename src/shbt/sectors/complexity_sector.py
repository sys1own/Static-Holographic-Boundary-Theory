from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.audit.audit_complexity_ceiling import (
    DEFAULT_PRECISION as COMPLEXITY_CEILING_DEFAULT_PRECISION,
    derive_complexity_ceiling,
)
from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.flavor_identity_resolver import FlavorIdentityAudit, build_flavor_identity_audit


DEFAULT_PRECISION = max(int(COMPLEXITY_CEILING_DEFAULT_PRECISION), 64)
_GUARD_DIGITS = 12
DNA_ALPHABET_CARDINALITY = 4
SHANNON_BITS_PER_NUCLEOTIDE = Decimal("2")
DUPLEX_COMPLEMENT_EFFICIENCY = Decimal("0.5")
PRIME_SYNC_SPREAD_TOLERANCE = Decimal("0.02")


def _decimal(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _format_decimal(value: Decimal, *, places: int = 18) -> str:
    if value.is_zero():
        return "0"
    adjusted = value.adjusted()
    if adjusted >= 6 or adjusted <= -4:
        return f"{value:.{places}E}"
    return f"{value:.{places}f}".rstrip("0").rstrip(".")


def _is_prime(candidate: int) -> bool:
    resolved_candidate = int(candidate)
    if resolved_candidate < 2:
        return False
    if resolved_candidate == 2:
        return True
    if resolved_candidate % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= resolved_candidate:
        if resolved_candidate % divisor == 0:
            return False
        divisor += 2
    return True


def _previous_prime(candidate: int) -> int:
    resolved_candidate = int(candidate)
    while resolved_candidate >= 2:
        if _is_prime(resolved_candidate):
            return resolved_candidate
        resolved_candidate -= 1
    return 2


def _next_prime(candidate: int) -> int:
    resolved_candidate = max(2, int(candidate))
    while not _is_prime(resolved_candidate):
        resolved_candidate += 1
    return resolved_candidate


def _surrounding_primes(value: Decimal) -> tuple[int, int]:
    resolved_value = float(value)
    lower_prime = _previous_prime(math.floor(resolved_value))
    upper_prime = _next_prime(math.ceil(resolved_value))
    if lower_prime == upper_prime:
        lower_prime = _previous_prime(lower_prime - 1)
        upper_prime = _next_prime(upper_prime + 1)
    return int(lower_prime), int(upper_prime)


@dataclass(frozen=True)
class PrimeSyncWindow:
    generation: int
    mass_ratio_to_m0: Decimal
    lower_prime: int
    upper_prime: int
    normalized_window_coordinate: Decimal

    @property
    def complementary_prime_gap(self) -> int:
        return int(self.upper_prime - self.lower_prime)

    @property
    def prime_window_center(self) -> Decimal:
        return (Decimal(self.lower_prime) + Decimal(self.upper_prime)) / Decimal(2)


@dataclass(frozen=True)
class PrimeSyncAudit:
    branch: tuple[int, int, int]
    beta_phase_lock: Decimal
    mass_step_ratio: Decimal
    windows: tuple[PrimeSyncWindow, ...]
    mean_window_coordinate: Decimal
    max_window_coordinate_spread: Decimal
    algebraic_rigidity_preserved: bool

    @property
    def prime_sync_verified(self) -> bool:
        return self.algebraic_rigidity_preserved and self.max_window_coordinate_spread <= PRIME_SYNC_SPREAD_TOLERANCE

    @property
    def statement(self) -> str:
        return "Algebraic rigidity remains phase-locked inside complementary prime windows."


@dataclass(frozen=True)
class ShannonEntropyLimitAudit:
    holographic_bits: Decimal
    shannon_bits_per_nucleotide: Decimal
    duplex_complement_efficiency: Decimal
    complexity_utilization_fraction: Decimal
    clock_skew_fraction: Decimal
    maximum_logical_storage_efficiency: Decimal
    maximum_logical_storage_bits: Decimal
    maximum_error_corrected_base_pairs: Decimal

    @property
    def within_holographic_budget(self) -> bool:
        return self.maximum_logical_storage_bits <= self.holographic_bits

    @property
    def holographic_ceiling_verified(self) -> bool:
        return self.within_holographic_budget and self.maximum_logical_storage_efficiency <= Decimal(1)

    @property
    def statement(self) -> str:
        return "The holographic bit budget caps duplex biological storage through a Shannon ceiling."


@dataclass(frozen=True)
class ComplexitySectorAudit:
    branch: tuple[int, int, int]
    flavor_identity: FlavorIdentityAudit
    prime_sync: PrimeSyncAudit
    shannon_limit: ShannonEntropyLimitAudit

    @property
    def sector_passed(self) -> bool:
        return (
            self.flavor_identity.mandatory_residue_verified
            and self.prime_sync.prime_sync_verified
            and self.shannon_limit.holographic_ceiling_verified
        )

    @property
    def statement(self) -> str:
        return "Physical residues stay branch-locked when scaled into duplex biological error correction."


def _build_prime_sync_audit(flavor_identity: FlavorIdentityAudit) -> PrimeSyncAudit:
    windows: list[PrimeSyncWindow] = []
    for residue in flavor_identity.generation_residues:
        if residue.generation <= 0:
            continue
        mass_ratio = _decimal(residue.mass_ratio_to_m0)
        lower_prime, upper_prime = _surrounding_primes(mass_ratio)
        normalized_window_coordinate = (mass_ratio - Decimal(lower_prime)) / Decimal(upper_prime - lower_prime)
        windows.append(
            PrimeSyncWindow(
                generation=int(residue.generation),
                mass_ratio_to_m0=+mass_ratio,
                lower_prime=int(lower_prime),
                upper_prime=int(upper_prime),
                normalized_window_coordinate=+normalized_window_coordinate,
            )
        )

    if not windows:
        raise ValueError("Prime-sync audit requires at least one nontrivial generation residue.")

    coordinates = tuple(window.normalized_window_coordinate for window in windows)
    mean_window_coordinate = sum(coordinates, Decimal("0")) / Decimal(len(coordinates))
    max_window_coordinate_spread = max(coordinates) - min(coordinates)

    return PrimeSyncAudit(
        branch=tuple(int(value) for value in flavor_identity.branch),
        beta_phase_lock=_decimal(flavor_identity.beta_phase_lock),
        mass_step_ratio=_decimal(flavor_identity.mass_step_ratio),
        windows=tuple(windows),
        mean_window_coordinate=+mean_window_coordinate,
        max_window_coordinate_spread=+max_window_coordinate_spread,
        algebraic_rigidity_preserved=bool(flavor_identity.mandatory_residue_verified),
    )


def _build_shannon_entropy_limit_audit(*, precision: int) -> ShannonEntropyLimitAudit:
    complexity_ceiling = derive_complexity_ceiling(precision=precision)
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        holographic_bits = _decimal(HOLOGRAPHIC_BITS)
        complexity_utilization_fraction = _decimal(complexity_ceiling.complexity_utilization_fraction)
        clock_skew_fraction = _decimal(complexity_ceiling.clock_skew)
        maximum_logical_storage_efficiency = DUPLEX_COMPLEMENT_EFFICIENCY * complexity_utilization_fraction
        maximum_logical_storage_bits = holographic_bits * maximum_logical_storage_efficiency
        maximum_error_corrected_base_pairs = maximum_logical_storage_bits / SHANNON_BITS_PER_NUCLEOTIDE
        context.prec = max(int(precision), DEFAULT_PRECISION)
    return ShannonEntropyLimitAudit(
        holographic_bits=+holographic_bits,
        shannon_bits_per_nucleotide=SHANNON_BITS_PER_NUCLEOTIDE,
        duplex_complement_efficiency=DUPLEX_COMPLEMENT_EFFICIENCY,
        complexity_utilization_fraction=+complexity_utilization_fraction,
        clock_skew_fraction=+clock_skew_fraction,
        maximum_logical_storage_efficiency=+maximum_logical_storage_efficiency,
        maximum_logical_storage_bits=+maximum_logical_storage_bits,
        maximum_error_corrected_base_pairs=+maximum_error_corrected_base_pairs,
    )


def build_complexity_sector_audit(*, precision: int = DEFAULT_PRECISION) -> ComplexitySectorAudit:
    flavor_identity = build_flavor_identity_audit()
    branch = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    if flavor_identity.branch != branch:
        raise AssertionError(
            f"Complexity sector audit is locked to the benchmark branch {branch}, received {flavor_identity.branch}."
        )

    return ComplexitySectorAudit(
        branch=branch,
        flavor_identity=flavor_identity,
        prime_sync=_build_prime_sync_audit(flavor_identity),
        shannon_limit=_build_shannon_entropy_limit_audit(precision=precision),
    )


def build_complexity_sector_report(*, precision: int = DEFAULT_PRECISION) -> str:
    audit = build_complexity_sector_audit(precision=precision)
    lines = [
        "Complexity Sector Audit",
        "=======================",
        f"Benchmark branch (k_l, k_q, K)       : {audit.branch}",
        f"DNA alphabet cardinality             : {DNA_ALPHABET_CARDINALITY}",
        f"Sector verdict                       : {'PASS' if audit.sector_passed else 'FAIL'}",
        f"Sector statement                     : {audit.statement}",
        "",
        "Prime-Sync Biological Error-Correction Map",
        "-------------------------------------------",
        f"beta phase lock                      : {_format_decimal(audit.prime_sync.beta_phase_lock, places=15)}",
        f"mass step ratio e^beta               : {_format_decimal(audit.prime_sync.mass_step_ratio, places=15)}",
        f"prime-sync mean coordinate           : {_format_decimal(audit.prime_sync.mean_window_coordinate, places=15)}",
        f"prime-sync spread                    : {_format_decimal(audit.prime_sync.max_window_coordinate_spread, places=15)}",
        f"prime-sync spread tolerance          : {_format_decimal(PRIME_SYNC_SPREAD_TOLERANCE, places=15)}",
        f"prime-sync verified                  : {audit.prime_sync.prime_sync_verified}",
        f"prime-sync statement                 : {audit.prime_sync.statement}",
        "",
        "g   m_g/m_0                prime window     normalized coordinate",
        "---------------------------------------------------------------",
    ]
    lines.extend(
        (
            f"{window.generation:<1}   "
            f"{_format_decimal(window.mass_ratio_to_m0, places=15):>18}   "
            f"({window.lower_prime:>2}, {window.upper_prime:>2})        "
            f"{_format_decimal(window.normalized_window_coordinate, places=15):>18}"
        )
        for window in audit.prime_sync.windows
    )
    lines.extend(
        [
            "",
            "Shannon Entropy Limit Audit",
            "---------------------------",
            f"holographic bits N                    : {_format_decimal(audit.shannon_limit.holographic_bits, places=18)}",
            f"Shannon bits per nucleotide           : {_format_decimal(audit.shannon_limit.shannon_bits_per_nucleotide, places=15)}",
            f"duplex complement efficiency          : {_format_decimal(audit.shannon_limit.duplex_complement_efficiency, places=15)}",
            f"branch-locked complexity utilization  : {_format_decimal(audit.shannon_limit.complexity_utilization_fraction, places=15)}",
            f"clock-skew redundancy fraction        : {_format_decimal(audit.shannon_limit.clock_skew_fraction, places=15)}",
            f"maximum logical storage efficiency    : {_format_decimal(audit.shannon_limit.maximum_logical_storage_efficiency, places=15)}",
            f"maximum logical biological bits       : {_format_decimal(audit.shannon_limit.maximum_logical_storage_bits, places=18)}",
            f"maximum error-corrected base pairs    : {_format_decimal(audit.shannon_limit.maximum_error_corrected_base_pairs, places=18)}",
            f"within holographic budget             : {audit.shannon_limit.within_holographic_budget}",
            f"holographic ceiling verified          : {audit.shannon_limit.holographic_ceiling_verified}",
            f"Shannon-limit statement               : {audit.shannon_limit.statement}",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map branch-fixed mass residues into a biological prime-sync and Shannon-ceiling audit."
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal precision used for the Shannon-entropy ceiling audit.",
    )
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    print(build_complexity_sector_report(precision=max(int(args.precision), 32)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
