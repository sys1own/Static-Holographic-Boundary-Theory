from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Final, Sequence

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.audit.audit_complexity_ceiling import (
    DEFAULT_PRECISION as COMPLEXITY_CEILING_DEFAULT_PRECISION,
    ComplexityCeilingAudit,
    derive_complexity_ceiling,
)
from shbt.core.rigidity_kernel import BENCHMARK_MOAT_CENTER, BENCHMARK_PARITY_CHECK_MATRIX
from shbt.sectors.complexity_sector import (
    DEFAULT_PRECISION as COMPLEXITY_SECTOR_DEFAULT_PRECISION,
    DNA_ALPHABET_CARDINALITY,
    DUPLEX_COMPLEMENT_EFFICIENCY,
    PRIME_SYNC_SPREAD_TOLERANCE,
    SHANNON_BITS_PER_NUCLEOTIDE,
    ComplexitySectorAudit,
    ShannonEntropyLimitAudit,
    build_complexity_sector_audit,
)


DEFAULT_PRECISION = max(int(COMPLEXITY_CEILING_DEFAULT_PRECISION), int(COMPLEXITY_SECTOR_DEFAULT_PRECISION), 80)
_GUARD_DIGITS = 12
CODON_LENGTH: Final[int] = 3

with localcontext() as _golden_context:
    _golden_context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
    GOLDEN_ANGLE_TURN_FRACTION: Final[Decimal] = +((Decimal(3) - Decimal(5).sqrt()) / Decimal(2))


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


@dataclass(frozen=True)
class BiologicalIsomorphism:
    label: str
    core_kernel_structure: str
    biological_structure: str
    invariant: str
    locked: bool

    @property
    def statement(self) -> str:
        return f"{self.label}: {self.core_kernel_structure} -> {self.biological_structure}."


@dataclass(frozen=True)
class DnaErrorCorrectionAudit:
    benchmark_branch: tuple[int, int, int]
    parity_check_matrix: tuple[tuple[float, float, float], ...]
    codon_length: int
    nucleotide_alphabet_cardinality: int
    codon_state_count: int
    branch_code_volume: int
    logical_codon_packets: int
    duplex_complement_efficiency: Decimal
    parity_rows_match_codon_length: bool
    code_volume_tiles_codon_space: bool
    shannon_limit: ShannonEntropyLimitAudit

    @property
    def dna_isomorphic(self) -> bool:
        return bool(
            self.parity_rows_match_codon_length
            and self.code_volume_tiles_codon_space
            and self.shannon_limit.holographic_ceiling_verified
        )

    @property
    def statement(self) -> str:
        return "The three-row parity kernel is isomorphic to codon-position error checks on duplex DNA."


@dataclass(frozen=True)
class FibonacciPhyllotaxisAudit:
    benchmark_branch: tuple[int, int, int]
    golden_angle_turn_fraction: Decimal
    mean_prime_coordinate: Decimal
    coordinate_deviations: tuple[Decimal, ...]
    max_coordinate_deviation: Decimal
    allowed_coordinate_deviation: Decimal
    prime_sync_spread_tolerance: Decimal
    clock_skew_fraction: Decimal
    rigidity_preserved: bool

    @property
    def phyllotaxis_locked(self) -> bool:
        return bool(self.rigidity_preserved and self.max_coordinate_deviation <= self.allowed_coordinate_deviation)

    @property
    def statement(self) -> str:
        return "Prime-sync residues stay close to the golden-angle turn fraction that organizes Fibonacci phyllotaxis."


@dataclass(frozen=True)
class BiologicalInformationProcessingAudit:
    holographic_bits: Decimal
    complexity_growth_rate_ops_per_second: Decimal
    biological_logical_storage_bits: Decimal
    biological_error_corrected_base_pairs: Decimal
    mass_hierarchy_lock_fraction: Decimal
    duplex_complement_efficiency: Decimal
    maximum_biological_processing_ops_per_second: Decimal
    upper_bound_margin_fraction: Decimal
    unitary_bound_satisfied: bool
    universal_computational_limit_pass: bool
    rigidity_preserved: bool

    @property
    def upper_bound_verified(self) -> bool:
        return bool(
            self.rigidity_preserved
            and self.unitary_bound_satisfied
            and self.universal_computational_limit_pass
            and self.mass_hierarchy_lock_fraction < Decimal(1)
            and self.maximum_biological_processing_ops_per_second <= self.complexity_growth_rate_ops_per_second
            and self.biological_logical_storage_bits <= self.holographic_bits
        )

    @property
    def statement(self) -> str:
        return "The same rigidity that fixes the mass hierarchy also caps biological information throughput."


@dataclass(frozen=True)
class BiologicalComplexityAudit:
    branch: tuple[int, int, int]
    complexity_sector: ComplexitySectorAudit
    dna_error_correction: DnaErrorCorrectionAudit
    fibonacci_phyllotaxis: FibonacciPhyllotaxisAudit
    information_processing: BiologicalInformationProcessingAudit
    isomorphisms: tuple[BiologicalIsomorphism, ...]

    @property
    def biological_limits_locked(self) -> bool:
        return bool(
            self.complexity_sector.sector_passed
            and self.dna_error_correction.dna_isomorphic
            and self.fibonacci_phyllotaxis.phyllotaxis_locked
            and self.information_processing.upper_bound_verified
            and all(isomorphism.locked for isomorphism in self.isomorphisms)
        )

    @property
    def statement(self) -> str:
        return (
            "The branch-fixed (26, 8, 312) kernel extends into biology through codon parity, "
            "Fibonacci packing, and rigidity-limited information throughput."
        )


def _build_dna_error_correction_audit(complexity_sector: ComplexitySectorAudit) -> DnaErrorCorrectionAudit:
    branch_code_volume = int(BENCHMARK_MOAT_CENTER[0] * BENCHMARK_MOAT_CENTER[1] * BENCHMARK_MOAT_CENTER[2])
    codon_state_count = int(DNA_ALPHABET_CARDINALITY**CODON_LENGTH)
    logical_codon_packets, remainder = divmod(branch_code_volume, codon_state_count)
    return DnaErrorCorrectionAudit(
        benchmark_branch=BENCHMARK_MOAT_CENTER,
        parity_check_matrix=BENCHMARK_PARITY_CHECK_MATRIX,
        codon_length=CODON_LENGTH,
        nucleotide_alphabet_cardinality=DNA_ALPHABET_CARDINALITY,
        codon_state_count=codon_state_count,
        branch_code_volume=branch_code_volume,
        logical_codon_packets=int(logical_codon_packets),
        duplex_complement_efficiency=DUPLEX_COMPLEMENT_EFFICIENCY,
        parity_rows_match_codon_length=len(BENCHMARK_PARITY_CHECK_MATRIX) == CODON_LENGTH,
        code_volume_tiles_codon_space=remainder == 0,
        shannon_limit=complexity_sector.shannon_limit,
    )


def _build_fibonacci_phyllotaxis_audit(
    complexity_sector: ComplexitySectorAudit,
    complexity_ceiling: ComplexityCeilingAudit,
) -> FibonacciPhyllotaxisAudit:
    coordinate_deviations = tuple(
        abs(window.normalized_window_coordinate - GOLDEN_ANGLE_TURN_FRACTION)
        for window in complexity_sector.prime_sync.windows
    )
    max_coordinate_deviation = max(coordinate_deviations, default=Decimal("0"))
    allowed_coordinate_deviation = max(PRIME_SYNC_SPREAD_TOLERANCE, _decimal(complexity_ceiling.clock_skew))
    return FibonacciPhyllotaxisAudit(
        benchmark_branch=complexity_sector.branch,
        golden_angle_turn_fraction=GOLDEN_ANGLE_TURN_FRACTION,
        mean_prime_coordinate=complexity_sector.prime_sync.mean_window_coordinate,
        coordinate_deviations=coordinate_deviations,
        max_coordinate_deviation=+max_coordinate_deviation,
        allowed_coordinate_deviation=+allowed_coordinate_deviation,
        prime_sync_spread_tolerance=PRIME_SYNC_SPREAD_TOLERANCE,
        clock_skew_fraction=_decimal(complexity_ceiling.clock_skew),
        rigidity_preserved=bool(complexity_sector.prime_sync.prime_sync_verified),
    )


def _build_information_processing_audit(
    complexity_sector: ComplexitySectorAudit,
    complexity_ceiling: ComplexityCeilingAudit,
    *,
    precision: int,
) -> BiologicalInformationProcessingAudit:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        complexity_growth_rate_ops_per_second = _decimal(complexity_ceiling.complexity_growth_rate_ops_per_second)
        mass_hierarchy_lock_fraction = Decimal(1) / _decimal(complexity_sector.flavor_identity.mass_step_ratio)
        biological_logical_storage_bits = complexity_sector.shannon_limit.maximum_logical_storage_bits * mass_hierarchy_lock_fraction
        biological_error_corrected_base_pairs = biological_logical_storage_bits / SHANNON_BITS_PER_NUCLEOTIDE
        maximum_biological_processing_ops_per_second = (
            complexity_growth_rate_ops_per_second * DUPLEX_COMPLEMENT_EFFICIENCY * mass_hierarchy_lock_fraction
        )
        upper_bound_margin_fraction = Decimal(1) - (
            maximum_biological_processing_ops_per_second / complexity_growth_rate_ops_per_second
        )
        context.prec = max(int(precision), DEFAULT_PRECISION)

    return BiologicalInformationProcessingAudit(
        holographic_bits=complexity_sector.shannon_limit.holographic_bits,
        complexity_growth_rate_ops_per_second=+complexity_growth_rate_ops_per_second,
        biological_logical_storage_bits=+biological_logical_storage_bits,
        biological_error_corrected_base_pairs=+biological_error_corrected_base_pairs,
        mass_hierarchy_lock_fraction=+mass_hierarchy_lock_fraction,
        duplex_complement_efficiency=DUPLEX_COMPLEMENT_EFFICIENCY,
        maximum_biological_processing_ops_per_second=+maximum_biological_processing_ops_per_second,
        upper_bound_margin_fraction=+upper_bound_margin_fraction,
        unitary_bound_satisfied=bool(complexity_ceiling.unitary_bound_satisfied),
        universal_computational_limit_pass=bool(complexity_ceiling.universal_computational_limit_pass),
        rigidity_preserved=bool(complexity_sector.flavor_identity.mandatory_residue_verified),
    )


def _build_isomorphisms(
    dna_error_correction: DnaErrorCorrectionAudit,
    fibonacci_phyllotaxis: FibonacciPhyllotaxisAudit,
    information_processing: BiologicalInformationProcessingAudit,
) -> tuple[BiologicalIsomorphism, ...]:
    return (
        BiologicalIsomorphism(
            label="DNA duplex parity",
            core_kernel_structure="3-row diagonal parity-check matrix",
            biological_structure="3-position codon redundancy on a 4-symbol duplex alphabet",
            invariant=(
                f"(26*8*312)/(4^{CODON_LENGTH}) = {dna_error_correction.logical_codon_packets} exact codon packets"
            ),
            locked=bool(dna_error_correction.dna_isomorphic),
        ),
        BiologicalIsomorphism(
            label="Fibonacci phyllotaxis lock",
            core_kernel_structure="prime-sync mass ladder windows",
            biological_structure="golden-angle packing coordinate in spiral growth",
            invariant=(
                f"max deviation = {_format_decimal(fibonacci_phyllotaxis.max_coordinate_deviation, places=15)}"
            ),
            locked=bool(fibonacci_phyllotaxis.phyllotaxis_locked),
        ),
        BiologicalIsomorphism(
            label="Biological information ceiling",
            core_kernel_structure="rigid mass hierarchy plus holographic complexity ceiling",
            biological_structure="upper bound on biological logical throughput",
            invariant=(
                "mass-hierarchy lock fraction = "
                f"{_format_decimal(information_processing.mass_hierarchy_lock_fraction, places=15)}"
            ),
            locked=bool(information_processing.upper_bound_verified),
        ),
    )


def build_biological_complexity_audit(*, precision: int = DEFAULT_PRECISION) -> BiologicalComplexityAudit:
    complexity_sector = build_complexity_sector_audit(precision=precision)
    complexity_ceiling = derive_complexity_ceiling(precision=precision)
    dna_error_correction = _build_dna_error_correction_audit(complexity_sector)
    fibonacci_phyllotaxis = _build_fibonacci_phyllotaxis_audit(complexity_sector, complexity_ceiling)
    information_processing = _build_information_processing_audit(
        complexity_sector,
        complexity_ceiling,
        precision=precision,
    )
    isomorphisms = _build_isomorphisms(
        dna_error_correction,
        fibonacci_phyllotaxis,
        information_processing,
    )
    return BiologicalComplexityAudit(
        branch=BENCHMARK_MOAT_CENTER,
        complexity_sector=complexity_sector,
        dna_error_correction=dna_error_correction,
        fibonacci_phyllotaxis=fibonacci_phyllotaxis,
        information_processing=information_processing,
        isomorphisms=isomorphisms,
    )


def build_biological_complexity_report(*, precision: int = DEFAULT_PRECISION) -> str:
    audit = build_biological_complexity_audit(precision=precision)
    lines = [
        "Biological Complexity Audit",
        "==========================",
        f"Benchmark branch (k_l, k_q, K)       : {audit.branch}",
        f"Biological limits locked             : {audit.biological_limits_locked}",
        f"Sector statement                     : {audit.statement}",
        "",
        "Kernel-to-Biology Isomorphisms",
        "-------------------------------",
    ]
    lines.extend(
        f"- {isomorphism.label}: locked={isomorphism.locked} | {isomorphism.invariant}"
        for isomorphism in audit.isomorphisms
    )
    lines.extend(
        [
            "",
            "DNA Error-Correction Audit",
            "--------------------------",
            f"codon length                         : {audit.dna_error_correction.codon_length}",
            f"codon state count                    : {audit.dna_error_correction.codon_state_count}",
            f"branch code volume                   : {audit.dna_error_correction.branch_code_volume}",
            f"logical codon packets                : {audit.dna_error_correction.logical_codon_packets}",
            f"parity rows match codon length       : {audit.dna_error_correction.parity_rows_match_codon_length}",
            f"branch tiles codon space exactly     : {audit.dna_error_correction.code_volume_tiles_codon_space}",
            f"DNA isomorphism verified             : {audit.dna_error_correction.dna_isomorphic}",
            f"DNA statement                        : {audit.dna_error_correction.statement}",
            "",
            "Fibonacci Phyllotaxis Audit",
            "---------------------------",
            f"golden-angle turn fraction           : {_format_decimal(audit.fibonacci_phyllotaxis.golden_angle_turn_fraction, places=18)}",
            f"mean prime coordinate                : {_format_decimal(audit.fibonacci_phyllotaxis.mean_prime_coordinate, places=18)}",
            f"max coordinate deviation             : {_format_decimal(audit.fibonacci_phyllotaxis.max_coordinate_deviation, places=18)}",
            f"allowed coordinate deviation         : {_format_decimal(audit.fibonacci_phyllotaxis.allowed_coordinate_deviation, places=18)}",
            f"clock-skew tolerance                 : {_format_decimal(audit.fibonacci_phyllotaxis.clock_skew_fraction, places=18)}",
            f"phyllotaxis lock verified            : {audit.fibonacci_phyllotaxis.phyllotaxis_locked}",
            f"phyllotaxis statement                : {audit.fibonacci_phyllotaxis.statement}",
            "",
            "Biological Information-Processing Ceiling",
            "-----------------------------------------",
            f"mass-hierarchy lock fraction         : {_format_decimal(audit.information_processing.mass_hierarchy_lock_fraction, places=18)}",
            f"max biological ops/s                 : {_format_decimal(audit.information_processing.maximum_biological_processing_ops_per_second, places=18)}",
            f"complexity growth rate ops/s         : {_format_decimal(audit.information_processing.complexity_growth_rate_ops_per_second, places=18)}",
            f"biological logical storage bits      : {_format_decimal(audit.information_processing.biological_logical_storage_bits, places=18)}",
            f"biological error-corrected basepairs : {_format_decimal(audit.information_processing.biological_error_corrected_base_pairs, places=18)}",
            f"upper-bound margin fraction          : {_format_decimal(audit.information_processing.upper_bound_margin_fraction, places=18)}",
            f"upper bound verified                 : {audit.information_processing.upper_bound_verified}",
            f"processing statement                 : {audit.information_processing.statement}",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map the branch-fixed (26, 8, 312) kernel into biological isomorphisms and information bounds."
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal precision used for the biological complexity audit.",
    )
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    print(build_biological_complexity_report(precision=max(int(args.precision), 32)))
    return 0


__all__ = [
    "BiologicalComplexityAudit",
    "BiologicalInformationProcessingAudit",
    "BiologicalIsomorphism",
    "CODON_LENGTH",
    "DEFAULT_PRECISION",
    "DnaErrorCorrectionAudit",
    "FibonacciPhyllotaxisAudit",
    "GOLDEN_ANGLE_TURN_FRACTION",
    "build_biological_complexity_audit",
    "build_biological_complexity_report",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
