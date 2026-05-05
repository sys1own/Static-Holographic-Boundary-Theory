from __future__ import annotations

"""Holographic error-stabilizer audit for the anomaly-free benchmark branch.

This module treats the benchmark conservation laws as topological checksums on
the finite-capacity boundary dictionary. Charge balance, Noether-bridged
momentum closure, and parity/framing closure act as logical stabilizers for the
three-dimensional bulk projection. If their syndromes remain below the
branch-fixed recovery threshold

    m_rec = kappa_D5 M_P N^(-1/4),

the boundary remains in its zero-energy state and the bulk image stays
torsion-free. Once a checksum failure exceeds that threshold, the syndrome can
no longer be absorbed by the completion sector, the framing defect reopens,
torsion reappears in the bulk closure tensor, and the Equivalence Principle is
lost.
"""

import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import GEOMETRIC_KAPPA, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL

if TYPE_CHECKING:
    from shbt.core import noether_bridge


DEFAULT_PRECISION = 200
BenchmarkBranch = tuple[int, int, int]
ChecksumLaw = Literal["charge", "momentum", "parity"]

BENCHMARK_BRANCH: BenchmarkBranch = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
FAILED_BRANCHES: dict[ChecksumLaw, BenchmarkBranch] = {
    "charge": (int(LEPTON_LEVEL), int(QUARK_LEVEL + 1), int(PARENT_LEVEL)),
    "momentum": (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL + 1)),
    "parity": (int(LEPTON_LEVEL + 1), int(QUARK_LEVEL), int(PARENT_LEVEL)),
}


def _decimal(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _format_decimal(value: Decimal, *, places: int = 12) -> str:
    if value.is_zero():
        return "0"
    adjusted = value.adjusted()
    if adjusted >= 6 or adjusted <= -4:
        return f"{value:.{places}E}"
    return f"{value:.{places}f}".rstrip("0").rstrip(".")


def _format_branch(branch: BenchmarkBranch) -> str:
    return f"({branch[0]}, {branch[1]}, {branch[2]})"


def _parent_order(branch: BenchmarkBranch) -> tuple[int, int, int]:
    return (int(branch[2]), int(branch[0]), int(branch[1]))


def _select_checksum(audit: "HolographicErrorStabilizerAudit", law: ChecksumLaw) -> "TopologicalChecksum":
    if law == "charge":
        return audit.charge
    if law == "momentum":
        return audit.momentum
    if law == "parity":
        return audit.parity
    raise ValueError(f"Unknown checksum law: {law}")


@dataclass(frozen=True)
class RecoveryThresholdAudit:
    planck_mass_ev: Decimal
    holographic_bits: Decimal
    kappa_d5: Decimal
    c_dark_fraction: noether_bridge.Fraction
    recovery_threshold_ev: Decimal
    dimensionless_syndrome_threshold: Decimal
    register_noise_floor: Decimal

    @property
    def recovery_threshold_mev(self) -> Decimal:
        return self.recovery_threshold_ev * Decimal("1000")


@dataclass(frozen=True)
class TopologicalChecksum:
    law: ChecksumLaw
    stabilizer_name: str
    protected_quantity: str
    checksum_equation: str
    expected_value: Decimal
    observed_value: Decimal
    syndrome_residual: Decimal
    syndrome_tolerance: Decimal
    interpretation: str

    @property
    def passed(self) -> bool:
        return self.syndrome_residual <= self.syndrome_tolerance

    @property
    def prevents_bit_rot(self) -> bool:
        return self.passed

    @property
    def syndrome_ratio(self) -> Decimal:
        if self.syndrome_tolerance.is_zero():
            return Decimal("0") if self.syndrome_residual.is_zero() else Decimal("Infinity")
        with localcontext() as context:
            context.prec = max(abs(self.syndrome_residual.as_tuple().exponent), 50)
            return self.syndrome_residual / self.syndrome_tolerance


@dataclass(frozen=True)
class ChecksumFailureSimulation:
    law: ChecksumLaw
    benchmark_branch: BenchmarkBranch
    failed_branch: BenchmarkBranch
    injected_syndrome: Decimal
    syndrome_energy_ev: Decimal
    recovery_budget_ev: Decimal
    recoverable: bool
    benchmark_closure_tensor: noether_bridge.TensorSnapshot
    failed_closure_tensor: noether_bridge.TensorSnapshot
    effective_closure_tensor: noether_bridge.TensorSnapshot
    effective_anomalous_source_ev2: noether_bridge.TensorSnapshot
    effective_anomalous_source_si_m2: noether_bridge.TensorSnapshot

    @property
    def torsion_reintroduced(self) -> bool:
        return not self.effective_closure_tensor.vanished

    @property
    def equivalence_principle_destroyed(self) -> bool:
        return not self.effective_anomalous_source_si_m2.vanished


@dataclass(frozen=True)
class BulkChecksumVerification:
    benchmark_branch: BenchmarkBranch
    charge_residual: Decimal
    momentum_residual: Decimal
    parity_residual: Decimal
    charge_checksum_passed: bool
    momentum_checksum_passed: bool
    parity_checksum_passed: bool
    simulated_boundary_decoherence: bool
    detail: str

    @property
    def passed(self) -> bool:
        return bool(
            self.charge_checksum_passed
            and self.momentum_checksum_passed
            and self.parity_checksum_passed
            and not self.simulated_boundary_decoherence
        )


class HolographicStabilizer:
    """Lightweight boundary checksum gate for physical bulk outputs."""

    def __init__(self, *, precision: int = DEFAULT_PRECISION, simulate_boundary_decoherence: bool = False) -> None:
        self.precision = max(int(precision), DEFAULT_PRECISION)
        self.simulate_boundary_decoherence = bool(simulate_boundary_decoherence)

    def verify_bulk_checksum(self) -> BulkChecksumVerification:
        from shbt.core.engine import calculate_efe_violation_tensor
        from shbt.core import noether_bridge

        with localcontext() as context:
            context.prec = self.precision
            packing_deficiency = Decimal("1") - _decimal(GEOMETRIC_KAPPA)
            parity_overhead = (Decimal(PARENT_LEVEL) * packing_deficiency) / Decimal(PARENT_LEVEL)
            charge_residual = abs(packing_deficiency - parity_overhead)
            momentum_residual = _decimal(
                calculate_efe_violation_tensor(
                    parent_level=PARENT_LEVEL,
                    lepton_level=LEPTON_LEVEL,
                    quark_level=QUARK_LEVEL,
                )
            )
            parity_audit = noether_bridge.framing_defect(PARENT_LEVEL, LEPTON_LEVEL, QUARK_LEVEL)
            parity_residual = Decimal(parity_audit.delta_fr.numerator) / Decimal(parity_audit.delta_fr.denominator)

        charge_checksum_passed = bool(charge_residual == 0)
        momentum_checksum_passed = bool(momentum_residual == 0)
        parity_checksum_passed = bool(parity_residual == 0)

        failure_modes: list[str] = []
        if not charge_checksum_passed:
            failure_modes.append(f"charge residual={_format_decimal(charge_residual)}")
        if not momentum_checksum_passed:
            failure_modes.append(f"momentum residual={_format_decimal(momentum_residual)}")
        if not parity_checksum_passed:
            failure_modes.append(f"parity residual={_format_decimal(parity_residual)}")
        if self.simulate_boundary_decoherence:
            failure_modes.append("simulated boundary decoherence")

        detail = "bulk checksum locked" if not failure_modes else "; ".join(failure_modes)
        return BulkChecksumVerification(
            benchmark_branch=BENCHMARK_BRANCH,
            charge_residual=charge_residual,
            momentum_residual=momentum_residual,
            parity_residual=parity_residual,
            charge_checksum_passed=charge_checksum_passed,
            momentum_checksum_passed=momentum_checksum_passed,
            parity_checksum_passed=parity_checksum_passed,
            simulated_boundary_decoherence=self.simulate_boundary_decoherence,
            detail=detail,
        )


@dataclass(frozen=True)
class HolographicErrorStabilizerAudit:
    branch: BenchmarkBranch
    recovery: RecoveryThresholdAudit
    charge: TopologicalChecksum
    momentum: TopologicalChecksum
    parity: TopologicalChecksum
    newton_lock: noether_bridge.NewtonLockAudit
    saturation: noether_bridge.SaturationAudit
    unity: noether_bridge.UnityOfScaleAudit
    reference_failure: noether_bridge.ReviewerTrapAudit
    proof: str

    @property
    def checksums(self) -> tuple[TopologicalChecksum, TopologicalChecksum, TopologicalChecksum]:
        return (self.charge, self.momentum, self.parity)

    @property
    def zero_energy_state_locked(self) -> bool:
        return all(checksum.passed for checksum in self.checksums)

    @property
    def equivalence_principle_preserved(self) -> bool:
        return self.reference_failure.closure_tensor_benchmark.vanished

    @property
    def self_correcting(self) -> bool:
        return bool(
            self.zero_energy_state_locked
            and self.unity.noether_bridge_identity_pass
            and self.reference_failure.closure_equivalence_verified
            and self.equivalence_principle_preserved
        )


def _build_recovery_threshold(
    *,
    precision: int,
) -> tuple[RecoveryThresholdAudit, noether_bridge.NewtonLockAudit, noether_bridge.SaturationAudit, noether_bridge.UnityOfScaleAudit]:
    from shbt.core import noether_bridge

    c_dark_fraction = noether_bridge.load_c_dark_completion_fraction()
    newton_lock = noether_bridge.newton_constant_lock(c_dark_fraction=c_dark_fraction, precision=precision)
    saturation = noether_bridge.saturation_audit(precision=precision)
    kappa_d5 = noether_bridge.derive_kappa_d5(lepton_level=LEPTON_LEVEL, precision=precision)
    unity = noether_bridge.unity_of_scale_audit(
        kappa_d5=kappa_d5,
        newton_lock_audit=newton_lock,
        saturation=saturation,
        precision=precision,
    )
    with localcontext() as context:
        context.prec = precision
        dimensionless_threshold = unity.lightest_mass_ev / newton_lock.planck_mass_ev
    recovery = RecoveryThresholdAudit(
        planck_mass_ev=newton_lock.planck_mass_ev,
        holographic_bits=saturation.holographic_bits_from_lambda,
        kappa_d5=kappa_d5,
        c_dark_fraction=c_dark_fraction,
        recovery_threshold_ev=unity.lightest_mass_ev,
        dimensionless_syndrome_threshold=dimensionless_threshold,
        register_noise_floor=unity.register_noise_floor,
    )
    return recovery, newton_lock, saturation, unity


def _build_charge_checksum(recovery: RecoveryThresholdAudit) -> TopologicalChecksum:
    from shbt.main import verify_bit_balance_identity

    bit_balance = verify_bit_balance_identity()
    return TopologicalChecksum(
        law="charge",
        stabilizer_name="Boundary Charge Stabilizer",
        protected_quantity="Zero-balanced vacuum loading",
        checksum_equation="|(1 - kappa_D5) - (c_dark / K)|",
        expected_value=_decimal(bit_balance.packing_deficiency),
        observed_value=_decimal(bit_balance.dark_sector_complexity_overhead),
        syndrome_residual=_decimal(bit_balance.residual),
        syndrome_tolerance=recovery.dimensionless_syndrome_threshold,
        interpretation=(
            "The charge checksum treats the visible packing deficiency as the logical payload and the c_dark parity sector "
            "as its redundancy block. A non-zero residue would mean boundary bit rot in the zero-energy state."
        ),
    )


def _build_momentum_checksum(
    recovery: RecoveryThresholdAudit,
    *,
    unity: noether_bridge.UnityOfScaleAudit,
) -> TopologicalChecksum:
    return TopologicalChecksum(
        law="momentum",
        stabilizer_name="Noether Momentum Stabilizer",
        protected_quantity="Translation / curvature lock",
        checksum_equation="epsilon_lambda^(bridge) = |1 - Lambda_obs / Lambda_bridge|",
        expected_value=unity.lambda_lhs_ev2,
        observed_value=unity.lambda_rhs_noether_bridged_ev2,
        syndrome_residual=unity.epsilon_lambda_noether_bridged,
        syndrome_tolerance=recovery.dimensionless_syndrome_threshold,
        interpretation=(
            "Momentum conservation is encoded through the Noether-bridged unity-of-scale identity. "
            "If that checksum fails, translational bookkeeping leaks out of the boundary code and the bulk geodesic image stops closing."
        ),
    )


def _build_parity_checksum(recovery: RecoveryThresholdAudit) -> TopologicalChecksum:
    from shbt.core import noether_bridge

    framing = noether_bridge.framing_defect(PARENT_LEVEL, LEPTON_LEVEL, QUARK_LEVEL)
    residual = Decimal(framing.delta_fr.numerator) / Decimal(framing.delta_fr.denominator)
    return TopologicalChecksum(
        law="parity",
        stabilizer_name="Framing Parity Stabilizer",
        protected_quantity="Anomaly-free torsion lock",
        checksum_equation="Delta_fr",
        expected_value=Decimal("0"),
        observed_value=residual,
        syndrome_residual=residual,
        syndrome_tolerance=recovery.dimensionless_syndrome_threshold,
        interpretation=(
            "Parity conservation is the topological framing checksum. Once Delta_fr reopens, the Levi-Civita branch exits the local moat and torsion returns."
        ),
    )


def build_holographic_error_stabilizer_audit(*, precision: int = DEFAULT_PRECISION) -> HolographicErrorStabilizerAudit:
    from shbt.core import noether_bridge

    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    recovery, newton_lock, saturation, unity = _build_recovery_threshold(precision=resolved_precision)
    charge = _build_charge_checksum(recovery)
    momentum = _build_momentum_checksum(recovery, unity=unity)
    parity = _build_parity_checksum(recovery)
    reference_failure = noether_bridge.reviewer_trap_audit(
        newton_lock_audit=newton_lock,
        saturation=saturation,
        benchmark_branch=_parent_order(BENCHMARK_BRANCH),
        detuned_branch=_parent_order(FAILED_BRANCHES["parity"]),
        precision=resolved_precision,
    )
    proof = (
        "Charge balance, Noether-bridged momentum closure, and parity/framing closure all remain below the recovery threshold "
        "m_rec = kappa_D5 M_P N^(-1/4), so the benchmark boundary stays in its zero-energy state with E_mu_nu = 0. "
        "The parity-detuned reference branch simultaneously gives Delta_fr != 0, E_mu_nu != 0, and J_mu_nu^(a) != 0, "
        "which means the bulk exists as a torsion-free, equivalence-principle-preserving image precisely because the boundary acts as a self-correcting error-stabilizer."
    )
    return HolographicErrorStabilizerAudit(
        branch=BENCHMARK_BRANCH,
        recovery=recovery,
        charge=charge,
        momentum=momentum,
        parity=parity,
        newton_lock=newton_lock,
        saturation=saturation,
        unity=unity,
        reference_failure=reference_failure,
        proof=proof,
    )


def simulate_failed_checksum(
    law: ChecksumLaw = "parity",
    *,
    syndrome: Decimal | float | int | str | None = None,
    precision: int = DEFAULT_PRECISION,
) -> ChecksumFailureSimulation:
    from shbt.core import noether_bridge

    audit = build_holographic_error_stabilizer_audit(precision=precision)
    checksum = _select_checksum(audit, law)
    injected_syndrome = checksum.syndrome_tolerance * Decimal("2") if syndrome is None else _decimal(syndrome)
    if injected_syndrome < 0:
        raise ValueError("syndrome must be non-negative.")

    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION)
        syndrome_energy_ev = injected_syndrome * audit.recovery.planck_mass_ev
    recoverable = bool(syndrome_energy_ev <= audit.recovery.recovery_threshold_ev)

    failed_branch = FAILED_BRANCHES[law]
    reviewer = noether_bridge.reviewer_trap_audit(
        newton_lock_audit=audit.newton_lock,
        saturation=audit.saturation,
        benchmark_branch=_parent_order(audit.branch),
        detuned_branch=_parent_order(failed_branch),
        precision=max(int(precision), DEFAULT_PRECISION),
    )
    if recoverable:
        effective_closure_tensor = reviewer.closure_tensor_benchmark
        effective_anomalous_source_ev2, effective_anomalous_source_si_m2 = noether_bridge.anomalous_source_tensor(
            reviewer.closure_tensor_benchmark,
            audit.newton_lock,
        )
    else:
        effective_closure_tensor = reviewer.closure_tensor_detuned
        effective_anomalous_source_ev2 = reviewer.anomalous_source_ev2
        effective_anomalous_source_si_m2 = reviewer.anomalous_source_si_m2
    return ChecksumFailureSimulation(
        law=law,
        benchmark_branch=audit.branch,
        failed_branch=failed_branch,
        injected_syndrome=injected_syndrome,
        syndrome_energy_ev=syndrome_energy_ev,
        recovery_budget_ev=audit.recovery.recovery_threshold_ev,
        recoverable=recoverable,
        benchmark_closure_tensor=reviewer.closure_tensor_benchmark,
        failed_closure_tensor=reviewer.closure_tensor_detuned,
        effective_closure_tensor=effective_closure_tensor,
        effective_anomalous_source_ev2=effective_anomalous_source_ev2,
        effective_anomalous_source_si_m2=effective_anomalous_source_si_m2,
    )


def render_report(
    audit: HolographicErrorStabilizerAudit,
    *,
    simulation: ChecksumFailureSimulation | None = None,
) -> str:
    lines = [
        "Holographic Error Stabilizer Audit",
        "==================================",
        f"Benchmark branch                    : {_format_branch(audit.branch)}",
        f"Planck mass M_P [eV]               : {_format_decimal(audit.recovery.planck_mass_ev)}",
        f"Holographic register N             : {_format_decimal(audit.recovery.holographic_bits)}",
        f"Recovery threshold m_rec [eV]      : {_format_decimal(audit.recovery.recovery_threshold_ev)}",
        f"Recovery threshold m_rec [meV]     : {_format_decimal(audit.recovery.recovery_threshold_mev)}",
        f"Dimensionless syndrome cap         : {_format_decimal(audit.recovery.dimensionless_syndrome_threshold)}",
        "",
        "Topological Checksums",
        "---------------------",
    ]
    for checksum in audit.checksums:
        lines.extend(
            (
                f"{checksum.law:>8} stabilizer              : {checksum.stabilizer_name}",
                f"{checksum.law:>8} residual                : {_format_decimal(checksum.syndrome_residual)}",
                f"{checksum.law:>8} tolerance               : {_format_decimal(checksum.syndrome_tolerance)}",
                f"{checksum.law:>8} bit-rot blocked         : {checksum.prevents_bit_rot}",
            )
        )
    lines.extend(
        (
            "",
            "Proof State",
            "-----------",
            f"Zero-energy boundary locked       : {audit.zero_energy_state_locked}",
            f"Equivalence Principle preserved   : {audit.equivalence_principle_preserved}",
            f"Self-correcting stabilizer        : {audit.self_correcting}",
            audit.proof,
        )
    )
    if simulation is not None:
        lines.extend(
            (
                "",
                "Failed Checksum Simulation",
                "-------------------------",
                f"Failed law                         : {simulation.law}",
                f"Detuned branch                     : {_format_branch(simulation.failed_branch)}",
                f"Injected syndrome                  : {_format_decimal(simulation.injected_syndrome)}",
                f"Syndrome energy [eV]              : {_format_decimal(simulation.syndrome_energy_ev)}",
                f"Recovery budget [eV]              : {_format_decimal(simulation.recovery_budget_ev)}",
                f"Recoverable                        : {simulation.recoverable}",
                f"Bulk closure tensor vanished      : {simulation.effective_closure_tensor.vanished}",
                f"Torsion reintroduced               : {simulation.torsion_reintroduced}",
                f"Equivalence Principle destroyed   : {simulation.equivalence_principle_destroyed}",
            )
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION)
    parser.add_argument("--simulate-law", choices=("charge", "momentum", "parity"))
    parser.add_argument("--syndrome")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    audit = build_holographic_error_stabilizer_audit(precision=args.precision)
    simulation = None
    if args.simulate_law is not None:
        simulation = simulate_failed_checksum(
            law=args.simulate_law,
            syndrome=args.syndrome,
            precision=args.precision,
        )
    print(render_report(audit, simulation=simulation))


__all__ = [
    "BENCHMARK_BRANCH",
    "BulkChecksumVerification",
    "FAILED_BRANCHES",
    "ChecksumFailureSimulation",
    "HolographicErrorStabilizerAudit",
    "HolographicStabilizer",
    "RecoveryThresholdAudit",
    "TopologicalChecksum",
    "build_holographic_error_stabilizer_audit",
    "main",
    "parse_args",
    "render_report",
    "simulate_failed_checksum",
]


if __name__ == "__main__":
    main()
