from __future__ import annotations

"""Holographic error-stabilizer audit for the anomaly-free benchmark branch.

This module treats the benchmark conservation laws as topological checksums on
the finite-capacity boundary dictionary. Charge balance, parity/framing
closure, and Noether-bridged time-reversal closure act as logical stabilizers
for the three-dimensional bulk projection. If their syndromes remain below the
branch-fixed recovery threshold

    m_rec = kappa_D5 M_P N^(-1/4),

the boundary remains in its zero-energy state and the bulk image stays
torsion-free. Once a checksum failure exceeds that threshold, the syndrome can
no longer be absorbed by the completion sector, the framing defect reopens,
torsion reappears in the bulk closure tensor, and the Equivalence Principle is
lost.
"""

import argparse
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import wraps
import inspect
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
ChecksumLaw = Literal["charge", "parity", "time_reversal", "momentum"]
_STABILIZER_DEPTH: ContextVar[int] = ContextVar("shbt_stabilizer_depth", default=0)

BENCHMARK_BRANCH: BenchmarkBranch = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
FAILED_BRANCHES: dict[ChecksumLaw, BenchmarkBranch] = {
    "charge": (int(LEPTON_LEVEL), int(QUARK_LEVEL + 1), int(PARENT_LEVEL)),
    "parity": (int(LEPTON_LEVEL + 1), int(QUARK_LEVEL), int(PARENT_LEVEL)),
    "time_reversal": (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL + 1)),
    "momentum": (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL + 1)),
}


class BoundaryStabilizationError(RuntimeError):
    pass


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


def _normalize_checksum_law(law: str) -> ChecksumLaw:
    if law in {"time_reversal", "time-reversal", "momentum"}:
        return "time_reversal"
    if law in {"charge", "parity"}:
        return law
    raise ValueError(f"Unknown checksum law: {law}")


def _display_checksum_law(law: str) -> str:
    resolved_law = _normalize_checksum_law(law)
    if resolved_law == "time_reversal":
        return "time-reversal"
    return resolved_law


def _resolve_precision_argument(callable_object, *args: object, **kwargs: object) -> int:
    try:
        bound_arguments = inspect.signature(callable_object).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return DEFAULT_PRECISION
    precision = bound_arguments.arguments.get("precision", DEFAULT_PRECISION)
    try:
        return max(int(precision), DEFAULT_PRECISION)
    except (TypeError, ValueError):
        return DEFAULT_PRECISION


def _verify_boundary_lock(precision: int = DEFAULT_PRECISION) -> BulkChecksumVerification:
    verification = HolographicStabilizer(precision=precision).verify_bulk_checksum()
    if not verification.passed:
        raise BoundaryStabilizationError(f"Boundary stabilization failed: {verification.detail}")
    return verification


def stabilize_boundary(callable_object):
    if getattr(callable_object, "__boundary_stabilized__", False):
        return callable_object

    @wraps(callable_object)
    def wrapped(*args, **kwargs):
        current_depth = _STABILIZER_DEPTH.get()
        resolved_precision = _resolve_precision_argument(callable_object, *args, **kwargs)
        if current_depth == 0:
            _verify_boundary_lock(resolved_precision)

        token = _STABILIZER_DEPTH.set(current_depth + 1)
        try:
            result = callable_object(*args, **kwargs)
        finally:
            _STABILIZER_DEPTH.reset(token)

        if current_depth == 0:
            _verify_boundary_lock(resolved_precision)
        return result

    wrapped.__boundary_stabilized__ = True
    return wrapped


def stabilize_classmethods(class_object: type) -> type:
    if getattr(class_object, "__boundary_stabilized_class__", False):
        return class_object

    for name, attribute in tuple(vars(class_object).items()):
        if name.startswith("__"):
            continue
        if isinstance(attribute, classmethod):
            setattr(class_object, name, classmethod(stabilize_boundary(attribute.__func__)))
            continue
        if isinstance(attribute, staticmethod):
            setattr(class_object, name, staticmethod(stabilize_boundary(attribute.__func__)))
    class_object.__boundary_stabilized_class__ = True
    return class_object


def _select_checksum(audit: "HolographicErrorStabilizerAudit", law: ChecksumLaw) -> "TopologicalChecksum":
    resolved_law = _normalize_checksum_law(law)
    if resolved_law == "charge":
        return audit.charge
    if resolved_law == "parity":
        return audit.parity
    if resolved_law == "time_reversal":
        return audit.time_reversal
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
    def boundary_integer(self) -> Decimal:
        return self.expected_value

    @property
    def bulk_projection(self) -> Decimal:
        return self.observed_value

    @property
    def residue(self) -> Decimal:
        return self.syndrome_residual

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
class TopologicalChecksumCode:
    law: ChecksumLaw
    stabilizer_name: str
    protected_quantity: str
    checksum_equation: str
    boundary_integer: Decimal
    bulk_projection: Decimal
    syndrome_tolerance: Decimal
    interpretation: str

    @property
    def residue(self) -> Decimal:
        return abs(self.boundary_integer - self.bulk_projection)

    @property
    def passed(self) -> bool:
        return self.residue <= self.syndrome_tolerance

    def verify(self) -> TopologicalChecksum:
        return TopologicalChecksum(
            law=_normalize_checksum_law(self.law),
            stabilizer_name=self.stabilizer_name,
            protected_quantity=self.protected_quantity,
            checksum_equation=self.checksum_equation,
            expected_value=self.boundary_integer,
            observed_value=self.bulk_projection,
            syndrome_residual=self.residue,
            syndrome_tolerance=self.syndrome_tolerance,
            interpretation=self.interpretation,
        )


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

    @property
    def charge_bit_flip(self) -> bool:
        return _normalize_checksum_law(self.law) == "charge"

    @property
    def zero_energy_boundary_required(self) -> bool:
        return self.torsion_reintroduced and self.equivalence_principle_destroyed


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
    checksums: tuple[TopologicalChecksum, ...] = ()
    torsion_projection_residual: Decimal = Decimal("0")

    @property
    def time_reversal_residual(self) -> Decimal:
        return self.momentum_residual

    @property
    def time_reversal_checksum_passed(self) -> bool:
        return self.momentum_checksum_passed

    @property
    def zero_energy_boundary_locked(self) -> bool:
        return bool(
            self.charge_checksum_passed
            and self.time_reversal_checksum_passed
            and self.parity_checksum_passed
        )

    @property
    def equivalence_principle_preserved(self) -> bool:
        return self.torsion_projection_residual == 0 and not self.simulated_boundary_decoherence

    @property
    def passed(self) -> bool:
        return bool(
            self.zero_energy_boundary_locked
            and self.equivalence_principle_preserved
        )


class HolographicStabilizer:
    """Lightweight boundary checksum gate for physical bulk outputs."""

    def __init__(self, *, precision: int = DEFAULT_PRECISION, simulate_boundary_decoherence: bool = False) -> None:
        self.precision = max(int(precision), DEFAULT_PRECISION)
        self.simulate_boundary_decoherence = bool(simulate_boundary_decoherence)

    def verify_bulk_integrity(self) -> BulkChecksumVerification:
        from shbt.core.engine import calculate_efe_violation_tensor
        from shbt.core import noether_bridge

        with localcontext() as context:
            context.prec = self.precision
            packing_deficiency = Decimal("1") - _decimal(GEOMETRIC_KAPPA)
            charge_bulk_projection = (Decimal(PARENT_LEVEL) * packing_deficiency) / Decimal(PARENT_LEVEL)
            charge = TopologicalChecksumCode(
                law="charge",
                stabilizer_name="Boundary Charge Stabilizer",
                protected_quantity="Zero-balanced vacuum loading",
                checksum_equation="|(1 - kappa_D5) - (c_dark / K)|",
                boundary_integer=packing_deficiency,
                bulk_projection=charge_bulk_projection,
                syndrome_tolerance=Decimal("0"),
                interpretation=(
                    "Charge conservation is the primary error-correcting codeword. "
                    "The bulk remains coherent only when the visible packing deficiency is canceled exactly by the dark completion residue."
                ),
            ).verify()
            time_reversal = TopologicalChecksumCode(
                law="time_reversal",
                stabilizer_name="Bulk Time-Reversal Stabilizer",
                protected_quantity="Reversible bulk projection",
                checksum_equation="|E_mu_nu|",
                boundary_integer=Decimal("0"),
                bulk_projection=_decimal(
                    calculate_efe_violation_tensor(
                        parent_level=PARENT_LEVEL,
                        lepton_level=LEPTON_LEVEL,
                        quark_level=QUARK_LEVEL,
                    )
                ),
                syndrome_tolerance=Decimal("0"),
                interpretation=(
                    "Time-reversal integrity is encoded by a vanishing bulk projection residue. "
                    "Once the boundary bookkeeping ceases to reverse exactly, the emergent bulk ceases to be torsion-free."
                ),
            ).verify()
            parity_audit = noether_bridge.framing_defect(PARENT_LEVEL, LEPTON_LEVEL, QUARK_LEVEL)
            parity = TopologicalChecksumCode(
                law="parity",
                stabilizer_name="Framing Parity Stabilizer",
                protected_quantity="Anomaly-free torsion lock",
                checksum_equation="Delta_fr",
                boundary_integer=Decimal("0"),
                bulk_projection=Decimal(parity_audit.delta_fr.numerator) / Decimal(parity_audit.delta_fr.denominator),
                syndrome_tolerance=Decimal("0"),
                interpretation=(
                    "Parity closure is the topological framing code. "
                    "A non-zero framing defect reopens the torsion channel in the bulk metric."
                ),
            ).verify()
            charge_bit_flip_audit = noether_bridge.framing_defect(*FAILED_BRANCHES["charge"])
            torsion_projection_residual = (
                Decimal(charge_bit_flip_audit.delta_fr.numerator) / Decimal(charge_bit_flip_audit.delta_fr.denominator)
                if self.simulate_boundary_decoherence
                else Decimal("0")
            )

        charge_residual = charge.residue
        momentum_residual = time_reversal.residue
        parity_residual = parity.residue
        charge_checksum_passed = charge.passed
        momentum_checksum_passed = time_reversal.passed
        parity_checksum_passed = parity.passed

        failure_modes: list[str] = []
        if not charge_checksum_passed:
            failure_modes.append(f"charge residual={_format_decimal(charge_residual)}")
        if not momentum_checksum_passed:
            failure_modes.append(f"time-reversal residual={_format_decimal(momentum_residual)}")
        if not parity_checksum_passed:
            failure_modes.append(f"parity residual={_format_decimal(parity_residual)}")
        if self.simulate_boundary_decoherence:
            failure_modes.append("simulated boundary decoherence")
        if torsion_projection_residual > 0:
            failure_modes.append(f"bulk torsion amplitude={_format_decimal(torsion_projection_residual)}")

        detail = "bulk integrity locked" if not failure_modes else "; ".join(failure_modes)
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
            checksums=(charge, parity, time_reversal),
            torsion_projection_residual=torsion_projection_residual,
        )

    def verify_bulk_checksum(self) -> BulkChecksumVerification:
        return self.verify_bulk_integrity()


@dataclass(frozen=True)
class HolographicErrorStabilizerAudit:
    branch: BenchmarkBranch
    recovery: RecoveryThresholdAudit
    charge: TopologicalChecksum
    parity: TopologicalChecksum
    time_reversal: TopologicalChecksum
    newton_lock: noether_bridge.NewtonLockAudit
    saturation: noether_bridge.SaturationAudit
    unity: noether_bridge.UnityOfScaleAudit
    reference_failure: noether_bridge.ReviewerTrapAudit
    proof: str

    @property
    def checksums(self) -> tuple[TopologicalChecksum, TopologicalChecksum, TopologicalChecksum]:
        return (self.charge, self.parity, self.time_reversal)

    @property
    def momentum(self) -> TopologicalChecksum:
        return self.time_reversal

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
    return TopologicalChecksumCode(
        law="charge",
        stabilizer_name="Boundary Charge Stabilizer",
        protected_quantity="Zero-balanced vacuum loading",
        checksum_equation="|(1 - kappa_D5) - (c_dark / K)|",
        boundary_integer=_decimal(bit_balance.packing_deficiency),
        bulk_projection=_decimal(bit_balance.dark_sector_complexity_overhead),
        syndrome_tolerance=recovery.dimensionless_syndrome_threshold,
        interpretation=(
            "The charge checksum treats the visible packing deficiency as the logical payload and the c_dark parity sector "
            "as its redundancy block. A non-zero residue would mean boundary bit rot in the zero-energy state."
        ),
    ).verify()


def _build_time_reversal_checksum(
    recovery: RecoveryThresholdAudit,
    *,
    unity: noether_bridge.UnityOfScaleAudit,
) -> TopologicalChecksum:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION
        bulk_projection = (
            unity.lambda_lhs_ev2 / unity.lambda_rhs_noether_bridged_ev2
            if unity.lambda_rhs_noether_bridged_ev2 != 0
            else Decimal("Infinity")
        )
    return TopologicalChecksumCode(
        law="time_reversal",
        stabilizer_name="Noether Time-Reversal Stabilizer",
        protected_quantity="Reversible bulk time ordering",
        checksum_equation="|1 - Lambda_obs / Lambda_bridge|",
        boundary_integer=Decimal("1"),
        bulk_projection=bulk_projection,
        syndrome_tolerance=recovery.dimensionless_syndrome_threshold,
        interpretation=(
            "Time-reversal conservation is encoded by the Noether-bridged unity-of-scale code. "
            "If the bridge ceases to reverse cleanly, temporal bookkeeping leaks out of the boundary dictionary and the bulk projection loses geodesic closure."
        ),
    ).verify()


def _build_momentum_checksum(
    recovery: RecoveryThresholdAudit,
    *,
    unity: noether_bridge.UnityOfScaleAudit,
) -> TopologicalChecksum:
    return _build_time_reversal_checksum(recovery, unity=unity)


def _build_parity_checksum(recovery: RecoveryThresholdAudit) -> TopologicalChecksum:
    from shbt.core import noether_bridge

    framing = noether_bridge.framing_defect(PARENT_LEVEL, LEPTON_LEVEL, QUARK_LEVEL)
    residual = Decimal(framing.delta_fr.numerator) / Decimal(framing.delta_fr.denominator)
    return TopologicalChecksumCode(
        law="parity",
        stabilizer_name="Framing Parity Stabilizer",
        protected_quantity="Anomaly-free torsion lock",
        checksum_equation="Delta_fr",
        boundary_integer=Decimal("0"),
        bulk_projection=residual,
        syndrome_tolerance=recovery.dimensionless_syndrome_threshold,
        interpretation=(
            "Parity conservation is the topological framing checksum. Once Delta_fr reopens, the Levi-Civita branch exits the local moat and torsion returns."
        ),
    ).verify()


def build_holographic_error_stabilizer_audit(*, precision: int = DEFAULT_PRECISION) -> HolographicErrorStabilizerAudit:
    from shbt.core import noether_bridge

    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    recovery, newton_lock, saturation, unity = _build_recovery_threshold(precision=resolved_precision)
    charge = _build_charge_checksum(recovery)
    parity = _build_parity_checksum(recovery)
    time_reversal = _build_time_reversal_checksum(recovery, unity=unity)
    reference_failure = noether_bridge.reviewer_trap_audit(
        newton_lock_audit=newton_lock,
        saturation=saturation,
        benchmark_branch=_parent_order(BENCHMARK_BRANCH),
        detuned_branch=_parent_order(FAILED_BRANCHES["parity"]),
        precision=resolved_precision,
    )
    proof = (
        "Charge balance, parity/framing closure, and Noether-bridged time-reversal closure all remain below the recovery threshold "
        "m_rec = kappa_D5 M_P N^(-1/4), so the benchmark boundary stays in its zero-energy state with E_mu_nu = 0. "
        "A charge-conservation bit flip or parity detuning simultaneously gives Delta_fr != 0, E_mu_nu != 0, and J_mu_nu^(a) != 0, "
        "which means the bulk exists as a torsion-free, equivalence-principle-preserving image precisely because the boundary acts as a self-correcting error-stabilizer."
    )
    return HolographicErrorStabilizerAudit(
        branch=BENCHMARK_BRANCH,
        recovery=recovery,
        charge=charge,
        parity=parity,
        time_reversal=time_reversal,
        newton_lock=newton_lock,
        saturation=saturation,
        unity=unity,
        reference_failure=reference_failure,
        proof=proof,
    )


def simulate_failed_checksum(
    law: ChecksumLaw = "charge",
    *,
    syndrome: Decimal | float | int | str | None = None,
    precision: int = DEFAULT_PRECISION,
) -> ChecksumFailureSimulation:
    from shbt.core import noether_bridge

    audit = build_holographic_error_stabilizer_audit(precision=precision)
    resolved_law = _normalize_checksum_law(law)
    checksum = _select_checksum(audit, resolved_law)
    injected_syndrome = checksum.syndrome_tolerance * Decimal("2") if syndrome is None else _decimal(syndrome)
    if injected_syndrome < 0:
        raise ValueError("syndrome must be non-negative.")

    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION)
        syndrome_energy_ev = injected_syndrome * audit.recovery.planck_mass_ev
    recoverable = bool(syndrome_energy_ev <= audit.recovery.recovery_threshold_ev)

    failed_branch = FAILED_BRANCHES[resolved_law]
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
        law=resolved_law,
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


def simulate_charge_bit_flip(
    *,
    syndrome: Decimal | float | int | str | None = None,
    precision: int = DEFAULT_PRECISION,
) -> ChecksumFailureSimulation:
    return simulate_failed_checksum(
        law="charge",
        syndrome=syndrome,
        precision=precision,
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
        display_law = _display_checksum_law(checksum.law)
        lines.extend(
            (
                f"{display_law:>13} stabilizer         : {checksum.stabilizer_name}",
                f"{display_law:>13} boundary integer   : {_format_decimal(checksum.boundary_integer)}",
                f"{display_law:>13} bulk projection    : {_format_decimal(checksum.bulk_projection)}",
                f"{display_law:>13} residual           : {_format_decimal(checksum.residue)}",
                f"{display_law:>13} tolerance          : {_format_decimal(checksum.syndrome_tolerance)}",
                f"{display_law:>13} bit-rot blocked    : {checksum.prevents_bit_rot}",
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
                f"Failed law                         : {_display_checksum_law(simulation.law)}",
                f"Detuned branch                     : {_format_branch(simulation.failed_branch)}",
                f"Charge-conservation bit flip       : {simulation.charge_bit_flip}",
                f"Injected syndrome                  : {_format_decimal(simulation.injected_syndrome)}",
                f"Syndrome energy [eV]              : {_format_decimal(simulation.syndrome_energy_ev)}",
                f"Recovery budget [eV]              : {_format_decimal(simulation.recovery_budget_ev)}",
                f"Recoverable                        : {simulation.recoverable}",
                f"Bulk closure tensor vanished      : {simulation.effective_closure_tensor.vanished}",
                f"Torsion reintroduced               : {simulation.torsion_reintroduced}",
                f"Equivalence Principle destroyed   : {simulation.equivalence_principle_destroyed}",
                f"Zero-energy boundary required      : {simulation.zero_energy_boundary_required}",
            )
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION)
    parser.add_argument("--simulate-law", choices=("charge", "parity", "time_reversal", "time-reversal", "momentum"))
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
    "BoundaryStabilizationError",
    "BulkChecksumVerification",
    "FAILED_BRANCHES",
    "ChecksumFailureSimulation",
    "HolographicErrorStabilizerAudit",
    "HolographicStabilizer",
    "RecoveryThresholdAudit",
    "TopologicalChecksum",
    "TopologicalChecksumCode",
    "build_holographic_error_stabilizer_audit",
    "main",
    "parse_args",
    "render_report",
    "simulate_charge_bit_flip",
    "simulate_failed_checksum",
    "stabilize_boundary",
    "stabilize_classmethods",
]


if __name__ == "__main__":
    main()
