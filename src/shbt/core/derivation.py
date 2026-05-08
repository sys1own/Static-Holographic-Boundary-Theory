from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import sqrt
from typing import Sequence

from shbt.constants import (
    LEPTON_LEVEL,
    PARENT_LEVEL,
    PLANCK2018_H0_KM_S_MPC as DERIVED_H0_KM_S_MPC,
    QUARK_LEVEL,
)

DEFAULT_PRECISION = 50
from shbt.verification.comparators import (
    CODATA_ALPHA_INVERSE_COMPARATOR,
    CODATA_PROTON_ELECTRON_MASS_RATIO_COMPARATOR,
    EmpiricalComparator,
    NUFIT_ATMOSPHERIC_MASS_SPLITTING_COMPARATOR,
    NUFIT_SOLAR_MASS_SPLITTING_COMPARATOR,
    PLANCK2018_H0_COMPARATOR,
    PLANCK2018_LAMBDA_COMPARATOR,
)


@dataclass(frozen=True)
class TensionAuditComponent:
    label: str
    predicted_value: Decimal
    observed_value: Decimal
    sigma: Decimal
    residual: Decimal
    normalized_residual: Decimal
    chi_squared: Decimal
    release: str


@dataclass(frozen=True)
class TensionAudit:
    label: str
    benchmark_branch: tuple[int, int, int]
    components: tuple[TensionAuditComponent, ...]
    chi_squared: Decimal
    degrees_of_freedom: int
    reduced_chi_squared: Decimal
    rms_pull: Decimal

    @property
    def comparison_count(self) -> int:
        return len(self.components)


def _decimal(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _sigma(value: Decimal | float | int | str) -> Decimal:
    resolved = abs(_decimal(value))
    floor = Decimal("1e-30")
    return resolved if resolved > floor else floor


def build_tension_component(
    *,
    label: str,
    predicted_value: Decimal | float | int | str,
    comparator: EmpiricalComparator,
) -> TensionAuditComponent:
    predicted = _decimal(predicted_value)
    observed = _decimal(comparator.value)
    sigma = _sigma(comparator.sigma)
    residual = predicted - observed
    normalized_residual = residual / sigma
    chi_squared = normalized_residual * normalized_residual
    return TensionAuditComponent(
        label=label,
        predicted_value=predicted,
        observed_value=observed,
        sigma=sigma,
        residual=residual,
        normalized_residual=normalized_residual,
        chi_squared=chi_squared,
        release=comparator.release,
    )


def build_tension_audit(
    *,
    label: str,
    benchmark_branch: tuple[int, int, int],
    components: Sequence[TensionAuditComponent],
) -> TensionAudit:
    resolved_components = tuple(components)
    chi_squared = sum((component.chi_squared for component in resolved_components), start=Decimal("0"))
    degrees_of_freedom = len(resolved_components)
    reduced_chi_squared = chi_squared / Decimal(degrees_of_freedom) if degrees_of_freedom else Decimal("0")
    rms_pull = _decimal(sqrt(float(reduced_chi_squared))) if degrees_of_freedom else Decimal("0")
    return TensionAudit(
        label=label,
        benchmark_branch=tuple(int(value) for value in benchmark_branch),
        components=resolved_components,
        chi_squared=chi_squared,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_squared=reduced_chi_squared,
        rms_pull=rms_pull,
    )


def build_flavor_tension_audit(
    *,
    kernel_splitting_ratio: Decimal | float | int | str,
    solar_delta_m21_ev2: Decimal | float | int | str | None = None,
    atmospheric_delta_m31_ev2: Decimal | float | int | str | None = None,
) -> TensionAudit:
    observed_ratio = _decimal(NUFIT_ATMOSPHERIC_MASS_SPLITTING_COMPARATOR.value) / _decimal(NUFIT_SOLAR_MASS_SPLITTING_COMPARATOR.value)
    relative_sigma = (
        (_decimal(NUFIT_ATMOSPHERIC_MASS_SPLITTING_COMPARATOR.sigma) / _decimal(NUFIT_ATMOSPHERIC_MASS_SPLITTING_COMPARATOR.value)) ** 2
        + (_decimal(NUFIT_SOLAR_MASS_SPLITTING_COMPARATOR.sigma) / _decimal(NUFIT_SOLAR_MASS_SPLITTING_COMPARATOR.value)) ** 2
    )
    ratio_sigma = abs(observed_ratio) * _decimal(sqrt(float(relative_sigma)))
    ratio_comparator = EmpiricalComparator(
        label="NuFIT Delta m31^2 / Delta m21^2",
        value=float(observed_ratio),
        sigma=float(ratio_sigma),
        release=NUFIT_ATMOSPHERIC_MASS_SPLITTING_COMPARATOR.release,
        reference=NUFIT_ATMOSPHERIC_MASS_SPLITTING_COMPARATOR.reference,
    )
    components = [
        build_tension_component(
            label="NuFIT Delta m31^2 / Delta m21^2",
            predicted_value=kernel_splitting_ratio,
            comparator=ratio_comparator,
        )
    ]
    if solar_delta_m21_ev2 is not None:
        components.append(
            build_tension_component(
                label=NUFIT_SOLAR_MASS_SPLITTING_COMPARATOR.label,
                predicted_value=solar_delta_m21_ev2,
                comparator=NUFIT_SOLAR_MASS_SPLITTING_COMPARATOR,
            )
        )
    if atmospheric_delta_m31_ev2 is not None:
        components.append(
            build_tension_component(
                label=NUFIT_ATMOSPHERIC_MASS_SPLITTING_COMPARATOR.label,
                predicted_value=atmospheric_delta_m31_ev2,
                comparator=NUFIT_ATMOSPHERIC_MASS_SPLITTING_COMPARATOR,
            )
        )
    return build_tension_audit(
        label="Flavor Tier-2 conformance audit",
        benchmark_branch=(LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL),
        components=components,
    )


def build_gravity_tension_audit(
    *,
    lambda_holo_si_m2: Decimal | float | int | str,
    h0_km_s_mpc: Decimal | float | int | str,
) -> TensionAudit:
    return build_tension_audit(
        label="Gravity Tier-2 conformance audit",
        benchmark_branch=(LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL),
        components=(
            build_tension_component(
                label=PLANCK2018_LAMBDA_COMPARATOR.label,
                predicted_value=lambda_holo_si_m2,
                comparator=PLANCK2018_LAMBDA_COMPARATOR,
            ),
            build_tension_component(
                label=PLANCK2018_H0_COMPARATOR.label,
                predicted_value=h0_km_s_mpc,
                comparator=PLANCK2018_H0_COMPARATOR,
            ),
        ),
    )


def build_zero_parameter_tension_audit(*, precision: int = DEFAULT_PRECISION) -> TensionAudit:
    from shbt.core.derivation_api import UniverseFactory

    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    physical_ledger = UniverseFactory.calculate_physical_ledger(precision=resolved_precision)
    lambda_surface = UniverseFactory.derive_lambda_surface(precision=resolved_precision)
    components = (
        build_tension_component(
            label=CODATA_ALPHA_INVERSE_COMPARATOR.label,
            predicted_value=physical_ledger.alpha_surface.alpha_inverse_decimal,
            comparator=CODATA_ALPHA_INVERSE_COMPARATOR,
        ),
        build_tension_component(
            label=CODATA_PROTON_ELECTRON_MASS_RATIO_COMPARATOR.label,
            predicted_value=physical_ledger.proton_ratio.mu_audit,
            comparator=CODATA_PROTON_ELECTRON_MASS_RATIO_COMPARATOR,
        ),
        build_tension_component(
            label=PLANCK2018_H0_COMPARATOR.label,
            predicted_value=DERIVED_H0_KM_S_MPC,
            comparator=PLANCK2018_H0_COMPARATOR,
        ),
        build_tension_component(
            label=PLANCK2018_LAMBDA_COMPARATOR.label,
            predicted_value=lambda_surface.lambda_holo_si_m2,
            comparator=PLANCK2018_LAMBDA_COMPARATOR,
        ),
    )
    return build_tension_audit(
        label="Zero-Parameter Tier-2 conformance audit",
        benchmark_branch=physical_ledger.vacuum.branch,
        components=components,
    )


__all__ = [
    "DEFAULT_PRECISION",
    "TensionAudit",
    "TensionAuditComponent",
    "build_flavor_tension_audit",
    "build_gravity_tension_audit",
    "build_tension_audit",
    "build_tension_component",
    "build_zero_parameter_tension_audit",
]
