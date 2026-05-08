from __future__ import annotations

from importlib import import_module
from typing import Any

_COMPARATORS_EXPORTS = {
    "ATMOSPHERIC_MASS_SPLITTING_NO_EV2",
    "ATMOSPHERIC_MASS_SPLITTING_NO_SIGMA_EV2",
    "CODATA_ALPHA_INVERSE_COMPARATOR",
    "CODATA_FINE_STRUCTURE_ALPHA_INVERSE",
    "CODATA_PROTON_ELECTRON_MASS_RATIO_COMPARATOR",
    "CODATA_PROTON_TO_ELECTRON_MASS_RATIO",
    "EmpiricalComparator",
    "NUFIT_ATMOSPHERIC_MASS_SPLITTING_COMPARATOR",
    "NUFIT_SOLAR_MASS_SPLITTING_COMPARATOR",
    "PLANCK2018_H0_COMPARATOR",
    "PLANCK2018_H0_KM_S_MPC",
    "PLANCK2018_H0_SIGMA_KM_S_MPC",
    "PLANCK2018_LAMBDA_COMPARATOR",
    "PLANCK2018_LAMBDA_SI_M2",
    "SOLAR_MASS_SPLITTING_EV2",
    "SOLAR_MASS_SPLITTING_SIGMA_EV2",
    "TIER2_COMPARATORS",
    "Tier2ComparatorBundle",
}

_LIVE_BRIDGE_EXPORTS = {
    "ADSLiteratureRecord",
    "DEFAULT_ADS_ENDPOINT",
    "DEFAULT_GWOSC_API_ROOT",
    "DEFAULT_GWOSC_CATALOG",
    "DEFAULT_SIMBAD_ENDPOINT",
    "DEFAULT_VIZIER_ENDPOINT",
    "JWSTFormationObservation",
    "LIGOStrainObservation",
    "LiveVerificationBridge",
    "ObservationTensionComponent",
    "SimbadObjectRecord",
    "TheoryKernelProfile",
    "TheoryTensionScore",
    "build_ads_search_url",
    "build_gwosc_catalog_events_url",
    "build_simbad_identifier_url",
    "build_vizier_catalog_url",
    "calculate_live_tension_score",
}

__all__ = sorted(_COMPARATORS_EXPORTS | _LIVE_BRIDGE_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _COMPARATORS_EXPORTS:
        module = import_module("shbt.verification.comparators")
        return getattr(module, name)
    if name in _LIVE_BRIDGE_EXPORTS:
        module = import_module("shbt.verification.live_bridge")
        return getattr(module, name)
    raise AttributeError(f"module 'shbt.verification' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
