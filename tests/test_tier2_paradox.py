from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

import shbt.constants as constants_module
import shbt.core.derivation as derivation_module
from shbt.core.derivation_api import UniverseFactory
import shbt.core.flavor_identity_resolver as flavor_module
import shbt.core.noether_bridge as noether_bridge
from shbt.verification.comparators import EmpiricalComparator


def test_constants_tier2_table_uses_comparator_namespace() -> None:
    tier2 = constants_module.TIER_2_OBSERVATIONAL_BOUNDARY_CONDITIONS

    assert "COMPARATOR_PLANCK2018_H0_KM_S_MPC" in tier2
    assert "COMPARATOR_PLANCK2018_LAMBDA_SI_M2" in tier2
    assert "COMPARATOR_CODATA_FINE_STRUCTURE_ALPHA_INVERSE" in tier2
    assert "PLANCK2018_H0_KM_S_MPC" not in tier2
    assert "PLANCK2018_LAMBDA_SI_M2" not in tier2


def test_universe_factory_tension_audit_accepts_compatibility_kwargs() -> None:
    audit = UniverseFactory.derive_tension_audit(precision=64, legacy_mode=True)

    assert audit.benchmark_branch == (26, 8, 312)
    assert audit.degrees_of_freedom == 4


def test_universe_factory_exposes_zero_parameter_tension_audit() -> None:
    audit = UniverseFactory.derive_tension_audit()
    labels = {component.label for component in audit.components}

    assert audit.benchmark_branch == (26, 8, 312)
    assert audit.degrees_of_freedom == 4
    assert labels == {
        "CODATA alpha^-1",
        "CODATA m_p/m_e",
        "Planck 2018 H0",
        "Planck 2018 Lambda",
    }
    assert audit.chi_squared >= Decimal("0")


def test_flavor_mandatory_residue_remains_tier1_locked_under_bad_conformance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_audit = derivation_module.build_tension_audit(
        label="synthetic bad fit",
        benchmark_branch=(26, 8, 312),
        components=(
            derivation_module.build_tension_component(
                label="synthetic",
                predicted_value=Decimal("10"),
                comparator=EmpiricalComparator(
                    label="synthetic",
                    value=1.0,
                    sigma=1.0,
                    release="test",
                ),
            ),
        ),
    )
    monkeypatch.setattr(flavor_module, "build_flavor_tension_audit", lambda **kwargs: fake_audit)

    audit = flavor_module.build_flavor_identity_audit()

    assert audit.tension_audit is fake_audit
    assert audit.mandatory_residue_verified is True
    assert audit.tension_audit.reduced_chi_squared > Decimal("1")


def test_gravity_report_exposes_tier2_conformance_audit() -> None:
    report = noether_bridge.build_gravity_side_rigidity_report()
    labels = {component.label for component in report.tension_audit.components}

    assert report.tension_audit.degrees_of_freedom == 2
    assert labels == {"Planck 2018 H0", "Planck 2018 Lambda"}
    assert report.tension_audit.chi_squared >= Decimal("0")


def test_universe_factory_tension_audit_uses_calling_class_residues() -> None:
    class SyntheticUniverseFactory(UniverseFactory):
        @classmethod
        def calculate_physical_ledger(cls, *, precision: int = derivation_module.DEFAULT_PRECISION):
            del cls, precision
            return SimpleNamespace(
                vacuum=SimpleNamespace(branch=(1, 2, 3)),
                alpha_surface=SimpleNamespace(alpha_inverse_decimal=Decimal("111")),
                proton_ratio=SimpleNamespace(mu_audit=Decimal("222")),
            )

        @classmethod
        def derive_lambda_surface(cls, *, precision: int = derivation_module.DEFAULT_PRECISION):
            del cls, precision
            return SimpleNamespace(lambda_holo_si_m2=Decimal("333"))

    audit = SyntheticUniverseFactory.derive_tension_audit()
    predicted = {component.label: component.predicted_value for component in audit.components}

    assert audit.degrees_of_freedom == 4
    assert predicted["CODATA alpha^-1"] == Decimal("111")
    assert predicted["CODATA m_p/m_e"] == Decimal("222")
    assert predicted["Planck 2018 Lambda"] == Decimal("333")
