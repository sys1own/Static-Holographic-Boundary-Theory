from __future__ import annotations

import math

from shbt.config_loader import ConfigLoader, GEOMETRIC_EMERGENCE_CLASSIFICATION
from shbt.core.master_transport import (
    build_master_transport_audit,
    derive_emergent_constants,
    render_master_transport_report,
)
from shbt.main import SECTOR_AUDIT_MODULES


def test_geometry_origin_synthesizes_zero_anchor_profile_defaults() -> None:
    loader = ConfigLoader()

    profile = loader.load_physics_profile()
    physics_constants = loader.load_physics_constants()
    classifications = loader.load_parameter_classifications()

    assert "tier_2" in profile
    assert profile["tier_2"]["planck_mass_ev"] == physics_constants["physical_constants"]["planck_mass_ev"]
    assert math.isclose(physics_constants["model"]["geometric_kappa"], 0.9887710512663789, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(physics_constants["physical_constants"]["planck_mass_ev"], 1.22089e28, rel_tol=5.0e-5)
    assert math.isclose(physics_constants["physical_constants"]["planck2018_lambda_si_m2"], 1.0891388337006832e-52, rel_tol=5.0e-4)
    assert classifications["model.geometric_kappa"] == GEOMETRIC_EMERGENCE_CLASSIFICATION
    assert classifications["physical_constants.planck_mass_ev"] == GEOMETRIC_EMERGENCE_CLASSIFICATION


def test_master_transport_equation_locks_gravity_flavor_and_cosmology() -> None:
    audit = build_master_transport_audit(flavor_shift_fraction=0.01)

    assert audit.kernel.dimension == 26
    assert audit.kernel.branch == (26, 8, 312)
    assert audit.zero_anchor_boot is True
    assert math.isclose(audit.emergent_constants.geometric_kappa, 0.9887710512663789, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(audit.gravity_view.planck_mass_ev, 1.22089e28, rel_tol=5.0e-5)
    assert math.isclose(audit.cosmology_view.lambda_si_m2, 1.0891388337006832e-52, rel_tol=5.0e-4)
    assert math.isclose(audit.cosmology_view.hubble_km_s_mpc, 67.36, rel_tol=5.0e-5)
    assert math.isclose(audit.cosmology_view.omega_lambda, 0.6847, rel_tol=5.0e-4)
    assert math.isclose(audit.rigidity_test.gravity_shift_fraction, 0.01, rel_tol=0.0, abs_tol=1.0e-12)
    assert audit.rigidity_test.gravity_lock_failed is True


def test_master_transport_report_and_sector_wiring_expose_zero_anchor_language() -> None:
    report = render_master_transport_report(build_master_transport_audit())

    assert "Master Transport Equation Audit" in report
    assert "Gravity sector failure        : True" in report
    assert "26-dimensional transport kernel" in report
    assert "A forced flavor-residue detuning" in report
    assert "shbt.core.master_transport" in SECTOR_AUDIT_MODULES["rigidity"]


def test_derived_emergent_constants_preserve_loader_benchmark_outputs() -> None:
    constants = derive_emergent_constants()

    assert math.isclose(constants.planck2018_alpha_s_mz, 0.1179, rel_tol=5.0e-5)
    assert math.isclose(constants.planck2018_alpha_em_inv_mz, 127.955, rel_tol=5.0e-5)
    assert math.isclose(constants.planck2018_sin2_theta_w_mz, 0.23122, rel_tol=5.0e-5)
    assert math.isclose(constants.codata_fine_structure_alpha_inverse, 137.035999084, rel_tol=5.0e-5)
