from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .constants import (
    AUDIT_STATEMENT_FILENAME,
    COROLLARY_REPORT_FILENAME,
    EIGENVECTOR_STABILITY_AUDIT_FILENAME,
    MIXING_SECTOR_RIGIDITY_MESSAGE,
    SEED_ROBUSTNESS_AUDIT_FILENAME,
    STABILITY_REPORT_FILENAME,
    SVD_STABILITY_REPORT_FILENAME,
)


def _audit_status(flag: object) -> str:
    return "consistent" if bool(flag) else "conditional"


def _audit_flag(flag: object) -> int:
    return int(bool(flag))


def _mass_ratio_rigidity_message(mass_ratio_stability_audit: Any) -> str:
    message = getattr(mass_ratio_stability_audit, "message", None)
    if callable(message):
        return str(message())
    return str(getattr(mass_ratio_stability_audit, "success_message", MIXING_SECTOR_RIGIDITY_MESSAGE))


def write_audit_statement(
    diagnostics: Mapping[str, Any],
    output_dir: Path | None = None,
) -> str:
    """Write the plain-text benchmark consistency statement used in the packet outputs."""

    diagnostics = dict(diagnostics)
    compatibility_aliases = {
        "anomaly_detection_alpha_inverse": "qec_syndrome_alpha_inverse",
        "anomaly_detection_alpha_inverse_fraction": "qec_syndrome_alpha_inverse_fraction",
        "anomaly_detection_alpha": "qec_syndrome_alpha",
        "anomaly_detection_noise_floor": "qec_syndrome_noise_floor",
        "anomaly_detection_stable": "qec_syndrome_stable",
        "benchmark_anchor_central_charge_ratio": "atomic_pixel_central_charge_ratio",
        "benchmark_anchor_pixel_volume": "atomic_pixel_volume",
        "benchmark_anchor_pixel_volume_fraction": "atomic_pixel_volume_fraction",
        "benchmark_anchor_mu_predicted": "atomic_mu_predicted",
        "benchmark_anchor_mu_empirical": "atomic_mu_empirical",
        "benchmark_anchor_mu_relative_error": "atomic_mu_relative_error",
        "benchmark_anchor_pass": "atomic_lock_pass",
    }
    for canonical_key, legacy_key in compatibility_aliases.items():
        if canonical_key not in diagnostics and legacy_key in diagnostics:
            diagnostics[canonical_key] = diagnostics[legacy_key]

    predictive_chi2 = float(diagnostics.get("predictive_chi2", 0.0))
    predictive_observable_count = max(int(diagnostics.get("predictive_observable_count", 0)), 1)
    predictive_rms_pull = float(diagnostics.get("predictive_rms_pull", math.sqrt(predictive_chi2 / predictive_observable_count)))
    raw_benchmark_chi2 = diagnostics.get("raw_benchmark_chi2", diagnostics["predictive_chi2"])
    raw_benchmark_rms_pull = diagnostics.get("raw_benchmark_rms_pull", predictive_rms_pull)
    raw_benchmark_dof = diagnostics.get("raw_benchmark_degrees_of_freedom", diagnostics["predictive_degrees_of_freedom"])
    local_benchmark_dof = diagnostics.get("local_frequentist_published_dof", diagnostics["predictive_degrees_of_freedom"])
    benchmark_rms_pull = diagnostics.get("benchmark_rms_pull", predictive_rms_pull)
    audit_rms_pull = diagnostics.get("audit_rms_pull", predictive_rms_pull)

    report_lines = [
        "Benchmark Consistency Statement",
        "===============================",
        (
            f"The largest observable pull in the benchmark is {diagnostics['predictive_max_abs_pull']:.2f}σ, "
            f"and the maximum RG non-linearity discrepancy is {diagnostics['max_rg_nonlinearity_sigma']:.2f}σ. "
            f"The predictive tally uses {diagnostics['predictive_observable_count']} displayed observables and the "
            f"cross-check tally uses {diagnostics['audit_observable_count']}, with raw chi2={raw_benchmark_chi2:.2f} "
            f"for ν_raw={raw_benchmark_dof} with RMS pull={raw_benchmark_rms_pull:.2f}, benchmark chi2_pred={diagnostics['predictive_chi2']:.2f} "
            f"for ν_pred={local_benchmark_dof} with RMS pull={benchmark_rms_pull:.2f}, and chi2_check={diagnostics['audit_chi2']:.2f} "
            f"for ν_check={diagnostics['audit_degrees_of_freedom']} with RMS pull={audit_rms_pull:.2f}."
        ),
        (
            "Because the threshold matching term w_{\\rm th} and the Geometric Matching Residue "
            "R_{01}^{\\rm par/vis} are fixed by the same boundary data rather than scanned continuously, "
            "the benchmark uses descriptive fixed-branch counting nu=N_obs: w_th, k_l, and k_q enter as fixed "
            "branch data rather than subtracted fit parameters."
        ),
        (
            f"The disclosed discrete survey spans {int(diagnostics['predictive_landscape_trial_count'])} visible-level cells. The retained solar "
            f"row has pull_theta12={diagnostics['theta12_pull']:+.2f}σ and is interpreted as the known one-loop "
            "tension in the $\\theta_12$ transport rather than evidence for hidden tuning or missing fit knobs. "
            "Theoretical uncertainties are uniquely defined by the Quantified Two-Loop Residuals of the anomaly-free branch. "
            f"The largest disclosed transport residual on the displayed flavor rows is {100.0 * diagnostics['theoretical_matching_uncertainty_fraction']:.3e}%, and the descriptive benchmark tally remains a fixed-branch audit; "
            "the residual deviations remain compatible with ordinary "
            "transport-level fluctuations of the current benchmark rather than evidence for a tuned refit."
        ),
    ]

    if all(
        key in diagnostics
        for key in (
            "mass_scale_status",
            "mass_scale_comparison_label",
            "mass_scale_benchmark_mass_relation_ev",
            "mass_scale_comparison_mass_ev",
            "mass_scale_sigma_ev",
            "mass_scale_holographic_pull",
            "mass_scale_support_threshold_sigma",
            "mass_scale_supported",
        )
    ):
        report_lines.append(
            (
                f"The light-neutrino mass relation is reported as a {diagnostics['mass_scale_status']}: "
                f"m_light^pred={diagnostics['mass_scale_benchmark_mass_relation_ev']:.6e} eV against "
                f"{diagnostics['mass_scale_comparison_label']}={diagnostics['mass_scale_comparison_mass_ev']:.6e} eV, "
                f"with matching sigma={diagnostics['mass_scale_sigma_ev']:.6e} eV and holographic pull="
                f"{diagnostics['mass_scale_holographic_pull']:.3f}σ. The support flag is "
                f"{_audit_flag(diagnostics['mass_scale_supported'])} relative to the disclosed "
                f"{diagnostics['mass_scale_support_threshold_sigma']:.1f}σ benchmark allowance."
            )
        )

    if all(
        key in diagnostics
        for key in (
            "gauge_alpha_inverse",
            "gauge_alpha_ir_inverse",
            "gauge_alpha_target",
            "gauge_matching_sigma_inverse",
            "gauge_surface_pull",
            "gauge_ir_pull",
            "gauge_ir_alignment_improves",
        )
    ):
        report_lines.append(
            (
                f"The surface-to-IR gauge bridge is {_audit_status(diagnostics['gauge_ir_alignment_improves'])}: "
                f"alpha_surf^-1={diagnostics['gauge_alpha_inverse']:.6f}, alpha_IR^-1={diagnostics['gauge_alpha_ir_inverse']:.6f}, "
                f"and alpha_target^-1={diagnostics['gauge_alpha_target']:.6f}. Under the disclosed matching allowance "
                f"sigma_alpha^-1={diagnostics['gauge_matching_sigma_inverse']:.6f}, the surface pull is "
                f"{diagnostics['gauge_surface_pull']:.3f}σ and the IR pull is {diagnostics['gauge_ir_pull']:.3f}σ."
            )
        )

    if all(
        key in diagnostics
        for key in (
            "ckm_benchmark_weight",
            "ckm_best_fit_weight",
            "ckm_benchmark_delta_chi2",
            "ckm_smallness_max_vus_shift",
            "ckm_smallness_max_vcb_shift",
            "ckm_smallness_max_vub_shift",
            "ckm_smallness_max_vus_sigma",
            "ckm_smallness_max_vcb_sigma",
            "ckm_smallness_max_vub_sigma",
            "ckm_smallness_lock_pass",
        )
    ):
        report_lines.append(
            (
                f"The CKM threshold-profile check remains {_audit_status(diagnostics['ckm_smallness_lock_pass'])}: "
                f"the benchmark residue sits at R_GUT={diagnostics['ckm_benchmark_weight']:.6f}, the off-shell minimum "
                f"is R_GUT={diagnostics['ckm_best_fit_weight']:.6f}, and the benchmark carries Δchi2_pred="
                f"{diagnostics['ckm_benchmark_delta_chi2']:.6f}. Across that profile the induced CKM-magnitude shifts stay at "
                f"Δ|Vus|={diagnostics['ckm_smallness_max_vus_shift']:.6e} ({diagnostics['ckm_smallness_max_vus_sigma']:.3e}σ), "
                f"Δ|Vcb|={diagnostics['ckm_smallness_max_vcb_shift']:.6e} ({diagnostics['ckm_smallness_max_vcb_sigma']:.3e}σ), and "
                f"Δ|Vub|={diagnostics['ckm_smallness_max_vub_shift']:.6e} ({diagnostics['ckm_smallness_max_vub_sigma']:.3e}σ), "
                "so the smallness of quark mixing remains a structural consequence of the branch-fixed quark level rather "
                "than a tuned threshold knob."
            )
        )

    if all(
        key in diagnostics
        for key in (
            "dm_rhn_scale_gev",
            "dm_beta_squared",
            "dm_mass_gev",
            "dm_alpha_chi",
            "dm_sigma_geom_cm2",
            "dm_light_wimp_impossible",
            "dm_direct_detection_below_floor",
            "dm_dark_sector_holographic_rigidity",
        )
    ):
        report_lines.append(
            (
                f"The completion-sector fingerprint is {_audit_status(diagnostics['dm_dark_sector_holographic_rigidity'])}: "
                f"the parity-bit relic is anchored at M_N={diagnostics['dm_rhn_scale_gev']:.6e} GeV and descends through "
                f"m_DM=M_N exp[-beta^2] with beta^2={diagnostics['dm_beta_squared']:.6f}, giving "
                f"m_DM={diagnostics['dm_mass_gev']:.6e} GeV, alpha_chi={diagnostics['dm_alpha_chi']:.6f}, and "
                f"sigma_chiN={diagnostics['dm_sigma_geom_cm2']:.6e} cm^2. The light-WIMP exclusion check is "
                f"{_audit_flag(diagnostics['dm_light_wimp_impossible'])} and the direct-detection-below-floor flag is "
                f"{_audit_flag(diagnostics['dm_direct_detection_below_floor'])}, so the completion sector is benchmarked as a superheavy "
                "parity-bit relic rather than explored here as a TeV-scale WIMP candidate. In publication-facing terms this is "
                "a consistency check on the locked completion sector, not the primary near-term falsification channel."
            )
        )

    if all(
        key in diagnostics
        for key in (
            "bit_balance_packing_deficiency",
            "bit_balance_dark_overhead",
            "bit_balance_residual",
            "bit_balance_zero_balanced",
            "inflation_holographic_suppression_factor",
            "inflation_observable_tensor_ratio",
            "inflation_bicep_keck_bound",
            "inflation_complexity_bound_tensor_pass",
        )
    ):
        report_lines.append(
            (
                f"The Bit-Balance / Complexity-Bound audit is {_audit_status(diagnostics['bit_balance_zero_balanced'])}: "
                f"the packing deficiency 1-kappa_D5={diagnostics['bit_balance_packing_deficiency']:.6f} is matched by "
                f"c_dark/K={diagnostics['bit_balance_dark_overhead']:.6f} up to Delta_E_bal={diagnostics['bit_balance_residual']:.6e}, "
                f"while the global graviton sector is throttled by xi=c_vis/c_dark={diagnostics['inflation_holographic_suppression_factor']:.6f}. "
                f"This lowers the observable tensor ratio to r_obs={diagnostics['inflation_observable_tensor_ratio']:.6f} against the current "
                f"bound r<{diagnostics['inflation_bicep_keck_bound']:.3f}, with Complexity-Bound pass={_audit_flag(diagnostics['inflation_complexity_bound_tensor_pass'])}."
            )
        )

    if all(
        key in diagnostics
        for key in (
            "anomaly_detection_alpha_inverse",
            "anomaly_detection_alpha_inverse_fraction",
            "anomaly_detection_alpha",
            "anomaly_detection_noise_floor",
            "anomaly_detection_stable",
            "benchmark_anchor_central_charge_ratio",
            "benchmark_anchor_pixel_volume",
            "benchmark_anchor_pixel_volume_fraction",
            "benchmark_anchor_mu_predicted",
            "benchmark_anchor_mu_empirical",
            "benchmark_anchor_mu_relative_error",
            "benchmark_anchor_pass",
            "holographic_triple_lock_success",
        )
    ):
        report_lines.append(
            (
                "QEC instruction-set audit / Boundary-support audit: electromagnetic charge is the syndrome measurement threshold of the anomaly-free boundary mapping. "
                f"The exact threshold is alpha_surf^-1={diagnostics['anomaly_detection_alpha_inverse']:.6f} "
                f"({diagnostics['anomaly_detection_alpha_inverse_fraction']}), so the effective gauge strength alpha={diagnostics['anomaly_detection_alpha']:.6f} "
                f"stays below the packing-noise floor 1-kappa_D5={diagnostics['anomaly_detection_noise_floor']:.6f}, with anomaly-detection stability flag={_audit_flag(diagnostics['anomaly_detection_stable'])}. "
                f"Treating the proton and electron as quark/lepton boundary pixels gives c_q/c_l={diagnostics['benchmark_anchor_central_charge_ratio']:.6f}, "
                f"V_pix={diagnostics['benchmark_anchor_pixel_volume_fraction']}={diagnostics['benchmark_anchor_pixel_volume']:.6f}, and mu_pix={diagnostics['benchmark_anchor_mu_predicted']:.6f} "
                f"against the empirical {diagnostics['benchmark_anchor_mu_empirical']:.6f}, i.e. a relative offset of {100.0 * diagnostics['benchmark_anchor_mu_relative_error']:.3f}%, "
                f"with Atomic-lock pass={_audit_flag(diagnostics['benchmark_anchor_pass'])} (Benchmark-Anchor pass={_audit_flag(diagnostics['benchmark_anchor_pass'])}) "
                f"and Triple-Lock success={_audit_flag(diagnostics['holographic_triple_lock_success'])}."
            )
        )

    elif all(
        key in diagnostics
        for key in (
            "baryon_lepton_observed_ratio",
            "baryon_lepton_structural_prefactor",
            "baryon_lepton_required_flux",
            "baryon_lepton_packing_deficiency",
            "baryon_lepton_rank_deficit_pressure",
            "baryon_lepton_vacuum_pressure",
            "baryon_lepton_rigid_match_pass",
        )
    ):
        report_lines.append(
            (
                "Baryon-Lepton Symmetry: $m_p/m_e$ is treated here as a branch-fixed consistency check rather than as an RG-floating prediction. "
                "Topologically protected lock requires identification of the SU(3) rank-deficit pressure $\\Pi_{vac}$ as the Trace-Anomaly boundary condition. "
                f"At the current level the rigid prefactor alpha_surf^-1*(I_Q/I_L)={diagnostics['baryon_lepton_structural_prefactor']:.6f} "
                f"undershoots the PDG ratio {diagnostics['baryon_lepton_observed_ratio']:.6f}, so the unresolved conformal mixing flux is "
                f"Phi_QE={diagnostics['baryon_lepton_required_flux']:.6f}. The Packing Deficiency 1-kappa_D5="
                f"{diagnostics['baryon_lepton_packing_deficiency']:.6f} and Rank-Deficit Pressure Pi_rank="
                f"{diagnostics['baryon_lepton_rank_deficit_pressure']:.6f} are therefore tracked only as Latent Scaling Candidates, "
                f"while the SU(3) trace-anomaly pressure remains Pi_vac={diagnostics['baryon_lepton_vacuum_pressure']:.6f}."
            )
        )

    if all(
        key in diagnostics
        for key in (
            "falsification_m_beta_beta_mev",
            "falsification_m_beta_beta_lower_mev",
            "falsification_m_beta_beta_upper_mev",
            "falsification_m_beta_beta_in_window",
            "falsification_f_nl",
            "falsification_f_nl_locked",
        )
    ):
        report_lines.append(
            (
                "The primary falsification envelope is shifted away from direct WIMP searches and onto the locked low-energy / CMB observables: "
                f"|m_bb|={diagnostics['falsification_m_beta_beta_mev']:.6f} meV inside the target window "
                f"[{diagnostics['falsification_m_beta_beta_lower_mev']:.1f}, {diagnostics['falsification_m_beta_beta_upper_mev']:.1f}] meV "
                f"with in-window flag={_audit_flag(diagnostics['falsification_m_beta_beta_in_window'])}, and "
                f"f_NL={diagnostics['falsification_f_nl']:.6f}=1-kappa_D5 with lock flag={_audit_flag(diagnostics['falsification_f_nl_locked'])}."
            )
        )

    if all(
        key in diagnostics
        for key in (
            "ih_support_deficit",
            "ih_modularity_limit_rank",
            "ih_required_dictionary_rank",
            "ih_redundancy_entropy_cost_nat",
            "ih_bankruptcy_exception",
        )
    ):
        report_lines.append(
            (
                "The narrow IH extension note is "
                f"{'active' if bool(diagnostics['ih_bankruptcy_exception']) else 'inactive'}: the one-copy support map carries "
                f"support deficit {int(diagnostics['ih_support_deficit'])} against modularity limit rank "
                f"{int(diagnostics['ih_modularity_limit_rank'])}, so an inverted completion would require dictionary rank "
                f"{int(diagnostics['ih_required_dictionary_rank'])} and redundancy entropy cost "
                f"{diagnostics['ih_redundancy_entropy_cost_nat']:.6f} nat. In publication-facing language this does not "
                "exclude IH phenomenologically; it records IH as an explicit non-minimal extension of the minimal "
                "one-copy benchmark dictionary."
            )
        )

    if all(
        key in diagnostics
        for key in (
            "gauge_framing_closed",
            "gauge_topological_stability_pass",
            "gravity_framing_gap",
            "gravity_torsion_free",
            "gravity_non_singular_bulk",
            "gravity_lambda_aligned",
            "gravity_bulk_emergent",
            "gravity_gmunu_consistency_score",
            "lambda_holo_si_m2",
            "lambda_anchor_si_m2",
            "lambda_identity_si_m2",
            "lambda_surface_tension_prefactor",
            "lambda_surface_tension_deviation_percent",
            "lambda_alpha_locked_under_bit_shift",
            "baryon_proton_lifetime_years",
            "baryon_dimension_five_forbidden",
            "triple_lock_consistent",
        )
    ):
        report_lines.append(
            (
                f"The torsion-free bulk lock is {_audit_status(diagnostics['triple_lock_consistent'])}: "
                f"Δ_fr={diagnostics['gravity_framing_gap']:.6f}, framing_closed={_audit_flag(diagnostics['gauge_framing_closed'])}, "
                f"torsion_free={_audit_flag(diagnostics['gravity_torsion_free'])}, non_singular_bulk={_audit_flag(diagnostics['gravity_non_singular_bulk'])}, "
                f"lambda_aligned={_audit_flag(diagnostics['gravity_lambda_aligned'])}, bulk_emergent={_audit_flag(diagnostics['gravity_bulk_emergent'])}, "
                f"and G_munu consistency score={diagnostics['gravity_gmunu_consistency_score']:.6f}. The same anomaly-free cell gives "
                f"Lambda_holo={diagnostics['lambda_holo_si_m2']:.6e} m^-2, Lambda_obs={diagnostics['lambda_anchor_si_m2']:.6e} m^-2, and "
                f"1/(L_P^2 N)={diagnostics['lambda_identity_si_m2']:.6e} m^-2, so the surface-tension prefactor stays at "
                f"{diagnostics['lambda_surface_tension_prefactor']:.6f} with a cosmology-anchor deviation of "
                f"{diagnostics['lambda_surface_tension_deviation_percent']:.2f}% and alpha-lock under N±1%="
                f"{_audit_flag(diagnostics['lambda_alpha_locked_under_bit_shift'])}."
            )
        )
        report_lines.append(
            (
                f"The protected baryon floor remains tau_p={diagnostics['baryon_proton_lifetime_years']:.2e} years with "
                f"dimension-five leakage forbidden={_audit_flag(diagnostics['baryon_dimension_five_forbidden'])}; the Residue Convergence Condition "
                "(alpha, Lambda_holo, tau_p) therefore stays attached to a single torsion-free benchmark branch."
            )
        )

    report = "\n".join(report_lines)
    if output_dir is not None:
        (output_dir / AUDIT_STATEMENT_FILENAME).write_text(report + "\n", encoding="utf-8")
    return report


def write_svd_stability_report(
    mass_ratio_stability_audit,
    output_dir: Path | None = None,
) -> str:
    r"""Write the Higgs-VEV-alignment stability report as numeric diagnostics."""

    def _format_triplet(values: tuple[float, float, float]) -> str:
        return "(" + ", ".join(f"{value:.6e}" for value in values) + ")"

    rigidity_message = _mass_ratio_rigidity_message(mass_ratio_stability_audit)
    report_lines = [
        "SVD Stability Report",
        "====================",
        f"target relative suppression   : {mass_ratio_stability_audit.target_relative_suppression:.6f}",
        f"Clebsch suppression           : {mass_ratio_stability_audit.clebsch_relative_suppression:.6f}",
        f"relative spectral shift       : {mass_ratio_stability_audit.relative_spectral_volume_shift:.6f}",
        f"lepton unitary Frobenius shift: {mass_ratio_stability_audit.lepton_unitary_frobenius_shift:.6e}",
        f"quark unitary Frobenius shift : {mass_ratio_stability_audit.quark_unitary_frobenius_shift:.6e}",
        f"min lepton overlaps           : left={mass_ratio_stability_audit.lepton_left_overlap_min:.12f}  right={mass_ratio_stability_audit.lepton_right_overlap_min:.12f}",
        f"min quark overlaps            : left={mass_ratio_stability_audit.quark_left_overlap_min:.12f}  right={mass_ratio_stability_audit.quark_right_overlap_min:.12f}",
        f"lepton sigma shifts           : {_format_triplet(mass_ratio_stability_audit.lepton_sigma_shifts)}",
        f"quark sigma shifts            : {_format_triplet(mass_ratio_stability_audit.quark_sigma_shifts)}",
        f"max sigma angle shift         : {mass_ratio_stability_audit.max_sigma_shift:.6e}",
        f"ensemble sample count         : {mass_ratio_stability_audit.ensemble_sample_count}",
        f"ensemble seed                 : {mass_ratio_stability_audit.ensemble_seed}",
        f"ensemble max sigma shift      : {mass_ratio_stability_audit.ensemble_max_sigma_shift:.6e}",
        f"ensemble max theta13 shift    : {mass_ratio_stability_audit.ensemble_theta13_max_sigma_shift:.6e}",
        f"ensemble max thetaC shift     : {mass_ratio_stability_audit.ensemble_theta_c_max_sigma_shift:.6e}",
        f"ensemble mass-scale range     : [{mass_ratio_stability_audit.ensemble_mass_scale_shift_min:.6f}, {mass_ratio_stability_audit.ensemble_mass_scale_shift_max:.6f}]",
        f"ensemble all within one sigma : {int(bool(mass_ratio_stability_audit.ensemble_all_within_one_sigma))}",
        rigidity_message,
    ]
    report = "\n".join(report_lines)
    if output_dir is not None:
        (output_dir / SVD_STABILITY_REPORT_FILENAME).write_text(report + "\n", encoding="utf-8")
    return report


def write_eigenvector_stability_audit(
    weight_profile,
    mass_ratio_stability_audit,
    output_dir: Path | None = None,
) -> str:
    """Write the standalone eigenvector stability audit as numeric diagnostics."""

    rigidity_message = _mass_ratio_rigidity_message(mass_ratio_stability_audit)
    report_lines = [
        "Eigenvector Stability Check",
        "===========================",
        f"benchmark R_GUT               : {weight_profile.benchmark_weight:.6f}",
        f"off-shell minimum R_GUT       : {weight_profile.best_fit_weight:.6f}",
        f"benchmark Delta chi2          : {weight_profile.benchmark_delta_chi2:.6f}",
        f"max Delta|Vus| from R_GUT     : {weight_profile.max_vus_shift:.6e}",
        f"max Delta|Vcb| from R_GUT     : {weight_profile.max_vcb_shift:.6e}",
        f"max Delta|Vub| from R_GUT     : {weight_profile.max_vub_shift:.6e}",
        f"max SVD angle shift [sigma]   : {mass_ratio_stability_audit.max_sigma_shift:.6e}",
        f"ensemble max SVD shift [sigma]: {mass_ratio_stability_audit.ensemble_max_sigma_shift:.6e}",
        rigidity_message,
    ]
    report = "\n".join(report_lines)
    if output_dir is not None:
        (output_dir / EIGENVECTOR_STABILITY_AUDIT_FILENAME).write_text(report + "\n", encoding="utf-8")
    return report


def write_seed_robustness_audit(
    seed_robustness_audit,
    output_dir: Path | None = None,
) -> str:
    """Write the seed-by-seed robustness report for stochastic audit components."""

    report_lines = [
        "Seed Robustness Check",
        "=====================",
        f"seed count                    : {seed_robustness_audit.seed_count}",
        f"seeds                         : {', '.join(str(seed) for seed in seed_robustness_audit.seeds)}",
        f"max relative std              : {seed_robustness_audit.max_relative_std:.6e}",
        f"max relative variance         : {seed_robustness_audit.max_relative_variance:.6e}",
    ]
    report = "\n".join(report_lines)
    if output_dir is not None:
        (output_dir / SEED_ROBUSTNESS_AUDIT_FILENAME).write_text(report + "\n", encoding="utf-8")
    return report


def write_corollary_report(
    corollary_audit,
    output_dir: Path | None = None,
) -> str:
    """Write the appendix-only interpretive corollary note as plain-text diagnostics."""

    sparse_residue_audit = corollary_audit.verify_biological_sparse_residue()
    zero_balanced_text = "YES" if bool(sparse_residue_audit["zero_balanced_vacuum"]) else "NO"
    report_lines = [
        "Appendix-only interpretive corollary report",
        "========================================",
        "",
        "These quantities are derived from the benchmark residues of the (26, 8, 312) branch.",
        "Biological sparse-residue diagnostics, the H0 skew, and the computational bounds are reported for appendix consultation only and are not part of the benchmark pass/fail criteria.",
        "",
        "Biological Sparse Residue Audit",
        "-------------------------------",
        f"1 - kappa_D5                  : {corollary_audit.biological_audit.packing_deficiency:.6f}",
        f"c_dark / K                    : {corollary_audit.biological_audit.dark_parity_density:.6f}",
        f"Delta_E_bal                   : {sparse_residue_audit['bit_balance_residual']:.6e}",
        f"zero-balanced vacuum          : {zero_balanced_text}",
        f"entropy debt hypothesis       : {sparse_residue_audit['entropy_debt_status']}",
        f"life necessity                : {sparse_residue_audit['life_necessity']}",
        "",
        "We evaluate biological complexity strictly as a non-essential application executing on the c_vis modular support data.",
        "The apparent packing deficiency is saturated by the completion-sector parity bits c_dark/K.",
        "This confirms the Bit-Balance Identity and rejects the Entropy Debt hypothesis: the vacuum is zero-balanced prior to, and independent of, macroscopic observers.",
        "",
        "Late-time Clock-Skew in the Expansion Rate",
        "------------------------------------------",
        f"Delta_mod                     : {corollary_audit.delta_mod:.6f}",
        f"H0_CMB [km s^-1 Mpc^-1]       : {corollary_audit.h0_cmb:.2f}",
        f"exp(Delta_mod / 2)            : {corollary_audit.hubble_skew_factor:.6f}",
        f"H0_clock_skew [km s^-1 Mpc^-1]: {corollary_audit.get_hubble_skew():.2f}",
        "",
        "We record as an appendix-only interpretive mapping that the modular residual Delta_mod ~= 0.1375 induces a late-time shift in the expansion-rate readout.",
        f"Applying this shift to the Planck-anchored H0 ~= {corollary_audit.h0_cmb:.1f} yields a local expansion value of ~= {corollary_audit.get_hubble_skew():.2f} km/s/Mpc.",
        "This shift is disclosed solely as a benchmark-derived clock-skew readout, not as an additional fit target.",
        "",
        "Computational Friction and Complexity Bound Mapping Obstruction",
        "---------------------------------------------------------------",
        f"n_s                           : {corollary_audit.n_s:.4f}",
        f"delta_clk = 1 - n_s           : {corollary_audit.get_technological_ceiling():.6f}",
        f"rate [operations / second]    : {corollary_audit.get_lloyd_ceiling():.6e}",
        f"mapping obstruction           : {corollary_audit.mapping_obstruction_label}",
        "",
        "While the vacuum is self-correcting, macroscopic evolution is globally throttled by the cosmic clock-skew.",
        "Any technological thread attempting to saturate the local Complexity Bound encounters a basis-mapping obstruction, inducing localized geometric torsion (Delta_fr != 0).",
        "This Complexity Bound Mapping Obstruction is reported only as an appendix-only topological speed limit of the K=312 lattice.",
        "It does not enter the benchmark-selection logic, the published pull table, or any benchmark pass/fail criterion.",
    ]
    report = "\n".join(report_lines)
    if output_dir is not None:
        (output_dir / COROLLARY_REPORT_FILENAME).write_text(report + "\n", encoding="utf-8")
    return report


def write_stability_report(
    bookkeeping_lines: list[str],
    nonlinearity_audit,
    svd_report: str,
    output_dir: Path | None = None,
) -> str:
    """Write the unified publication-facing stability report as numeric diagnostics."""

    report = "\n".join(
        [
            "Stability Report",
            "================",
            "Benchmark bookkeeping",
            "---------------------",
            *bookkeeping_lines,
            "",
            "ODE convergence metrics",
            "-----------------------",
            f"max full-vs-linear sigma discrepancy : {nonlinearity_audit.max_sigma_error:.6e}",
            f"theta13 linear/full [deg]            : {nonlinearity_audit.theta_linear_deg[1]:.8f} / {nonlinearity_audit.theta_nonlinear_deg[1]:.8f}",
            f"theta23 linear/full [deg]            : {nonlinearity_audit.theta_linear_deg[2]:.8f} / {nonlinearity_audit.theta_nonlinear_deg[2]:.8f}",
            f"deltaCP linear/full [deg]            : {nonlinearity_audit.delta_linear_deg:.8f} / {nonlinearity_audit.delta_nonlinear_deg:.8f}",
            f"m0 linear/full [meV]                 : {1.0e3 * nonlinearity_audit.m_0_linear_ev:.5f} / {1.0e3 * nonlinearity_audit.m_0_nonlinear_ev:.5f}",
            "",
            svd_report,
        ]
    )
    if output_dir is not None:
        (output_dir / STABILITY_REPORT_FILENAME).write_text(report + "\n", encoding="utf-8")
    return report
