from __future__ import annotations

"""Synchronize README and manuscript macros from the live SHBT audit ledger."""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from scipy.constants import electron_mass, proton_mass

if __package__ in (None, ""):
    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.config_loader import ConfigLoader
from shbt.constants import SU2_DIMENSION, SU2_DUAL_COXETER, SU3_DIMENSION, SU3_DUAL_COXETER
from shbt.core.derivation_api import (
    DEFAULT_PRECISION as DERIVATION_DEFAULT_PRECISION,
    TopologicalVacuum,
    UniverseFactory,
    decimal_pi,
)
from shbt.core.engine import quark_branching_index, wzw_central_charge_fraction
from shbt.main import derive_transport_curvature_audit
from shbt.paths import ProjectPaths


DEFAULT_RESIDUALS_PATH = ProjectPaths.RESULTS / "residuals.json"
DEFAULT_README_PATH = ProjectPaths.ROOT / "README.md"
DEFAULT_PHYSICS_CONSTANTS_PATH = ProjectPaths.PAPERS / "physics_constants.tex"
DEFAULT_PHYSICS_PROFILE_PATH = (ProjectPaths.CONFIG / "physics_profiles" / "standard_model.yaml").resolve()
DEFAULT_UNIVERSAL_CONSTANTS_PATH = DEFAULT_PHYSICS_PROFILE_PATH
DEFAULT_PRECISION = max(int(DERIVATION_DEFAULT_PRECISION), 50)
_GUARD_DIGITS = 12

README_LEDGER_TABLE_HEADER = "| Observable | Derived From | Predicted Value | CODATA / anchor |"
README_SYNC_START = "<!-- sync-system:derivation-ledger:start -->"
README_SYNC_END = "<!-- sync-system:derivation-ledger:end -->"
PHYSICS_CONSTANTS_SYNC_START = "% sync-system:residual-macros:start"
PHYSICS_CONSTANTS_SYNC_END = "% sync-system:residual-macros:end"


@dataclass(frozen=True)
class SyncSnapshot:
    benchmark_tuple: tuple[int, int, int]
    epsilon_lambda: float
    exact_epsilon_lambda: float
    register_noise_floor: float
    exact_register_noise_floor: float
    gauge_topological_alpha_inverse: float
    gauge_codata_alpha_inverse: float
    gauge_two_loop_residual_fraction: float
    gauge_two_loop_residual_percent: float
    gauge_two_loop_residual_pull: float
    delta_s_red_nat: float
    mass_scale_two_loop_fraction: float
    lepton_theta12_two_loop_deg: float
    lepton_theta13_two_loop_deg: float
    lepton_theta23_two_loop_deg: float
    lepton_delta_two_loop_deg: float
    quark_theta12_two_loop_deg: float
    quark_theta13_two_loop_deg: float
    quark_theta23_two_loop_deg: float
    quark_delta_two_loop_deg: float
    proton_electron_mass_ratio: float
    proton_electron_mass_ratio_codata: float
    neutrino_floor_mev: float


@dataclass(frozen=True)
class UniversalConstantsSnapshot:
    lepton_level: int
    quark_level: int
    parent_level: int
    g_sm: int
    lepton_descendant: int
    quark_descendant: int
    topological_alpha_inverse: float
    topological_alpha_inverse_numerator: int
    topological_alpha_inverse_denominator: int
    planck_mass_ev: float
    planck_length_m: float
    lambda_obs_si_m2: float


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Expected '{key}' to be a mapping.")
    return value


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


_SYNC_TOP_LEVEL_PATHS = frozenset(
    {
        ProjectPaths.RESULTS.name,
        ProjectPaths.PAPERS.name,
        ProjectPaths.CONFIG.name,
        ProjectPaths.DATA.name,
        ProjectPaths.SCRIPTS.name,
        ProjectPaths.SRC.name,
        "README.md",
    }
)


def _resolve_sync_path(path: Path | str, *, base_dir: Path) -> Path:
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path.resolve()
    cwd_relative_path = resolved_path.resolve()
    if cwd_relative_path.exists():
        return cwd_relative_path
    repo_relative_path = (ProjectPaths.ROOT / resolved_path).resolve()
    if repo_relative_path.exists():
        return repo_relative_path
    if resolved_path.parts and resolved_path.parts[0] in _SYNC_TOP_LEVEL_PATHS:
        return repo_relative_path
    return (Path(base_dir) / resolved_path).resolve()


def _build_topological_vacuum(universal_snapshot: UniversalConstantsSnapshot) -> TopologicalVacuum:
    return TopologicalVacuum(
        lepton_level=universal_snapshot.lepton_level,
        quark_level=universal_snapshot.quark_level,
        parent_level=universal_snapshot.parent_level,
        generation_count=universal_snapshot.g_sm,
    )


def _build_universal_constants_snapshot(config_loader: ConfigLoader) -> UniversalConstantsSnapshot:
    universal_constants = config_loader.load_universal_constants()
    tier_1 = _require_mapping(universal_constants, "tier_1")
    tier_2 = _require_mapping(universal_constants, "tier_2")

    lepton_level = int(tier_1["lepton_level"])
    quark_level = int(tier_1["quark_level"])
    parent_level = int(tier_1["parent_level"])
    g_sm = int(tier_1["g_sm"])
    lepton_descendant = parent_level // (2 * lepton_level)
    quark_descendant = parent_level // (3 * quark_level)
    alpha_numerator = g_sm * parent_level
    alpha_denominator = lepton_level + quark_level
    topological_alpha_inverse = alpha_numerator / alpha_denominator
    lambda_obs_si_m2 = float(tier_2["planck2018_lambda_si_m2"])

    return UniversalConstantsSnapshot(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        g_sm=g_sm,
        lepton_descendant=lepton_descendant,
        quark_descendant=quark_descendant,
        topological_alpha_inverse=topological_alpha_inverse,
        topological_alpha_inverse_numerator=alpha_numerator,
        topological_alpha_inverse_denominator=alpha_denominator,
        planck_mass_ev=float(tier_2["planck_mass_ev"]),
        planck_length_m=float(tier_2["planck_length_m"]),
        lambda_obs_si_m2=lambda_obs_si_m2,
    )


def _format_markdown_float(value: float, *, digits: int = 6) -> str:
    numeric_value = float(value)
    if numeric_value == 0.0:
        return "0"
    if abs(numeric_value) >= 1.0e4 or abs(numeric_value) < 1.0e-3:
        return f"{numeric_value:.{digits}e}"
    return f"{numeric_value:.{digits}f}".rstrip("0").rstrip(".")


def _format_latex_float(value: float, *, digits: int = 10) -> str:
    numeric_value = float(value)
    if numeric_value == 0.0:
        return "0"
    if abs(numeric_value) >= 1.0e4 or abs(numeric_value) < 1.0e-3:
        coefficient_text, exponent_text = f"{numeric_value:.{digits}e}".split("e")
        coefficient = coefficient_text.rstrip("0").rstrip(".")
        exponent = int(exponent_text)
        return rf"{coefficient}\times10^{{{exponent}}}"
    return f"{numeric_value:.{digits}f}".rstrip("0").rstrip(".")


def _resolve_project_path(path: Path) -> Path:
    return _resolve_sync_path(path, base_dir=ProjectPaths.ROOT)


def _load_residual_payload(path: Path) -> dict[str, Any]:
    resolved_path = _resolve_project_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Residual ledger not found at {resolved_path}. Run the verifier to generate results/residuals.json first."
        )
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Residual ledger must decode to a JSON object.")
    return payload


def _derive_kappa_d5(
    universal_snapshot: UniversalConstantsSnapshot,
    *,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    return UniverseFactory.derive_kappa_d5(
        precision=precision,
        vacuum=_build_topological_vacuum(universal_snapshot),
    ).kappa


def _derive_holographic_bits(
    universal_snapshot: UniversalConstantsSnapshot,
    *,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        three_pi = Decimal(3) * decimal_pi(context.prec)
        planck_length_m = _decimal(universal_snapshot.planck_length_m)
        lambda_obs_si_m2 = _decimal(universal_snapshot.lambda_obs_si_m2)
        bits = three_pi / (planck_length_m * planck_length_m * lambda_obs_si_m2)
        context.prec = precision
        return +bits


def _derive_neutrino_floor_mev(
    universal_snapshot: UniversalConstantsSnapshot,
    *,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        kappa = _derive_kappa_d5(universal_snapshot, precision=context.prec)
        holographic_bits = _derive_holographic_bits(universal_snapshot, precision=context.prec)
        planck_mass_ev = _decimal(universal_snapshot.planck_mass_ev)
        neutrino_floor_ev = kappa * planck_mass_ev / holographic_bits.sqrt().sqrt()
        neutrino_floor_mev = neutrino_floor_ev * Decimal(1000)
        context.prec = precision
        return +neutrino_floor_mev


def _branch_pixel_volume_fraction(universal_snapshot: UniversalConstantsSnapshot) -> Fraction:
    return Fraction(SU3_DUAL_COXETER, quark_branching_index(universal_snapshot.parent_level, universal_snapshot.quark_level))


def _derive_proton_electron_mass_ratio(
    universal_snapshot: UniversalConstantsSnapshot,
    *,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    derivation = UniverseFactory.derive_proton_ratio(
        precision=precision,
        vacuum=_build_topological_vacuum(universal_snapshot),
    )
    return _decimal(derivation.mu_audit)


def _codata_proton_electron_mass_ratio() -> Decimal:
    return _decimal(proton_mass) / _decimal(electron_mass)


def build_sync_snapshot(payload: dict[str, Any]) -> SyncSnapshot:
    return build_sync_snapshot_with_universal_constants(payload)


def build_sync_snapshot_with_universal_constants(
    payload: dict[str, Any],
    *,
    universal_snapshot: UniversalConstantsSnapshot | None = None,
) -> SyncSnapshot:
    if universal_snapshot is None:
        universal_snapshot = _build_universal_constants_snapshot(ConfigLoader())

    expected_benchmark_tuple = (
        universal_snapshot.lepton_level,
        universal_snapshot.quark_level,
        universal_snapshot.parent_level,
    )
    benchmark_tuple = tuple(int(value) for value in payload.get("benchmark_tuple", expected_benchmark_tuple))
    if len(benchmark_tuple) != 3:
        raise ValueError("benchmark_tuple must contain exactly three integers.")
    if benchmark_tuple != expected_benchmark_tuple:
        raise ValueError(
            "Residual benchmark_tuple "
            f"{benchmark_tuple} does not match the configured universal constants {expected_benchmark_tuple}."
        )

    unity = _require_mapping(payload, "unity_of_scale_identity")
    gauge = _require_mapping(payload, "gauge_residual_bookkeeping")
    informational_costs = _require_mapping(payload, "informational_costs")
    mixing_angle_drifts = _require_mapping(payload, "mixing_angle_drifts_deg")

    reported_alpha_inverse = float(gauge["topological_alpha_inverse"])
    if not math.isclose(
        reported_alpha_inverse,
        universal_snapshot.topological_alpha_inverse,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "Residual gauge_residual_bookkeeping.topological_alpha_inverse "
            f"{reported_alpha_inverse} does not match the configured universal constants "
            f"{universal_snapshot.topological_alpha_inverse}."
        )

    transport_curvature = derive_transport_curvature_audit(
        lepton_level=int(expected_benchmark_tuple[0]),
        quark_level=int(expected_benchmark_tuple[1]),
        parent_level=int(expected_benchmark_tuple[2]),
    )

    proton_electron_mass_ratio = _derive_proton_electron_mass_ratio(universal_snapshot)
    neutrino_floor_mev = _derive_neutrino_floor_mev(universal_snapshot)

    return SyncSnapshot(
        benchmark_tuple=benchmark_tuple,
        epsilon_lambda=float(unity["epsilon_lambda"]),
        exact_epsilon_lambda=float(unity.get("exact_epsilon_lambda", unity["epsilon_lambda"])),
        register_noise_floor=float(unity["register_noise_floor"]),
        exact_register_noise_floor=float(unity.get("exact_register_noise_floor", unity["register_noise_floor"])),
        gauge_topological_alpha_inverse=reported_alpha_inverse,
        gauge_codata_alpha_inverse=float(gauge["codata_alpha_inverse"]),
        gauge_two_loop_residual_fraction=float(gauge["two_loop_residual_fraction"]),
        gauge_two_loop_residual_percent=float(gauge["two_loop_residual_percent"]),
        gauge_two_loop_residual_pull=float(gauge["two_loop_residual_pull"]),
        delta_s_red_nat=float(informational_costs["delta_s_red_nat"]),
        mass_scale_two_loop_fraction=float(payload["mass_scale_two_loop_fraction"]),
        lepton_theta12_two_loop_deg=float(mixing_angle_drifts["theta12"]),
        lepton_theta13_two_loop_deg=float(mixing_angle_drifts["theta13"]),
        lepton_theta23_two_loop_deg=float(mixing_angle_drifts["theta23"]),
        lepton_delta_two_loop_deg=float(mixing_angle_drifts["delta_cp"]),
        quark_theta12_two_loop_deg=float(transport_curvature.quark_theta_two_loop[0]),
        quark_theta13_two_loop_deg=float(transport_curvature.quark_theta_two_loop[1]),
        quark_theta23_two_loop_deg=float(transport_curvature.quark_theta_two_loop[2]),
        quark_delta_two_loop_deg=float(transport_curvature.quark_delta_two_loop),
        proton_electron_mass_ratio=float(proton_electron_mass_ratio),
        proton_electron_mass_ratio_codata=float(_codata_proton_electron_mass_ratio()),
        neutrino_floor_mev=float(neutrino_floor_mev),
    )


def _render_readme_derivation_table(
    snapshot: SyncSnapshot,
    universal_snapshot: UniversalConstantsSnapshot,
) -> str:
    visible_support = universal_snapshot.lepton_level + universal_snapshot.quark_level
    lepton_central_charge = wzw_central_charge_fraction(
        universal_snapshot.lepton_level,
        SU2_DIMENSION,
        SU2_DUAL_COXETER,
    )
    quark_central_charge = wzw_central_charge_fraction(
        universal_snapshot.quark_level,
        SU3_DIMENSION,
        SU3_DUAL_COXETER,
    )
    branch_pixel_volume = _branch_pixel_volume_fraction(universal_snapshot)

    lines = [
        README_LEDGER_TABLE_HEADER,
        "| :--- | :--- | :--- | :--- |",
        (
            f"| $\\alpha^{{-1}}$ | ${universal_snapshot.g_sm} \\times {universal_snapshot.parent_level} / {visible_support}$ | "
            f"$\\approx {_format_markdown_float(snapshot.gauge_topological_alpha_inverse, digits=3)}$ | "
            f"${_format_markdown_float(snapshot.gauge_codata_alpha_inverse, digits=3)}$ (Two-Loop Residual) |"
        ),
        (
            "| $\\mu$ ($m_p/m_e$) | "
            "$(c_q/c_{\\ell}) V_{\\rm px}^{-1} \\Pi_{\\rm vac}^2 / [(1-\\kappa_{D_5})\\kappa_{D_5}^{1/3}]$ <br> "
            f"($c_q={_format_fraction(quark_central_charge)}, "
            f"c_{{\\ell}}={_format_fraction(lepton_central_charge)}, "
            f"V_{{\\rm px}}={_format_fraction(branch_pixel_volume)}$) | "
            f"$\\approx {_format_markdown_float(snapshot.proton_electron_mass_ratio, digits=2)}$ | "
            f"${_format_markdown_float(snapshot.proton_electron_mass_ratio_codata, digits=3)}$ (Atomic Lock) |"
        ),
        (
            "| $m_{\\nu}$ | $\\kappa_{D_5} M_P N^{-1/4}$ | "
            f"$\\approx {_format_markdown_float(snapshot.neutrino_floor_mev, digits=2)}$ meV | "
            f"$\\sim {_format_markdown_float(snapshot.neutrino_floor_mev, digits=1)}$ meV (Theory-Fixed) |"
        ),
    ]
    return "\n".join(lines)


def _render_readme_sync_block(snapshot: SyncSnapshot) -> str:
    lepton_tuple = (
        _format_markdown_float(snapshot.lepton_theta12_two_loop_deg),
        _format_markdown_float(snapshot.lepton_theta13_two_loop_deg),
        _format_markdown_float(snapshot.lepton_theta23_two_loop_deg),
        _format_markdown_float(snapshot.lepton_delta_two_loop_deg),
    )
    quark_tuple = (
        _format_markdown_float(snapshot.quark_theta12_two_loop_deg),
        _format_markdown_float(snapshot.quark_theta13_two_loop_deg),
        _format_markdown_float(snapshot.quark_theta23_two_loop_deg),
        _format_markdown_float(snapshot.quark_delta_two_loop_deg),
    )
    lines = [
        "### Machine-Synced Residual Ledger",
        README_SYNC_START,
        "| Audit quantity | JSON key / source | Synced value |",
        "| --- | --- | --- |",
        (
            "| Unity-of-Scale residue | `unity_of_scale_identity.epsilon_lambda` | "
            f"`{_format_markdown_float(snapshot.epsilon_lambda)}` |"
        ),
        (
            "| Exact Unity-of-Scale residue | `unity_of_scale_identity.exact_epsilon_lambda` | "
            f"`{_format_markdown_float(snapshot.exact_epsilon_lambda)}` |"
        ),
        (
            "| Register noise floor | `unity_of_scale_identity.register_noise_floor` | "
            f"`{_format_markdown_float(snapshot.register_noise_floor)}` |"
        ),
        (
            "| Gauge-side `alpha^{-1}` residue fraction | `gauge_residual_bookkeeping.two_loop_residual_fraction` | "
            f"`{_format_markdown_float(snapshot.gauge_two_loop_residual_fraction)}` |"
        ),
        (
            "| Gauge-side `alpha^{-1}` residual pull | `gauge_residual_bookkeeping.two_loop_residual_pull` | "
            f"`{_format_markdown_float(snapshot.gauge_two_loop_residual_pull)}` |"
        ),
        (
            "| Gauge topological anchor | `gauge_residual_bookkeeping.topological_alpha_inverse` | "
            f"`{_format_markdown_float(snapshot.gauge_topological_alpha_inverse)}` vs `CODATA = {_format_markdown_float(snapshot.gauge_codata_alpha_inverse)}` |"
        ),
        (
            "| Redundancy entropy cost $\\Delta S_{\\rm red}$ | `informational_costs.delta_s_red_nat` | "
            f"`{_format_markdown_float(snapshot.delta_s_red_nat)}` nat |"
        ),
        (
            "| PMNS two-loop drifts $(\\theta_{12},\\theta_{13},\\theta_{23},\\delta)$ | `mixing_angle_drifts_deg.*` | "
            f"`{lepton_tuple}` deg |"
        ),
        (
            "| CKM two-loop drifts $(\\theta_{12},\\theta_{13},\\theta_{23},\\delta)$ | "
            "`benchmark_tuple -> derive_transport_curvature_audit()` | "
            f"`{quark_tuple}` deg |"
        ),
        (
            "| Mass-scale two-loop fraction | `mass_scale_two_loop_fraction` | "
            f"`{_format_markdown_float(snapshot.mass_scale_two_loop_fraction)}` |"
        ),
        README_SYNC_END,
    ]
    return "\n".join(lines)


def _derivation_section_bounds(readme_text: str) -> tuple[int, int]:
    ledger_header = "## Derivation Ledger"
    if ledger_header not in readme_text:
        raise ValueError("README.md is missing the '## Derivation Ledger' section.")
    ledger_start = readme_text.index(ledger_header)
    next_section_index = readme_text.find("\n## ", ledger_start + len(ledger_header))
    ledger_end = len(readme_text) if next_section_index == -1 else next_section_index
    return ledger_start, ledger_end


def _replace_or_insert_readme_table(readme_text: str, table: str) -> str:
    ledger_start, ledger_end = _derivation_section_bounds(readme_text)
    section_text = readme_text[ledger_start:ledger_end]
    section_lines = section_text.splitlines()
    table_lines = table.splitlines()

    try:
        table_header_index = section_lines.index(README_LEDGER_TABLE_HEADER)
    except ValueError:
        insertion_anchor = "### Tier Classification"
        try:
            insertion_index = section_lines.index(insertion_anchor)
        except ValueError:
            insertion_index = len(section_lines)
        new_lines = section_lines[:insertion_index]
        if new_lines and new_lines[-1] != "":
            new_lines.append("")
        new_lines.extend(table_lines)
        if insertion_index < len(section_lines):
            new_lines.append("")
        new_lines.extend(section_lines[insertion_index:])
    else:
        table_end_index = table_header_index
        while table_end_index < len(section_lines) and section_lines[table_end_index].startswith("|"):
            table_end_index += 1
        new_lines = section_lines[:table_header_index] + table_lines + section_lines[table_end_index:]

    updated_section = "\n".join(new_lines).rstrip() + "\n"
    return readme_text[:ledger_start] + updated_section + readme_text[ledger_end:]


def _replace_or_insert_readme_block(readme_text: str, block: str) -> str:
    if README_SYNC_START in readme_text and README_SYNC_END in readme_text:
        managed_block = "\n".join(block.splitlines()[1:])
        return re.sub(
            rf"{re.escape(README_SYNC_START)}.*?{re.escape(README_SYNC_END)}",
            lambda _: managed_block,
            readme_text,
            flags=re.DOTALL,
        )

    ledger_start, _ = _derivation_section_bounds(readme_text)
    insertion_anchor = "### Tier Classification"
    insertion_index = readme_text.find(insertion_anchor, ledger_start)
    if insertion_index == -1:
        next_section_index = readme_text.find("\n## ", ledger_start + len("## Derivation Ledger"))
        insertion_index = len(readme_text) if next_section_index == -1 else next_section_index
    return readme_text[:insertion_index].rstrip() + "\n\n" + block + "\n\n" + readme_text[insertion_index:].lstrip()


def sync_readme(
    *,
    readme_path: Path,
    snapshot: SyncSnapshot,
    universal_snapshot: UniversalConstantsSnapshot | None = None,
) -> Path:
    resolved_path = _resolve_project_path(readme_path)
    readme_text = resolved_path.read_text(encoding="utf-8")
    if universal_snapshot is None:
        universal_snapshot = _build_universal_constants_snapshot(ConfigLoader())
    updated_text = _replace_or_insert_readme_table(
        readme_text,
        _render_readme_derivation_table(snapshot, universal_snapshot),
    )
    updated_text = _replace_or_insert_readme_block(updated_text, _render_readme_sync_block(snapshot))
    resolved_path.write_text(updated_text, encoding="utf-8")
    return resolved_path


def _replace_macro(
    text: str,
    macro_name: str,
    macro_value: str,
    *,
    append_if_missing: bool = True,
) -> str:
    patterns = (
        re.compile(
            rf"^(?P<prefix>\\(?:newcommand|providecommand|renewcommand)\{{\\{re.escape(macro_name)}\}}\{{)(?P<value>.*?)(?P<suffix>\}})$",
            flags=re.MULTILINE,
        ),
        re.compile(
            rf"^(?P<prefix>\\def\\{re.escape(macro_name)}\{{)(?P<value>.*?)(?P<suffix>\}})$",
            flags=re.MULTILINE,
        ),
        re.compile(
            rf"^(?P<prefix>\\{re.escape(macro_name)}\{{)(?P<value>.*?)(?P<suffix>\}})$",
            flags=re.MULTILINE,
        ),
    )

    def _replacement(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{macro_value}{match.group('suffix')}"

    for pattern in patterns:
        updated_text, count = pattern.subn(_replacement, text, count=1)
        if count:
            return updated_text

    if not append_if_missing:
        return text
    return text + f"\n\\providecommand{{\\{macro_name}}}{{{macro_value}}}\n"


def _render_managed_macro_block(snapshot: SyncSnapshot) -> str:
    lines = [
        PHYSICS_CONSTANTS_SYNC_START,
        rf"\providecommand{{\unityResidueEpsilonLambda}}{{{_format_latex_float(snapshot.epsilon_lambda)}}}",
        rf"\providecommand{{\exactUnityResidueEpsilonLambda}}{{{_format_latex_float(snapshot.exact_epsilon_lambda)}}}",
        rf"\providecommand{{\unityRegisterNoiseFloor}}{{{_format_latex_float(snapshot.register_noise_floor)}}}",
        rf"\providecommand{{\exactUnityRegisterNoiseFloor}}{{{_format_latex_float(snapshot.exact_register_noise_floor)}}}",
        rf"\providecommand{{\gaugeTwoLoopResidualFraction}}{{{_format_latex_float(snapshot.gauge_two_loop_residual_fraction)}}}",
        rf"\providecommand{{\gaugeTwoLoopResidualPercent}}{{{_format_latex_float(snapshot.gauge_two_loop_residual_percent)}}}",
        rf"\providecommand{{\gaugeTwoLoopResidualPull}}{{{_format_latex_float(snapshot.gauge_two_loop_residual_pull)}}}",
        rf"\providecommand{{\deltaSRedNat}}{{{_format_latex_float(snapshot.delta_s_red_nat)}}}",
        rf"\providecommand{{\massScaleTwoLoopFraction}}{{{_format_latex_float(snapshot.mass_scale_two_loop_fraction)}}}",
        PHYSICS_CONSTANTS_SYNC_END,
    ]
    return "\n".join(lines)


def _replace_or_insert_physics_constants_block(physics_text: str, block: str) -> str:
    if PHYSICS_CONSTANTS_SYNC_START in physics_text and PHYSICS_CONSTANTS_SYNC_END in physics_text:
        return re.sub(
            rf"{re.escape(PHYSICS_CONSTANTS_SYNC_START)}.*?{re.escape(PHYSICS_CONSTANTS_SYNC_END)}",
            lambda _: block,
            physics_text,
            flags=re.DOTALL,
        )
    return physics_text.rstrip() + "\n\n" + block + "\n"


def sync_physics_constants(
    *,
    physics_constants_path: Path,
    snapshot: SyncSnapshot,
    universal_snapshot: UniversalConstantsSnapshot | None = None,
) -> Path:
    resolved_path = _resolve_project_path(physics_constants_path)
    physics_text = resolved_path.read_text(encoding="utf-8")
    if universal_snapshot is None:
        universal_snapshot = _build_universal_constants_snapshot(ConfigLoader())

    macro_updates = {
        "alphaSurfBenchmarkDecimal": _format_latex_float(universal_snapshot.topological_alpha_inverse),
        "alphaSurfBenchmarkExact": rf"\dfrac{{{universal_snapshot.topological_alpha_inverse_numerator}}}{{{universal_snapshot.topological_alpha_inverse_denominator}}}",
        "alphaSurfBenchmarkRounded": f"{universal_snapshot.topological_alpha_inverse:.3f}",
        "benchmarkLeptonDescendant": str(universal_snapshot.lepton_descendant),
        "benchmarkLeptonLevel": str(universal_snapshot.lepton_level),
        "benchmarkLambdaMetersInverseSquared": _format_latex_float(universal_snapshot.lambda_obs_si_m2),
        "benchmarkParentLevel": str(universal_snapshot.parent_level),
        "benchmarkPlanckMassEv": _format_latex_float(universal_snapshot.planck_mass_ev),
        "benchmarkQuarkDescendant": str(universal_snapshot.quark_descendant),
        "benchmarkQuarkLevel": str(universal_snapshot.quark_level),
        "benchmarkVisibleBranch": f"({universal_snapshot.lepton_level},{universal_snapshot.quark_level},{universal_snapshot.parent_level})",
        "benchmarkVisiblePair": f"({universal_snapshot.lepton_level},{universal_snapshot.quark_level})",
        "leptonThetaTwelveBetaTwoLoop": _format_latex_float(snapshot.lepton_theta12_two_loop_deg),
        "leptonThetaThirteenBetaTwoLoop": _format_latex_float(snapshot.lepton_theta13_two_loop_deg),
        "leptonThetaTwentyThreeBetaTwoLoop": _format_latex_float(snapshot.lepton_theta23_two_loop_deg),
        "leptonDeltaBetaTwoLoop": _format_latex_float(snapshot.lepton_delta_two_loop_deg),
        "mZeroBenchmarkMeV": _format_latex_float(snapshot.neutrino_floor_mev),
        "PredictedAlphaInverse": _format_latex_float(universal_snapshot.topological_alpha_inverse),
        "PredictedMassRatio": _format_latex_float(snapshot.proton_electron_mass_ratio),
        "PredictedNeutrinoFloorMeV": _format_latex_float(snapshot.neutrino_floor_mev),
        "quarkThetaTwelveBetaTwoLoop": _format_latex_float(snapshot.quark_theta12_two_loop_deg),
        "quarkThetaThirteenBetaTwoLoop": _format_latex_float(snapshot.quark_theta13_two_loop_deg),
        "quarkThetaTwentyThreeBetaTwoLoop": _format_latex_float(snapshot.quark_theta23_two_loop_deg),
        "quarkDeltaBetaTwoLoop": _format_latex_float(snapshot.quark_delta_two_loop_deg),
    }
    for macro_name, macro_value in macro_updates.items():
        physics_text = _replace_macro(physics_text, macro_name, macro_value)

    legacy_macro_updates = {
        "FineStructureInverse": _format_latex_float(universal_snapshot.topological_alpha_inverse),
        "NeutrinoFloor": _format_latex_float(snapshot.neutrino_floor_mev),
    }
    for macro_name, macro_value in legacy_macro_updates.items():
        physics_text = _replace_macro(
            physics_text,
            macro_name,
            macro_value,
            append_if_missing=False,
        )

    physics_text = _replace_or_insert_physics_constants_block(
        physics_text,
        _render_managed_macro_block(snapshot),
    )
    resolved_path.write_text(physics_text, encoding="utf-8")
    return resolved_path


def synchronize_system(
    *,
    residuals_path: Path = DEFAULT_RESIDUALS_PATH,
    readme_path: Path = DEFAULT_README_PATH,
    physics_constants_path: Path = DEFAULT_PHYSICS_CONSTANTS_PATH,
    physics_profile_path: Path | None = None,
    universal_constants_path: Path = DEFAULT_UNIVERSAL_CONSTANTS_PATH,
) -> tuple[Path, Path]:
    payload = _load_residual_payload(residuals_path)
    resolved_physics_profile_path = _resolve_sync_path(
        physics_profile_path if physics_profile_path is not None else universal_constants_path,
        base_dir=ProjectPaths.CONFIG,
    )
    universal_snapshot = _build_universal_constants_snapshot(
        ConfigLoader(physics_profile_path=resolved_physics_profile_path)
    )
    snapshot = build_sync_snapshot_with_universal_constants(
        payload,
        universal_snapshot=universal_snapshot,
    )
    updated_readme = sync_readme(
        readme_path=Path(readme_path),
        snapshot=snapshot,
        universal_snapshot=universal_snapshot,
    )
    updated_physics_constants = sync_physics_constants(
        physics_constants_path=Path(physics_constants_path),
        snapshot=snapshot,
        universal_snapshot=universal_snapshot,
    )
    return updated_readme, updated_physics_constants


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals-path", type=Path, default=DEFAULT_RESIDUALS_PATH)
    parser.add_argument("--readme-path", type=Path, default=DEFAULT_README_PATH)
    parser.add_argument("--physics-constants-path", type=Path, default=DEFAULT_PHYSICS_CONSTANTS_PATH)
    parser.add_argument("--physics-profile-path", type=Path, default=DEFAULT_PHYSICS_PROFILE_PATH)
    parser.add_argument("--universal-constants-path", dest="legacy_universal_constants_path", type=Path, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    updated_readme, updated_physics_constants = synchronize_system(
        residuals_path=args.residuals_path,
        readme_path=args.readme_path,
        physics_constants_path=args.physics_constants_path,
        physics_profile_path=args.legacy_universal_constants_path or args.physics_profile_path,
    )
    print(f"README synced                 : {updated_readme}")
    print(f"physics_constants synced      : {updated_physics_constants}")
    return 0


__all__ = [
    "DEFAULT_PHYSICS_CONSTANTS_PATH",
    "DEFAULT_PHYSICS_PROFILE_PATH",
    "DEFAULT_README_PATH",
    "DEFAULT_RESIDUALS_PATH",
    "DEFAULT_UNIVERSAL_CONSTANTS_PATH",
    "SyncSnapshot",
    "UniversalConstantsSnapshot",
    "build_sync_snapshot",
    "build_sync_snapshot_with_universal_constants",
    "_build_universal_constants_snapshot",
    "main",
    "parse_args",
    "sync_physics_constants",
    "sync_readme",
    "synchronize_system",
]


if __name__ == "__main__":
    raise SystemExit(main())
