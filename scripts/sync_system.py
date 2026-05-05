from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.main import derive_transport_curvature_audit
from shbt.paths import ProjectPaths


DEFAULT_RESIDUALS_PATH = ProjectPaths.RESULTS / "residuals.json"
DEFAULT_README_PATH = ProjectPaths.ROOT / "README.md"
DEFAULT_PHYSICS_CONSTANTS_PATH = ProjectPaths.PAPERS / "physics_constants.tex"

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


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Expected '{key}' to be a mapping in residual payload.")
    return value


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


def _load_residual_payload(path: Path) -> dict[str, Any]:
    resolved_path = Path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Residual ledger not found at {resolved_path}. Run the verifier to generate results/residuals.json first."
        )
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Residual ledger must decode to a JSON object.")
    return payload


def build_sync_snapshot(payload: dict[str, Any]) -> SyncSnapshot:
    benchmark_tuple = tuple(int(value) for value in payload.get("benchmark_tuple", (26, 8, 312)))
    if len(benchmark_tuple) != 3:
        raise ValueError("benchmark_tuple must contain exactly three integers.")

    unity = _require_mapping(payload, "unity_of_scale_identity")
    gauge = _require_mapping(payload, "gauge_residual_bookkeeping")
    informational_costs = _require_mapping(payload, "informational_costs")
    mixing_angle_drifts = _require_mapping(payload, "mixing_angle_drifts_deg")

    transport_curvature = derive_transport_curvature_audit(
        lepton_level=int(benchmark_tuple[0]),
        quark_level=int(benchmark_tuple[1]),
        parent_level=int(benchmark_tuple[2]),
    )

    return SyncSnapshot(
        benchmark_tuple=benchmark_tuple,
        epsilon_lambda=float(unity["epsilon_lambda"]),
        exact_epsilon_lambda=float(unity.get("exact_epsilon_lambda", unity["epsilon_lambda"])),
        register_noise_floor=float(unity["register_noise_floor"]),
        exact_register_noise_floor=float(unity.get("exact_register_noise_floor", unity["register_noise_floor"])),
        gauge_topological_alpha_inverse=float(gauge["topological_alpha_inverse"]),
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
    )


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


def _replace_or_insert_readme_block(readme_text: str, block: str) -> str:
    ledger_header = "## Derivation Ledger"
    if ledger_header not in readme_text:
        raise ValueError("README.md is missing the '## Derivation Ledger' section.")
    if README_SYNC_START in readme_text and README_SYNC_END in readme_text:
        managed_block = "\n".join(block.splitlines()[1:])
        return re.sub(
            rf"{re.escape(README_SYNC_START)}.*?{re.escape(README_SYNC_END)}",
            lambda _: managed_block,
            readme_text,
            flags=re.DOTALL,
        )

    insertion_anchor = "### Tier Classification"
    ledger_start = readme_text.index(ledger_header)
    insertion_index = readme_text.find(insertion_anchor, ledger_start)
    if insertion_index == -1:
        next_section_index = readme_text.find("\n## ", ledger_start + len(ledger_header))
        insertion_index = len(readme_text) if next_section_index == -1 else next_section_index
    return readme_text[:insertion_index].rstrip() + "\n\n" + block + "\n\n" + readme_text[insertion_index:].lstrip()


def sync_readme(*, readme_path: Path, snapshot: SyncSnapshot) -> Path:
    resolved_path = Path(readme_path)
    readme_text = resolved_path.read_text(encoding="utf-8")
    updated_text = _replace_or_insert_readme_block(readme_text, _render_readme_sync_block(snapshot))
    resolved_path.write_text(updated_text, encoding="utf-8")
    return resolved_path


def _replace_macro(text: str, macro_name: str, macro_value: str) -> str:
    pattern = re.compile(
        rf"^(?P<prefix>\\(?:newcommand|providecommand)\{{\\{re.escape(macro_name)}\}}\{{)(?P<value>.*?)(?P<suffix>\}})$",
        flags=re.MULTILINE,
    )

    def _replacement(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{macro_value}{match.group('suffix')}"

    updated_text, count = pattern.subn(_replacement, text, count=1)
    return updated_text if count else text + f"\n\\providecommand{{\\{macro_name}}}{{{macro_value}}}\n"


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


def sync_physics_constants(*, physics_constants_path: Path, snapshot: SyncSnapshot) -> Path:
    resolved_path = Path(physics_constants_path)
    physics_text = resolved_path.read_text(encoding="utf-8")

    macro_updates = {
        "alphaSurfBenchmarkDecimal": _format_latex_float(snapshot.gauge_topological_alpha_inverse),
        "alphaSurfBenchmarkRounded": f"{snapshot.gauge_topological_alpha_inverse:.3f}",
        "leptonThetaTwelveBetaTwoLoop": _format_latex_float(snapshot.lepton_theta12_two_loop_deg),
        "leptonThetaThirteenBetaTwoLoop": _format_latex_float(snapshot.lepton_theta13_two_loop_deg),
        "leptonThetaTwentyThreeBetaTwoLoop": _format_latex_float(snapshot.lepton_theta23_two_loop_deg),
        "leptonDeltaBetaTwoLoop": _format_latex_float(snapshot.lepton_delta_two_loop_deg),
        "quarkThetaTwelveBetaTwoLoop": _format_latex_float(snapshot.quark_theta12_two_loop_deg),
        "quarkThetaThirteenBetaTwoLoop": _format_latex_float(snapshot.quark_theta13_two_loop_deg),
        "quarkThetaTwentyThreeBetaTwoLoop": _format_latex_float(snapshot.quark_theta23_two_loop_deg),
        "quarkDeltaBetaTwoLoop": _format_latex_float(snapshot.quark_delta_two_loop_deg),
    }
    for macro_name, macro_value in macro_updates.items():
        physics_text = _replace_macro(physics_text, macro_name, macro_value)

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
) -> tuple[Path, Path]:
    payload = _load_residual_payload(Path(residuals_path))
    snapshot = build_sync_snapshot(payload)
    updated_readme = sync_readme(readme_path=Path(readme_path), snapshot=snapshot)
    updated_physics_constants = sync_physics_constants(
        physics_constants_path=Path(physics_constants_path),
        snapshot=snapshot,
    )
    return updated_readme, updated_physics_constants


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals-path", type=Path, default=DEFAULT_RESIDUALS_PATH)
    parser.add_argument("--readme-path", type=Path, default=DEFAULT_README_PATH)
    parser.add_argument("--physics-constants-path", type=Path, default=DEFAULT_PHYSICS_CONSTANTS_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    updated_readme, updated_physics_constants = synchronize_system(
        residuals_path=args.residuals_path,
        readme_path=args.readme_path,
        physics_constants_path=args.physics_constants_path,
    )
    print(f"README synced                 : {updated_readme}")
    print(f"physics_constants synced      : {updated_physics_constants}")
    return 0


__all__ = [
    "DEFAULT_PHYSICS_CONSTANTS_PATH",
    "DEFAULT_README_PATH",
    "DEFAULT_RESIDUALS_PATH",
    "SyncSnapshot",
    "build_sync_snapshot",
    "main",
    "parse_args",
    "sync_physics_constants",
    "sync_readme",
    "synchronize_system",
]


if __name__ == "__main__":
    raise SystemExit(main())
