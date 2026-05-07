from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import (
    BENCHMARK_C_DARK_RESIDUE,
    GEOMETRIC_KAPPA,
    HOLOGRAPHIC_BITS,
    LEPTON_LEVEL,
    PARENT_LEVEL,
    QUARK_LEVEL,
    R_GUT,
    SU2_DUAL_COXETER,
)
from shbt.main import BENCHMARK_VEV_RATIO, surface_tension_gauge_alpha_inverse, topological_mass_coordinate_ev


BENCHMARK_BRANCH = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))


def _format_float(value: float, *, digits: int = 12) -> str:
    resolved_value = float(value)
    if not math.isfinite(resolved_value):
        return str(resolved_value)
    magnitude = abs(resolved_value)
    if math.isclose(magnitude, 0.0, rel_tol=0.0, abs_tol=0.0):
        return "0"
    if magnitude >= 1.0e6 or magnitude < 1.0e-4:
        return f"{resolved_value:.6e}"
    return f"{resolved_value:.{digits}f}".rstrip("0").rstrip(".")


def _ordered_unique(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


@dataclass(frozen=True)
class MetaKernel:
    branch: tuple[int, int, int]
    vev_residue_fraction: Fraction
    vev_residue_display: str
    gauge_alpha_inverse: float
    gauge_coupling_squared: float
    gauge_kinetic_prefactor: float
    neutrino_mass_coordinate_ev: float
    kappa_d5: float
    holographic_bit_budget: float
    gut_threshold_residue: float
    gut_threshold_display: str
    c_dark_completion: float
    statement: str


@dataclass(frozen=True)
class LagrangianTerm:
    name: str
    sector: str
    coefficient: str
    operator: str
    interpretation: str

    @property
    def expression(self) -> str:
        return f"{self.coefficient} {self.operator}"


@dataclass(frozen=True)
class QFTTranslation:
    kernel: MetaKernel
    terms: tuple[LagrangianTerm, ...]
    lagrangian_density: str
    statement: str


@dataclass(frozen=True)
class GraphEdge:
    source: str
    relation: str
    target: str

    @property
    def node_string(self) -> str:
        return f"{self.source} -[{self.relation}]-> {self.target}"


@dataclass(frozen=True)
class GraphTranslation:
    kernel: MetaKernel
    nodes: tuple[str, ...]
    edges: tuple[GraphEdge, ...]
    node_strings: tuple[str, ...]
    statement: str


def build_meta_kernel() -> MetaKernel:
    branch = BENCHMARK_BRANCH
    vev_residue_fraction = Fraction(BENCHMARK_VEV_RATIO)
    gauge_alpha_inverse = float(surface_tension_gauge_alpha_inverse())
    gauge_coupling_squared = float((4.0 * math.pi) / gauge_alpha_inverse)
    gauge_kinetic_prefactor = float(gauge_alpha_inverse / (16.0 * math.pi))
    neutrino_mass_coordinate = float(topological_mass_coordinate_ev())
    vev_residue_display = f"{8 * QUARK_LEVEL}/{PARENT_LEVEL}"
    gut_threshold_display = f"{QUARK_LEVEL}/{LEPTON_LEVEL + SU2_DUAL_COXETER}"
    statement = (
        "The benchmark branch (26, 8, 312) compiles into a reusable meta-language: "
        "the same kernel can be rendered as QFT operators or as AGI-facing node-string graphs."
    )
    return MetaKernel(
        branch=branch,
        vev_residue_fraction=vev_residue_fraction,
        vev_residue_display=vev_residue_display,
        gauge_alpha_inverse=gauge_alpha_inverse,
        gauge_coupling_squared=gauge_coupling_squared,
        gauge_kinetic_prefactor=gauge_kinetic_prefactor,
        neutrino_mass_coordinate_ev=neutrino_mass_coordinate,
        kappa_d5=float(GEOMETRIC_KAPPA),
        holographic_bit_budget=float(HOLOGRAPHIC_BITS),
        gut_threshold_residue=float(R_GUT),
        gut_threshold_display=gut_threshold_display,
        c_dark_completion=float(BENCHMARK_C_DARK_RESIDUE),
        statement=statement,
    )


def build_qft_translation(*, kernel: MetaKernel | None = None) -> QFTTranslation:
    resolved_kernel = build_meta_kernel() if kernel is None else kernel
    terms = (
        LagrangianTerm(
            name="gauge_kinetic",
            sector="gauge",
            coefficient=f"-({_format_float(resolved_kernel.gauge_alpha_inverse)}/(16*pi))",
            operator="F^a_{mu nu} F_a^{mu nu}",
            interpretation="The visible gauge-density residue fixes the gauge kinetic normalization.",
        ),
        LagrangianTerm(
            name="fermion_kinetic",
            sector="matter",
            coefficient="+1",
            operator="bar(psi) i gamma^mu D_mu psi",
            interpretation="Matter transport stays canonically normalized on the anomaly-free branch.",
        ),
        LagrangianTerm(
            name="yukawa_alignment",
            sector="higgs",
            coefficient=f"-({resolved_kernel.vev_residue_display}) y_f",
            operator="bar(psi_L) H psi_R + h.c.",
            interpretation="The algebraic mass hierarchy appears as the branch-fixed Yukawa alignment residue.",
        ),
        LagrangianTerm(
            name="majorana_mass",
            sector="neutrino",
            coefficient=f"-(1/2) ({_format_float(resolved_kernel.neutrino_mass_coordinate_ev)} eV)",
            operator="nu_L^T C nu_L + h.c.",
            interpretation="The topological neutrino coordinate compiles into a Majorana closure term.",
        ),
        LagrangianTerm(
            name="gut_threshold_matching",
            sector="matching",
            coefficient=f"+({resolved_kernel.gut_threshold_display})",
            operator="O_threshold",
            interpretation="The branch-fixed transport residue becomes the threshold-matching Wilson weight.",
        ),
        LagrangianTerm(
            name="dark_completion",
            sector="completion",
            coefficient=f"+{_format_float(resolved_kernel.c_dark_completion)}",
            operator="O_dark",
            interpretation="The positive parity-sink residue closes the neutral completion channel.",
        ),
    )
    lagrangian_density = "\n".join(
        (
            f"L_eff[{resolved_kernel.branch}] =",
            *(f"  {term.expression}" for term in terms),
            (
                f"  [kappa_D5={_format_float(resolved_kernel.kappa_d5)}, "
                f"g_surf^2={_format_float(resolved_kernel.gauge_coupling_squared)}, "
                f"N_holo={_format_float(resolved_kernel.holographic_bit_budget)}]"
            ),
        )
    )
    statement = (
        "The translation function tau.to_qft() turns the branch residues into operator-level QFT syntax "
        "without reopening phenomenological fit directions."
    )
    return QFTTranslation(
        kernel=resolved_kernel,
        terms=terms,
        lagrangian_density=lagrangian_density,
        statement=statement,
    )


def build_graph_translation(*, kernel: MetaKernel | None = None) -> GraphTranslation:
    resolved_kernel = build_meta_kernel() if kernel is None else kernel
    kernel_node = f"kernel:{resolved_kernel.branch}"
    gauge_node = f"residue:alpha_surface^-1={_format_float(resolved_kernel.gauge_alpha_inverse)}"
    vev_node = f"residue:vev={resolved_kernel.vev_residue_display}"
    mass_node = f"residue:m_nu={_format_float(resolved_kernel.neutrino_mass_coordinate_ev)}eV"
    threshold_node = f"residue:R_GUT={resolved_kernel.gut_threshold_display}"
    complexity_node = f"residue:N_holo={_format_float(resolved_kernel.holographic_bit_budget)}"
    completion_node = f"residue:c_dark={_format_float(resolved_kernel.c_dark_completion)}"

    qft_gauge_node = "qft:gauge_kinetic"
    qft_yukawa_node = "qft:yukawa_alignment"
    qft_majorana_node = "qft:majorana_mass"
    qft_threshold_node = "qft:threshold_matching"
    qft_completion_node = "qft:dark_completion"

    agi_world_node = "agi:world_model"
    agi_alignment_node = "agi:representation_alignment"
    agi_memory_node = "agi:memory_budget"
    agi_error_correction_node = "agi:error_correction"
    agi_guard_node = "agi:null_channel_guard"
    agi_router_node = "agi:router"

    edges = (
        GraphEdge(kernel_node, "fixes", gauge_node),
        GraphEdge(kernel_node, "fixes", vev_node),
        GraphEdge(kernel_node, "fixes", mass_node),
        GraphEdge(kernel_node, "fixes", threshold_node),
        GraphEdge(kernel_node, "fixes", complexity_node),
        GraphEdge(kernel_node, "fixes", completion_node),
        GraphEdge(gauge_node, "normalizes", qft_gauge_node),
        GraphEdge(vev_node, "aligns", qft_yukawa_node),
        GraphEdge(mass_node, "closes", qft_majorana_node),
        GraphEdge(threshold_node, "weights", qft_threshold_node),
        GraphEdge(completion_node, "stabilizes", qft_completion_node),
        GraphEdge(qft_gauge_node, "exposes", agi_world_node),
        GraphEdge(qft_yukawa_node, "exposes", agi_alignment_node),
        GraphEdge(qft_majorana_node, "exposes", agi_error_correction_node),
        GraphEdge(complexity_node, "bounds", agi_memory_node),
        GraphEdge(qft_completion_node, "guards", agi_guard_node),
        GraphEdge(qft_threshold_node, "routes", agi_router_node),
    )
    node_strings = (
        f"{kernel_node} -> {gauge_node} -> {qft_gauge_node} -> {agi_world_node}",
        f"{kernel_node} -> {vev_node} -> {qft_yukawa_node} -> {agi_alignment_node}",
        f"{kernel_node} -> {mass_node} -> {qft_majorana_node} -> {agi_error_correction_node}",
        f"{kernel_node} -> {threshold_node} -> {qft_threshold_node} -> {agi_router_node}",
        f"{kernel_node} -> {complexity_node} -> {agi_memory_node}",
        f"{kernel_node} -> {completion_node} -> {qft_completion_node} -> {agi_guard_node}",
    )
    nodes = _ordered_unique(
        [
            *(edge.source for edge in edges),
            *(edge.target for edge in edges),
        ]
    )
    statement = (
        "The translation function tau.to_graph() rewrites SHBT residues as node-string graphs that AGI systems can use "
        "for routing, memory-budgeting, alignment, and null-channel guards."
    )
    return GraphTranslation(
        kernel=resolved_kernel,
        nodes=nodes,
        edges=edges,
        node_strings=node_strings,
        statement=statement,
    )


def render_qft_translation(translation: QFTTranslation) -> str:
    lines = [
        "Meta-Compiler QFT Translation",
        "============================",
        f"Benchmark kernel           : {translation.kernel.branch}",
        f"VEV residue               : {translation.kernel.vev_residue_display}",
        f"Gauge alpha inverse       : {_format_float(translation.kernel.gauge_alpha_inverse)}",
        f"Neutrino coordinate [eV]  : {_format_float(translation.kernel.neutrino_mass_coordinate_ev)}",
        f"Holographic bit budget    : {_format_float(translation.kernel.holographic_bit_budget)}",
        "",
        "Lagrangian Terms",
        "----------------",
        *(f"- {term.name}: {term.expression}" for term in translation.terms),
        "",
        translation.lagrangian_density,
        "",
        translation.statement,
    ]
    return "\n".join(lines)


def render_graph_translation(translation: GraphTranslation) -> str:
    lines = [
        "Meta-Compiler Graph Translation",
        "==============================",
        f"Benchmark kernel : {translation.kernel.branch}",
        f"Node count        : {len(translation.nodes)}",
        f"Edge count        : {len(translation.edges)}",
        "",
        "Node Strings",
        "------------",
        *(f"- {node_string}" for node_string in translation.node_strings),
        "",
        translation.statement,
    ]
    return "\n".join(lines)


class TranslationFunction:
    def __init__(self, *, kernel: MetaKernel | None = None) -> None:
        self.kernel = build_meta_kernel() if kernel is None else kernel

    def to_qft(self, *, structured: bool = False) -> QFTTranslation | str:
        translation = build_qft_translation(kernel=self.kernel)
        return translation if structured else render_qft_translation(translation)

    def to_graph(self, *, structured: bool = False) -> GraphTranslation | str:
        translation = build_graph_translation(kernel=self.kernel)
        return translation if structured else render_graph_translation(translation)


MetaCompiler = TranslationFunction


def build_meta_compiler() -> TranslationFunction:
    return TranslationFunction()


tau = TranslationFunction()
τ = tau


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile the SHBT benchmark kernel into accessible frameworks.")
    parser.add_argument(
        "--format",
        choices=("qft", "graph", "both"),
        default="both",
        help="Choose whether to emit the QFT translation, the AGI graph translation, or both.",
    )
    args = parser.parse_args(tuple(argv) if argv is not None else None)

    if args.format in {"qft", "both"}:
        print(tau.to_qft())
    if args.format == "both":
        print("")
    if args.format in {"graph", "both"}:
        print(tau.to_graph())
    return 0


__all__ = [
    "BENCHMARK_BRANCH",
    "GraphEdge",
    "GraphTranslation",
    "LagrangianTerm",
    "MetaCompiler",
    "MetaKernel",
    "QFTTranslation",
    "TranslationFunction",
    "build_graph_translation",
    "build_meta_compiler",
    "build_meta_kernel",
    "build_qft_translation",
    "main",
    "render_graph_translation",
    "render_qft_translation",
    "tau",
    "τ",
]


if __name__ == "__main__":
    raise SystemExit(main())
