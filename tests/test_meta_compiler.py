from __future__ import annotations

import math

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.main import surface_tension_gauge_alpha_inverse, topological_mass_coordinate_ev
from shbt.meta_compiler import (
    MetaCompiler,
    build_graph_translation,
    build_meta_kernel,
    build_qft_translation,
    tau,
    τ,
)


def test_meta_kernel_locks_benchmark_branch_and_residues() -> None:
    kernel = build_meta_kernel()

    assert kernel.branch == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
    assert kernel.vev_residue_display == "64/312"
    assert math.isclose(kernel.gauge_alpha_inverse, surface_tension_gauge_alpha_inverse(), rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(kernel.neutrino_mass_coordinate_ev, topological_mass_coordinate_ev(), rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(kernel.holographic_bit_budget, HOLOGRAPHIC_BITS, rel_tol=0.0, abs_tol=0.0)


def test_tau_to_qft_compiles_branch_into_lagrangian_terms() -> None:
    translation = build_qft_translation()
    rendered = tau.to_qft()

    assert [term.name for term in translation.terms] == [
        "gauge_kinetic",
        "fermion_kinetic",
        "yukawa_alignment",
        "majorana_mass",
        "gut_threshold_matching",
        "dark_completion",
    ]
    assert translation.kernel.branch == (26, 8, 312)
    assert "L_eff[(26, 8, 312)]" in translation.lagrangian_density
    assert "64/312" in translation.lagrangian_density
    assert "F^a_{mu nu} F_a^{mu nu}" in rendered
    assert "O_threshold" in rendered
    assert "O_dark" in rendered


def test_tau_to_graph_compiles_branch_into_node_string_paths() -> None:
    translation = build_graph_translation()
    rendered = tau.to_graph()

    assert translation.kernel.branch == (26, 8, 312)
    assert len(translation.nodes) >= 12
    assert len(translation.edges) >= 12
    assert any("kernel:(26, 8, 312)" in node_string for node_string in translation.node_strings)
    assert any("qft:gauge_kinetic" in node_string for node_string in translation.node_strings)
    assert any("agi:memory_budget" in node_string for node_string in translation.node_strings)
    assert "Meta-Compiler Graph Translation" in rendered


def test_meta_compiler_aliases_are_exposed() -> None:
    assert isinstance(tau, MetaCompiler)
    assert τ is tau
