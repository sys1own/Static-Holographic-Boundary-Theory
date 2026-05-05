from __future__ import annotations

"""Informational-economy audit for the minimal one-copy character dictionary."""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pub.algebra import su2_total_quantum_dimension
    from pub.constants import LEPTON_LEVEL
else:
    from .algebra import su2_total_quantum_dimension
    from .constants import LEPTON_LEVEL


BOUNDARY_CHARACTERS = (0, 1, 2)
NORMAL_ORDERING_ASSIGNMENT = (0, 1, 2)
INVERTED_ORDERING_ASSIGNMENT = (1, 1, 0)
SINGULAR_FLOOR_TARGET = 1.0e-15


@dataclass(frozen=True)
class AsymptoticCharacterSupport:
    bulk_state_index: int
    genus_label: int
    support_slot: str
    tau_limit: str
    q_limit: float
    leading_term: str


@dataclass(frozen=True)
class DictionaryConfigurationAudit:
    label: str
    genus_assignment: tuple[int, int, int]
    support_map: tuple[AsymptoticCharacterSupport, ...]
    transition_matrix: np.ndarray
    overlap_matrix: np.ndarray
    determinant: float
    rank: int
    singular_values: tuple[float, float, float]
    sigma_min: float
    condition_number: float
    support_deficit: int
    required_dictionary_rank: int
    nullspace_witness: np.ndarray | None

    @property
    def full_rank(self) -> bool:
        return bool(self.rank == self.overlap_matrix.shape[0])


@dataclass(frozen=True)
class InformationalEconomyReport:
    tau_limit: str
    q_limit: float
    one_copy_boundary_basis: tuple[str, str, str]
    normal_ordering: DictionaryConfigurationAudit
    inverted_ordering: DictionaryConfigurationAudit
    beta_genus_spacing: float
    relaxed_proxy_gap: float
    redundancy_entropy_cost_nat: float
    unique_preferred_ordering: str
    verdict: str


def asymptotic_character_support(*, bulk_state_index: int, genus_label: int) -> AsymptoticCharacterSupport:
    resolved_genus_label = int(genus_label)
    return AsymptoticCharacterSupport(
        bulk_state_index=int(bulk_state_index),
        genus_label=resolved_genus_label,
        support_slot=rf"\chi_{{{resolved_genus_label}}}",
        tau_limit="tau -> i∞",
        q_limit=0.0,
        leading_term=rf"q^{{h_{{{resolved_genus_label}}}-c/24}} e^{{i {resolved_genus_label} phi}}",
    )


def support_identification_limit(genus_assignment: Sequence[int]) -> tuple[AsymptoticCharacterSupport, ...]:
    return tuple(
        asymptotic_character_support(bulk_state_index=index, genus_label=genus_label)
        for index, genus_label in enumerate(tuple(int(label) for label in genus_assignment), start=1)
    )


def transition_matrix_from_assignment(
    genus_assignment: Sequence[int],
    *,
    boundary_characters: Sequence[int] = BOUNDARY_CHARACTERS,
) -> np.ndarray:
    resolved_boundary_characters = tuple(int(label) for label in boundary_characters)
    support_rows = {rf"\chi_{{{label}}}": index for index, label in enumerate(resolved_boundary_characters)}
    support_map = support_identification_limit(genus_assignment)
    matrix = np.zeros((len(resolved_boundary_characters), len(support_map)), dtype=float)
    for column_index, support in enumerate(support_map):
        row_index = support_rows.get(support.support_slot)
        if row_index is None:
            raise ValueError(
                f"Support slot {support.support_slot} is not present in the one-copy boundary basis {resolved_boundary_characters}."
            )
        matrix[row_index, column_index] = 1.0
    return np.asarray(matrix, dtype=float)


def overlap_matrix_from_transition(transition_matrix: np.ndarray) -> np.ndarray:
    resolved_transition_matrix = np.asarray(transition_matrix, dtype=float)
    return np.asarray(resolved_transition_matrix.conjugate().T @ resolved_transition_matrix, dtype=float)


def _nullspace_witness(transition_matrix: np.ndarray) -> np.ndarray | None:
    resolved_transition_matrix = np.asarray(transition_matrix, dtype=float)
    duplicate_columns: dict[tuple[float, ...], list[int]] = {}
    for column_index in range(resolved_transition_matrix.shape[1]):
        key = tuple(float(value) for value in resolved_transition_matrix[:, column_index])
        duplicate_columns.setdefault(key, []).append(column_index)
    for indices in duplicate_columns.values():
        if len(indices) < 2:
            continue
        witness = np.zeros(resolved_transition_matrix.shape[1], dtype=float)
        witness[indices[0]] = 1.0 / math.sqrt(2.0)
        witness[indices[1]] = -1.0 / math.sqrt(2.0)
        return np.asarray(witness, dtype=float)
    return None


def build_dictionary_configuration_audit(
    *,
    label: str,
    genus_assignment: Sequence[int],
) -> DictionaryConfigurationAudit:
    resolved_assignment = tuple(int(label_value) for label_value in genus_assignment)
    transition_matrix = transition_matrix_from_assignment(resolved_assignment)
    overlap_matrix = overlap_matrix_from_transition(transition_matrix)
    singular_values_array = np.asarray(np.linalg.svd(overlap_matrix, compute_uv=False), dtype=float)
    rank = int(np.linalg.matrix_rank(overlap_matrix))
    sigma_min = float(np.min(singular_values_array)) if singular_values_array.size else 0.0
    condition_number = float(np.linalg.cond(overlap_matrix)) if singular_values_array.size else 1.0
    support_deficit = int(overlap_matrix.shape[0] - rank)
    required_dictionary_rank = int(overlap_matrix.shape[0] + support_deficit)
    return DictionaryConfigurationAudit(
        label=str(label),
        genus_assignment=resolved_assignment,
        support_map=support_identification_limit(resolved_assignment),
        transition_matrix=np.asarray(transition_matrix, dtype=float),
        overlap_matrix=np.asarray(overlap_matrix, dtype=float),
        determinant=float(np.linalg.det(overlap_matrix)),
        rank=rank,
        singular_values=tuple(float(value) for value in singular_values_array),
        sigma_min=sigma_min,
        condition_number=condition_number,
        support_deficit=support_deficit,
        required_dictionary_rank=required_dictionary_rank,
        nullspace_witness=_nullspace_witness(transition_matrix),
    )


def build_informational_economy_report() -> InformationalEconomyReport:
    normal_ordering = build_dictionary_configuration_audit(
        label="NO",
        genus_assignment=NORMAL_ORDERING_ASSIGNMENT,
    )
    inverted_ordering = build_dictionary_configuration_audit(
        label="IO/IH",
        genus_assignment=INVERTED_ORDERING_ASSIGNMENT,
    )
    beta = 0.5 * math.log(su2_total_quantum_dimension(int(LEPTON_LEVEL)))
    relaxed_proxy_gap = float(beta * beta)
    redundancy_entropy_cost_nat = float(inverted_ordering.support_deficit * math.log(2.0))

    assert normal_ordering.rank == 3, "The normal-ordering overlap map must remain full-rank in the one-copy dictionary."
    assert inverted_ordering.rank == 2, "The inverted-ordering overlap map must remain rank-deficient in the one-copy dictionary."
    assert inverted_ordering.sigma_min < SINGULAR_FLOOR_TARGET, (
        "The inverted-ordering smallest singular value must stay below the 1e-15 structural floor."
    )
    assert math.isclose(redundancy_entropy_cost_nat, math.log(2.0), rel_tol=0.0, abs_tol=0.0), (
        "The one-copy IO/IH redundancy entropy cost must remain Delta S_red = ln 2."
    )

    verdict = (
        "Informational Economy Report: PASS — NO is the unique preferred structural realization of the one-copy dictionary. "
        "The IO/IH support map is rank-deficient in the tau -> i∞ support-identification limit, "
        "so accommodating it requires a non-minimal extra support slot and incurs Delta S_red = ln 2."
    )
    return InformationalEconomyReport(
        tau_limit="tau -> i∞",
        q_limit=0.0,
        one_copy_boundary_basis=tuple(rf"\chi_{{{label}}}" for label in BOUNDARY_CHARACTERS),
        normal_ordering=normal_ordering,
        inverted_ordering=inverted_ordering,
        beta_genus_spacing=float(beta),
        relaxed_proxy_gap=relaxed_proxy_gap,
        redundancy_entropy_cost_nat=redundancy_entropy_cost_nat,
        unique_preferred_ordering="NO",
        verdict=verdict,
    )


def _render_matrix(matrix: np.ndarray) -> str:
    resolved_matrix = np.asarray(matrix, dtype=float)
    return np.array2string(
        resolved_matrix,
        formatter={"float_kind": lambda value: f"{float(value):.0f}"},
        separator=", ",
    )


def _render_assignment(assignment: Sequence[int]) -> str:
    return "(" + ", ".join(str(int(value)) for value in assignment) + ")"


def render_report(report: InformationalEconomyReport) -> str:
    def _render_audit(audit: DictionaryConfigurationAudit) -> list[str]:
        support_slots = ", ".join(support.support_slot for support in audit.support_map)
        nullspace_text = "none"
        if audit.nullspace_witness is not None:
            nullspace_text = np.array2string(
                np.asarray(audit.nullspace_witness, dtype=float),
                formatter={"float_kind": lambda value: f"{float(value):+.6f}"},
                separator=", ",
            )
        return [
            f"{audit.label} genus assignment         : {_render_assignment(audit.genus_assignment)}",
            f"{audit.label} asymptotic support map   : {support_slots}",
            f"{audit.label} transition matrix T      : {_render_matrix(audit.transition_matrix)}",
            f"{audit.label} overlap matrix O=T^dag T : {_render_matrix(audit.overlap_matrix)}",
            (
                f"{audit.label} spectrum                : rank={audit.rank}, det={audit.determinant:.6e}, "
                f"sv={audit.singular_values}, sigma_min={audit.sigma_min:.6e}, kappa2={audit.condition_number:.6e}"
            ),
            f"{audit.label} nullspace witness       : {nullspace_text}",
        ]

    lines = [
        "Informational Economy Report",
        "============================",
        f"Support-identification limit  : {report.tau_limit}, q -> {report.q_limit:.1f}",
        (
            "Asymptotic character logic    : chi_g(tau,phi) ~ q^(h_g-c/24) e^(i g phi), so the tau -> i∞ limit isolates primary support slots."
        ),
        f"One-copy boundary basis       : {report.one_copy_boundary_basis}",
        "",
        "Dictionary Construction",
        "-----------------------",
        *_render_audit(report.normal_ordering),
        *_render_audit(report.inverted_ordering),
        "",
        "Rank-Deficiency Proof",
        "---------------------",
        f"NO overlap verdict            : full-rank={int(report.normal_ordering.full_rank)} with rank {report.normal_ordering.rank}",
        (
            "IO/IH overlap verdict         : "
            f"full-rank={int(report.inverted_ordering.full_rank)} with rank {report.inverted_ordering.rank} and "
            f"sigma_min={report.inverted_ordering.sigma_min:.6e} < {SINGULAR_FLOOR_TARGET:.0e}"
        ),
        (
            "IO/IH non-minimality          : "
            f"support deficit={report.inverted_ordering.support_deficit}, required dictionary rank={report.inverted_ordering.required_dictionary_rank}"
        ),
        "",
        "Penalty Quantification",
        "----------------------",
        f"beta = 0.5 ln D_{int(LEPTON_LEVEL)}         : {report.beta_genus_spacing:.12f}",
        f"Relaxed proxy gap beta^2      : {report.relaxed_proxy_gap:.12f}",
        f"Redundancy entropy cost       : Delta S_red = ln 2 = {report.redundancy_entropy_cost_nat:.12f}",
        "",
        report.verdict,
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    parse_args(argv)
    print(render_report(build_informational_economy_report()))


if __name__ == "__main__":
    main()
