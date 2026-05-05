from __future__ import annotations

from shbt.core.algebra import (
    LOW_SU3_WEIGHTS,
    ModularKernel,
    charge_embedding,
    interference_holonomy_phase,
    jarlskog_invariant,
    pdg_parameters,
    pdg_unitary,
    polar_unitary,
    predict_delta_cp,
    rotation_23,
    su2_conformal_weight,
    su2_modular_s,
    su2_quantum_dimension,
    su2_total_quantum_dimension,
    su3_conformal_weight,
    su3_low_weight_block,
    su3_modular_s_entry,
    su3_weight_vector,
)


def sequence_bit_loading(*, lepton_level: int, quark_level: int) -> tuple[tuple[int, int], ...]:
    lepton_block = ModularKernel(int(lepton_level), "lepton").restricted_block()
    quark_block = ModularKernel(int(quark_level), "quark").restricted_block()
    ordering = [
        (
            (row_index, column_index),
            float(abs(lepton_block[row_index, column_index]) ** 2 * abs(quark_block[row_index, column_index]) ** 2),
        )
        for row_index in range(lepton_block.shape[0])
        for column_index in range(lepton_block.shape[1])
    ]
    ordering.sort(key=lambda item: (item[1], -item[0][0], -item[0][1]), reverse=True)
    return tuple(coordinate for coordinate, _ in ordering)

__all__ = [
    "LOW_SU3_WEIGHTS",
    "ModularKernel",
    "charge_embedding",
    "interference_holonomy_phase",
    "jarlskog_invariant",
    "pdg_parameters",
    "pdg_unitary",
    "polar_unitary",
    "predict_delta_cp",
    "rotation_23",
    "sequence_bit_loading",
    "su2_conformal_weight",
    "su2_modular_s",
    "su2_quantum_dimension",
    "su2_total_quantum_dimension",
    "su3_conformal_weight",
    "su3_low_weight_block",
    "su3_modular_s_entry",
    "su3_weight_vector",
]
