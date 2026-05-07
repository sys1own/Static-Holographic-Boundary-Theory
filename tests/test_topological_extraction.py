from __future__ import annotations

import numpy as np
import pytest

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.derivation_api import TopologicalVacuum
from shbt.core.topological_extraction import (
    BENCHMARK_BOUNDARY_BRANCH,
    INTERNAL_EIGENVALUE_TOLERANCE,
    RADAU_IIA_METHOD,
    extract_mass_hierarchy_eigenvalue,
    initialize_radau_iia_boundary_problem,
    search_stable_transport_residues,
)


def test_radau_problem_is_initialized_on_benchmark_boundary_manifold() -> None:
    problem = initialize_radau_iia_boundary_problem()

    assert problem.vacuum.branch == BENCHMARK_BOUNDARY_BRANCH == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
    assert problem.solver_method == RADAU_IIA_METHOD
    assert problem.transport_operator.shape == (3, 3)
    assert np.allclose(problem.transport_operator, np.diag(np.diag(problem.transport_operator)))
    assert np.isclose(np.linalg.norm(problem.initial_state), 1.0)


def test_transport_operator_extracts_mass_hierarchy_without_external_anchor() -> None:
    extraction = search_stable_transport_residues()

    assert extraction.stable_residues[0] > 0
    assert extraction.stable_residues[1] > extraction.stable_residues[0]
    assert extraction.stable_residues[2] == extraction.problem.mass_hierarchy_mu
    assert extraction.mass_hierarchy_mode.eigenvalue == extract_mass_hierarchy_eigenvalue()
    assert extraction.mass_hierarchy_mode.relative_error <= INTERNAL_EIGENVALUE_TOLERANCE
    assert float(extraction.mass_hierarchy_mode.eigenvalue) == pytest.approx(
        float(extraction.problem.mass_hierarchy_mu),
        rel=1e-15,
    )
    assert extraction.numerical_eigenvalues[-1] == pytest.approx(float(extraction.problem.mass_hierarchy_mu), rel=1e-15)


def test_topological_extraction_rejects_nonbenchmark_branch() -> None:
    with pytest.raises(ValueError, match=r"anomaly-free \(26, 8, 312\) boundary manifold"):
        initialize_radau_iia_boundary_problem(vacuum=TopologicalVacuum(27, 8, 312))
