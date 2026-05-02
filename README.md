# Static Holographic Boundary Theory (SHBT)

SHBT is a formal flavor construction built on the fixed-parent $SO(10)_{312}$ boundary benchmark. This repository treats Standard Model flavor observables as mandatory residues of 4D gravity on a finite-capacity horizon rather than as coefficients to be optimized by a detached fit sector.

## Project Overview

SHBT provides a mathematically rigid benchmark for Standard Model flavor data derived from the fixed-parent $SO(10)_{312}$ boundary construction. Unlike models organized around numerical optimization, the benchmark branch $(26, 8, 312)$ is retained by the **Derived Uniqueness Theorem**, operationally audited through the framing-closure condition

$$
\Delta_{\text{fr}} = 0
$$

Within the retained one-copy branch, modular closure, boundary neutrality, and finite-capacity holography are solved together. The result is a benchmark in which the displayed observables are structural outputs of anomaly-free boundary data.

## Mathematical Rigidity

The SHBT benchmark contains **Zero Free Parameters** in the strict sense used by the manuscript and supplement. Once the following topological residues are fixed, the construction retains no continuous adjustment sector:

- **Topological Coordinate Triplet:** $(26,8,312)$.
- **Representational Admissibility Constraint:**

  $$
  \frac{\langle \Sigma_{126}\rangle}{\langle \phi_{10}\rangle}=\frac{64}{312}.
  $$

- **Current-Algebra Neutrality (Matter Weight):**

  $$
  G_{SM}=15
  $$.

- **Observational Boundary Condition ($N$):** the finite holographic information budget anchored to the observed cosmological constant $\Lambda_{obs}$.

This is the sense in which the benchmark is **unfittable**. A discrete detuning of the branch coordinates is not a mild parameter variation; it exits the anomaly-free branch. For example, a one-step shift such as $k_{\ell}:26\to27$ forces

$$
\Delta_{fr}\neq 0,
$$

reopens the framing anomaly, and renders the completed boundary partition function physically non-normalizable.

## Universal Source Code & Verifier (`tn.py`)

This repository acts as the universal source code for the SHBT benchmark. The repository-root driver `../tn.py` forwards to `pub/tn.py`, which serves as the integrated professional verifier for the branch-fixed construction.

The verifier:

- reproduces the manuscript tables and benchmark artifacts from disclosed branch residues,
- carries the full coupled one-loop PMNS/CKM transport,
- uses SciPy's implicit Radau IIA solver,
- exports the Local Moat Audit, residual diagnostics, and referee-facing evidence packet,
- and operates at the $10^{-12}$ ODE-tolerance level used by the computational audit.

The code should therefore be read as a numerical verifier of branch-fixed analytic identities, not as a generic fit engine with floating Yukawa or threshold knobs.

### SVD Rigidity Shield

The verifier demonstrates the **SVD Rigidity Shield**: the scalar matching sector dresses singular values without relaxing the primary flavor eigenvectors. In the runtime audit, the PMNS and CKM mixing angles remain stable at the $10^{-12}\sigma$ level even under $\pm10\%$ deformations of the matched mass-sector settings. This is the repository's explicit numerical statement of **Eigenvector Rigidity**.

## Key Results

- **Neutrino Floor:**

  $$
  m_{\nu}\approx 2.9\,\text{meV},
  $$

  with

  $$
  |m_{\beta\beta}|\approx 5.6\,\text{meV}.
  $$

- **Topological Baryogenesis:**

  $$
  \eta_{B}\approx 6.4\times10^{-10}.
  $$

- **Unity of Scale Identity:**

  $$
  \Lambda_{holo}=\frac{3\pi}{\kappa_{D_5}^{4}}G_{N}m_{\nu}^{4}.
  $$

- **Rigid CKM / PMNS Textures:** the PMNS kernel descends from the $SU(2)_{26}$ modular block, while the CKM magnitudes and transported apex are locked by the branch-fixed threshold closure.

## Computational Audit Statement

All numerical values, benchmark tables, and $\chi^{2}$ pulls quoted in the manuscript are derived from a reproducible evaluation of the $SO(10)_{312}$ branch residues contained in this repository. The audit stack in `pub/tn.py` reproduces the benchmark from disclosed structural inputs rather than from hidden numerical retuning.

On the boundary side, the reference coset is treated as the unique **GKO Virasoro-orthogonal complement** of the visible stress tensor. On the gravity side, the anomaly-free branch is read through the Levi-Civita connection of the emergent bulk metric. The formal completion residue $c_{dark}$ is recorded as the **Topological Gravitational Ghost**: a bookkeeping completion sector required for modular closure and boundary neutrality, not a freely scanned particle-relic sector.

## Usage

Run the full local moat audit and residual audit from the repository root:

```bash
python tn.py --output-dir results/
```

This command writes the benchmark-facing artifacts into `results/`, including the moat scan, residual diagnostics, and publication-facing exports.

The key local-isolation visualization is:

```text
results/framing_gap_moat_heatmap.png
```

which acts as the algebraic visualization of local branch isolation around the anomaly-free island.

## Source Trail

The primary manuscript-aligned source files in `pub/` are:

- `tn.py` — universal verifier and artifact generator,
- `tn.tex` — main manuscript source,
- `supplementary.tex` — Local Moat Audit, SVD Rigidity Shield, residual audits, and numerical stability appendices,
- `gravity.tex` — gravity-side derivations, including the holographic closure chain and the Unity of Scale Identity.

In short, SHBT does not fit flavor by introducing new knobs. It audits a rigid anomaly-free branch and treats the displayed Standard Model observables as mandatory residues of $SO(10)_{312}$ boundary closure.
