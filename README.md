# Static Holographic Boundary Theory (SHBT)

## A Formal Theory of Flavor from $SO(10)_{312}$ Boundary Anomalies

> SHBT presents the displayed Standard Model observables as mandatory residues of 4D gravity on a finite-capacity horizon, not as fittable coefficients in a phenomenological optimizer.

This repository contains the publication-facing benchmark engine, manuscript sources, supplementary audits, and exported artifacts for the anomaly-free fixed-parent branch

$$
(k_{\ell},k_q,K)=(26,8,312).
$$

Within the minimal canonical completion used here, the retained branch is isolated by the **Derived Uniqueness Theorem** and operationally audited by framing closure,

$$
\Delta_{\rm fr}=0.
$$

The central claim of the project is that the displayed flavor observables are not floating Yukawa targets. They are rigid outputs of boundary anomaly cancellation, modular-$T$ closure, finite-capacity holography, and the branch-fixed threshold lock. In this sense the repository is best read as a **verifier of a mathematically rigid benchmark** rather than a generic fit engine.

On the gravity side, the effective 4D image is written on the anomaly-free branch through the Levi-Civita connection of the emergent bulk metric. On the boundary side, the reference coset is treated as the unique **GKO Virasoro-orthogonal complement** of the visible stress tensor. The formal completion residue $c_{\rm dark}$ is labeled the **Topological Gravitational Ghost (TGG)**: a bookkeeping completion sector required for modular closure and boundary neutrality, not a freely scanned WIMP-like hidden sector.

## Abstract Summary

SHBT provides a mathematically rigid benchmark for Standard Model flavor data from the fixed parent $SO(10)_{312}$ boundary construction. The benchmark is retained over a disclosed 9,801-cell **Local Moat Audit** of the visible lattice, with the branch $(26,8,312)$ selected by framing closure and fixed-parent modular consistency rather than by numerical optimization. Once the Topological Coordinate triplet $(26,8,312)$, the representational admissibility constraint $64/312$, the matter weight $G_{\rm SM}=15$, and the observational boundary condition $N$ are fixed, the construction retains no detached continuous adjustment sector. The displayed Standard Model observables are therefore treated as mandatory residues of 4D gravity on a finite-capacity horizon rather than fittable coefficients.

## Mathematical Rigidity: Why the Benchmark is “Unfittable”

The benchmark is intentionally **unfittable** in the strict sense used throughout the manuscript and supplement.

- The branch $(26,8,312)$ is the anomaly-free local survivor of the fixed-parent audit.
- The branch-selection theorem is the **Derived Uniqueness Theorem**, operationally audited by the framing condition $\Delta_{\rm fr}=0$.
- Any discrete detuning of the structural coordinates reopens the framing anomaly. For example, a one-step shift such as $k_{\ell}: 26\to27$ renders the completed boundary partition function physically non-normalizable.
- The benchmark therefore does not admit a detached continuous fitting sector: changing the coordinates does not “improve the fit”; it exits the anomaly-free branch.

In the repository’s own language, the Standard Model is recovered as the **unique survivor of mathematical consistency** on the retained one-copy branch.

## Core Branch Data

| Quantity | Benchmark value | Interpretation |
| --- | --- | --- |
| Topological Coordinate triplet | $(26,8,312)$ | Anomaly-free fixed-parent branch |
| Matter weight | $G_{\rm SM}=15$ | Current-Algebra Neutrality count of visible charged Weyl channels |
| Representational admissibility ratio | $64/312$ | Scalar Ward Identity / basis-invertibility constraint |
| Geometric residue | $\kappa_{D_5}=0.988771051266$ | $D_5$ simplex hyperarea invariant |
| Formal completion residue | $c_{\rm dark}=24\,\Delta_{\rm mod}(26,8)\approx 3.3008$ | Topological Gravitational Ghost / modular-closure datum |

## Core Identities

### Unity of Scale Identity

$$
\Lambda_{\rm holo}=\frac{3\pi}{\kappa_{D_5}^4}G_N m_\nu^4.
$$

This identity is the gravity-side closure statement tying the holographic vacuum loading, Newton coupling, and lightest neutrino floor to the same anomaly-free branch datum.

### Representational Admissibility

$$
\frac{\langle \Sigma_{126}\rangle}{\langle \phi_{10}\rangle}=\frac{64}{312}.
$$

This ratio is fixed by the **Scalar Ward Identity** and serves as the benchmark structural matching point for the charged-sector map.

### Current-Algebra Neutrality

$$
G_{\rm SM}=15.
$$

The Standard Model generation weight is not an adjustable normalization. It is the visible-current neutrality count obtained by removing the singlet $\nu^c$ direction from one visible $16_F$.

### Threshold Lock

$$
M_{\rm match}\equiv M_{126}^{\rm match}=M_N e^{-\beta^2}.
$$

This closed threshold equation is what transports the CKM apex angle $\gamma$ without reopening a phenomenological threshold scan.

## Architecture of the Engine (`pub/tn.py`)

The repository’s numerical core lives in `pub/tn.py`, with the repository-root wrapper `tn.py` forwarding directly to `pub.tn.main`.

The engine acts as a **professional verifier** for the fixed branch rather than as a generic fit optimizer:

- It reproduces the manuscript tables and audit artifacts from the disclosed residues without manual tuning.
- It carries the full coupled one-loop PMNS/CKM transport with **SciPy’s implicit Radau IIA solver**.
- It is audited at the **$10^{-12}$ ODE-tolerance level**, with tighter tolerance sweeps used as a reproducibility cross-check.
- It exports the **RG Consistency Audit**, Standard Residual Pulls, matching-residual diagnostics, moat scans, and referee-facing evidence packets.
- It keeps the ultraviolet RCFT kernel and the low-energy phenomenology layer synchronized through the same branch-fixed transport map.

### The SVD Rigidity Shield

One of the key numerical results of the engine is the **SVD Rigidity Shield**. The scalar-sector matching ratio enters only through singular-value dressing; it does not relax the topological eigenvectors. In the supplementary runtime audit, the PMNS and CKM eigenvectors remain stable at the $10^{-12}\sigma$ scale even under a $\pm10\%$ deformation of the matched mass-sector settings. Concretely, the reported bound is

$$
\max_a |\Delta\theta_a|/\sigma_a < 5.6\times10^{-12},
$$

so the Higgs/VEV sector can renormalize singular values without turning the benchmark into a fit problem.

## Key Results

- **CKM / PMNS post-dictions:** The PMNS kernel descends from the $SU(2)_{26}$ modular block, while the CKM magnitudes arise from the coset-tunneling structure and the branch-fixed threshold closure.
- **Absolute mass scale:** The anomaly-free branch gives the theory-fixed neutrino floor

  $$
  m_\nu\approx 2.9\,\mathrm{meV},
  $$

  with the benchmark effective Majorana mass

  $$
  |m_{\beta\beta}|\approx 5.6\,\mathrm{meV}.
  $$

- **Topological baryogenesis:** The branch-fixed baryogenesis identity yields

  $$
  \eta_B\approx 6.4\times10^{-10}.
  $$

- **Normal Ordering benchmark:** Within the Minimal One-Copy Dictionary used here, Normal Ordering is the retained benchmark realization.
- **No detached hidden retuning sector:** The completion residue $c_{\rm dark}$ is a modular-closure datum and parity sink, not a free phenomenological dark-sector dial.

## Reproducing the Local Moat Audit and Residual Audit

From the **repository root**, run:

```bash
python tn.py --output-dir results/
```

This command drives the main benchmark pipeline and writes the repository-facing audit artifacts into `results/`.

In particular, it reproduces:

- the fixed-parent **Local Moat Audit** over the visible lattice,
- the benchmark **Residual Audit** / RG Consistency Audit outputs,
- the exported tables and figures used by the manuscript-facing supplement,
- the moat and anomaly maps that make the branch isolation visually explicit.

If you want the explicit benchmark-residue detuning pass only, the wrapper also supports:

```bash
python tn.py --residue-check --output-dir results/
```

The installed console entry points declared in `pyproject.toml` are:

```bash
so10-312-review --output-dir results/
so10-312-referee --output-dir results/
```

## Key Visualization

The most important branch-isolation figure is:

- `results/framing_gap_moat_heatmap.png`

This figure is the repository’s algebraic visualization of local branch isolation. It highlights the anomaly-free island inside the disclosed moat and makes the framing-closure statement $\Delta_{\rm fr}=0$ directly visible as a structural selection rule rather than a best-fit contour.

## Repository Map

- `../tn.py` — repository-root wrapper for the publication-facing driver.
- `tn.py` — main SHBT verifier and artifact generator.
- `tn.tex` — manuscript source.
- `supplementary.tex` — supplementary audit source.
- `gravity.tex` — gravity-side derivations, including the Unity of Scale Identity and Topological EFE backup.
- `config/benchmark_v1.yaml` — benchmark configuration.
- `data/nufit_5_3.json` — archived neutrino input intervals used by the phenomenology layer.
- `tests/` — regression tests for the verifier and export pipeline.

## Scope

SHBT is **static** in the deliberately narrow sense used by the manuscript: it is a fixed topological mapping from boundary RCFT data to bulk flavor observables on the retained anomaly-free branch. It is not presented here as a generic landscape fit, a freely scanning beyond-the-Standard-Model sector, or a complete dynamical quantum-gravity dual. The load-bearing claims are the anomaly-filtered benchmark construction, the PMNS/CKM texture map, the finite-capacity neutrino floor, the topological baryogenesis identity, and the disclosed benchmark-consistency audit.

## Primary Source Files in this Repository

For the load-bearing statements summarized above, the canonical source files are:

- `tn.tex` — abstract, Topological Coordinate statement, computational audit statement, and benchmark summary.
- `supplementary.tex` — Local Moat Audit, SVD Rigidity Shield, tolerance sweeps, Topological Identity Matrix, and residual audits.
- `gravity.tex` — gravity-side closure analysis, holographic surface tension, and Unity of Scale Identity.
- `tn.py` — numerical implementation of the branch-fixed identities and audit exports.

In short: this repository does not try to fit flavor by adding knobs. It documents, audits, and exports a rigid branch in which the observed benchmark data are treated as residues of anomaly-free $SO(10)_{312}$ boundary closure.
