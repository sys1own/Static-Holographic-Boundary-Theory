# Static Holographic Boundary Theory: 
## Universal Source Code v1.0

Static Holographic Boundary Theory (SHBT) is an executable proof architecture for a **Zero Free Parameter** boundary construction. In SHBT, Standard Model observables are treated as mandatory residues of effective 4D gravity on a finite-capacity horizon, not as outputs of a detached fit sector. The retained construction is anchored to the unique anomaly-free branch $(26, 8, 312)$, and every benchmark artifact in this repository is read as a consequence of that branch-fixed closure.

The benchmark constants are tracked in `pub/constants.py` as a strict three-tier catalog.

- `TIER_1_TOPOLOGICAL_COORDINATES` contains the branch-defining discrete coordinates: `LEPTON_LEVEL = 26`, `QUARK_LEVEL = 8`, `PARENT_LEVEL = 312`, and `G_SM = 15`.
- `TIER_2_OBSERVATIONAL_BOUNDARY_CONDITIONS` contains the external observational anchors supplied to the verifier, including the Planck 2018 late-time cosmology inputs and the electroweak gauge data.
- `TIER_3_DERIVED_RESIDUES` contains the quantities fixed once Tier 1 is chosen and Tier 2 is supplied, including `GEOMETRIC_KAPPA`, `HOLOGRAPHIC_BITS`, `R_GUT`, and the positive `c_{\rm dark}` completion residue.

This is the precise sense in which SHBT retains no continuous phenomenological fit coordinates: Tier 1 identifies the unique anomaly-free cell, Tier 2 is boundary data rather than model freedom, and Tier 3 is computed consequence rather than adjustable input. SHBT is therefore presented here as executable proof, not as a fittable model.


## Framework Identity

SHBT formalizes the claim that the Standard Model is the visible residue of a finite-information holographic boundary. The branch $(26, 8, 312)$ is not an empirically chosen coordinate triple; it is the unique anomaly-free survivor of the local moat after Diophantine closure, framing closure, and Virasoro-orthogonal completion are imposed simultaneously. The repository is organized to prove that uniqueness, propagate it through transport, and expose every remaining residual as a disclosed audit quantity.

This is the precise sense in which SHBT is a Zero Free Parameter architecture. A detuning does not generate a nearby admissible model. It exits the only branch that preserves anomaly cancellation, boundary neutrality, and bulk consistency.

## The Reviewer Trap

The reviewer trap is the gravity-side statement that off-shell detuning is not a harmless robustness exercise. In `noether_bridge.py`, the benchmark branch satisfies

$$
\Delta_{\rm fr} = 0,
$$

so the closure tensor vanishes on-shell and the anomalous bulk source remains absent. Detune any coordinate and the framing defect reopens, the closure tensor becomes non-zero, an anomalous source $J_{\mu\nu}^{(a)}$ is induced, and the Equivalence Principle in the bulk is destroyed. In SHBT, attempted coordinate detuning is therefore a proof of failure, not a path to a better fit. The sealing clause is explicit: if a coordinate detuning drives any Tier 3 Derived Residue beyond the quantified two-loop limits disclosed in `results/residuals.json`, then the detuned proposal is treated as falsified rather than as a nearby admissible variant of the benchmark branch.

## Unity of Scale

The gravity sector hardening makes explicit that the cosmological constant and the neutrino floor are locked by the same finite bit-capacity:

$$
N = \frac{3\pi}{L_P^2 \Lambda_{\rm holo}},
\qquad
m_{\nu} = \kappa_{D_5} M_P N^{-1/4},
\qquad
\Lambda_{\rm holo} = \frac{3\pi}{\kappa_{D_5}^4} G_N m_{\nu}^4.
$$

This repository treats that chain as a theorem to be audited, not as an ansatz to be curve-fit. The gravity proof engines close the Unity of Scale Identity to better than $O(10^{-124})$ precision and, in the current checked-in audit, export $\epsilon_{\Lambda} = 1.0 \times 10^{-199}$ against the one-register floor $1/N \sim 3.0 \times 10^{-123}$. The same audit enforces the Newton Constant Lock that ties the emergent gravitational coupling to the positive $c_{\rm dark}$ completion residue required by boundary neutrality.

## Modular Proof Engines

The repository is partitioned into modular proof engines. These modules do not introduce adjustable phenomenology; they certify distinct sectors of the same branch-fixed architecture.

### Gravity Sector

- `noether_bridge.py` proves the gravity-side rigidity chain, including the Unity of Scale Identity, the Newton Constant Lock, the reviewer trap, and the branch-sensitive closure tensor audit.
- `holographic_tension_verifier.py` is the publication-facing tension verifier. It confirms that finite boundary capacity forces a non-zero neutrino floor and that the anomaly-free branch satisfies the transported cosmological bound.

### Cosmology Sector

- `baryon_asymmetry.py` verifies the topological Sakharov conditions and reproduces the branch-fixed baryon asymmetry target $\eta_B \approx 6.4 \times 10^{-10}$ without free baryogenesis parameters.
- `precision_cosmology_engine.py` computes the late-time Hubble Gradient from boundary entropy debt and shows how the local uplift is sourced by positive holographic surface tension rather than by an appended fluid sector.

### Flavor Sector

- `flavor_identity_resolver.py` verifies the rigid Modular Genus Ladder,

$$ m_g = m_0 e^{\beta g} $$

  and demonstrates that the observed neutrino splitting hierarchy is a mandatory residue of the $SU(2)_{26}$ phase lock.
- `eigenvector_rigidity.py` demonstrates Eigenvector Rigidity under deformation. The current hardened audit keeps the maximum sigma-weighted angle drift below $10^{-12}\sigma$, with the measured drift at $\sim 3.73 \times 10^{-13}\sigma$.

### Algebraic Rigidity

- `uniqueness_theorem.py` formally certifies the lexicographic elimination of off-shell coordinates and isolates $(26, 8, 312)$ as the unique Minimal Anomaly-Free Local Survivor.
- `minimality_proof.py` proves that $SO(10)$ is the minimal parent Lie algebra that simultaneously furnishes the direct renormalizable Majorana channel and the required Virasoro-orthogonal completion residue.

### Numerical Hardening

- `stiff_transport_audit.py` proves the mathematical necessity of the implicit Radau IIA solver for stiff holographic transport. Explicit RK45 is audited as non-certified at the $10^{-12}$ floor, while duplicate Radau runs remain reproducible below $10^{-12}\sigma$.
- `character_dictionary.py` quantifies the informational cost of Inverted Ordering in the one-copy dictionary and proves that accommodating it requires an additional support slot with redundancy cost

$$ \Delta S_{\rm red} = \ln 2 $$

## Technical Rigidities

SHBT does not use manual uncertainty floors as a substitute for missing structure. The definitive machine-readable source for the **Quantified Two-Loop Residuals** is `results/residuals.json`, exported directly by the branch-fixed transport audit. That JSON records the Unity-of-Scale residue `epsilon_lambda`, the signed benchmark mixing-angle drifts and per-observable transport residual fractions, and the one-copy informational cost `Delta S_red`.

`results/residuals.json` is a disclosed audit quantity, not a reviewer-adjustable error bar. Any deviation in `results/residuals.json` from the benchmark export signifies that the computation has left the anomaly-free `(26, 8, 312)` shell and moved onto an unphysical, off-shell branch.

The same policy applies across the stack:

- branch residues are disclosed rather than tuned,
- solver tolerances are justified by stiffness audits rather than convenience,
- hierarchy penalties are measured as informational costs rather than rhetorical preferences,
- and publication tables are generated from the verifier rather than hand-maintained.

## Infrastructure & Integrity

The repository includes a production-grade diagnostic stack to ensure the mathematical rigidity of the $(26, 8, 312)$ branch is never compromised by environment drift or manual intervention. These scripts extend the verifier with cryptographic locking, visual moat diagnostics, and formal exception testing.

### Cryptographic Verification (`verify_benchmark_integrity.py`)

`verify_benchmark_integrity.py` automatically audits the SHA-256 hash of `benchmark_v1.yaml` against the mandated theoretical benchmark. This check prevents "silent drift" in the 8-observable flavor residues by refusing to treat a modified configuration as the published branch.

### Topological Visualization (`plot_local_moat.py`)

`plot_local_moat.py` generates a high-precision map of the Local Moat surrounding the anomaly-free island and writes `results/local_moat_topology.png`. The figure makes the moat visually explicit by showing the steep rise of the framing anomaly $\Delta_{\rm fr}$ under any off-shell coordinate shift away from $(26, 8, 312)$.

### Formal Exception Testing (`test_anomaly_logic.py`)

`test_anomaly_logic.py` provides a dedicated formal verification suite for Reviewer Trap scenarios. It asserts that the engine raises `AnomalyClosureError` and halts execution when presented with unphysical anomalous coordinates, ensuring that future optimizations cannot silently relax the branch-closure logic.

## Operational Usage

Run the universal verifier from the repository root:

```bash
python tn.py --output-dir results/
```

The repository-root driver forwards into `pub/tn.py` and regenerates the benchmark-facing artifacts, numerical audits, manuscript exports, and reviewer packets from the disclosed branch data. The intended use is verification of a rigid branch-fixed theorem stack, not parameter search.

Interpret `results/residuals.json` as the benchmark's audit ledger:

- `unity_of_scale_identity.epsilon_lambda` is the closed Unity-of-Scale residue, not a tunable tolerance.
- `gauge_residual_bookkeeping` discloses the separate gauge-coupling closure residue, including the topological mismatch, fractional residual, pull, and closure status.
- `theoretical_uncertainty_fractions` and `transport_residuals` are the disclosed two-loop transport residuals actually used by the verifier, not adjustable padding for the pull table.
- `informational_costs.delta_s_red_nat` is the finite one-copy redundancy cost for forcing inverted support, not a fit parameter.

Read the JSON as a branch diagnosis rather than as an error-budget knob. The publication-facing summary in `results/benchmark_diagnostics.json` and `results/final/benchmark_diagnostics.json` now mirrors the same nested `unity_of_scale_identity.epsilon_lambda` payload. If these artifacts differ from the benchmark export, then the run is no longer evaluating the physical branch and should be treated as an unphysical off-shell computation.

## Derivation Ledger

The standalone `derive_*.py` scripts are publication-facing spot checks for the same benchmark branch. Where an external CODATA comparator exists, the script prints it directly; where no external CODATA entry exists, the script instead audits the branch-fixed theorem closure against the live `tn.py` benchmark.

| Script | Ledger output | Comparator / anchor | What the comparison seals |
| --- | --- | --- | --- |
| `python pub/derive_universe.py` | `alpha_surf^-1 = 2340/17` from the branch-fixed gauge-density ratio | CODATA `alpha^-1 = 137.035999084` | Confirms that the disclosed gauge-side residue is reported against the standard electromagnetic reference rather than hidden behind a retuned threshold. |
| `python pub/derive_universe.py` | `kappa_D5`, `m_nu = kappa_D5 M_P N^{-1/4}`, and `epsilon_Lambda` from the Unity of Scale Identity | No standalone CODATA constant; audited against the theorem closure and the live `tn.py` benchmark export | Confirms that the D5 hyperarea residue, finite-capacity mass bridge, and gravity-side closure remain branch-fixed derived residues rather than floating normalizations. |
| `python pub/derive_proton_ratio.py` | `mu_audit = (c_q/c_l) V_{\rm px}^{-1} \mathcal P_\mu = (896/99) \mathcal P_\mu`, equivalently `(c_q/c_l) \Pi_{\rm branch}^{SU(3)_8}` with `c_q = 64/11` and `c_l = 39/14` | CODATA `m_p/m_e` from `scipy.constants.proton_mass / scipy.constants.electron_mass` | Makes the proton/electron mass-ratio audit transparent by comparing the branch-derived residue to the standard CODATA mass ratio in the script output itself. |

Treat these ledgers as sealed diagnostics. They are not auxiliary fit scripts: they are short-form disclosures of the same branch-fixed outputs exported by the main verifier.

## Reproducibility

The benchmark configuration is locked by the checked-in YAML at `config/benchmark_v1.yaml` and the standard package configuration in `pyproject.toml`.

- **Benchmark lockfile:** `config/benchmark_v1.yaml`
- **SHA-256:** `737667c8d0a2925f09ae89e40a68f7d26d2177df383f4a220e7d9c2c6b55dbf4`
- **Build backend:** `setuptools.build_meta`
- **Python requirement:** `>=3.11`
- **Pinned scientific stack:** `PyYAML==6.0.3`, `Jinja2==3.1.6`, `mpmath==1.3.0`, `numpy==1.26.4`, `scipy==1.12.0`, `matplotlib==3.8.3`, `sympy==1.12`

### Constant Tiers

- **Topological Coordinates** are the branch-fixed integers and discrete support counts carried by `TIER_1_TOPOLOGICAL_COORDINATES`: `(k_{\ell}, k_q, K) = (26, 8, 312)` together with `G_SM = 15`. Changing these is not recalibration; it is a coordinate detuning that moves the run off the published branch.
- **Observational Boundary Conditions** are the external anchors carried by `TIER_2_OBSERVATIONAL_BOUNDARY_CONDITIONS`, such as `\Lambda_{\rm obs}`, the late-time cosmology inputs, and the electroweak comparison data. These are supplied to the verifier as boundary data, not opened as hidden fit directions.
- **Derived Residues** are the theorem outputs carried by `TIER_3_DERIVED_RESIDUES` and the generated audit artifacts: `kappa_{D_5}`, `N_{\rm holo}`, `R_{\rm GUT}`, `c_{\rm dark}`, `epsilon_\Lambda`, the two-loop transport residual fractions, and the informational costs written under `results/`. These quantities are consequences of Tier 1 plus Tier 2 and are therefore the objects that must remain numerically stable under rerun.

### Reviewer Trap Clause

The repository's falsification rule is strict. If any **Derived Residue** drifts beyond the quantified two-loop limits disclosed in `results/residuals.json` during a coordinate detuning, then SHBT counts that detuned run as falsified. In particular, a failure of `unity_of_scale_identity.epsilon_lambda`, the gauge residual bookkeeping, the disclosed transport residual fractions, or the one-copy informational costs is not interpreted as a better neighboring fit. It is interpreted as proof that the computation has exited the anomaly-free shell and no longer represents the physical `(26, 8, 312)` branch.

These locks matter. The benchmark is only meaningful if the same structural residues, the same transport tolerances, and the same reporting stack reproduce the same published artifacts. Reproducibility in SHBT is therefore a proof obligation, not a convenience feature.

## Repository Position

SHBT is a universal source-code implementation of a hardened branch-fixed boundary theory. It is designed to answer a narrow question with maximal rigidity: whether the anomaly-free $(26, 8, 312)$ branch can carry flavor structure, cosmological sourcing, and gravity closure without reopening a free-parameter sector. This repository answers that question by executable audit.
