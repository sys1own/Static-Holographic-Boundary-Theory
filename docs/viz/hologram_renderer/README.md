# Prime-Lattice Hologram Renderer

This directory contains an interactive 4D renderer for the SHBT prime-lattice.
It visualizes Noether-bridge decoherence around the benchmark kernel
`(26, 8, 312)` and lets you detune the lattice up to `10^-15` to witness the
local moat open and the manifold state collapse.

## 4D Encoding

- **Scene 1 — Prime lattice occupancy**
  - `x = k_q`
  - `y = k_l / 2` (the prime-index coordinate)
  - `z = K - 312`
  - `color = M_pi` state
  - `marker size = bit-loading density ρ_bit = K / (k_l + k_q)`
- **Scene 2 — Noether-bridge moat response**
  - `x = k_q`
  - `y = k_l / 2`
  - `z = log10(1 + Δ_moat / 10^-18)`
  - `color = M_pi` state

The renderer defines the visual manifold state as

`M_pi = (ρ_bit / ρ_bit,benchmark) * exp(-Δ_moat / 10^-16)`

so the benchmark branch remains bright and loaded at zero detuning, while even
`10^-15` detuning visibly drives collapse.

## Usage

Generate the interactive HTML artifact:

```bash
python docs/viz/hologram_renderer/render_hologram.py
```

Write to a custom path:

```bash
python docs/viz/hologram_renderer/render_hologram.py --output docs/viz/hologram_renderer/index.html
```

The default output is `docs/viz/hologram_renderer/index.html`.

## Interaction

- Use the detuning slider to move from the locked benchmark to `1.0e-15`.
- Hover any lattice node to inspect `M_pi`, `ρ_bit`, moat divergence, closure amplitude, and Axiom IX closure status.
- Prime-indexed kernels are identified in the hover state via the `k_l / 2` coordinate and Axiom IX status.
