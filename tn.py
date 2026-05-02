"""Repository-root wrapper for the publication-facing SO(10)_312 RG consistency driver.

Quickstart for `sys1own/Static-Holographic-Boundary-Theory`:
    Run the RG Consistency Audit Driver:
        `python tn.py`
        `so10-312-review`
        `so10-312-referee`
    Run only the benchmark-residue detuning check:
        `python tn.py --residue-check --output-dir results/`
    Reproduce manuscript-facing artifacts:
        `python tn.py --manuscript-dir . --output-dir output/`

This wrapper forwards directly to `pub.tn.main`. Benchmark mass-coordinate
diagnostics live in `pub.tn`: manual detuning is reported through the
Benchmark Consistency Audit / RG Consistency Audit flow rather than raising a
hard benchmark exception from the mass-lock helper.
"""

from pub.tn import main


if __name__ == "__main__":
    main()
