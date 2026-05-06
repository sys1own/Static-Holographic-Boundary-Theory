from __future__ import annotations

"""Build the SHBT executable paper from proofs, synced constants, and LaTeX."""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.core.evolutionary_engine import DEFAULT_PRECISION as DERIVATION_DEFAULT_PRECISION, EvolutionaryEngine
from shbt.paths import ProjectPaths


TARGET_AUDIT_SECTORS = ("gravity", "cosmology", "flavor", "rigidity")
PREFERRED_LATEX_BUILD_ORDER = ("supplementary.tex", "tn.tex", "gravity.tex")
MANUSCRIPT_TARGET_TEX = "gravity.tex"
MANUSCRIPT_TARGET_PDF = "gravity.pdf"
SUPPORTED_LATEX_BACKENDS = ("pdflatex",)


@dataclass(frozen=True)
class LatexInstallation:
    backend: str
    command: str


@dataclass(frozen=True)
class TopologicalConstants:
    derivation_ledger: str
    alpha_surface_inverse: float
    kappa_d5: float
    neutrino_floor_mev: float
    epsilon_lambda: float
    lambda_holo_si_m2: float


@dataclass(frozen=True)
class BuildResult:
    proof_output_dir: Path
    latex_artifact_dir: Path
    manuscript_dir: Path
    latex_backend: str | None
    compiled_pdfs: tuple[Path, ...]
    final_pdf: Path | None


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


@lru_cache(maxsize=1)
def _load_sync_system_module() -> Any:
    script_path = ProjectPaths.SCRIPTS / "sync_system.py"
    spec = importlib.util.spec_from_file_location("sync_system", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load sync_system from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_latex_installation(backend: str = "auto") -> LatexInstallation:
    if backend == "auto":
        candidates = SUPPORTED_LATEX_BACKENDS
    else:
        candidates = (backend,)

    for candidate in candidates:
        command = shutil.which(candidate)
        if command is not None:
            return LatexInstallation(backend=candidate, command=command)

    if backend == "auto":
        raise RuntimeError(
            "No local LaTeX installation detected. Install `pdflatex` and rerun `python scripts/build_manuscript.py`."
        )
    raise RuntimeError(
        f"Requested LaTeX backend `{backend}` is not available on PATH. Install it or rerun with `--latex-backend auto`."
    )


def is_standalone_latex_source(path: Path) -> bool:
    if path.suffix != ".tex" or not path.is_file():
        return False
    return "\\documentclass" in path.read_text(encoding="utf-8")


def discover_compile_targets(manuscript_dir: Path) -> tuple[Path, ...]:
    resolved_manuscript_dir = Path(manuscript_dir)
    standalone_sources = {
        source_path.name: source_path
        for source_path in resolved_manuscript_dir.glob("*.tex")
        if is_standalone_latex_source(source_path)
    }
    ordered_sources = [standalone_sources.pop(name) for name in PREFERRED_LATEX_BUILD_ORDER if name in standalone_sources]
    ordered_sources.extend(standalone_sources[name] for name in sorted(standalone_sources))
    if not ordered_sources:
        raise FileNotFoundError(f"No standalone LaTeX sources were found in {resolved_manuscript_dir}.")
    return tuple(ordered_sources)


def run_verifier(args: Sequence[str]) -> None:
    from shbt import main as shbt_main

    shbt_main.main(list(args))


def run_entire_proof(*, output_dir: Path, manuscript_dir: Path, validate_text: bool = False) -> None:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    (resolved_output_dir / "final").mkdir(parents=True, exist_ok=True)

    common_args = ["--output-dir", str(resolved_output_dir), "--manuscript-dir", str(manuscript_dir)]
    if validate_text:
        common_args.append("--validate-text")

    for sector in TARGET_AUDIT_SECTORS:
        run_verifier([*common_args, "--sector", sector])
    run_verifier(common_args)


def generate_topological_constants(*, precision: int = DERIVATION_DEFAULT_PRECISION) -> TopologicalConstants:
    resolved_precision = max(int(precision), int(DERIVATION_DEFAULT_PRECISION))
    alpha = EvolutionaryEngine.derive_alpha_surface(precision=resolved_precision)
    kappa = EvolutionaryEngine.derive_kappa_d5(precision=resolved_precision)
    mass = EvolutionaryEngine.derive_mass_bridge(precision=resolved_precision, kappa=kappa.kappa)
    unity = EvolutionaryEngine.derive_unity_of_scale(precision=resolved_precision, kappa=kappa.kappa, mass_bridge=mass)
    lambda_surface = EvolutionaryEngine.derive_lambda_surface(precision=resolved_precision)
    derivation_ledger = EvolutionaryEngine.generate_ledger(kind="universe", precision=resolved_precision)
    return TopologicalConstants(
        derivation_ledger=derivation_ledger,
        alpha_surface_inverse=float(alpha.alpha_inverse_decimal),
        kappa_d5=float(kappa.kappa),
        neutrino_floor_mev=float(mass.neutrino_floor_mev),
        epsilon_lambda=float(unity.epsilon_lambda),
        lambda_holo_si_m2=float(lambda_surface.lambda_holo_si_m2),
    )


def synchronize_manuscript_sources(*, output_dir: Path, manuscript_dir: Path) -> tuple[Path, Path]:
    module = _load_sync_system_module()
    return module.synchronize_system(
        residuals_path=Path(output_dir) / "residuals.json",
        readme_path=ProjectPaths.ROOT / "README.md",
        physics_constants_path=Path(manuscript_dir) / "physics_constants.tex",
    )


def compile_gravity_manuscript(pdflatex_command: str, document_path: Path, *, workdir: Path) -> Path:
    resolved_workdir = Path(workdir)
    resolved_document_path = Path(document_path)
    if not resolved_document_path.is_file():
        raise FileNotFoundError(f"Executable-paper source not found: {resolved_document_path}")

    command = (
        str(pdflatex_command),
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        resolved_document_path.name,
    )
    try:
        subprocess.run(command, cwd=resolved_workdir, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"LaTeX compilation failed for {resolved_document_path.name} using `pdflatex`.") from exc

    pdf_path = resolved_workdir / resolved_document_path.with_suffix(".pdf").name
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Expected compiled PDF at {pdf_path}, but it was not created.")
    return pdf_path


def build_manuscript(
    *,
    manuscript_dir: Path = ProjectPaths.PAPERS,
    output_dir: Path = ProjectPaths.RESULTS,
    latex_backend: str = "auto",
    validate_text: bool = False,
) -> BuildResult:
    resolved_manuscript_dir = Path(manuscript_dir)
    resolved_output_dir = Path(output_dir)
    latex_artifact_dir = resolved_output_dir / "final"

    if latex_backend not in {"auto", "pdflatex"}:
        raise ValueError("build_manuscript only supports `pdflatex` for executable-paper generation.")
    if not resolved_manuscript_dir.is_dir():
        raise FileNotFoundError(f"Manuscript directory not found: {resolved_manuscript_dir}")

    print("[build] Running sector proofs and universal audit")
    run_entire_proof(
        output_dir=resolved_output_dir,
        manuscript_dir=resolved_manuscript_dir,
        validate_text=validate_text,
    )

    print("[build] Triggering EvolutionaryEngine derivation ledger")
    topological_constants = generate_topological_constants()
    print(
        "[build] Constants ready: "
        f"alpha_surf^-1={topological_constants.alpha_surface_inverse:.9f}, "
        f"kappa_D5={topological_constants.kappa_d5:.9f}, "
        f"m_nu={topological_constants.neutrino_floor_mev:.9f} meV"
    )

    print("[build] Freezing live constants into manuscript sources")
    synchronize_manuscript_sources(
        output_dir=resolved_output_dir,
        manuscript_dir=resolved_manuscript_dir,
    )

    try:
        latex_installation = resolve_latex_installation("pdflatex" if latex_backend == "auto" else latex_backend)
    except RuntimeError:
        print("Mathematical verification successful; PDF generation skipped due to missing LaTeX compiler.")
        return BuildResult(
            proof_output_dir=resolved_output_dir,
            latex_artifact_dir=latex_artifact_dir,
            manuscript_dir=resolved_manuscript_dir,
            latex_backend=None,
            compiled_pdfs=(),
            final_pdf=None,
        )

    target_document = resolved_manuscript_dir / MANUSCRIPT_TARGET_TEX
    print(f"[build] Compiling executable paper: {_display_path(target_document)}")
    compiled_pdf = compile_gravity_manuscript(
        latex_installation.command,
        target_document,
        workdir=resolved_manuscript_dir,
    )

    return BuildResult(
        proof_output_dir=resolved_output_dir,
        latex_artifact_dir=latex_artifact_dir,
        manuscript_dir=resolved_manuscript_dir,
        latex_backend=latex_installation.backend,
        compiled_pdfs=(compiled_pdf,),
        final_pdf=compiled_pdf,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript-dir", type=Path, default=ProjectPaths.PAPERS)
    parser.add_argument("--output-dir", type=Path, default=ProjectPaths.RESULTS)
    parser.add_argument("--latex-backend", choices=("auto", "pdflatex"), default="auto")
    parser.add_argument(
        "--validate-text",
        action="store_true",
        help="Pass `--validate-text` through to the universal audit before freezing manuscript constants.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_manuscript(
        manuscript_dir=args.manuscript_dir.expanduser(),
        output_dir=args.output_dir.expanduser(),
        latex_backend=args.latex_backend,
        validate_text=args.validate_text,
    )

    if result.final_pdf is None:
        return

    print("")
    print(f"[build] Completed executable paper PDF: {_display_path(result.final_pdf)}")


if __name__ == "__main__":
    main()
