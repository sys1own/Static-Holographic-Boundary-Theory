from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.paths import ProjectPaths


TARGET_AUDIT_SECTORS = ("gravity", "cosmology", "flavor", "rigidity")
PREFERRED_LATEX_BUILD_ORDER = ("supplementary.tex", "tn.tex", "gravity.tex")
CANONICAL_JOURNAL_TEX = "tn.tex"
CANONICAL_JOURNAL_PDF = "tn.pdf"
SUPPORTED_LATEX_BACKENDS = ("latexmk", "pdflatex")


@dataclass(frozen=True)
class LatexInstallation:
    backend: str
    command: str


@dataclass(frozen=True)
class BuildResult:
    proof_output_dir: Path
    latex_artifact_dir: Path
    manuscript_dir: Path
    latex_backend: str
    compiled_pdfs: tuple[Path, ...]
    final_pdf: Path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


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
            "No local LaTeX installation detected. Install `latexmk` (preferred) or `pdflatex` and rerun `python scripts/build_manuscript.py`."
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


def mirror_manuscript_artifacts(build_output_dir: Path, *, latex_results_dir: Path = ProjectPaths.RESULTS) -> Path:
    source_dir = Path(build_output_dir) / "final"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Universal audit did not produce manuscript artifacts in {source_dir}.")

    target_dir = Path(latex_results_dir) / "final"
    target_dir.mkdir(parents=True, exist_ok=True)

    if source_dir.resolve() == target_dir.resolve():
        return target_dir

    for child in source_dir.iterdir():
        destination = target_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
            continue
        shutil.copy2(child, destination)
    return target_dir


def refresh_manuscript_physics_constants(*, latex_artifact_dir: Path, manuscript_dir: Path) -> Path:
    source_path = Path(latex_artifact_dir) / "physics_constants.tex"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Expected generated manuscript macro export at {source_path}. Run the universal audit before compiling the papers."
        )

    target_path = Path(manuscript_dir) / "physics_constants.tex"
    shutil.copy2(source_path, target_path)
    return target_path


def _latex_commands(installation: LatexInstallation, document_name: str) -> tuple[tuple[str, ...], ...]:
    if installation.backend == "latexmk":
        return (
            (
                installation.command,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                document_name,
            ),
        )

    return (
        (
            installation.command,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            document_name,
        ),
        (
            installation.command,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            document_name,
        ),
    )


def compile_latex_document(installation: LatexInstallation, document_path: Path, *, workdir: Path) -> Path:
    resolved_workdir = Path(workdir)
    source_name = Path(document_path).name
    try:
        for command in _latex_commands(installation, source_name):
            subprocess.run(command, cwd=resolved_workdir, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"LaTeX compilation failed for {source_name} using `{installation.backend}`.") from exc

    pdf_path = resolved_workdir / Path(source_name).with_suffix(".pdf")
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

    if not resolved_manuscript_dir.is_dir():
        raise FileNotFoundError(f"Manuscript directory not found: {resolved_manuscript_dir}")

    latex_installation = resolve_latex_installation(latex_backend)

    print(f"[build] LaTeX backend: {latex_installation.backend}")
    print("[build] Running sector proofs and universal audit")
    run_entire_proof(
        output_dir=resolved_output_dir,
        manuscript_dir=resolved_manuscript_dir,
        validate_text=validate_text,
    )

    print("[build] Refreshing manuscript-facing artifact mirror")
    latex_artifact_dir = mirror_manuscript_artifacts(resolved_output_dir, latex_results_dir=ProjectPaths.RESULTS)
    refresh_manuscript_physics_constants(
        latex_artifact_dir=latex_artifact_dir,
        manuscript_dir=resolved_manuscript_dir,
    )

    print("[build] Compiling standalone papers")
    compiled_pdfs = tuple(
        compile_latex_document(latex_installation, document_path, workdir=resolved_manuscript_dir)
        for document_path in discover_compile_targets(resolved_manuscript_dir)
    )

    final_pdf = resolved_manuscript_dir / CANONICAL_JOURNAL_PDF
    if not final_pdf.is_file():
        final_pdf = compiled_pdfs[0]

    return BuildResult(
        proof_output_dir=resolved_output_dir,
        latex_artifact_dir=latex_artifact_dir,
        manuscript_dir=resolved_manuscript_dir,
        latex_backend=latex_installation.backend,
        compiled_pdfs=compiled_pdfs,
        final_pdf=final_pdf,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript-dir", type=Path, default=ProjectPaths.PAPERS)
    parser.add_argument("--output-dir", type=Path, default=ProjectPaths.RESULTS)
    parser.add_argument("--latex-backend", choices=("auto", *SUPPORTED_LATEX_BACKENDS), default="auto")
    parser.add_argument(
        "--validate-text",
        action="store_true",
        help="Pass `--validate-text` through to the universal audit before LaTeX compilation.",
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

    supporting_pdfs = [pdf_path for pdf_path in result.compiled_pdfs if pdf_path != result.final_pdf]
    print("")
    print(f"[build] Completed journal PDF: {_display_path(result.final_pdf)}")
    if supporting_pdfs:
        supporting_paths = ", ".join(_display_path(pdf_path) for pdf_path in supporting_pdfs)
        print(f"[build] Supporting PDFs: {supporting_paths}")


if __name__ == "__main__":
    main()
