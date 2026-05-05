from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_manuscript.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_manuscript", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_latex_installation_requires_local_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    monkeypatch.setattr(module.shutil, "which", lambda _command: None)

    with pytest.raises(RuntimeError, match=r"No local LaTeX installation detected"):
        module.resolve_latex_installation()


def test_discover_compile_targets_orders_standalone_documents(tmp_path: Path) -> None:
    module = _load_script_module()
    manuscript_dir = tmp_path / "papers"
    manuscript_dir.mkdir()

    (manuscript_dir / "tn.tex").write_text(r"\documentclass{article}\begin{document}tn\end{document}", encoding="utf-8")
    (manuscript_dir / "supplementary.tex").write_text(
        r"\documentclass{article}\begin{document}supp\end{document}",
        encoding="utf-8",
    )
    (manuscript_dir / "gravity.tex").write_text(
        r"\documentclass{article}\begin{document}gravity\end{document}",
        encoding="utf-8",
    )
    (manuscript_dir / "appendix.tex").write_text(
        r"\documentclass{article}\begin{document}appendix\end{document}",
        encoding="utf-8",
    )
    (manuscript_dir / "physics_constants.tex").write_text(r"\providecommand{\demo}{1}", encoding="utf-8")

    ordered_targets = module.discover_compile_targets(manuscript_dir)

    assert tuple(path.name for path in ordered_targets) == (
        "supplementary.tex",
        "tn.tex",
        "gravity.tex",
        "appendix.tex",
    )


def test_run_entire_proof_invokes_all_sector_audits_before_full_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    calls: list[tuple[str, ...]] = []
    manuscript_dir = tmp_path / "papers"
    manuscript_dir.mkdir()
    output_dir = tmp_path / "results"

    monkeypatch.setattr(module, "run_verifier", lambda args: calls.append(tuple(args)))

    module.run_entire_proof(output_dir=output_dir, manuscript_dir=manuscript_dir, validate_text=True)

    common_args = ("--output-dir", str(output_dir), "--manuscript-dir", str(manuscript_dir), "--validate-text")
    assert calls == [
        (*common_args, "--sector", "gravity"),
        (*common_args, "--sector", "cosmology"),
        (*common_args, "--sector", "flavor"),
        (*common_args, "--sector", "rigidity"),
        common_args,
    ]
    assert (output_dir / "final").is_dir()


def test_build_manuscript_runs_audit_then_compiles_papers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    manuscript_dir = tmp_path / "papers"
    manuscript_dir.mkdir()
    output_dir = tmp_path / "proof-results"
    for source_name in ("supplementary.tex", "tn.tex", "gravity.tex"):
        (manuscript_dir / source_name).write_text(
            r"\documentclass{article}\begin{document}demo\end{document}",
            encoding="utf-8",
        )

    recorded_events: list[tuple[object, ...]] = []
    generated_artifact_dir = tmp_path / "latex-results" / "final"

    monkeypatch.setattr(
        module,
        "resolve_latex_installation",
        lambda _backend="auto": module.LatexInstallation(backend="latexmk", command="latexmk"),
    )

    def fake_run_entire_proof(*, output_dir: Path, manuscript_dir: Path, validate_text: bool) -> None:
        recorded_events.append(("proof", output_dir, manuscript_dir, validate_text))

    def fake_mirror(build_output_dir: Path, *, latex_results_dir: Path) -> Path:
        recorded_events.append(("mirror", build_output_dir, latex_results_dir))
        generated_artifact_dir.mkdir(parents=True, exist_ok=True)
        (generated_artifact_dir / "physics_constants.tex").write_text("% generated\n", encoding="utf-8")
        return generated_artifact_dir

    def fake_refresh(*, latex_artifact_dir: Path, manuscript_dir: Path) -> Path:
        recorded_events.append(("refresh", latex_artifact_dir, manuscript_dir))
        target_path = manuscript_dir / "physics_constants.tex"
        target_path.write_text("% synced\n", encoding="utf-8")
        return target_path

    def fake_compile(installation, document_path: Path, *, workdir: Path) -> Path:
        recorded_events.append(("compile", installation.backend, document_path.name, workdir))
        pdf_path = workdir / f"{document_path.stem}.pdf"
        pdf_path.write_text("pdf", encoding="utf-8")
        return pdf_path

    monkeypatch.setattr(module, "run_entire_proof", fake_run_entire_proof)
    monkeypatch.setattr(module, "mirror_manuscript_artifacts", fake_mirror)
    monkeypatch.setattr(module, "refresh_manuscript_physics_constants", fake_refresh)
    monkeypatch.setattr(module, "compile_latex_document", fake_compile)

    result = module.build_manuscript(
        manuscript_dir=manuscript_dir,
        output_dir=output_dir,
        latex_backend="auto",
        validate_text=True,
    )

    assert result.latex_backend == "latexmk"
    assert result.final_pdf == manuscript_dir / "tn.pdf"
    assert tuple(path.name for path in result.compiled_pdfs) == (
        "supplementary.pdf",
        "tn.pdf",
        "gravity.pdf",
    )
    assert recorded_events[0] == ("proof", output_dir, manuscript_dir, True)
    assert recorded_events[1] == ("mirror", output_dir, module.ProjectPaths.RESULTS)
    assert recorded_events[2] == ("refresh", generated_artifact_dir, manuscript_dir)
    assert [event[2] for event in recorded_events if event[0] == "compile"] == [
        "supplementary.tex",
        "tn.tex",
        "gravity.tex",
    ]
