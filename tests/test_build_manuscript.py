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


def test_synchronize_manuscript_sources_uses_current_sync_system_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    manuscript_dir = tmp_path / "papers"
    manuscript_dir.mkdir()
    output_dir = tmp_path / "proof-results"
    output_dir.mkdir()

    recorded_call: dict[str, Path] = {}

    class FakeSyncModule:
        @staticmethod
        def synchronize_system(
            *,
            residuals_path: Path,
            readme_path: Path,
            physics_constants_path: Path,
        ) -> tuple[Path, Path]:
            recorded_call.update(
                residuals_path=residuals_path,
                readme_path=readme_path,
                physics_constants_path=physics_constants_path,
            )
            return readme_path, physics_constants_path

    monkeypatch.setattr(module, "_load_sync_system_module", lambda: FakeSyncModule)

    updated_readme, updated_physics_constants = module.synchronize_manuscript_sources(
        output_dir=output_dir,
        manuscript_dir=manuscript_dir,
    )

    assert updated_readme == module.ProjectPaths.ROOT / "README.md"
    assert updated_physics_constants == manuscript_dir / "physics_constants.tex"
    assert recorded_call == {
        "residuals_path": output_dir / "residuals.json",
        "readme_path": module.ProjectPaths.ROOT / "README.md",
        "physics_constants_path": manuscript_dir / "physics_constants.tex",
    }


def test_build_manuscript_runs_executable_paper_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    manuscript_dir = tmp_path / "papers"
    manuscript_dir.mkdir()
    output_dir = tmp_path / "proof-results"
    (manuscript_dir / "gravity.tex").write_text(
        r"\documentclass{article}\begin{document}gravity\end{document}",
        encoding="utf-8",
    )

    recorded_events: list[tuple[object, ...]] = []

    def fake_run_entire_proof(*, output_dir: Path, manuscript_dir: Path, validate_text: bool) -> None:
        recorded_events.append(("proof", output_dir, manuscript_dir, validate_text))
        (output_dir / "final").mkdir(parents=True, exist_ok=True)

    def fake_generate_topological_constants(*, precision: int = module.DERIVATION_DEFAULT_PRECISION):
        recorded_events.append(("constants", precision))
        return module.TopologicalConstants(
            derivation_ledger="ledger",
            alpha_surface_inverse=137.6470588235,
            kappa_d5=0.123456789,
            neutrino_floor_mev=2.83,
            epsilon_lambda=1.0e-199,
            lambda_holo_si_m2=1.0e-52,
        )

    def fake_sync(*, output_dir: Path, manuscript_dir: Path) -> tuple[Path, Path]:
        recorded_events.append(("sync", output_dir, manuscript_dir))
        readme_path = tmp_path / "README.md"
        physics_constants_path = manuscript_dir / "physics_constants.tex"
        readme_path.write_text("# synced\n", encoding="utf-8")
        physics_constants_path.write_text("% synced\n", encoding="utf-8")
        return readme_path, physics_constants_path

    def fake_resolve(_backend: str = "auto") -> module.LatexInstallation:
        recorded_events.append(("resolve", _backend))
        return module.LatexInstallation(backend="pdflatex", command="pdflatex")

    def fake_compile(command: str, document_path: Path, *, workdir: Path) -> Path:
        recorded_events.append(("compile", command, document_path.name, workdir))
        pdf_path = workdir / "gravity.pdf"
        pdf_path.write_text("pdf", encoding="utf-8")
        return pdf_path

    monkeypatch.setattr(module, "run_entire_proof", fake_run_entire_proof)
    monkeypatch.setattr(module, "generate_topological_constants", fake_generate_topological_constants)
    monkeypatch.setattr(module, "synchronize_manuscript_sources", fake_sync)
    monkeypatch.setattr(module, "resolve_latex_installation", fake_resolve)
    monkeypatch.setattr(module, "compile_gravity_manuscript", fake_compile)

    result = module.build_manuscript(
        manuscript_dir=manuscript_dir,
        output_dir=output_dir,
        latex_backend="auto",
        validate_text=True,
    )

    assert result.latex_backend == "pdflatex"
    assert result.final_pdf == manuscript_dir / "gravity.pdf"
    assert tuple(path.name for path in result.compiled_pdfs) == ("gravity.pdf",)
    assert recorded_events == [
        ("proof", output_dir, manuscript_dir, True),
        ("constants", module.DERIVATION_DEFAULT_PRECISION),
        ("sync", output_dir, manuscript_dir),
        ("resolve", "pdflatex"),
        ("compile", "pdflatex", "gravity.tex", manuscript_dir),
    ]


def test_build_manuscript_reports_missing_pdflatex_after_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    manuscript_dir = tmp_path / "papers"
    manuscript_dir.mkdir()
    output_dir = tmp_path / "proof-results"
    (manuscript_dir / "gravity.tex").write_text(
        r"\documentclass{article}\begin{document}gravity\end{document}",
        encoding="utf-8",
    )

    recorded_events: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        module,
        "run_entire_proof",
        lambda *, output_dir, manuscript_dir, validate_text: recorded_events.append(
            ("proof", output_dir, manuscript_dir, validate_text)
        ),
    )
    monkeypatch.setattr(
        module,
        "generate_topological_constants",
        lambda *, precision=module.DERIVATION_DEFAULT_PRECISION: (
            recorded_events.append(("constants", precision))
            or module.TopologicalConstants("ledger", 137.0, 0.1, 2.8, 1.0e-199, 1.0e-52)
        ),
    )
    monkeypatch.setattr(
        module,
        "synchronize_manuscript_sources",
        lambda *, output_dir, manuscript_dir: recorded_events.append(("sync", output_dir, manuscript_dir))
        or (tmp_path / "README.md", manuscript_dir / "physics_constants.tex"),
    )

    def fake_resolve(_backend: str = "auto"):
        recorded_events.append(("resolve", _backend))
        raise RuntimeError("pdflatex missing")

    monkeypatch.setattr(module, "resolve_latex_installation", fake_resolve)

    result = module.build_manuscript(
        manuscript_dir=manuscript_dir,
        output_dir=output_dir,
        latex_backend="auto",
        validate_text=False,
    )

    captured = capsys.readouterr()
    assert "Mathematical proofs complete; LaTeX compiler not found for PDF generation" in captured.out
    assert result.latex_backend is None
    assert result.compiled_pdfs == ()
    assert result.final_pdf is None
    assert recorded_events == [
        ("proof", output_dir, manuscript_dir, False),
        ("constants", module.DERIVATION_DEFAULT_PRECISION),
        ("sync", output_dir, manuscript_dir),
        ("resolve", "pdflatex"),
    ]
