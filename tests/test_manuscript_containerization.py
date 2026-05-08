from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_dockerfile_provisions_internal_tectonic_environment() -> None:
    dockerfile = (ROOT_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    assert "SHBT_CONTAINERIZED_LATEX=1" in dockerfile
    assert "SHBT_INTERNAL_LATEX_BACKEND=tectonic" in dockerfile
    assert "tectonic" in dockerfile
    assert "scripts/build_manuscript.py" in dockerfile


def test_workflow_builds_and_uploads_executable_manuscript_pdf() -> None:
    workflow = (ROOT_DIR / ".github" / "workflows" / "shbt-verify.yml").read_text(encoding="utf-8")

    assert "Build SHBT Manuscript Container" in workflow
    assert "docker build -t shbt-manuscript:latest ." in workflow
    assert "python scripts/build_manuscript.py --latex-backend auto --compile-only --output-dir results/" in workflow
    assert "Upload Executable Manuscript PDF" in workflow
    assert "results/final/gravity.pdf" in workflow
    assert "if-no-files-found: error" in workflow
