from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from shbt.core.math_utils import verify_gko_orthogonality


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_verify_gko_orthogonality_is_available_from_core_math_utils() -> None:
    audit = verify_gko_orthogonality()

    assert audit.parent_level > 0
    assert audit.lepton_level > 0
    assert audit.quark_level > 0
    assert audit.orthogonality_verified


def test_ontic_cascade_import_does_not_load_shbt_main() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import shbt.core.ontic_cascade; print('shbt.main' in sys.modules)",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"
