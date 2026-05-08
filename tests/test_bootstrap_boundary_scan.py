from __future__ import annotations

import shbt.core.bootstrap as bootstrap_module



def test_scan_boundary_configurations_defaults_to_benchmark_window() -> None:
    assert bootstrap_module.scan_boundary_configurations() == {
        24: 0.222,
        25: 0.222,
        26: 0.0,
        27: 0.222,
        28: 0.222,
    }



def test_scan_boundary_configurations_accepts_keyword_axes() -> None:
    assert bootstrap_module.scan_boundary_configurations(
        lepton_levels=[24, 26],
        quark_levels=[8],
        parent_levels=[312],
    ) == {
        24: 0.222,
        26: 0.0,
    }
