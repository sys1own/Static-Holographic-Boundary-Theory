from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from shbt.constants import LIGHT_SPEED_M_PER_S
from shbt.observational_bridge import ObservationalBridge, TensionAnomaly


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_observational_bridge_ingests_ligo_and_jwst_trigger_files(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "01_ligo.json",
        [
            {
                "source": "LIGO/Virgo",
                "event_name": "GW-test",
                "redshift": "0.01",
                "luminosity_distance_mpc": "44.479592",
                "peak_strain": "1e-21",
            }
        ],
    )
    (tmp_path / "02_jwst.csv").write_text(
        "source,event_name,redshift,observed_expansion_rate_km_s_mpc,galaxy_count\n"
        "JWST,JADES-1,12,68.5,14\n",
        encoding="utf-8",
    )

    bridge = ObservationalBridge(trigger_directory=tmp_path, precision=50)
    triggers = bridge.load_external_triggers()

    assert len(triggers) == 2
    assert triggers[0].trigger_kind == "ligo"
    assert triggers[0].peak_strain == Decimal("1e-21")
    assert triggers[1].trigger_kind == "jwst"
    assert triggers[1].galaxy_count == Decimal("14")
    assert triggers[1].observed_expansion_rate_km_s_mpc == Decimal("68.5")


def test_observational_bridge_maps_ligo_distance_and_jwst_counts_to_expansion_proxies(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "01_ligo.json",
        {
            "source": "LIGO/Virgo",
            "event_name": "GW-anchor",
            "redshift": "0.01",
            "luminosity_distance_mpc": "44.479592",
            "peak_strain": "8e-22",
        },
    )
    _write_json(
        tmp_path / "02_jwst.json",
        {
            "source": "JWST",
            "event_name": "JADES-2",
            "redshift": "12",
            "galaxy_count": "103",
            "reference_galaxy_count": "100",
        },
    )

    bridge = ObservationalBridge(trigger_directory=tmp_path, precision=50)
    audit = bridge.calculate_holographic_tension(raise_on_anomaly=False)

    ligo_diagnostic, jwst_diagnostic = audit.diagnostics
    expected_ligo_hubble = (Decimal(str(LIGHT_SPEED_M_PER_S / 1000.0)) * Decimal("0.01")) / Decimal("44.479592")

    assert ligo_diagnostic.observed_expansion_rate_km_s_mpc is not None
    assert abs(ligo_diagnostic.observed_expansion_rate_km_s_mpc - expected_ligo_hubble) < Decimal("1e-20")
    assert jwst_diagnostic.observed_expansion_rate_km_s_mpc is not None
    assert abs(
        jwst_diagnostic.observed_expansion_rate_km_s_mpc
        - (jwst_diagnostic.predicted_expansion_rate_km_s_mpc * Decimal("100") / Decimal("103"))
    ) < Decimal("1e-20")
    assert not audit.anomaly_detected
    assert audit.multi_messenger
    assert jwst_diagnostic.included_in_high_redshift_audit


def test_observational_bridge_raises_tension_anomaly_for_high_z_multimessenger_moat_violation(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "01_ligo.json",
        {
            "source": "LIGO/Virgo",
            "event_name": "GW-baseline",
            "redshift": "0.01",
            "luminosity_distance_mpc": "44.479592",
            "peak_strain": "9e-22",
        },
    )
    _write_json(
        tmp_path / "02_jwst.json",
        {
            "source": "JWST",
            "event_name": "JADES-tension",
            "redshift": "12",
            "observed_expansion_rate_km_s_mpc": "75",
            "galaxy_count": "9",
        },
    )

    bridge = ObservationalBridge(trigger_directory=tmp_path, precision=50)

    with pytest.raises(TensionAnomaly) as excinfo:
        bridge.calculate_holographic_tension()

    audit = excinfo.value.audit
    assert audit.anomaly_detected
    assert audit.multi_messenger
    assert audit.moat_violation_count == 1
    diagnostic = audit.high_redshift_diagnostics[0]
    assert diagnostic.exceeds_moat_bounds
    assert diagnostic.fractional_tension is not None
    assert diagnostic.fractional_tension > audit.moat_bounds.maximum_fractional_tension
    assert diagnostic.equivalent_moat_shift is not None
    assert diagnostic.equivalent_moat_shift > audit.moat_bounds.published_visible_moat_radius
