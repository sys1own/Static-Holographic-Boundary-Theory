from __future__ import annotations

import json
from pathlib import Path

import pytest

from shbt.http_bridge import HTTPBridgeError, fetch_csv_rows, fetch_json, stage_resource, stage_trigger_payload
from shbt.observational_bridge import ObservationalBridge


def _build_fetcher(payloads: dict[str, tuple[str, str]]) -> object:
    def fetcher(url: str, _headers: dict[str, str], _timeout: float) -> tuple[int, dict[str, str], str]:
        body, content_type = payloads[url]
        return 200, {"Content-Type": content_type}, body

    return fetcher


def test_fetch_json_uses_fake_fetcher() -> None:
    fetcher = _build_fetcher(
        {
            "https://example.test/gw.json": (
                json.dumps({"source": "LIGO/Virgo", "event_name": "GW-http", "redshift": "0.01"}),
                "application/json",
            )
        }
    )

    payload = fetch_json("https://example.test/gw.json", fetcher=fetcher)

    assert payload["event_name"] == "GW-http"
    assert payload["source"] == "LIGO/Virgo"


def test_fetch_csv_rows_uses_fake_fetcher() -> None:
    fetcher = _build_fetcher(
        {
            "https://example.test/jwst.csv": (
                "source,event_name,redshift\nJWST,JADES-http,12\n",
                "text/csv; charset=utf-8",
            )
        }
    )

    rows = fetch_csv_rows("https://example.test/jwst.csv", fetcher=fetcher)

    assert rows == ({"source": "JWST", "event_name": "JADES-http", "redshift": "12"},)


def test_stage_resource_infers_suffix_from_content_type(tmp_path: Path) -> None:
    fetcher = _build_fetcher(
        {
            "https://example.test/triggers": (
                json.dumps({"source": "JWST", "event_name": "remote", "redshift": "12"}),
                "application/json; charset=utf-8",
            )
        }
    )

    mirrored = stage_resource("https://example.test/triggers", tmp_path, fetcher=fetcher)

    assert mirrored == tmp_path / "triggers.json"
    assert json.loads(mirrored.read_text(encoding="utf-8"))["event_name"] == "remote"


def test_stage_trigger_payload_rejects_unsupported_content_type(tmp_path: Path) -> None:
    fetcher = _build_fetcher({"https://example.test/archive": ("plain text", "text/plain")})

    with pytest.raises(HTTPBridgeError):
        stage_trigger_payload("https://example.test/archive", tmp_path, fetcher=fetcher)


def test_observational_bridge_can_stage_and_ingest_remote_triggers(tmp_path: Path) -> None:
    fetcher = _build_fetcher(
        {
            "https://example.test/gw.json": (
                json.dumps(
                    {
                        "source": "LIGO/Virgo",
                        "event_name": "GW-remote",
                        "redshift": "0.01",
                        "luminosity_distance_mpc": "44.479592",
                        "peak_strain": "1e-21",
                    }
                ),
                "application/json",
            ),
            "https://example.test/jwst.csv": (
                "source,event_name,redshift,observed_expansion_rate_km_s_mpc,galaxy_count\n"
                "JWST,JADES-remote,12,68.5,14\n",
                "text/csv",
            ),
        }
    )

    bridge = ObservationalBridge(trigger_directory=tmp_path, precision=50)
    triggers = bridge.ingest_remote_triggers(
        ["https://example.test/gw.json", "https://example.test/jwst.csv"],
        fetcher=fetcher,
    )

    assert len(triggers) == 2
    assert triggers[0].event_id == "GW-remote"
    assert triggers[1].event_id == "JADES-remote"
