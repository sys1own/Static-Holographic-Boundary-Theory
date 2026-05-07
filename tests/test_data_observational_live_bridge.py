from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.observational.live_bridge import (
    DEFAULT_H0_ADS_FIELDS,
    DEFAULT_JWST_ADS_QUERY,
    SystemDecoherenceWarning,
    build_ads_search_url,
    build_live_gwosc_events_url,
    calculate_live_h0_tension,
)


def _build_fetcher(payloads: dict[str, tuple[str, str]]) -> object:
    def fetcher(url: str, _headers: dict[str, str], _timeout: float) -> tuple[int, dict[str, str], str]:
        body, content_type = payloads[url]
        return 200, {"Content-Type": content_type}, body

    return fetcher


def test_calculate_live_h0_tension_combines_jwst_ads_and_ligo_gwosc_sources() -> None:
    ads_url = build_ads_search_url(DEFAULT_JWST_ADS_QUERY, rows=3, fields=DEFAULT_H0_ADS_FIELDS)
    ligo_url = build_live_gwosc_events_url("GWTC")

    ads_fetcher = _build_fetcher(
        {
            ads_url: (
                json.dumps(
                    {
                        "response": {
                            "docs": [
                                {
                                    "bibcode": "2026ApJ...123..456A",
                                    "title": ["JWST strong-lens expansion audit"],
                                    "pubdate": "2026-04-01",
                                    "author": ["Ada Lovelace"],
                                    "citation_count": 4,
                                    "abstract": "Using JWST lensing we infer H0 = 74.3 ± 1.2 km s^-1 Mpc^-1.",
                                }
                            ]
                        }
                    }
                ),
                "application/json",
            )
        }
    )
    ligo_fetcher = _build_fetcher(
        {
            ligo_url: (
                json.dumps(
                    {
                        "results": [
                            {
                                "commonName": "GW240109_050431",
                                "catalog": {"shortName": "GWTC"},
                                "default_parameters": [
                                    {"name": "redshift", "best": "0.010"},
                                    {"name": "luminosity_distance", "best": "43.0"},
                                    {"name": "network_matched_filter_snr", "best": "19.8"},
                                ],
                            },
                            {
                                "commonName": "GW240216_050213",
                                "catalog": {"shortName": "GWTC"},
                                "default_parameters": [
                                    {"name": "redshift", "best": "0.011"},
                                    {"name": "luminosity_distance", "best": "46.0"},
                                ],
                            },
                        ]
                    }
                ),
                "application/json",
            )
        }
    )

    with pytest.warns(SystemDecoherenceWarning):
        report = calculate_live_h0_tension(
            ads_rows=3,
            ads_token="unit-test-token",
            ligo_url=ligo_url,
            ads_fetcher=ads_fetcher,
            ligo_fetcher=ligo_fetcher,
        )

    measurements = {measurement.channel: measurement for measurement in report.measurements}
    assert measurements["jwst"].observation_id == "2026ApJ...123..456A"
    assert measurements["jwst"].observed_h0_km_s_mpc == Decimal("74.3")
    assert measurements["ligo"].observation_id == "GW240109_050431"
    assert measurements["ligo"].observed_h0_km_s_mpc > Decimal("65")
    assert measurements["ligo"].observed_h0_km_s_mpc < Decimal("75")
    assert report.max_fractional_drift > Decimal("1e-15")
    assert report.drift_to_moat_ratio > Decimal("1")
    assert report.decoherence_triggered is True


def test_calculate_live_h0_tension_skips_ads_docs_without_extractable_h0() -> None:
    ads_url = build_ads_search_url(DEFAULT_JWST_ADS_QUERY, rows=2, fields=DEFAULT_H0_ADS_FIELDS)
    ligo_url = build_live_gwosc_events_url("GWTC")

    ads_fetcher = _build_fetcher(
        {
            ads_url: (
                json.dumps(
                    {
                        "response": {
                            "docs": [
                                {
                                    "bibcode": "2026ApJ...111..111A",
                                    "title": ["JWST calibrates high-redshift supernovae"],
                                    "pubdate": "2026-04-02",
                                    "author": ["No Number"],
                                    "citation_count": 1,
                                    "abstract": "We revisit the H0 tension with improved systematics.",
                                },
                                {
                                    "bibcode": "2026ApJ...222..222B",
                                    "title": ["JWST local ladder reanalysis"],
                                    "pubdate": "2026-03-18",
                                    "author": ["Grace Hopper"],
                                    "citation_count": 2,
                                    "abstract": "The inferred H0 = 71.6 ± 1.5 km s^-1 Mpc^-1 in this JWST sample.",
                                },
                            ]
                        }
                    }
                ),
                "application/json",
            )
        }
    )
    ligo_fetcher = _build_fetcher(
        {
            ligo_url: (
                json.dumps(
                    {
                        "results": [
                            {
                                "commonName": "GW240109_050431",
                                "catalog": {"shortName": "GWTC"},
                                "default_parameters": [
                                    {"name": "redshift", "best": "0.010"},
                                    {"name": "luminosity_distance", "best": "43.0"},
                                ],
                            }
                        ]
                    }
                ),
                "application/json",
            )
        }
    )

    report = calculate_live_h0_tension(
        ads_rows=2,
        ads_token="unit-test-token",
        ligo_url=ligo_url,
        ads_fetcher=ads_fetcher,
        ligo_fetcher=ligo_fetcher,
        warn_on_decoherence=False,
    )

    measurements = {measurement.channel: measurement for measurement in report.measurements}
    assert measurements["jwst"].observation_id == "2026ApJ...222..222B"
    assert measurements["jwst"].observed_h0_km_s_mpc == Decimal("71.6")
