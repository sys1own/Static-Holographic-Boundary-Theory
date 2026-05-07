from __future__ import annotations

import json
from decimal import Decimal

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.verification.live_bridge import (
    LiveVerificationBridge,
    ADSLiteratureRecord,
    JWSTFormationObservation,
    LIGOStrainObservation,
    build_ads_search_url,
    build_gwosc_catalog_events_url,
    build_simbad_identifier_url,
    build_vizier_catalog_url,
    calculate_live_tension_score,
)


def _build_fetcher(payloads: dict[str, tuple[str, str]]) -> object:
    def fetcher(url: str, _headers: dict[str, str], _timeout: float) -> tuple[int, dict[str, str], str]:
        body, content_type = payloads[url]
        return 200, {"Content-Type": content_type}, body

    return fetcher


def test_live_bridge_fetches_vizier_jwst_rows_and_resolves_simbad() -> None:
    vizier_url = build_vizier_catalog_url("J/ApJS/267/44/table1", columns=("ID", "z", "Mstar"), limit=2)
    simbad_url = build_simbad_identifier_url("JADES-1")
    fetcher = _build_fetcher(
        {
            vizier_url: (
                "#Catalog rows\n"
                "ID\tz\tMstar\n"
                "JADES-1\t12.3\t9.1\n"
                "JADES-2\t10.8\t8.7\n",
                "text/tab-separated-values",
            ),
            simbad_url: (
                "Identifier: GN-z11\nObject type: Galaxy\nRedshift: 10.6\n",
                "text/plain",
            ),
        }
    )

    bridge = LiveVerificationBridge(precision=50)
    observations = bridge.fetch_vizier_jwst_catalog(
        "J/ApJS/267/44/table1",
        columns=("ID", "z", "Mstar"),
        limit=2,
        resolve_with_simbad=True,
        max_simbad_lookups=1,
        fetcher=fetcher,
    )

    assert len(observations) == 2
    assert observations[0].identifier == "JADES-1"
    assert observations[0].canonical_identifier == "GN-z11"
    assert observations[0].object_type == "Galaxy"
    assert observations[0].formation_density_proxy > 0
    assert observations[1].canonical_identifier is None


def test_live_bridge_fetches_ads_literature_with_fake_fetcher() -> None:
    url = build_ads_search_url("title:JWST AND body:\"gravitational waves\"", rows=2)
    fetcher = _build_fetcher(
        {
            url: (
                json.dumps(
                    {
                        "response": {
                            "docs": [
                                {
                                    "bibcode": "2026ApJ...123..456A",
                                    "title": ["JWST and gravitational-wave follow-up"],
                                    "pubdate": "2026-04-01",
                                    "author": ["Ada Lovelace", "Katherine Johnson"],
                                    "citation_count": 7,
                                }
                            ]
                        }
                    }
                ),
                "application/json",
            )
        }
    )

    records = LiveVerificationBridge(precision=50).fetch_ads_literature(
        'title:JWST AND body:"gravitational waves"',
        rows=2,
        token="unit-test-token",
        fetcher=fetcher,
    )

    assert len(records) == 1
    assert records[0].bibcode == "2026ApJ...123..456A"
    assert records[0].authors == ("Ada Lovelace", "Katherine Johnson")
    assert records[0].citation_count == 7


def test_live_bridge_fetches_ligo_events_from_gwosc_style_payload() -> None:
    url = build_gwosc_catalog_events_url("GWTC")
    fetcher = _build_fetcher(
        {
            url: (
                json.dumps(
                    {
                        "results": [
                            {
                                "commonName": "GW150914",
                                "catalog": {"shortName": "GWTC"},
                                "peak_strain": "1e-21",
                                "redshift": "0.09",
                                "luminosity_distance_mpc": "410",
                            }
                        ]
                    }
                ),
                "application/json",
            )
        }
    )

    observations = LiveVerificationBridge(precision=50).fetch_ligo_events(fetcher=fetcher)

    assert len(observations) == 1
    assert observations[0].event_id == "GW150914"
    assert observations[0].catalog == "GWTC"
    assert observations[0].peak_strain == Decimal("1e-21")


def test_live_bridge_computes_theory_tension_score() -> None:
    bridge = LiveVerificationBridge(precision=50)
    jwst = (
        JWSTFormationObservation(
            identifier="JADES-1",
            catalog="J/ApJS/267/44/table1",
            redshift=Decimal("12.1"),
            stellar_mass=Decimal("8.0"),
            formation_density_proxy=Decimal("8.0"),
        ),
        JWSTFormationObservation(
            identifier="JADES-2",
            catalog="J/ApJS/267/44/table1",
            redshift=Decimal("11.4"),
            stellar_mass=Decimal("10.0"),
            formation_density_proxy=Decimal("10.0"),
        ),
    )
    ligo = (
        LIGOStrainObservation(
            event_id="GW150914",
            catalog="GWTC",
            peak_strain=Decimal("1e-21"),
            network_snr=Decimal("24"),
            redshift=Decimal("0.09"),
            luminosity_distance_mpc=Decimal("410"),
        ),
    )
    literature = (
        ADSLiteratureRecord(
            bibcode="2026ApJ...123..456A",
            title=("JWST and gravitational-wave follow-up",),
            pubdate="2026-04-01",
            authors=("Ada Lovelace",),
            citation_count=7,
        ),
    )

    score = bridge.compute_tension_score(jwst, ligo, literature=literature)

    assert score.benchmark_branch == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
    assert score.kernel_profile.jwst_formation_peak > Decimal("0.7")
    assert score.kernel_profile.ligo_strain_peak > Decimal("0.7")
    assert score.comparison_count == 3
    assert score.chi_square > 0
    assert Decimal("0") <= score.tension_score <= Decimal("100")
    assert score.multi_messenger is True
    assert score.data_sources == ("GWTC", "J/ApJS/267/44/table1", "NASA ADS")


def test_calculate_live_tension_score_orchestrates_remote_fetches() -> None:
    vizier_url = build_vizier_catalog_url("J/ApJS/267/44/table1", columns=("ID", "z", "Mstar"), limit=1)
    ligo_url = build_gwosc_catalog_events_url("GWTC")
    ads_url = build_ads_search_url("title:JWST", rows=1)

    vizier_fetcher = _build_fetcher(
        {
            vizier_url: (
                "ID\tz\tMstar\nJADES-1\t12.0\t9.1\n",
                "text/tab-separated-values",
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
                                "commonName": "GW150914",
                                "catalog": {"shortName": "GWTC"},
                                "network_snr": "24",
                                "luminosity_distance_mpc": "410",
                            }
                        ]
                    }
                ),
                "application/json",
            )
        }
    )
    ads_fetcher = _build_fetcher(
        {
            ads_url: (
                json.dumps(
                    {
                        "response": {
                            "docs": [
                                {
                                    "bibcode": "2026ApJ...123..456A",
                                    "title": ["JWST paper"],
                                    "pubdate": "2026-04-01",
                                    "author": ["Ada Lovelace"],
                                    "citation_count": 7,
                                }
                            ]
                        }
                    }
                ),
                "application/json",
            )
        }
    )

    score = calculate_live_tension_score(
        "J/ApJS/267/44/table1",
        jwst_columns=("ID", "z", "Mstar"),
        jwst_limit=1,
        ligo_url=ligo_url,
        ads_query="title:JWST",
        ads_rows=1,
        ads_token="unit-test-token",
        vizier_fetcher=vizier_fetcher,
        ligo_fetcher=ligo_fetcher,
        ads_fetcher=ads_fetcher,
    )

    assert score.jwst_observations[0].identifier == "JADES-1"
    assert score.ligo_observations[0].event_id == "GW150914"
    assert score.literature[0].bibcode == "2026ApJ...123..456A"
    assert score.statement.startswith("The (26, 8, 312) kernel returns chi^2=")
