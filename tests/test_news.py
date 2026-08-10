#!/usr/bin/env python3
"""
Tests for the News Intelligence Layer (Phase 2).

Covers:
  * rule-based enrichment (category / severity / relevance / location)
  * fetchers (AirNow official alerts + NewsAPI media) with mocked HTTP
  * deduplication
  * persistence (upsert / nearby query / stale deactivation)
  * API: /news/nearby + news folded into the safety assessment
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Any, Dict
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest
from fastapi.testclient import TestClient

import main
from models import (
    NewsArticle,
    PatientProfile,
    WeatherReading,
    AirQualityReading,
    SymptomLog,
)
from services import news_processor as proc
from services.news_fetcher import (
    AirNowFetcher,
    NewsAPIFetcher,
    NewsAggregator,
    RawNewsItem,
)
from services import news_store
from services.news_scheduler import NewsScheduler
import services.news_scheduler as scheduler_mod


# ==========================================================================
# Fakes
# ==========================================================================
class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeClient:
    """Minimal httpx.AsyncClient stand-in with per-URL canned responses."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.mapping.get(url, FakeResponse({}, status=404))


class FakeScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class FakeScalarsResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _FakeScalars(self._rows)


class _FakeScalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class GenericResult:
    rowcount = 1


class FakeSession:
    """Handles selects for the entities the news layer + safety engine touch."""

    def __init__(
        self,
        profile=None,
        weather_rows=None,
        aq_rows=None,
        news_rows=None,
        symptom_rows=None,
    ):
        self.profile = profile
        self.weather_rows = weather_rows or []
        self.aq_rows = aq_rows or []
        self.news_rows = news_rows or []
        self.symptom_rows = symptom_rows or []
        self.added = []
        self.commits = 0

    def execute(self, statement):
        try:
            entity = statement.column_descriptions[0]["entity"]
        except Exception:
            entity = None
        if entity is PatientProfile:
            return FakeScalarResult([self.profile] if self.profile else [])
        if entity is WeatherReading:
            return FakeScalarsResult(self.weather_rows)
        if entity is AirQualityReading:
            return FakeScalarsResult(self.aq_rows)
        if entity is NewsArticle:
            return FakeScalarsResult(self.news_rows)
        if entity is SymptomLog:
            return FakeScalarsResult(self.symptom_rows)
        return GenericResult()

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        pass


class FakeCache:
    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.hits = 0

    async def get(self, key):
        if key in self.store:
            self.hits += 1
            return self.store[key]
        return None

    async def set(self, key, value, ttl=None):
        self.store[key] = value
        return True

    async def delete(self, key):
        self.store.pop(key, None)
        return True

    async def delete_pattern(self, pattern):
        prefix = pattern.split(":")[0] + ":"
        keys = [k for k in self.store if k.startswith(prefix)]
        for k in keys:
            self.store.pop(k, None)
        return len(keys)


def make_news_row(**overrides):
    defaults = dict(
        id=1,
        external_id="test:1",
        source_name="EPA AirNow",
        source_type="official_alert",
        title="Wildfire smoke advisory near Portland",
        summary="Smoke from the forest fire is affecting air quality.",
        url=None,
        published_at=datetime.utcnow(),
        fetched_at=datetime.utcnow(),
        latitude=40.7128,
        longitude=-74.0060,
        event_category="wildfire",
        severity=85,
        respiratory_relevance=90,
        is_active=True,
        raw_metadata={},
    )
    defaults.update(overrides)
    return type("NewsRow", (), defaults)()


def make_weather_row(**overrides):
    defaults = dict(
        pm2_5=5.0,
        pm10=10.0,
        ozone=20.0,
        nitrogen_dioxide=8.0,
        aqi=25,
        temperature_celsius=20.0,
        humidity=45.0,
        wind_kph=12.0,
        wind_direction="SW",
        timestamp=datetime.utcnow(),
        grid_cell_id=1,
    )
    defaults.update(overrides)
    return type("WeatherRow", (), defaults)()


def make_aq_row(**overrides):
    defaults = dict(
        aqi=30,
        no2_gt=8.0,
        pt08_s5_o3=20.0,
        temperature=20.0,
        relative_humidity=45.0,
        timestamp=datetime.utcnow(),
        grid_cell_id=1,
    )
    defaults.update(overrides)
    return type("AQRow", (), defaults)()


def _client(session):
    def override_get_db():
        yield session

    main.app.dependency_overrides[main.get_db] = override_get_db
    main._test_cache = FakeCache()
    main.get_cache = lambda: _async(main._test_cache)
    return TestClient(main.app)


async def _async(obj):
    return obj


@pytest.fixture(autouse=True)
def _reset_main():
    yield
    main.app.dependency_overrides.clear()
    if hasattr(main, "get_cache_original"):
        main.get_cache = main.get_cache_original


# ==========================================================================
# Rule-based processor
# ==========================================================================
def test_categorize_wildfire():
    assert proc.categorize_event("Wildfire burns near Sacramento") == "wildfire"


def test_categorize_health_advisory():
    assert (
        proc.categorize_event("Air quality alert for sensitive groups in Los Angeles")
        == "health_advisory"
    )


def test_categorize_pollen():
    assert proc.categorize_event("Ragweed pollen season peaks") == "pollen"


def test_categorize_general_fallback():
    assert proc.categorize_event("City council meets Tuesday") == "general"


def test_severity_official_base_plus_urgency():
    # official base 70 + "warning" 20 = 90
    assert (
        proc.score_severity(
            "Warning: hazardous air in Chicago", source_type="official_alert"
        )
        >= 90
    )


def test_severity_capped_at_100():
    text = "Emergency warning hazardous severe critical in Denver"
    assert proc.score_severity(text, source_type="official_alert") == 100


def test_relevance_wildfire_asthma_boost():
    # wildfire 90 + asthma 15 = 100
    assert (
        proc.score_respiratory_relevance(
            "Wildfire smoke worsens asthma in Seattle", category="wildfire"
        )
        == 100
    )


def test_relevance_general_low():
    assert (
        proc.score_respiratory_relevance("New park opens downtown", category="general")
        == 20
    )


def test_extract_location_from_gazetteer():
    lat, lon = proc.extract_location("Smoke plume over Denver", "advisory")
    assert (lat, lon) == pytest.approx((39.7392, -104.9903), abs=0.01)


def test_extract_location_prefers_source_coords():
    lat, lon = proc.extract_location(
        "Alert near Los Angeles", "", source_lat=34.05, source_lon=-118.24
    )
    assert (lat, lon) == pytest.approx((34.05, -118.24), abs=0.001)


def test_extract_location_none_when_unknown():
    lat, lon = proc.extract_location("Unusual event somewhere far away")
    assert lat is None and lon is None


def test_process_article_end_to_end():
    result = proc.process_article(
        external_id="n:1",
        source_name="EPA",
        source_type="official_alert",
        title="Wildfire smoke advisory near Portland",
        summary="Asthma patients should stay indoors.",
    )
    assert result["event_category"] == "wildfire"
    assert result["severity"] >= 70
    assert result["respiratory_relevance"] >= 90
    assert result["latitude"] is not None  # Portland in gazetteer


# ==========================================================================
# Fetchers
# ==========================================================================
def test_airnow_skips_without_key():
    fetcher = AirNowFetcher(api_key="")
    items = asyncio.run(fetcher.fetch(FakeClient({})))
    assert items == []


def test_airnow_emits_only_unhealthy():
    fetcher = AirNowFetcher(api_key="key", max_items=10)
    rows = [
        {
            "Latitude": 40.7,
            "Longitude": -74.0,
            "AQI": 150,
            "ParameterName": "PM2.5",
            "ReportingArea": "NYC",
            "Category": {"Name": "Unhealthy"},
            "DateObserved": "2026-08-10",
            "HourObserved": 12,
        },
        {
            "Latitude": 40.7,
            "Longitude": -74.0,
            "AQI": 50,
            "ParameterName": "PM2.5",
            "ReportingArea": "NYC",
            "Category": {"Name": "Good"},
            "DateObserved": "2026-08-10",
            "HourObserved": 11,
        },
    ]
    client = FakeClient({"https://airnowapi.org/aq/data/": FakeResponse(rows)})
    items = asyncio.run(fetcher.fetch(client))
    assert len(items) == 1
    assert items[0].source_type == "official_alert"
    assert items[0].lat == 40.7
    assert items[0].external_id.startswith("airnow:")


def test_newsapi_skips_without_key():
    fetcher = NewsAPIFetcher(api_key="")
    assert asyncio.run(fetcher.fetch(FakeClient({}))) == []


def test_newsapi_parses_articles():
    fetcher = NewsAPIFetcher(api_key="key", max_items=5)
    payload = {
        "status": "ok",
        "articles": [
            {
                "source": {"name": "Reuters"},
                "title": "Wildfire smoke blankets Seattle",
                "description": "Air quality unhealthy",
                "url": "https://x.com/a",
                "publishedAt": "2026-08-10T12:00:00Z",
            },
        ],
    }
    client = FakeClient({"https://newsapi.org/v2/everything": FakeResponse(payload)})
    items = asyncio.run(fetcher.fetch(client))
    assert len(items) == 1
    assert items[0].source_name == "Reuters"
    assert items[0].source_type == "news_media"
    assert items[0].external_id == "newsapi:https://x.com/a"
    assert items[0].published_at is not None


def test_aggregator_dedup_near_identical_titles():
    now = datetime.utcnow()
    items = [
        RawNewsItem(
            "a",
            "S1",
            "news_media",
            "Wildfire smoke over Portland",
            url="http://x/1",
            published_at=now,
        ),
        RawNewsItem(
            "b",
            "S2",
            "news_media",
            "Wildfire smoke over Portland",
            url="http://x/2",
            published_at=now,
        ),
        RawNewsItem(
            "c",
            "S3",
            "news_media",
            "Pollen counts rise in Miami",
            url="http://x/3",
            published_at=now,
        ),
    ]
    agg = NewsAggregator(fetchers=[])
    out = agg._dedupe(items)
    assert len(out) == 2  # one of the two wildfire titles dropped


# ==========================================================================
# Store
# ==========================================================================
def test_haversine_identity_and_distance():
    assert news_store.haversine_km(40.0, -74.0, 40.0, -74.0) == 0.0
    # NYC -> Philadelphia is ~130 km
    d = news_store.haversine_km(40.7128, -74.0060, 39.9526, -75.1652)
    assert 120 < d < 145


def test_upsert_articles_commits():
    db = FakeSession()
    count = news_store.upsert_articles(
        db,
        [
            {
                "external_id": "n:1",
                "source_name": "EPA",
                "source_type": "official_alert",
                "title": "Alert",
                "summary": "",
                "url": None,
                "published_at": datetime.utcnow(),
                "latitude": 40.7,
                "longitude": -74.0,
                "event_category": "wildfire",
                "severity": 80,
                "respiratory_relevance": 90,
                "raw_metadata": {},
            }
        ],
    )
    assert count >= 1
    assert db.commits >= 1


def test_get_active_nearby_filters_by_radius():
    rows = [
        make_news_row(id=1, latitude=40.7128, longitude=-74.0060),  # ~0 km
        make_news_row(id=2, latitude=40.7, longitude=-74.0),  # ~1 km
        make_news_row(id=3, latitude=45.0, longitude=-100.0),  # far away
    ]
    db = FakeSession(news_rows=rows)
    results = news_store.get_active_articles_nearby(db, 40.7128, -74.0060, radius_km=10)
    assert len(results) == 2
    assert all(r["distance_km"] <= 10 for r in results)


def test_deactivate_stale():
    db = FakeSession()
    count = news_store.deactivate_stale_articles(db, older_than_hours=24)
    assert count >= 0  # generic result rowcount
    assert db.commits >= 1


# ==========================================================================
# Scheduler
# ==========================================================================
def test_scheduler_tick_noop_without_keys():
    async def run():
        with patch.object(scheduler_mod, "get_cache", lambda: _async(FakeCache())):
            await NewsScheduler(interval_minutes=5).tick()

    asyncio.run(run())  # must not raise


# ==========================================================================
# API
# ==========================================================================
def test_news_nearby_endpoint_and_cache():
    rows = [
        make_news_row(
            id=1,
            latitude=40.7128,
            longitude=-74.0060,
            title="Wildfire smoke advisory in NYC",
            event_category="wildfire",
            severity=85,
            respiratory_relevance=90,
        )
    ]
    session = FakeSession(news_rows=rows)
    client = _client(session)

    r1 = client.get(
        "/api/v1/news/nearby",
        params={"lat": 40.7128, "lon": -74.006, "radius_km": 25},
    )
    assert r1.status_code == 200
    body = r1.json()
    assert len(body) == 1
    assert body[0]["event_category"] == "wildfire"
    assert body[0]["distance_km"] <= 25
    assert "title" in body[0]

    # second call served from cache
    r2 = client.get(
        "/api/v1/news/nearby",
        params={"lat": 40.7128, "lon": -74.006, "radius_km": 25},
    )
    assert r2.json() == body
    assert main._test_cache.hits >= 1


def test_safety_assessment_news_reduces_score():
    session = FakeSession(
        weather_rows=[make_weather_row()],
        aq_rows=[make_aq_row()],
        news_rows=[],  # baseline: no news
    )
    client = _client(session)
    baseline = client.get(
        "/api/v1/patients/hank/safety-assessment",
        params={"lat": 40.7128, "lon": -74.006},
    ).json()

    # Now with a severe nearby wildfire event (invalidate the cache so
    # the new event actually reaches the scorer)
    session.news_rows = [
        make_news_row(
            id=1,
            latitude=40.7128,
            longitude=-74.0060,
            title="Wildfire smoke advisory in NYC",
            event_category="wildfire",
            severity=90,
            respiratory_relevance=95,
        )
    ]
    main._test_cache.store = {}
    with_news = client.get(
        "/api/v1/patients/hank/safety-assessment",
        params={"lat": 40.7128, "lon": -74.006},
    ).json()

    assert with_news["safety_score"] < baseline["safety_score"]
    assert with_news["nearby_events"]
    assert with_news["nearby_events"][0]["category"] == "wildfire"
    texts = " ".join(r["text"].lower() for r in with_news["recommendations"])
    assert "wildfire" in texts


def test_safety_assessment_news_does_not_break_no_news():
    session = FakeSession(
        weather_rows=[make_weather_row()],
        aq_rows=[make_aq_row()],
        news_rows=[],
    )
    client = _client(session)
    body = client.get(
        "/api/v1/patients/iris/safety-assessment",
        params={"lat": 40.7128, "lon": -74.006},
    ).json()
    assert body["data_status"] == "available"
    assert body["nearby_events"] == []
    assert 0 <= body["safety_score"] <= 100


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
