#!/usr/bin/env python3
"""
API tests for Phase 3: symptom logging + insights endpoints.

Uses stubbed DB (entity-detecting FakeSession) + FakeCache so no live
PostGIS/Redis is needed. Verifies:

  * POST /patients/{id}/symptoms — 201, auto weather snapshot, cache
    invalidation, validation (422)
  * GET /patients/{id}/insights — empty history and populated history
  * safety assessment reflects history (score drops when similar
    conditions are in the patient's symptom log)
"""

import sys
import os
from datetime import datetime
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest
from fastapi.testclient import TestClient

import main
from models import (
    PatientProfile,
    WeatherReading,
    AirQualityReading,
    SymptomLog,
    NewsArticle,
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class ScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class ScalarsResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _Scalars(self._rows)


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class GenericResult:
    rowcount = 1


class FakeSession:
    def __init__(
        self,
        profile=None,
        weather_rows=None,
        aq_rows=None,
        symptom_rows=None,
        news_rows=None,
    ):
        self.profile = profile
        self.weather_rows = weather_rows or []
        self.aq_rows = aq_rows or []
        self.symptom_rows = symptom_rows or []
        self.news_rows = news_rows or []
        self.added = []
        self.commits = 0

    def execute(self, statement):
        try:
            entity = statement.column_descriptions[0]["entity"]
        except Exception:
            entity = None
        if entity is PatientProfile:
            return ScalarResult([self.profile] if self.profile else [])
        if entity is WeatherReading:
            return ScalarsResult(self.weather_rows)
        if entity is AirQualityReading:
            return ScalarsResult(self.aq_rows)
        if entity is SymptomLog:
            return ScalarsResult(self.symptom_rows)
        if entity is NewsArticle:
            return ScalarsResult(self.news_rows)
        return GenericResult()

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        # Simulate DB population: id + server-side defaults
        if getattr(obj, "id", None) is None:
            obj.id = 1
        if getattr(obj, "logged_at", None) is None:
            obj.logged_at = datetime.utcnow()


class FakeCache:
    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.deleted: list = []

    async def get(self, key):
        return self.store.get(key)

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
        self.deleted.append(pattern)
        return len(keys)


def make_weather_row(**overrides):
    defaults = dict(
        pm2_5=45.0,
        pm10=62.0,
        ozone=80.0,
        nitrogen_dioxide=35.0,
        aqi=135,
        temperature_celsius=28.0,
        humidity=88.0,
        wind_kph=6.0,
        wind_direction="SW",
        timestamp=datetime.utcnow(),
        grid_cell_id=1,
    )
    defaults.update(overrides)
    return type("WeatherRow", (), defaults)()


def make_symptom_row(severity, hour, **snapshot):
    defaults = dict(
        id=1,
        severity=severity,
        symptom_type="coughing",
        weather_snapshot=snapshot or {"aqi": 125, "pm25": 40, "humidity": 78},
        logged_at=datetime(2026, 8, 1, hour, 0),
        lat=40.7128,
        lon=-74.006,
    )
    return type("SymptomRow", (), defaults)()


async def _async(obj):
    return obj


def _client(session):
    def override_get_db():
        yield session

    main.app.dependency_overrides[main.get_db] = override_get_db
    main._test_cache = FakeCache()
    main.get_cache = lambda: _async(main._test_cache)
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def _reset():
    yield
    main.app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# Symptom logging
# --------------------------------------------------------------------------
def test_log_symptom_201_with_snapshot_and_cache_invalidation():
    session = FakeSession(
        weather_rows=[make_weather_row()],
        aq_rows=[],
    )
    client = _client(session)

    resp = client.post(
        "/api/v1/patients/pat1/symptoms",
        json={
            "symptom_type": "coughing",
            "severity": 3,
            "lat": 40.7128,
            "lon": -74.006,
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["symptom_type"] == "coughing"
    assert body["severity"] == 3
    # snapshot captured from the weather row
    assert body["weather_snapshot"]["aqi"] == 135
    assert body["weather_snapshot"]["pm25"] == 45.0
    assert body["logged_at"]
    # a SymptomLog was added + committed
    assert any(isinstance(o, SymptomLog) for o in session.added)
    assert session.commits >= 1
    # patient + insights caches invalidated
    assert "safety_assessment:pat1:*" in main._test_cache.deleted
    assert "insights:pat1:*" in main._test_cache.deleted


def test_log_symptom_invalid_type_422():
    session = FakeSession(weather_rows=[make_weather_row()])
    client = _client(session)
    resp = client.post(
        "/api/v1/patients/pat2/symptoms",
        json={"symptom_type": "headache", "severity": 3, "lat": 40.0, "lon": -74.0},
    )
    assert resp.status_code == 422


def test_log_symptom_invalid_severity_422():
    session = FakeSession(weather_rows=[make_weather_row()])
    client = _client(session)
    resp = client.post(
        "/api/v1/patients/pat3/symptoms",
        json={"symptom_type": "coughing", "severity": 9, "lat": 40.0, "lon": -74.0},
    )
    assert resp.status_code == 422


# --------------------------------------------------------------------------
# Insights
# --------------------------------------------------------------------------
def test_insights_empty_history_graceful():
    session = FakeSession()
    client = _client(session)
    resp = client.get("/api/v1/patients/pat4/insights")
    assert resp.status_code == 200
    body = resp.json()
    assert body["top_triggers"] == []
    assert body["safest_times"] == []
    assert body["riskiest_times"] == []
    assert "Log more symptoms" in body["recent_trend"]


def test_insights_with_history_returns_triggers():
    session = FakeSession(
        profile=PatientProfile(
            id=1,
            user_id="pat5",
            conditions=["asthma"],
        ),
        symptom_rows=[
            make_symptom_row(1, 7, aqi=40, pm25=6, humidity=45, temperature=18),
            make_symptom_row(1, 7, aqi=42, pm25=7, humidity=46, temperature=18),
            make_symptom_row(2, 8, aqi=80, pm25=22, humidity=55, temperature=21),
            make_symptom_row(4, 17, aqi=150, pm25=60, humidity=75, temperature=26),
            make_symptom_row(5, 17, aqi=180, pm25=85, humidity=80, temperature=27),
        ],
    )
    client = _client(session)
    resp = client.get("/api/v1/patients/pat5/insights")
    assert resp.status_code == 200
    body = resp.json()
    assert any(t["factor"] == "PM2.5" for t in body["top_triggers"])
    assert body["safest_times"][0]["hour"] == 7
    assert body["riskiest_times"][0]["hour"] == 17


# --------------------------------------------------------------------------
# Safety assessment reflects history
# --------------------------------------------------------------------------
def test_assessment_reflects_similar_history():
    session = FakeSession(
        weather_rows=[make_weather_row()],
        aq_rows=[],
        symptom_rows=[
            make_symptom_row(4, 8, aqi=130, pm25=42, humidity=85),
            make_symptom_row(4, 9, aqi=140, pm25=48, humidity=88),
            make_symptom_row(4, 10, aqi=135, pm25=45, humidity=86),
        ],
    )
    client = _client(session)
    # current conditions (from weather row) are aqi 135 / pm25 45 / hum 88 —
    # very similar to the symptom-history conditions
    resp = client.get(
        "/api/v1/patients/pat6/safety-assessment",
        params={"lat": 40.7128, "lon": -74.006},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["component_scores"]["history"] < 100
    # history factor surfaced
    assert any(f["factor"] == "Personal history" for f in body["contributing_factors"])


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
