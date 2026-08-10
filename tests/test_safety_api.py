#!/usr/bin/env python3
"""
Integration tests for the safety-assessment API endpoint.

Uses FastAPI TestClient with a stubbed database session and cache so the
full request flow can be exercised without a live PostGIS/Redis:

  * patient auto-registration
  * environmental snapshot building from weather/AQ readings
  * risk scoring + recommendations
  * response serialization
  * cache write-then-read round trip

Run:  pytest tests/test_safety_api.py -v
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import main
from models import PatientProfile, WeatherReading


# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------
class FakeCache:
    """In-memory Redis-cache stand-in with the same async interface."""

    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[Any]:
        if key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self.store[key] = value
        return True


class FakeScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class FakeScalarsResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return FakeScalars(self._rows)

    def all(self):
        return self._rows


class FakeScalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeSession:
    """Stub DB session returning controlled rows per entity type."""

    def __init__(self, profile=None, weather_rows=None, aq_rows=None):
        self.profile = profile
        self.weather_rows = weather_rows or []
        self.aq_rows = aq_rows or []
        self.added = []
        self.committed = 0
        self.refreshed = 0

    def execute(self, statement):
        # Detect entity from the select statement
        try:
            descriptions = statement.column_descriptions
            entity = descriptions[0]["entity"] if descriptions else None
        except Exception:
            entity = None
        if entity is PatientProfile:
            return FakeScalarResult([self.profile] if self.profile else [])
        if entity is WeatherReading:
            return FakeScalarsResult(self.weather_rows)
        # AirQualityReading and anything else -> empty
        return FakeScalarsResult(self.aq_rows)

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed += 1

    def refresh(self, obj):
        self.refreshed += 1


def make_weather_row(**overrides):
    """Fake WeatherReading with the attributes the snapshot builder reads."""
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
        grid_cell_id=123,
    )
    defaults.update(overrides)
    return type("WeatherRow", (), defaults)()


def make_aq_row(**overrides):
    """Fake AirQualityReading with the attributes the snapshot builder reads."""
    defaults = dict(
        aqi=140,
        no2_gt=40.0,
        pt08_s5_o3=85.0,
        temperature=28.0,
        relative_humidity=88.0,
        timestamp=datetime.utcnow(),
        grid_cell_id=123,
    )
    defaults.update(overrides)
    return type("AQRow", (), defaults)()


@pytest.fixture(autouse=True)
def reset_app_state():
    """Reset module-level mocks and metrics between tests."""
    main.get_cache_original = getattr(main, "get_cache", None)
    main._test_cache = FakeCache()
    main.get_cache = lambda: _async(main._test_cache)
    yield
    if main.get_cache_original is not None:
        main.get_cache = main.get_cache_original


async def _async(obj):
    return obj


def _client(session: FakeSession):
    def override_get_db():
        yield session

    main.app.dependency_overrides[main.get_db] = override_get_db
    client = TestClient(main.app)
    return client


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_auto_registers_patient_and_returns_assessment():
    session = FakeSession(
        profile=None,
        weather_rows=[make_weather_row()],
    )
    client = _client(session)

    resp = client.get(
        "/api/v1/patients/alice/safety-assessment",
        params={"lat": 40.7128, "lon": -74.0060},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["user_id"] == "alice"
    assert isinstance(body["safety_score"], int)
    assert 0 <= body["safety_score"] <= 100
    assert body["risk_level"] in ("low", "moderate", "high", "very_high")
    assert body["summary"]
    assert body["data_status"] in ("available", "partial", "unavailable")
    assert body["component_scores"]
    assert body["contributions"]
    assert isinstance(body["recommendations"], list)
    assert body["current_conditions"]["pm25"] == 45.0
    assert body["generated_at"]

    # Auto-registration happened
    assert session.committed >= 1
    assert len(session.added) == 1
    assert isinstance(session.added[0], PatientProfile)


def test_cache_round_trip_second_call_is_cached():
    session = FakeSession(weather_rows=[make_weather_row()])
    client = _client(session)

    r1 = client.get(
        "/api/v1/patients/bob/safety-assessment",
        params={"lat": 40.7128, "lon": -74.0060},
    )
    assert r1.status_code == 200

    # Second call should be served from cache (no additional DB work)
    r2 = client.get(
        "/api/v1/patients/bob/safety-assessment",
        params={"lat": 40.7128, "lon": -74.0060},
    )
    assert r2.status_code == 200
    assert r2.json()["safety_score"] == r1.json()["safety_score"]
    # cache hit recorded
    assert main._test_cache.hits == 1


def test_existing_profile_used_and_route_risk_computed():
    profile = PatientProfile(
        user_id="carol",
        conditions=["copd"],
        aqi_threshold=80,
        pm25_threshold=25.0,
        alert_radius_km=30.0,
    )
    session = FakeSession(
        profile=profile,
        weather_rows=[make_weather_row()],
    )
    client = _client(session)

    resp = client.get(
        "/api/v1/patients/carol/safety-assessment",
        params={
            "lat": 40.7128,
            "lon": -74.0060,
            "dest_lat": 40.7282,
            "dest_lon": -73.9849,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "carol"
    # No auto-registration for existing profile
    assert session.committed == 0
    # Route risk present
    assert body["route_risk"] is not None
    assert "route_risk_score" in body["route_risk"]
    assert len(body["route_risk"]["segments"]) == 3


def test_validation_error_on_bad_coordinates():
    session = FakeSession()
    client = _client(session)
    resp = client.get(
        "/api/v1/patients/dave/safety-assessment",
        params={"lat": 91.0, "lon": -74.0060},
    )
    assert resp.status_code == 422


def test_data_status_partial_when_weather_only():
    session = FakeSession(weather_rows=[make_weather_row()])
    client = _client(session)
    body = client.get(
        "/api/v1/patients/erin/safety-assessment",
        params={"lat": 40.7128, "lon": -74.0060},
    ).json()
    assert body["data_status"] == "partial"
    assert any("partial" in r["text"].lower() for r in body["recommendations"])


def test_data_status_available_with_both_sources():
    session = FakeSession(
        weather_rows=[make_weather_row()],
        aq_rows=[make_aq_row()],
    )
    client = _client(session)
    body = client.get(
        "/api/v1/patients/frank/safety-assessment",
        params={"lat": 40.7128, "lon": -74.0060},
    ).json()
    assert body["data_status"] == "available"
    assert body["current_conditions"]["aq_reading_count"] == 1


def test_data_status_unavailable_with_caution():
    session = FakeSession()  # no weather, no aq readings
    client = _client(session)
    body = client.get(
        "/api/v1/patients/grace/safety-assessment",
        params={"lat": 40.7128, "lon": -74.0060},
    ).json()
    assert body["data_status"] == "unavailable"
    texts = " ".join(r["text"].lower() for r in body["recommendations"])
    assert "no recent monitoring data" in texts
    assert "favorable" not in texts  # must NOT claim safe without data


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
