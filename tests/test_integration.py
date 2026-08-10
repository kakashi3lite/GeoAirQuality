"""Integration tests requiring a live PostGIS + Redis (CI-only).

These tests exercise the REAL database and cache, so they validate
more than unit tests can:

  * the Alembic migration chain applied to a real PostGIS server
    (catches invalid SQL — e.g. bad index expressions)
  * the geography expression indexes created by migration 003
  * the safety-assessment endpoint end-to-end: auto-registration,
    meter-accurate spatial fetch, scoring, and caching against
    real PostGIS + Redis.

Skipped unless RUN_INTEGRATION=1 is set. CI runs `alembic upgrade head`
before pytest, then invokes this file with RUN_INTEGRATION=1.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 and DATABASE_URL to run integration tests",
)


def test_migration_schema_and_geography_indexes():
    """Verify migrations applied and geography indexes exist."""
    from sqlalchemy import create_engine, inspect

    engine = create_engine(os.environ["DATABASE_URL"])
    inspector = inspect(engine)

    expected_tables = {
        "patient_profiles",
        "symptom_logs",
        "spatial_grids",
        "air_quality_readings",
        "weather_readings",
        "aggregated_data",
        "data_sources",
    }
    tables = set(inspector.get_table_names())
    missing = expected_tables - tables
    assert not missing, f"tables missing — run `alembic upgrade head`: {missing}"

    # GiST expression indexes from migration 003 (meter-accurate queries)
    aq_indexes = {i["name"] for i in inspector.get_indexes("air_quality_readings")}
    weather_indexes = {i["name"] for i in inspector.get_indexes("weather_readings")}
    assert "idx_aq_location_geog" in aq_indexes
    assert "idx_weather_location_geog" in weather_indexes
    engine.dispose()


def test_safety_assessment_end_to_end():
    """Full request flow against real PostGIS + Redis."""
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    import main as main_mod

    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    main_mod.SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False
    )

    with main_mod.SessionLocal() as db:
        # Idempotent: reset the CI sample row, then insert a clean-air
        # weather reading near NYC so the assessment has real data.
        db.execute(
            text("DELETE FROM weather_readings WHERE station_id = 'ci-station'")
        )
        grid = db.execute(
            text(
                """
                SELECT id FROM spatial_grids
                ORDER BY ST_Distance(
                    geometry,
                    ST_SetSRID(ST_MakePoint(-74.006, 40.7128), 4326)
                )
                LIMIT 1
                """
            )
        ).first()
        db.execute(
            text(
                """
                INSERT INTO weather_readings (
                    station_id, country, location_name, latitude, longitude,
                    location, grid_cell_id, timestamp, last_updated,
                    temperature_celsius, humidity, wind_kph,
                    pm2_5, pm10, ozone, nitrogen_dioxide,
                    aqi, is_validated, quality_score
                ) VALUES (
                    'ci-station', 'US', 'CI Test', 40.7128, -74.006,
                    ST_SetSRID(ST_MakePoint(-74.006, 40.7128), 4326), :gid,
                    now(), now(),
                    20.0, 45.0, 12.0,
                    5.0, 10.0, 20.0, 8.0,
                    25, true, 1.0
                )
                """
            ),
            {"gid": grid[0] if grid else None},
        )
        db.commit()

    client = TestClient(main_mod.app)
    resp = client.get(
        "/api/v1/patients/ci-patient/safety-assessment",
        params={"lat": 40.7128, "lon": -74.006},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user_id"] == "ci-patient"
    assert body["data_status"] in ("available", "partial")
    assert 0 <= body["safety_score"] <= 100
    assert body["summary"]
    assert isinstance(body["recommendations"], list)

    # Second call should be a Redis cache hit with identical score
    resp2 = client.get(
        "/api/v1/patients/ci-patient/safety-assessment",
        params={"lat": 40.7128, "lon": -74.006},
    )
    assert resp2.status_code == 200
    assert resp2.json()["safety_score"] == body["safety_score"]
    engine.dispose()
