#!/usr/bin/env python3
"""Seed realistic sensor data for load testing.

Inserts weather + air-quality readings around NYC (dense) and scattered
across the US so the geography-cast ST_DWithin queries hit the GiST
expression indexes with real rows.

Usage:
    DATABASE_URL=postgresql://geoair_user:geoair_pass@localhost:5433/geoairquality \\
        python tests/load/seed_data.py --weather 20000 --aq 10000
"""

import argparse
import datetime
import os
import random
import time

from sqlalchemy import create_engine, text


def _insert_weather(engine, count: int) -> None:
    """Insert weather readings clustered around NYC + US cities."""
    cities = [
        # (lat, lon, name, weight)
        (40.7128, -74.0060, "New York", 0.45),
        (40.7580, -73.9855, "Times Square", 0.20),
        (40.6501, -73.9496, "Brooklyn", 0.15),
        (40.7831, -73.9712, "Manhattan", 0.10),
        (34.0522, -118.2437, "Los Angeles", 0.03),
        (41.8781, -87.6298, "Chicago", 0.03),
        (29.7604, -95.3698, "Houston", 0.02),
        (33.4484, -112.0740, "Phoenix", 0.02),
    ]
    now = datetime.datetime.utcnow()
    rows = []
    for i in range(count):
        city = random.choices(cities, weights=[c[3] for c in cities])[0]
        lat = city[0] + random.uniform(-0.5, 0.5)
        lon = city[1] + random.uniform(-0.5, 0.5)
        ts = now - datetime.timedelta(minutes=random.randint(0, 60 * 24))
        rows.append(
            (
                f"st{i:07d}",
                "US",
                city[2],
                round(lat, 5),
                round(lon, 5),
                ts,
                ts,
                round(random.uniform(10, 32), 1),  # temperature_celsius
                round(random.uniform(50, 90), 1),  # temperature_fahrenheit
                round(random.uniform(20, 95), 1),  # humidity
                round(random.uniform(990, 1030), 1),  # pressure_mb
                round(random.uniform(0, 25), 1),  # wind_kph
                random.randint(0, 359),
                round(random.uniform(0, 250), 1),  # pm2_5
                round(random.uniform(0, 300), 1),  # pm10
                round(random.uniform(0, 1.5), 3),  # carbon_monoxide
                round(random.uniform(0, 0.12), 4),  # ozone
                round(random.uniform(0, 0.08), 4),  # nitrogen_dioxide
                round(random.uniform(0, 0.06), 4),  # sulphur_dioxide
                round(random.uniform(0, 300), 1),  # aqi
                random.choice(["Good", "Moderate", "Unhealthy", "Hazardous"]),
                True,
                round(random.uniform(0.5, 1.0), 3),
            )
        )
        if len(rows) >= 2000:
            _flush_weather(engine, rows)
            rows.clear()
    _flush_weather(engine, rows)


def _flush_weather(engine, rows) -> None:
    if not rows:
        return
    stmt = """
        INSERT INTO weather_readings (
            station_id, country, location_name, latitude, longitude, location,
            grid_cell_id, timestamp, last_updated, temperature_celsius,
            temperature_fahrenheit, humidity, pressure_mb, wind_kph,
            wind_degree, pm2_5, pm10, carbon_monoxide, ozone,
            nitrogen_dioxide, sulphur_dioxide, aqi, aqi_category,
            is_validated, quality_score, created_at, updated_at
        )
        SELECT
            s.station_id, s.country, s.location_name, s.latitude, s.longitude,
            ST_SetSRID(ST_MakePoint(s.longitude, s.latitude), 4326),
            NULL, s.timestamp, s.last_updated, s.temperature_celsius,
            s.temperature_fahrenheit, s.humidity, s.pressure_mb, s.wind_kph,
            s.wind_degree, s.pm2_5, s.pm10, s.carbon_monoxide, s.ozone,
            s.nitrogen_dioxide, s.sulphur_dioxide, s.aqi, s.aqi_category,
            s.is_validated, s.quality_score, NOW(), NOW()
        FROM (VALUES
            {values}
        ) AS s(
            station_id, country, location_name, latitude, longitude, timestamp,
            last_updated, temperature_celsius, temperature_fahrenheit, humidity,
            pressure_mb, wind_kph, wind_degree, pm2_5, pm10, carbon_monoxide,
            ozone, nitrogen_dioxide, sulphur_dioxide, aqi, aqi_category,
            is_validated, quality_score
        )
    """
    placeholders = []
    params = {}
    for idx, r in enumerate(rows):
        p = f"p{idx}"
        placeholders.append(
            f"(:{p}_station_id, :{p}_country, :{p}_name, :{p}_lat, :{p}_lon, "
            f":{p}_ts, :{p}_lu, :{p}_tc, :{p}_tf, :{p}_hum, :{p}_pres, "
            f":{p}_wind, :{p}_deg, :{p}_pm25, :{p}_pm10, :{p}_co, :{p}_o3, "
            f":{p}_no2, :{p}_so2, :{p}_aqi, :{p}_cat, :{p}_valid, :{p}_q)"
        )
        params.update(
            {
                f"{p}_station_id": r[0],
                f"{p}_country": r[1],
                f"{p}_name": r[2],
                f"{p}_lat": r[3],
                f"{p}_lon": r[4],
                f"{p}_ts": r[5],
                f"{p}_lu": r[6],
                f"{p}_tc": r[7],
                f"{p}_tf": r[8],
                f"{p}_hum": r[9],
                f"{p}_pres": r[10],
                f"{p}_wind": r[11],
                f"{p}_deg": r[12],
                f"{p}_pm25": r[13],
                f"{p}_pm10": r[14],
                f"{p}_co": r[15],
                f"{p}_o3": r[16],
                f"{p}_no2": r[17],
                f"{p}_so2": r[18],
                f"{p}_aqi": r[19],
                f"{p}_cat": r[20],
                f"{p}_valid": r[21],
                f"{p}_q": r[22],
            }
        )
    with engine.begin() as conn:
        conn.execute(text(stmt.format(values=",\n".join(placeholders))), params)


def _insert_aq(engine, count: int) -> None:
    """Insert air-quality readings (raw sensor model) around NYC."""
    now = datetime.datetime.utcnow()
    rows = []
    for i in range(count):
        lat = 40.7128 + random.uniform(-0.5, 0.5)
        lon = -74.0060 + random.uniform(-0.5, 0.5)
        ts = now - datetime.timedelta(minutes=random.randint(0, 60 * 24))
        rows.append(
            (
                f"aq{i:07d}",
                round(lat, 5),
                round(lon, 5),
                ts,
                round(random.uniform(0.1, 5.0), 3),  # co_gt
                round(random.uniform(500, 2500), 1),  # pt08_s1_co
                round(random.uniform(0, 100), 2),  # nmhc_gt
                round(random.uniform(0, 30), 3),  # c6h6_gt
                round(random.uniform(100, 1500), 1),  # pt08_s2_nmhc
                round(random.uniform(0, 300), 1),  # nox_gt
                round(random.uniform(100, 1500), 1),  # pt08_s3_nox
                round(random.uniform(0, 200), 2),  # no2_gt
                round(random.uniform(100, 1500), 1),  # pt08_s4_no2
                round(random.uniform(100, 2000), 1),  # pt08_s5_o3
                round(random.uniform(10, 32), 1),  # temperature
                round(random.uniform(20, 95), 1),  # relative_humidity
                round(random.uniform(0, 30), 1),  # absolute_humidity
                round(random.uniform(0, 300), 1),  # aqi
                random.choice(["Good", "Moderate", "Unhealthy", "Hazardous"]),
                True,
                round(random.uniform(0.5, 1.0), 3),
            )
        )
        if len(rows) >= 2000:
            _flush_aq(engine, rows)
            rows.clear()
    _flush_aq(engine, rows)


def _flush_aq(engine, rows) -> None:
    if not rows:
        return
    stmt = """
        INSERT INTO air_quality_readings (
            sensor_id, latitude, longitude, location, grid_cell_id, timestamp,
            co_gt, pt08_s1_co, nmhc_gt, c6h6_gt, pt08_s2_nmhc, nox_gt,
            pt08_s3_nox, no2_gt, pt08_s4_no2, pt08_s5_o3, temperature,
            relative_humidity, absolute_humidity, aqi, aqi_category,
            is_validated, quality_score, created_at, updated_at
        )
        SELECT
            s.sensor_id, s.latitude, s.longitude,
            ST_SetSRID(ST_MakePoint(s.longitude, s.latitude), 4326),
            NULL, s.timestamp, s.co_gt, s.pt08_s1_co, s.nmhc_gt, s.c6h6_gt,
            s.pt08_s2_nmhc, s.nox_gt, s.pt08_s3_nox, s.no2_gt, s.pt08_s4_no2,
            s.pt08_s5_o3, s.temperature, s.relative_humidity,
            s.absolute_humidity, s.aqi, s.aqi_category, s.is_validated,
            s.quality_score, NOW(), NOW()
        FROM (VALUES
            {values}
        ) AS s(
            sensor_id, latitude, longitude, timestamp, co_gt, pt08_s1_co,
            nmhc_gt, c6h6_gt, pt08_s2_nmhc, nox_gt, pt08_s3_nox, no2_gt,
            pt08_s4_no2, pt08_s5_o3, temperature, relative_humidity,
            absolute_humidity, aqi, aqi_category, is_validated, quality_score
        )
    """
    placeholders = []
    params = {}
    for idx, r in enumerate(rows):
        p = f"p{idx}"
        placeholders.append(
            f"(:{p}_sensor, :{p}_lat, :{p}_lon, :{p}_ts, :{p}_co, :{p}_s1co, "
            f":{p}_nmhc, :{p}_b, :{p}_s2nmhc, :{p}_nox, :{p}_s3nox, :{p}_no2, "
            f":{p}_s4no2, :{p}_s5o3, :{p}_temp, :{p}_rh, :{p}_ah, :{p}_aqi, "
            f":{p}_cat, :{p}_valid, :{p}_q)"
        )
        params.update(
            {
                f"{p}_sensor": r[0],
                f"{p}_lat": r[1],
                f"{p}_lon": r[2],
                f"{p}_ts": r[3],
                f"{p}_co": r[4],
                f"{p}_s1co": r[5],
                f"{p}_nmhc": r[6],
                f"{p}_b": r[7],
                f"{p}_s2nmhc": r[8],
                f"{p}_nox": r[9],
                f"{p}_s3nox": r[10],
                f"{p}_no2": r[11],
                f"{p}_s4no2": r[12],
                f"{p}_s5o3": r[13],
                f"{p}_temp": r[14],
                f"{p}_rh": r[15],
                f"{p}_ah": r[16],
                f"{p}_aqi": r[17],
                f"{p}_cat": r[18],
                f"{p}_valid": r[19],
                f"{p}_q": r[20],
            }
        )
    with engine.begin() as conn:
        conn.execute(text(stmt.format(values=",\n".join(placeholders))), params)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", type=int, default=20000)
    ap.add_argument("--aq", type=int, default=10000)
    args = ap.parse_args()

    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://geoair_user:geoair_pass@localhost:5433/geoairquality",
    )
    engine = create_engine(url, pool_pre_ping=True)
    t0 = time.time()
    print(f"Seeding {args.weather} weather + {args.aq} AQ readings ...")
    _insert_weather(engine, args.weather)
    _insert_aq(engine, args.aq)
    with engine.connect() as conn:
        w = conn.execute(text("SELECT count(*) FROM weather_readings")).scalar()
        a = conn.execute(text("SELECT count(*) FROM air_quality_readings")).scalar()
    print(f"Done in {time.time() - t0:.1f}s. weather={w} air_quality={a}")


if __name__ == "__main__":
    main()
