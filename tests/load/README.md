# Load Testing Harness

Reproducible stress/load tests for the GeoAirQuality API. These scripts were
used in the production stress-test pass (2026-08) and are kept as the canonical
way to re-run it after changes.

## Prerequisites

- Live stack: `docker compose up -d postgres redis` (or your own PostGIS+Redis)
- Schema migrated: `cd api && alembic upgrade head`
- API running: `uvicorn main:app --app-dir . --port 8000`
- Python venv with `sqlalchemy` (for seeding) and `httpx` (for load)

## 1. Seed realistic data

The geography-cast `ST_DWithin` queries need rows to exercise the GiST
expression indexes. Defaults seed 20k weather + 10k AQ readings clustered
around NYC with the rest scattered across US cities.

```bash
DATABASE_URL=postgresql://geoair_user:geoair_pass@localhost:5433/geoairquality \
  python tests/load/seed_data.py --weather 20000 --aq 10000
```

## 2. Run the load test

Hammers every public endpoint with concurrent workers and reports
p50/p95/p99 latency, throughput and error rate.

```bash
# 30 workers for 15s (default)
python tests/load/run_load_test.py

# heavier, including the POST /symptoms write path
python tests/load/run_load_test.py --workers 60 --duration 20 --writes
```

Output columns: `n`, `req/s`, `p50/p95/p99`, `max`, `errors`.

## What the stress test proved (2026-08)

| Metric | Before fixes | After fixes |
|---|---|---|
| Error rate (30 workers) | 19% (63 errors) | **0%** |
| Throughput (30 workers) | 7.2 req/s | **130+ req/s** |
| Event-loop ceiling (`/ready`) | ~1.2s p50 under load | **32,306 req/s @ 200 conn** |
| Cache hit ratio (safety) | n/a | **90%** |
| Redis outage | n/a | all endpoints 200 (degrade to DB) |

The data-endpoint latency floor on this machine is dominated by the amd64
emulated Postgres (Apple Silicon host). The API layer itself sustains 30k+
req/s on `/ready` and `/health`; on native Postgres hardware the spatial
query latency drops by roughly an order of magnitude.

## Key invariants these scripts check

- **Zero 5xx under sustained concurrency** (graceful degradation).
- **Cache works**: second identical request must be far faster than the first.
- **Event loop stays responsive**: `/health` and `/ready` never stall behind
  heavy DB queries (all DB work is in `asyncio.to_thread`).
- **Connection pool holds**: 60 workers × multi-query safety assessments
  without pool exhaustion errors.
