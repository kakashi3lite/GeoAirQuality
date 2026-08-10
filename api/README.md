# API Service — FastAPI + PostGIS + Redis

The backend serving personalized safety assessments, air quality/weather
data, and news intelligence.

> Design docs: see [`docs/design/`](../docs/design/).
> Full API reference is served at `/docs` (Swagger) when running.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` · `/ready` | Probes (K8s liveness/readiness) |
| `GET` | `/api/v1/air-quality/readings` | AQ readings within radius (meter-accurate) |
| `GET` | `/api/v1/air-quality/grid/{grid_id}` | AQ readings for a grid cell |
| `GET` | `/api/v1/weather/readings` | Weather readings within radius |
| `GET` | `/api/v1/aggregated/grid/{grid_id}` | Aggregated stats per grid |
| `GET` | `/api/v1/patients/{user_id}/safety-assessment` | ⭐ **Hero**: personalized safety score + recommendations |
| `GET` | `/api/v1/news/nearby` | Curated air-quality news near a location |
| `GET` | `/metrics` | Prometheus metrics |

## Run locally

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload        # needs PostGIS + Redis (docker-compose up -d postgres redis)
```

## Environment variables

| Var | Default | Notes |
|-----|---------|-------|
| `DATABASE_URL` | dev fallback (warns) | PostGIS connection; production MUST set via Secret |
| `REDIS_URL` | `redis://localhost:6379/0` | Cache |
| `AIRNOW_API_KEY` | unset | Official alerts (skips if unset) |
| `NEWSAPI_KEY` | unset | Media articles (skips if unset) |
| `NEWS_FETCH_INTERVAL_MINUTES` | `30` | Background news loop |
| `NEWS_RETENTION_DAYS` | `7` | Deactivate articles older than N days |

## Architecture notes

- **Meter-accurate spatial queries** use geography-cast `ST_DWithin`
  (migration `003` adds GiST expression indexes).
- **Caching** is explicit (keyed on query params, not the DB session) with
  Redis TTLs; news upserts invalidate affected caches.
- **Event-loop safety**: synchronous DB work runs via `asyncio.to_thread`.
- Models in [`models.py`](models.py); services in [`services/`](services/):
  - `safety_engine.py` — condition-calibrated risk scoring
  - `recommendation_engine.py` — rule-based actionable guidance
  - `news_fetcher.py` / `news_processor.py` / `news_store.py` / `news_scheduler.py` — news intelligence
- Migrations: [`migrations/`](migrations) (Alembic, `001`–`004`).
