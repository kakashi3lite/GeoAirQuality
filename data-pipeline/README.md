# Data Pipeline — Ingestion & Processing

Ingests and cleans air quality + weather data, computes AQI, and writes
processed datasets.

## Current state

| Script | Status | Notes |
|--------|--------|-------|
| `ingest_simple_minimal.py` | ✅ **Active** | Dask-free; what the Dockerfile runs |
| `ingest.py` | ⚠️ Stale | Full Dask/GeoPandas pipeline — **dask is temporarily disabled** (version compatibility); see [`docs/guides/pipeline.md`](../docs/guides/pipeline.md) |

## Run

```bash
cd data-pipeline
pip install -r requirements.txt
python ingest_simple_minimal.py   # generates sample data + processed CSVs
```

## Environment variables

`DASK_WORKERS`, `GRID_RESOLUTION`, `OUTPUT_FORMAT` (see
[`docs/guides/pipeline.md`](../docs/guides/pipeline.md)).

## Sample datasets

Root-level `AirQuality.csv` and `GlobalWeatherRepository.csv` are the
canonical sample datasets referenced by the pipeline and the research
notebook — keep them at the repo root (the notebook fetches them by
stable path).
