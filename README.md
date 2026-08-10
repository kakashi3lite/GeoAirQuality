# GeoAirQuality

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![PostGIS](https://img.shields.io/badge/PostGIS-3.3-blue.svg)](https://postgis.net/)

A real-time air quality platform that answers one question for
respiratory patients: **"Is it safe for ME to go outside right now?"**

It combines **PostGIS spatial analytics**, **weather + news
intelligence**, and a **condition-calibrated safety engine** into a
single actionable score with plain-language recommendations.

---

## 🧭 Repository map

```
GeoAirQuality/
├── README.md            ← you are here
├── api/                 ✅ FastAPI service (safety engine, news, endpoints)
├── data-pipeline/       ⚠️ ingestion (simple active; dask disabled)
├── deploy/              ✅ k8s, helm, nginx, monitoring (consolidated)
├── docs/                ✅ documentation hub
├── data/                datasets (raw/processed — generated, untracked)
├── tests/               ✅ unit + integration suites
├── frontend/  ml-pipeline/  api/auth/   📋 design stubs → docs/design/
├── Makefile             shared local/CI commands
└── .gitlab-ci.yml · .github/workflows/ci.yml   dual-platform CI/CD
```

## 📊 Module status

| Module | Status | Notes |
|--------|--------|-------|
| **API service** (`api/`) | ✅ Implemented | 8 endpoints, PostGIS, Redis cache |
| **Safety engine** (`services/safety_engine.py`) | ✅ Implemented | Condition-calibrated risk score (Phase 1) |
| **News intelligence** (`services/news_*`) | ✅ Implemented | AirNow + NewsAPI feed the score (Phase 2) |
| **Data pipeline** (`data-pipeline/`) | ⚠️ Partial | `ingest_simple_minimal.py` active; Dask disabled |
| **CI/CD** (`.gitlab-ci.yml`, `.github/`) | ✅ Implemented | lint→test→build→scan→publish→deploy |
| **Security** | ✅ Hardened | 0-leak audit, gitleaks blocking in CI |
| **Auth** (`api/auth/`) | 📋 Design only | JWT/RBAC spec in `docs/design/auth.md` |
| **ML pipeline** (`ml-pipeline/`) | 📋 Design only | Forecasters spec in `docs/design/ml-pipeline.md` |
| **Frontend** (`frontend/`) | 📋 Design only | React/MapLibre spec in `docs/design/frontend.md` |

## 🚀 Quick start

```bash
# Full stack (PostGIS + Redis + API + pipeline + monitoring)
docker-compose up --build

# Or local:
make install          # pip deps for api + pipeline
make test             # unit tests (no services needed)
cd api && uvicorn main:app --reload
```

Then open http://localhost:8000/docs (Swagger) and try the hero endpoint:

```bash
curl "http://localhost:8000/api/v1/patients/demo/safety-assessment?lat=40.7128&lon=-74.0060"
```

## 📚 Documentation

Start at the [**docs hub**](docs/README.md) — it links architecture,
design specs, and operational guides.

| Section | Link |
|---------|------|
| Blue Ocean strategy (verified research) | [docs/strategy/blue-ocean.md](docs/strategy/blue-ocean.md) |
| Architecture & roadmap | [docs/architecture](docs/architecture/) |
| Design specs (auth / ML / frontend) | [docs/design](docs/design/) |
| Deployment · Database · Performance · Pipeline · CI/CD · Security | [docs/guides](docs/guides/) |
| API service | [api/README.md](api/README.md) |
| Deploy assets | [deploy/README.md](deploy/README.md) |
| Tests | [tests/README.md](tests/README.md) |

## 🧪 Testing

```bash
make lint             # enforced lint (services/ + tests/)
make test             # supported suite (54 tests)
make test-integration # CI-only: needs PostGIS + Redis (RUN_INTEGRATION=1)
```

## 📄 License

MIT — see [LICENSE](LICENSE).

---

**GeoAirQuality** — personalized environmental safety for respiratory health.
