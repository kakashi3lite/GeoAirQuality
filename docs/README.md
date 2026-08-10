# GeoAirQuality — Documentation Hub

A real-time air quality platform with personalized safety guidance for
respiratory patients. This hub is the single entry point to all
documentation.

```
docs/
├── README.md                 ← you are here
├── architecture/             what & why (roadmaps)
├── design/                   planned modules (specs, status-labeled)
├── guides/                   how to operate
└── implementation/           phase-by-phase build plans
```

## 📖 Sections

| Section | Contents |
|---------|----------|
| [**Architecture**](architecture/ARCHITECTURE_ROADMAP.md) | System design, scalability, monetization strategy |
| [**Product Roadmap**](architecture/PRODUCT_ROADMAP.md) | Phase-by-phase product vision, KPIs, competitive strategy |
| [**Design specs**](design/) | Planned modules: [Auth](design/auth.md) · [ML Pipeline](design/ml-pipeline.md) · [Frontend](design/frontend.md) · **[UI/UX reference](design/uiux.md)** — **📋 not yet implemented** |
| [**Guides**](guides/) | [Database](guides/database.md) · [Deployment](guides/deployment.md) · [Performance](guides/performance.md) · [Pipeline](guides/pipeline.md) · [CI/CD](guides/ci-cd.md) · [Security](guides/security.md) |
| [**Implementation plans**](implementation/) | [Phase 1](implementation/implementation-phase1.md) · **[Phase 3 — Insights & Routes](implementation/implementation-phase3.md)** 📋 planned |

## 🔗 Quick links

| Where | What |
|-------|------|
| [API service](../api/README.md) | FastAPI endpoints, run instructions, env vars |
| [Data pipeline](../data-pipeline/README.md) | Ingestion & processing |
| [Deploy assets](../deploy/README.md) | k8s, Helm, nginx, monitoring |
| [Tests](../tests/README.md) | What each test file covers |
| [Root README](../README.md) | Project map + module status table |

## 🚦 Status legend

| Icon | Meaning |
|------|---------|
| ✅ | Implemented & tested |
| ⚠️ | Partially implemented (see notes) |
| 📋 | Design spec only — planned |
