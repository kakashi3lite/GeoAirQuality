# Frontend — Breathe (Patient Safety Dashboard)

> **STATUS: ✅ Implemented (2026-08-10).** React + TypeScript + Vite +
> Tailwind + TanStack Query + MapLibre GL + Framer Motion.

The "Breathe" dashboard answers the patient's core question first:
*"Is it safe for ME to go outside right now, and what should I do?"*

## Run

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000 (proxies /api → :8000)
npm run build        # type-check + production build
```

> Requires the API running on `localhost:8000` (see `../api/README.md`).

## Structure

```
src/
├── App.tsx / main.tsx / context/AppContext.tsx
├── hooks/      useApi.ts (safety, news, insights, log-symptom)
├── services/   api.ts (typed axios client for the 8 endpoints)
├── types/      api.ts (TS mirrors of the API response models)
└── components/
    ├── layout/   AmbientBackground, Header, BottomNav, LocationGate
    ├── home/     SafetyScoreCard, ContributingFactors,
    │             RecommendationsList, NearbyAlertsStrip, RouteRiskPreview
    ├── map/      MapPage (MapLibre GL + event markers)
    ├── log/      LogPage (symptom picker + severity slider)
    ├── insights/ InsightsPage (triggers, times, trend)
    └── shared/   DataHonestyBanner, ScoreGauge, RiskBadge, CategoryIcon
```

Design references: `../docs/design/wireframe-blueprint.md` (this build),
`../docs/design/uiux.md` (principles), `../docs/design/frontend.md` (stack).

