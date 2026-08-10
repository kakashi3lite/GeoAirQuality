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

## Deploy to Cloudflare Pages

The SPA is Cloudflare-Pages-ready out of the box:

| File | Purpose |
|------|---------|
| `wrangler.jsonc` | Pages project config (`geoairquality-breathe`, output `dist/`) |
| `functions/api/[[path]].js` | Edge proxy → forwards `/api/*` to `API_ORIGIN` |
| `public/_headers` | Security headers + strict CSP (OSM tiles, geolocation) |
| `public/_redirects` | SPA fallback (`/* → /index.html`) |

```bash
npm ci && npm run build
npx wrangler pages deploy dist --project-name=geoairquality-breathe --branch=production
```

Set `API_ORIGIN` on the Pages project to the public backend URL
(see `../docs/guides/deployment.md` → "Cloudflare Deployment"). Both CI
pipelines (GitLab + GitHub) have a `pages` deploy job.

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

