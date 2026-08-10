# UI/UX Reference — Patient Safety Dashboard

**Status:** 📋 Planning reference for the frontend (design spec:
[`frontend.md`](frontend.md)).
**Audience:** respiratory patients (asthma, COPD, bronchitis, allergy),
many of whom are older adults. Every decision below optimizes for that
audience.

---

## 1. Design principles

1. **One-screen answer.** The patient's core question is *"Is it safe
   for ME to go outside right now?"* — the home screen must answer it in
   under 3 seconds, no scrolling required.
2. **Actionable, not just informative.** Never show a raw number without
   "what to do next". Every card ends in a verb.
3. **Personal, not generic.** The score, factors, and wording are
   calibrated to the patient's condition ("your asthma").
4. **Honest about data.** If monitoring data is missing (`data_status:
   unavailable`), the UI must say so — never imply "safe".
5. **Calm & accessible.** No alarmist flashing; WCAG AA; colorblind-safe
   AQI; large touch targets; readable type.
6. **Progressive detail.** Home = answer. Tap = why. Tap = source.

---

## 2. Information architecture

```
Home (Safety Dashboard)          ← hero: answer first
├── Safety Score card             score / risk / summary / recommendation
├── Why now?                      contributing factors (tap → details)
├── What to do                    actionable recommendations
├── Near you                      nearby news events (tap → feed)
├── Map tab                       AQI heatmap + event + route overlays
├── Log tab                       quick symptom logging
└── Insights tab                  triggers, safest times, trend
```

Bottom navigation (4 tabs) sized for one-handed use; home is the
default and the only screen a patient *must* see.

---

## 3. Core screens (wireframes)

### 3.1 Safety Dashboard (home)

```
┌──────────────────────────────────────────────┐
│  Good morning, Sam            ☀ 08:00 · NYC   │
│                                              │
│   ┌────────────────────────────────────┐     │
│   │   SAFETY SCORE             72/100   │     │
│   │   ● MODERATE RISK                   │     │
│   │   "Limit prolonged outdoor          │     │
│   │    exertion. Keep your rescue       │     │
│   │    inhaler accessible."             │     │
│   └────────────────────────────────────┘     │
│                                              │
│  WHY NOW?                                    │
│  ▸ PM2.5 45 µg/m³ — above your threshold (60%)│
│  ▸ Wildfire smoke 12 km away (25%)           │
│  ▸ Humidity 88% (15%)                        │
│                                              │
│  WHAT TO DO                                  │
│  ✔ Wear an N95 mask if you go out            │
│  ✔ Avoid downtown — smoke advisory           │
│  ✔ Walk before 10 AM                          │
│                                              │
│  ⚠ 1 active alert near you        [View]     │
└──────────────────────────────────────────────┘
```

**Rules**
- Score card is the first element; it is *always* on screen.
- Risk color drives a full-width accent (green/amber/orange/red/purple)
  **plus** a label — never color alone.
- "Why now?" factors are ranked with contribution %.
- Recommendations are single verbs, short, tappable for detail.

### 3.2 Map view

```
┌──────────────────────────────────────────────┐
│  🔍 [Search location...]        [Layer: ⬡]   │
│  ┌────────────────────────────────────────┐  │
│  │  AQI heatmap (green→purple)           │  │
│  │      🔥 Wildfire smoke                │  │
│  │      🏭 Industrial                    │  │
│  │  ⭐ You  ·  🚶 Planned route          │  │
│  └────────────────────────────────────────┘  │
│  Route risk: 68/100 — worst segment downtown │
└──────────────────────────────────────────────┘
```

- Events render as category icons (wildfire 🔥, industrial 🏭,
  traffic 🚗, pollen 🌿, advisory ⚠) — colorblind-safe.
- Route corridor shows per-segment safety (origin/midpoint/dest).
- Layer toggles: AQI, weather, events, pollen.

### 3.3 Log & Insights tabs

**Log** — three taps to log: pick symptom → severity slider → save.
Auto-snapshots conditions; shows "logged with AQI 145 at 8:04 AM".

**Insights** — triggers ranked by correlation, safest/riskiest hour
chips, and a 30-day trend line. Copy is plain:
*"You reported symptoms 12× when PM2.5 was above 40."*

---

## 4. Component → API mapping

| UI component | API |
|--------------|-----|
| Safety score card | `GET /patients/{id}/safety-assessment?lat=&lon=` |
| Why now? factors | `contributing_factors` (ranked, with %) |
| What to do | `recommendations[]` (type: general/precaution/location/timing/route) |
| Data honesty banner | `data_status` (available/partial/unavailable) |
| Nearby alerts list | `nearby_events[]` |
| Route corridor | `route_risk.segments[]` (origin/midpoint/destination) |
| Map event markers | `GET /news/nearby?lat=&lon=&radius_km=` |
| Symptom logging | `POST /patients/{id}/symptoms` |
| Insights tab | `GET /patients/{id}/insights` (Phase 3) |

## 5. Visual system

| Token | Value | Notes |
|-------|-------|-------|
| AQI good | `#00e400` | green |
| AQI moderate | `#ffff00` | amber (use with dark text — contrast) |
| AQI unhealthy | `#ff7e00` | orange |
| AQI very unhealthy | `#ff0000` | red |
| AQI hazardous | `#8f3f97` | purple |
| Type | ≥16px body, ≥44px touch targets | WCAG 2.4.7 |
| Contrast | AA (4.5:1 text) | seniors + low-vision |
| Motion | `pulse-slow`, no flashing | photosensitive-safe |

Reference: `frontend.md` already defines `air.*` Tailwind tokens and the
MapLibre heatmap layer — reuse them.

## 6. States every component needs

| State | Behavior |
|-------|----------|
| Loading | skeleton, not spinner-only |
| Cached/stale | subtle "updated X min ago"; 5-min TTL matches API |
| `data_status: partial` | note "some readings missing" |
| `data_status: unavailable` | amber banner + "no data in your area" — **never** a green score |
| Error | retry + offline fallback to last cached score |
| Empty history | insights show "log symptoms to see your triggers" |

## 7. Open questions for the build

1. Map provider: MapLibre GL (self-hosted tiles) — confirm tile budget.
2. Push notifications for imminent trigger conditions (Phase 3+).
3. PWA offline: cache last assessment + news feed.
4. Language/regionalization (AQI is US-centric; EU uses CAQI/EAQI).

## 8. Relationship to other docs

- [`frontend.md`](frontend.md) — full technical spec (React, MapLibre,
  TanStack Query, Socket.IO)
- [`implementation-phase3.md`](../implementation/implementation-phase3.md)
  — the API this UI consumes next
- [`../architecture/ARCHITECTURE_ROADMAP.md`](../architecture/ARCHITECTURE_ROADMAP.md)
  — Phase 2 dashboard plan + WebXR AR overlays
