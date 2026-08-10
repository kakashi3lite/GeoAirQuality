# Phase 3 — Insights, Routes & Polish

**Status:** 📋 Planned (next after news intelligence)
**Goal:** complete the patient experience — symptom logging, personal
trigger learning, route-aware safety, and an insights API. The safety
score stops being generic and becomes *personalized to this patient's
history*.

---

## Why this phase

Phases 1–2 answer *"is it safe right now, here?"* using live AQI,
weather, and news. Phase 3 answers the deeper questions:

- *"Why do I always cough on high-PM2.5 days?"* → personal trigger learning
- *"When should I schedule my walk?"* → safest/riskiest times
- *"Is the route I planned OK?"* → route corridor risk
- *"Am I getting better or worse?"* → trend over time

---

## Work items (with acceptance criteria)

### 1. Symptom logging — `POST /api/v1/patients/{user_id}/symptoms`

**Contract**
```json
POST /api/v1/patients/{user_id}/symptoms
{ "symptom_type": "coughing", "severity": 3, "lat": 40.71, "lon": -74.00 }
→ 201 { "id": 12, "symptom_type": "coughing", "severity": 3,
        "weather_snapshot": { "aqi": 145, "pm25": 45, "humidity": 88, ... },
        "logged_at": "..." }
```

**Behavior**
- Auto-captures the current environmental snapshot (reuse
  `_build_environmental_snapshot`) into `weather_snapshot` JSONB.
- Inserts a `SymptomLog` row with a spatial point.
- Invalidates the patient's `safety_assessment:*` cache entries
  (`cache.delete_pattern(f"safety_assessment:{user_id}:*")`).
- Validation: `symptom_type` in allowlist, `severity` 1–5, valid coords.

**Acceptance:** 201 with snapshot populated; bad input → 422; cache invalidated.

### 2. History component in the safety engine

Replace the `score_history_component` stub (currently returns neutral 100):

- Query the patient's `SymptomLog` rows from the last 30 days.
- Build a simple trigger model:
  - Bucket symptoms by environmental conditions (AQI band, humidity band).
  - If the patient logged symptoms at conditions similar to *now* →
    apply a penalty to the history component (0–100).
  - Example: symptoms at AQI 120+ 4× in 30 days, current AQI 115 →
    history score ~60 → pulls overall score down via 15% weight.
- Conservative by default: no data → neutral 100 (unchanged).

**Acceptance:** unit tests prove penalty only when similar-condition
symptoms exist; no symptoms → 100.

### 3. Insights endpoint — `GET /api/v1/patients/{user_id}/insights`

**Contract**
```json
GET /api/v1/patients/{user_id}/insights
→ {
  "top_triggers": [{ "factor": "PM2.5 > 40", "correlation": 0.78, "occurrences": 12 }],
  "safest_times":  [{ "hour": 7, "day": "Sunday", "avg_aqi": 35 }],
  "riskiest_times":[{ "hour": 17, "day": "Wednesday", "avg_aqi": 95 }],
  "recent_trend":  "Symptoms down 20% this month as AQI improved",
  "period_days": 30
}
```

**Behavior**
- Pearson correlation between logged symptom severity and each
  environmental factor (PM2.5, AQI, humidity, temperature) — pure
  statistics, no ML.
- Safest/riskiest hour buckets from the patient's symptom history +
  nearby AQI.
- Cached 15 min per user.

**Acceptance:** deterministic outputs; empty history → sensible empty
response, not an error.

### 4. Route risk refinement

Already have origin/midpoint/destination sampling (5 km radius). Refine:
- Return a recommended corridor (`safest_segment` + `worst_segment`)
  and, when a news event intersects the corridor, note it.
- Keep the worst-segment-dominates rule.

**Acceptance:** route risk includes segment labels + optional news
intersections.

### 5. Scheduler + retention polish

- Wire `NEWS_ARTICLES_ACTIVE` gauge to an actual count query.
- Log per-source fetch stats (add `NEWS_ARTICLES_PER_SOURCE` gauge).
- Keep Redis-lock + cache-invalidation behavior (already done in Phase 2).

**Acceptance:** metrics visible in `/metrics` after a cycle.

---

## Files affected

| File | Change |
|------|--------|
| `api/main.py` | +2 endpoints (symptoms, insights); wire history into assessment |
| `api/services/safety_engine.py` | implement `score_history_component` |
| `api/services/insights.py` | **new** — correlation + time-bucket analysis |
| `api/services/news_scheduler.py` | per-source metrics + active-count gauge |
| `tests/test_symptoms.py` | **new** — logging + snapshot + validation |
| `tests/test_insights.py` | **new** — correlation math, buckets, empty history |
| `tests/test_safety_engine.py` | history-component cases |
| `tests/test_safety_api.py` | assessment now includes history effect |

## Verification

1. `make lint && make test` → all suites green (target 65+ tests).
2. `curl -X POST .../symptoms` → 201 with snapshot; repeat assessment
   reflects history when symptoms exist at similar conditions.
3. Insights endpoint returns triggers + safest/riskiest times.
4. Milestone tag: `milestone/insights-routes`.

## Deferred (future)

- Email/push notifications when a trigger condition is imminent.
- Collaborative filtering ("patients like you report…").
- Forecast-driven "safe window" (needs the ML pipeline).
