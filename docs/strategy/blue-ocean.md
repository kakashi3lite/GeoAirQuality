# Blue Ocean Strategy — The Personal Safety Copilot for Respiratory Patients

**Status:** 📋 Strategic thesis (verified competitive research, Aug 2026)
**Purpose:** define the uncontested market space this dashboard should own,
and the plan to occupy it before anyone else can.

---

## 1. Executive summary

Air quality apps compete on **data** (who has the most sensors and the
prettiest map). Health apps compete on **tracking** (who logs the most
symptoms). **Nobody competes on the answer.** A respiratory patient does
not want an AQI number — they want to know:

> *"Is it safe for **ME** to go outside right now, and what should I do?"*

That question sits in **uncontested market space**. Verified research
(below) shows every major product is either (a) a generic AQI display,
(b) a health tracker without live-air context, or (c) B2B or defunct.
This dashboard — with its **condition-calibrated safety score, actionable
recommendations, news-fused personal risk, symptom-trigger learning, and
route planning** — is the first to answer the patient's actual question.

**Blue ocean position:**
> *"We are not an air quality app. We are the personal safety copilot
> for the 576+ million people living with asthma and COPD."*

---

## 2. Competitive research (verified Aug 2026)

| Player | What it does | Monetization | The gap for a patient |
|--------|--------------|--------------|------------------------|
| **IQAir AirVisual** (5M+ downloads, 4.7★) | Real-time + 7-day AQI, 2D/3D maps, 6 pollutants, pollen, wildfire, news | Free app; hardware + AirVisual Pro | Health guidance is **generic AQI-band boilerplate** ("sensitive groups"), not *your* condition |
| **BreezoMeter** | "Health recommendations" API — used by Samsung Health etc. | B2B API | **Acquired by Google** → redirects to Google Maps Platform; consumer surface gone |
| **Plume Labs** | Street-level cleanest-route maps, "Clean-Air Coaching" | Free | **Frozen** — consumer app last updated May 2022; pivoted B2B |
| **PurpleAir** | 35,000+ community sensors, free public map | Hardware ($200–400) | No health guidance; low-cost sensors are **not regulatory-grade** (EPA) |
| **EPA AirNow** | US official AQI, forecasts, flag program | Tax-funded | US-only, **no personalization, no condition-specific action guidance** |
| **Weather apps** (Apple/Google/AccuWeather) | AQI as a number + color in the weather feed | Ads / subscriptions | Purely informational — no health framing, no advice hierarchy |
| **AsthmaMD** | Symptom/peak-flow journal, manual trigger log, doctor export | Free | **Abandoned (2017)**; no live AQ data, no route planning |
| **Propeller Health** (ResMed) | Inhaler-attached sensor, adherence, clinician dashboard | Enterprise B2B | **No consumer self-serve**; no live-air personalization, no routing |

**Market context:** ~363M people with asthma + ~213M with COPD globally —
yet the intersection of *personalized safety decisions × live
environmental intelligence × symptom learning* is unoccupied.

---

## 3. Strategy canvas — the value curve

Higher = more delivered to the patient.

```
10 │                              ┌── GeoAirQuality (blue ocean)
   │                              │
 8 │        IQAir ────────────────┤
   │                              
 6 │  Weather apps ───────────────┤
   │                              
 4 │         Propeller ───────────┤
   │                              
 2 │  ────────────────────────────┘
   │
 0 └──────────────────────────────────────────────────────────────
      AQI   Map  Forecast  Health  Personal- Action-  Trigger News-  Route  Clinician Honest
      data  cov.           advis.  ization  ability learning fused  plan.  sharing  no-data
```

- **IQAir / weather apps**: high on raw data + map, flatline near zero on
  personalization, action, triggers, news-fused risk, routing, clinician sharing.
- **Propeller**: high on tracking/clinical, near zero on live air context.
- **GeoAirQuality**: deliberately *lower* on raw data breadth (we consume
  external sources) and *far higher* on the decision layer — the shape
  of the curve, not its height, is the blue ocean.

---

## 4. ERRC grid (Eliminate · Reduce · Raise · Create)

| **ELIMINATE** (industry assumptions we drop) | **REDUCE** (below industry standard) |
|---|---|
| Raw AQI as the hero — replaced by a decision | Time-to-answer (one screen, <3s) |
| Map-first UI — replaced by answer-first | Setup friction (auto-register) |
| Generic "sensitive groups" boilerplate | Jargon (µg/m³, band names) |
| Ad- / data-selling business model | Reliance on a single data source |
| | Cost to the patient (open source, free core) |

| **RAISE** (above industry standard) | **CREATE** (never offered together) |
|---|---|
| Personalization to the *actual condition* | **Condition-calibrated Safety Score (0–100)** |
| Actionability — "what to do", not "what it means" | **Symptom ↔ environment trigger learning** |
| Honesty — `data_status: unavailable` never implies "safe" | **News-fused personal risk** (wildfire/industrial → *your* score) |
| Transparency — rule-based, auditable logic | **Route planning** with per-segment safety + event intersections |
| Insight depth — triggers, safest/riskiest times, trends | **Clinician-shareable insights** + open, self-hostable platform |

---

## 5. Six paths to uncontested space

1. **Alternative industries** — weather apps *and* health-tracking apps;
   we are neither — we are decision support.
2. **Strategic groups** — consumer AQI apps (IQAir) vs enterprise B2B
   (BreezoMeter/Google, Propeller/ResMed); we are a **self-serve patient
   tool with clinical-grade, auditable logic** — the empty middle.
3. **Buyer group** — the *decision maker* is not just the patient:
   **clinicians** (referral), **schools** (personalized flag program),
   **employers** (outdoor-worker safety), **payers** (avoidable
   exacerbations). One platform, multiple buyers.
4. **Complementary offerings** — IQAir ties to its purifiers, Propeller
   to its inhaler. We are **device-agnostic**: we guide decisions about
   *when to go out, which route, whether to pre-medicate, whether to
   turn on the purifier* — complementing all devices.
5. **Functional → emotional** — competitors sell a measurement; we sell
   **peace of mind** ("you're clear until 4pm, take the riverside path").
6. **Time** — everyone shows *now*; we add *"your safest window"*
   (forecast) and *30-day trend* (are you getting better?).

---

## 6. Why this is defensible (moats)

1. **The data flywheel** — trigger-learning gets *more accurate per
   patient over time*; switching costs rise with every logged symptom.
2. **Trust via transparency** — rule-based scoring is auditable and
   explainable ("why 72? because PM2.5 + wildfire smoke + your history"),
   unlike black-box ML. Essential for health-adjacent claims and
   clinician adoption.
3. **Spatial engine (PostGIS)** — the platform already ingests arbitrary
   geo-data sources; adding pollen, noise, UV, heat is incremental.
4. **Open source + self-hostable** — enterprises, health systems, and
   researchers can deploy privately; a credibility + distribution asset
   no consumer competitor has.
5. **Referral loop** — a clinician-shareable insights report turns every
   happy patient into a doctor who recommends the platform.

---

## 7. Target market & beachhead

- **Beachhead (Phase 1)**: US respiratory patients + clinicians who treat
  them (asthma/COPD). Free PWA, no install.
- **Expansion**: workplace safety (outdoor workforce, construction,
  schools), then Europe (EAQI support is an explicit roadmap item).
- **Sizing anchor**: 576M+ patients globally; IQAir alone proves 5M+
  users want air-quality guidance — the underserved, personalized layer
  sits on top of that demand.

## 8. Business model (blue ocean = new value, new revenue)

- **Free core** — the safety score and recommendations (a public good).
- **Premium** — clinician-shareable reports, multi-condition profiles,
  family/caregiver sharing, advanced insights.
- **B2B2C** — white-label for health systems, insurers, and employers
  (workplace safety dashboards) built on the same engine.

## 9. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| **Health-claim liability** | Non-diagnostic framing; rule-based + auditable; clinician co-development; disclaimers |
| **Sensor data quality** | `data_status` honesty; regulatory-grade sources (EPA) + community data labeled by class (per EPA Air Sensor Toolbox) |
| **Distribution** (apps are a discovery problem) | Clinician referral loop, patient-advocacy partnerships (Asthma + Lung UK, American Lung Assoc.), PWA |
| **Big-tech entry** (Google owns BreezoMeter) | Own the *decision layer*, not the data layer; moat is per-patient learning + trust, not sensor count |
| **Regulatory (EU/California health-app rules)** | Build consent/privacy-first from day one; EU AQI support on roadmap |

## 10. Roadmap to own the space

| Milestone | Action | Why it matters for the blue ocean |
|-----------|--------|----------------------------------|
| ✅ Done | Safety score, news fusion, trigger learning, route risk, insights | The decision layer exists (verified: no competitor combines these) |
| **Next** | **PWA dashboard** (per `docs/design/uiux.md`) | The answer-first UI is the *product* — currently headless |
| Next | **Clinician report export** (`GET /insights` → PDF) | Activates the referral loop + premium tier |
| Next | **Safe-window forecast** (ML pipeline, `docs/design/ml-pipeline.md`) | Owns the *time* path — "go before 4pm" |
| Later | White-label for employers/schools | New buyer groups; multiplies reach |
| Later | EU AQI, pollen + heat layers | Geographic + complementary-feature expansion |

## 11. The one-line pitch

> **"Stop guessing. We tell you if it's safe — for you — right now, what
> to do, and which route to take. And we learn your triggers so tomorrow
> is safer than today."**

---

## Sources

- Apple iTunes Lookup API & Google Play listings (IQAir, Plume, AsthmaMD)
- breezometer.com → redirect to Google Maps Platform (environment products)
- purpleair.com; EPA Air Sensor Toolbox (low-cost sensor guidance)
- airnow.gov (US AQI, flag program, EnviroFlash)
- resmed.com (Propeller Health)
- WHO fact sheets (asthma ~363M; COPD ~213M / 3rd-leading cause of death)
- Research performed 2026-08-10; URLs verified during this session.
