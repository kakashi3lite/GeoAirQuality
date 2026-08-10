#!/usr/bin/env python3
"""
Unit tests for the personalized Safety Scoring Engine.

Tests the ORM-free scoring and recommendation logic (no database or
Redis required). These validate the core patient-facing behavior:

  * Per-condition threshold calibration (asthma vs COPD vs allergy)
  * Weighted multi-factor scoring math
  * Risk level mapping
  * Contribution attribution
  * Recommendation template triggering
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from services.safety_engine import (
    PatientContext,
    EnvironmentalSnapshot,
    RiskScorer,
    risk_level_from_score,
    data_status_from_snapshot,
    WEIGHT_AQI,
    WEIGHT_WEATHER,
    WEIGHT_NEWS,
    WEIGHT_HISTORY,
)
from services.recommendation_engine import RecommendationEngine


def make_context(conditions=("asthma",), **overrides):
    defaults = dict(
        user_id="test-patient",
        conditions=list(conditions),
        aqi_threshold=100.0,
        pm25_threshold=35.4,
        o3_threshold=70.0,
        no2_threshold=100.0,
        alert_radius_km=25.0,
    )
    defaults.update(overrides)
    return PatientContext(**defaults)


def clean_snapshot(**overrides):
    """Snapshot with realistic clean-air values (safe by default)."""
    defaults = dict(
        aqi=25.0,
        pm25=5.0,
        pm10=10.0,
        o3=20.0,
        no2=8.0,
        temperature=20.0,
        humidity=45.0,
        wind_speed=12.0,
        wind_direction="SW",
        source_timestamp=datetime.utcnow(),
        reading_count=10,
        aq_reading_count=10,
    )
    defaults.update(overrides)
    return EnvironmentalSnapshot(**defaults)


# ----------------------------------------------------------------------
# Risk level mapping
# ----------------------------------------------------------------------
def test_risk_level_mapping():
    assert risk_level_from_score(100) == "low"
    assert risk_level_from_score(80) == "low"
    assert risk_level_from_score(79) == "moderate"
    assert risk_level_from_score(60) == "moderate"
    assert risk_level_from_score(59) == "high"
    assert risk_level_from_score(40) == "high"
    assert risk_level_from_score(39) == "very_high"
    assert risk_level_from_score(0) == "very_high"


# ----------------------------------------------------------------------
# Condition thresholds
# ----------------------------------------------------------------------
def test_condition_thresholds_copd_more_protective():
    asthma = make_context(conditions=("asthma",)).thresholds()
    copd = make_context(conditions=("copd",)).thresholds()
    # COPD is more sensitive to particles
    assert copd.pm25 < asthma.pm25
    assert copd.pm10 < asthma.pm10


def test_multi_condition_merges_most_protective():
    merged = make_context(conditions=("asthma", "allergy")).thresholds()
    # min(pm25: 35 asthma, 40 allergy) = 35
    assert merged.pm25 == 35.0
    # min(pm10: 100 asthma, 54 allergy) = 54
    assert merged.pm10 == 54.0


# ----------------------------------------------------------------------
# AQI component scoring
# ----------------------------------------------------------------------
def test_clean_air_scores_100():
    context = make_context()
    scorer = RiskScorer(context)
    score, factors = scorer.score_aqi_component(clean_snapshot())
    assert score == 100.0
    assert factors == []


def test_elevated_pm25_reduces_score():
    context = make_context(conditions=("asthma",))
    scorer = RiskScorer(context)
    snapshot = clean_snapshot(pm25=50.0)  # above asthma threshold 35
    score, factors = scorer.score_aqi_component(snapshot)
    assert score < 100.0
    assert any(f["factor"] == "PM2.5" for f in factors)
    # ratio 50/35 = 1.43 -> between 70 and 40
    assert 40.0 <= score <= 70.0


def test_worst_pollutant_dominates():
    context = make_context()
    scorer = RiskScorer(context)
    snapshot = clean_snapshot(pm25=200.0, no2=5.0)
    score, _ = scorer.score_aqi_component(snapshot)
    # PM2.5 ratio ~5.7 -> 10; NO2 fine -> 100. Worst dominates.
    assert score == 10.0


def test_missing_data_is_neutral():
    context = make_context()
    scorer = RiskScorer(context)
    score, factors = scorer.score_aqi_component(EnvironmentalSnapshot())
    assert score == 100.0
    assert factors == []


# ----------------------------------------------------------------------
# Weather component scoring
# ----------------------------------------------------------------------
def test_weather_penalties():
    context = make_context(conditions=("asthma",))
    scorer = RiskScorer(context)
    # humid (asthma max 85) + stagnant + hot
    snapshot = clean_snapshot(
        humidity=95.0, wind_speed=2.0, temperature=38.0, aqi=150.0
    )
    score, factors = scorer.score_weather_component(snapshot)
    assert score < 100.0
    names = {f["factor"] for f in factors}
    assert {"Humidity", "Temperature", "Stagnant air"} <= names


def test_weather_clean_no_penalty():
    context = make_context()
    scorer = RiskScorer(context)
    score, factors = scorer.score_weather_component(clean_snapshot())
    assert score == 100.0
    assert factors == []


# ----------------------------------------------------------------------
# News component scoring (Phase 2 readiness)
# ----------------------------------------------------------------------
def test_news_no_events_neutral():
    context = make_context()
    scorer = RiskScorer(context)
    score, factors = scorer.score_news_component([])
    assert score == 100.0
    assert factors == []


def test_news_severe_nearby_event_reduces_score():
    context = make_context()
    scorer = RiskScorer(context)
    events = [
        {
            "title": "Wildfire smoke advisory",
            "category": "wildfire",
            "severity": 90,
            "respiratory_relevance": 90,
            "distance_km": 5.0,
        }
    ]
    score, factors = scorer.score_news_component(events)
    assert score < 100.0
    assert len(factors) == 1
    # far event has less impact
    far_score, _ = scorer.score_news_component([{**events[0], "distance_km": 24.0}])
    assert far_score > score


# ----------------------------------------------------------------------
# Full assessment
# ----------------------------------------------------------------------
def test_assessment_clean_conditions_low_risk():
    context = make_context(conditions=("asthma",))
    scorer = RiskScorer(context)
    result = scorer.assess(clean_snapshot())
    assert result["safety_score"] == 100
    assert result["risk_level"] == "low"
    assert result["component_scores"]["aqi"] == 100.0
    assert result["contributing_factors"] == []
    assert sum(result["contributions"].values()) == 0


def test_assessment_polluted_high_risk():
    context = make_context(conditions=("asthma",))
    scorer = RiskScorer(context)
    snapshot = clean_snapshot(pm25=120.0, aqi=165.0, humidity=95.0)
    result = scorer.assess(snapshot)
    assert result["safety_score"] < 60
    assert result["risk_level"] in ("high", "very_high")
    assert result["contributing_factors"]
    # contributions should sum to ~100
    assert sum(result["contributions"].values()) in (99, 100, 101)


def test_assessment_weights_are_consistent():
    # weights must sum to 1.0
    total = WEIGHT_AQI + WEIGHT_WEATHER + WEIGHT_NEWS + WEIGHT_HISTORY
    assert abs(total - 1.0) < 1e-9


# ----------------------------------------------------------------------
# Recommendation engine
# ----------------------------------------------------------------------
def test_recommendation_low_risk_favorable():
    context = make_context(conditions=("asthma",))
    scorer = RiskScorer(context)
    engine = RecommendationEngine(context)
    assessment = scorer.assess(clean_snapshot())
    recs = engine.generate(assessment, clean_snapshot())
    assert any("favorable" in r["text"].lower() for r in recs)


def test_recommendation_pm25_mask_advice():
    context = make_context(conditions=("asthma",))
    scorer = RiskScorer(context)
    engine = RecommendationEngine(context)
    snapshot = clean_snapshot(pm25=60.0)
    assessment = scorer.assess(snapshot)
    recs = engine.generate(assessment, snapshot)
    texts = " ".join(r["text"].lower() for r in recs)
    assert "n95" in texts
    assert "asthma" in texts  # condition-personalized


def test_recommendation_news_wildfire_location_advice():
    context = make_context(conditions=("asthma",))
    scorer = RiskScorer(context)
    engine = RecommendationEngine(context)
    snapshot = clean_snapshot()
    events = [
        {
            "title": "Wildfire smoke advisory",
            "category": "wildfire",
            "severity": 90,
            "respiratory_relevance": 90,
            "distance_km": 5.0,
        }
    ]
    assessment = scorer.assess(snapshot, news_events=events)
    recs = engine.generate(assessment, snapshot, news_events=events)
    texts = " ".join(r["text"].lower() for r in recs)
    assert "wildfire" in texts
    assert "avoid" in texts


def test_recommendation_route_comparison():
    context = make_context(conditions=("asthma",))
    scorer = RiskScorer(context)
    engine = RecommendationEngine(context)
    snapshot = clean_snapshot()
    assessment = scorer.assess(snapshot)
    route_risk = {"route_risk_score": 90, "worst_segment": "origin", "segments": []}
    recs = engine.generate(assessment, snapshot, route_risk=route_risk)
    texts = " ".join(r["text"].lower() for r in recs)
    assert "route" in texts


def test_recommendation_very_high_stay_indoors():
    context = make_context(conditions=("asthma",))
    scorer = RiskScorer(context)
    engine = RecommendationEngine(context)
    snapshot = clean_snapshot(pm25=500.0, aqi=400.0, humidity=98.0, temperature=45.0)
    assessment = scorer.assess(snapshot)
    recs = engine.generate(assessment, snapshot)
    texts = " ".join(r["text"].lower() for r in recs)
    assert "stay indoors" in texts


# ----------------------------------------------------------------------
# Data availability honesty
# ----------------------------------------------------------------------
def test_data_status_helper():
    assert data_status_from_snapshot(clean_snapshot()) == "available"
    assert data_status_from_snapshot(clean_snapshot(aq_reading_count=0)) == "partial"
    assert data_status_from_snapshot(EnvironmentalSnapshot()) == "unavailable"


def test_recommendation_unavailable_data_caution():
    context = make_context(conditions=("asthma",))
    scorer = RiskScorer(context)
    engine = RecommendationEngine(context)
    snapshot = EnvironmentalSnapshot()  # no data at all
    assessment = scorer.assess(snapshot)
    # Engine must NOT claim it is safe when there is no data
    recs = engine.generate(assessment, snapshot, data_status="unavailable")
    texts = " ".join(r["text"].lower() for r in recs)
    assert "no recent monitoring data" in texts
    assert "favorable" not in texts


def test_recommendation_partial_data_note():
    context = make_context(conditions=("asthma",))
    engine = RecommendationEngine(context)
    snapshot = clean_snapshot(aq_reading_count=0)
    assessment = RiskScorer(context).assess(snapshot)
    recs = engine.generate(assessment, snapshot, data_status="partial")
    texts = " ".join(r["text"].lower() for r in recs)
    assert "partial" in texts


if __name__ == "__main__":
    # Minimal runner: executes each test_* function, reports failures.
    import traceback

    failures = 0
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
