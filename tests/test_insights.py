#!/usr/bin/env python3
"""
Tests for the personal insights service (Phase 3).

Covers the pure statistics: Pearson correlation, trigger ranking,
hour-bucket safest/riskiest times, and the 30-day trend — including
empty-history behavior.
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from services.insights import (
    _pearson,
    compute_top_triggers,
    compute_time_buckets,
    compute_trend,
    compute_insights,
)


def _sym(severity, hour, **snapshot):
    return {
        "severity": severity,
        "logged_at": datetime(2026, 8, 1, hour, 0),
        "weather_snapshot": snapshot,
    }


# --------------------------------------------------------------------------
# Pearson
# --------------------------------------------------------------------------
def test_pearson_perfect_positive():
    assert _pearson([1, 2, 3, 4], [2, 4, 6, 8]) > 0.99


def test_pearson_perfect_negative():
    assert _pearson([1, 2, 3, 4], [8, 6, 4, 2]) < -0.99


def test_pearson_undefined_short_or_flat():
    assert _pearson([1], [2]) == 0.0
    assert _pearson([3, 3, 3], [1, 2, 3]) == 0.0  # zero variance on x


# --------------------------------------------------------------------------
# Top triggers
# --------------------------------------------------------------------------
def test_compute_top_triggers_finds_pm25_correlation():
    # severity rises with PM2.5 -> PM2.5 should rank first with r > 0.5
    symptoms = [
        _sym(1, 8, aqi=40, pm25=5, humidity=40, temperature=18),
        _sym(2, 9, aqi=70, pm25=20, humidity=50, temperature=20),
        _sym(3, 10, aqi=110, pm25=38, humidity=60, temperature=22),
        _sym(4, 11, aqi=150, pm25=55, humidity=70, temperature=24),
    ]
    triggers = compute_top_triggers(symptoms)
    assert triggers, "expected at least one trigger"
    pm = next((t for t in triggers if t["factor"] == "PM2.5"), None)
    assert pm is not None
    assert pm["correlation"] > 0.5
    assert pm["occurrences"] == 4


def test_compute_top_triggers_empty():
    assert compute_top_triggers([]) == []


def test_compute_top_triggers_requires_min_occurrences():
    symptoms = [_sym(1, 8, pm25=5), _sym(2, 9, pm25=40)]
    assert compute_top_triggers(symptoms) == []  # only 2 samples < 3


# --------------------------------------------------------------------------
# Time buckets
# --------------------------------------------------------------------------
def test_time_buckets_safest_and_riskiest():
    symptoms = [
        _sym(1, 7, aqi=40),  # morning, mild
        _sym(1, 7, aqi=45),
        _sym(2, 7, aqi=50),
        _sym(5, 17, aqi=150),  # evening, severe
        _sym(5, 17, aqi=160),
    ]
    safest, riskiest = compute_time_buckets(symptoms, limit=1)
    assert safest == [{"hour": 7, "avg_severity": 1.33}]
    assert riskiest == [{"hour": 17, "avg_severity": 5.0}]


def test_time_buckets_empty():
    safest, riskiest = compute_time_buckets([])
    assert safest == [] and riskiest == []


# --------------------------------------------------------------------------
# Trend
# --------------------------------------------------------------------------
def test_compute_trend_improving():
    # newer (first half) lower severity than older
    symptoms = [
        _sym(2, 10, aqi=60),
        _sym(2, 11, aqi=60),  # newer
        _sym(4, 12, aqi=90),
        _sym(4, 13, aqi=90),  # older
    ]
    text = compute_trend(symptoms)
    assert "down" in text


def test_compute_trend_insufficient_data():
    assert "Log more symptoms" in compute_trend([])
    assert "Log more symptoms" in compute_trend([_sym(2, 10, aqi=60)])


# --------------------------------------------------------------------------
# Full insights
# --------------------------------------------------------------------------
def test_compute_insights_empty_history():
    result = compute_insights([])
    assert result["top_triggers"] == []
    assert result["safest_times"] == []
    assert result["riskiest_times"] == []
    assert "Log more symptoms" in result["recent_trend"]
    assert result["period_days"] == 30


def test_compute_insights_full():
    symptoms = [
        _sym(1, 7, aqi=40, pm25=6, humidity=45, temperature=18),
        _sym(1, 7, aqi=42, pm25=7, humidity=46, temperature=18),
        _sym(2, 8, aqi=80, pm25=22, humidity=55, temperature=21),
        _sym(4, 17, aqi=150, pm25=60, humidity=75, temperature=26),
        _sym(5, 17, aqi=180, pm25=85, humidity=80, temperature=27),
    ]
    result = compute_insights(symptoms, period_days=30, limit=1)
    assert result["period_days"] == 30
    assert result["safest_times"] and result["riskiest_times"]
    assert result["safest_times"][0]["hour"] == 7
    assert result["riskiest_times"][0]["hour"] == 17
    assert any(t["factor"] == "PM2.5" for t in result["top_triggers"])
    assert isinstance(result["recent_trend"], str)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
