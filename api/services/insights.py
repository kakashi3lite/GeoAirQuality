"""Personal insights: trigger correlation, time buckets, and trend.

Pure statistics over the patient's symptom history (no ML). Deterministic
and unit-testable. Consumes symptom dicts shaped like:

    {"severity": 3, "logged_at": datetime, "weather_snapshot": {...}}

`weather_snapshot` may carry aqi / pm25 / humidity / temperature.
"""

import logging
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Factors we can correlate with symptom severity (label -> snapshot key)
_TRIGGER_FACTORS = {
    "AQI": "aqi",
    "PM2.5": "pm25",
    "Humidity": "humidity",
    "Temperature": "temperature",
}


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient; 0.0 when undefined (n<2 or zero variance)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def compute_top_triggers(
    symptoms: List[Dict[str, Any]], min_occurrences: int = 3
) -> List[Dict[str, Any]]:
    """Rank environmental factors by correlation with symptom severity.

    Only factors with >= min_occurrences paired samples are considered;
    weak correlations (|r| < 0.2) are dropped to avoid noise.
    """
    triggers: List[Dict[str, Any]] = []
    for label, key in _TRIGGER_FACTORS.items():
        pairs = []
        for s in symptoms:
            snap = s.get("weather_snapshot") or {}
            value = snap.get(key)
            if value is None:
                continue
            pairs.append((float(value), float(s.get("severity", 3))))
        if len(pairs) < min_occurrences:
            continue
        corr = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
        if abs(corr) >= 0.2:
            triggers.append(
                {
                    "factor": label,
                    "correlation": round(corr, 2),
                    "occurrences": len(pairs),
                }
            )
    triggers.sort(key=lambda t: abs(t["correlation"]), reverse=True)
    return triggers


def _hour_buckets(symptoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate symptom severity by hour-of-day."""
    by_hour: Dict[int, List[float]] = defaultdict(list)
    for s in symptoms:
        logged = s.get("logged_at")
        if logged is None:
            continue
        by_hour[logged.hour].append(float(s.get("severity", 3)))
    buckets = [
        {
            "hour": hour,
            "avg_severity": round(mean(sev), 2),
            "occurrences": len(sev),
        }
        for hour, sev in by_hour.items()
    ]
    buckets.sort(key=lambda b: (b["avg_severity"], b["hour"]))
    return buckets


def compute_time_buckets(
    symptoms: List[Dict[str, Any]], limit: int = 3
) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """Safest (lowest avg severity) and riskiest (highest) hours."""
    buckets = _hour_buckets(symptoms)
    if not buckets:
        return [], []
    safest = [
        {"hour": b["hour"], "avg_severity": b["avg_severity"]} for b in buckets[:limit]
    ]
    riskiest = [
        {"hour": b["hour"], "avg_severity": b["avg_severity"]}
        for b in reversed(buckets[-limit:])
    ]
    return safest, riskiest


def compute_trend(symptoms: List[Dict[str, Any]]) -> str:
    """Plain-language 30-day severity trend (newer half vs older half)."""
    n = len(symptoms)
    if n < 2:
        return "Log more symptoms to see your 30-day trend."
    half = n // 2
    newer = [float(s.get("severity", 3)) for s in symptoms[:half]]
    older = [float(s.get("severity", 3)) for s in symptoms[half:]]
    avg_new, avg_old = mean(newer), mean(older)
    if avg_old <= 0:
        return "Log more symptoms to see your 30-day trend."
    pct = (avg_old - avg_new) / avg_old * 100.0
    if pct >= 5:
        return f"Your symptom severity is down {pct:.0f}% compared with two weeks ago."
    if pct <= -5:
        return (
            f"Your symptom severity is up {abs(pct):.0f}% compared with two weeks ago."
        )
    return "Your symptom severity is holding steady compared with two weeks ago."


def compute_insights(
    symptoms: List[Dict[str, Any]],
    period_days: int = 30,
    limit: int = 3,
) -> Dict[str, Any]:
    """Full insights payload from a patient's symptom history.

    Symptoms should be sorted newest-first (the trend splits on the
    midpoint of that ordering).
    """
    ordered = sorted(
        symptoms,
        key=lambda s: s.get("logged_at") or datetime.min,
        reverse=True,
    )
    safest, riskiest = compute_time_buckets(ordered, limit=limit)
    return {
        "top_triggers": compute_top_triggers(ordered),
        "safest_times": safest,
        "riskiest_times": riskiest,
        "recent_trend": compute_trend(ordered),
        "period_days": period_days,
    }
