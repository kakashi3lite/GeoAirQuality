"""Personalized Safety Scoring Engine.

Computes a weighted multi-factor safety score (0-100) calibrated to a
patient's specific respiratory condition. Factors:

    * AQI / pollutant levels        - 40% weight
    * Weather (temp, humidity, wind) - 20% weight
    * Active news events            - 25% weight (stubbed until Phase 2)
    * Personal symptom history      - 15% weight (stubbed until Phase 3)

Scoring philosophy: 100 = perfectly safe for THIS patient, 0 = extremely
dangerous for THIS patient. Within each component the worst factor
dominates (conservative by design -- appropriate for health guidance).

The engine is ORM-free: it operates on plain dataclasses so the scoring
logic is fully unit-testable without a database.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Component weights (must sum to 1.0)
WEIGHT_AQI = 0.40
WEIGHT_WEATHER = 0.20
WEIGHT_NEWS = 0.25
WEIGHT_HISTORY = 0.15


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


def risk_level_from_score(score: float) -> str:
    """Map a 0-100 safety score to a risk level."""
    if score >= 80:
        return RiskLevel.LOW.value
    if score >= 60:
        return RiskLevel.MODERATE.value
    if score >= 40:
        return RiskLevel.HIGH.value
    return RiskLevel.VERY_HIGH.value


@dataclass
class ConditionThresholds:
    """Environmental thresholds calibrated per respiratory condition.

    Lower pollutant thresholds = more protective. When a patient has
    multiple conditions, the merged thresholds take the most protective
    (worst-case) value for each metric.
    """
    pm25: float = 35.0
    pm10: float = 100.0
    o3: float = 70.0
    no2: float = 100.0
    humidity_min: float = 15.0
    humidity_max: float = 85.0
    temp_min: float = -10.0
    temp_max: float = 35.0
    label: str = "condition"


CONDITION_THRESHOLDS: Dict[str, ConditionThresholds] = {
    "asthma": ConditionThresholds(
        pm25=35.0, pm10=100.0, o3=70.0, no2=100.0,
        humidity_min=20.0, humidity_max=85.0,
        temp_min=5.0, temp_max=32.0, label="asthma",
    ),
    "copd": ConditionThresholds(
        pm25=25.0, pm10=50.0, o3=60.0, no2=100.0,
        humidity_min=15.0, humidity_max=90.0,
        temp_min=0.0, temp_max=30.0, label="COPD",
    ),
    "bronchitis": ConditionThresholds(
        pm25=35.0, pm10=100.0, o3=70.0, no2=100.0,
        humidity_min=20.0, humidity_max=90.0,
        temp_min=5.0, temp_max=32.0, label="bronchitis",
    ),
    "allergy": ConditionThresholds(
        pm25=40.0, pm10=54.0, o3=80.0, no2=120.0,
        humidity_min=10.0, humidity_max=95.0,
        temp_min=-5.0, temp_max=35.0, label="allergies",
    ),
}


@dataclass
class PatientContext:
    """Lightweight, ORM-free patient profile for the scoring engine."""
    user_id: str
    conditions: List[str] = field(default_factory=lambda: ["asthma"])
    aqi_threshold: float = 100.0
    pm25_threshold: float = 35.4
    o3_threshold: float = 70.0
    no2_threshold: float = 100.0
    alert_radius_km: float = 25.0

    @classmethod
    def from_profile(cls, profile: Any) -> "PatientContext":
        """Build a PatientContext from a SQLAlchemy PatientProfile row."""
        conditions = list(profile.conditions or ["asthma"])
        return cls(
            user_id=profile.user_id,
            conditions=conditions,
            aqi_threshold=float(profile.aqi_threshold or 100.0),
            pm25_threshold=float(profile.pm25_threshold or 35.4),
            o3_threshold=float(profile.o3_threshold or 70.0),
            no2_threshold=float(profile.no2_threshold or 100.0),
            alert_radius_km=float(profile.alert_radius_km or 25.0),
        )

    def thresholds(self) -> ConditionThresholds:
        """Merge per-condition thresholds; most protective value wins."""
        merged = ConditionThresholds(
            label=", ".join(self.conditions) if self.conditions else "condition"
        )
        for cond in self.conditions:
            t = CONDITION_THRESHOLDS.get(cond.lower())
            if t is None:
                continue
            merged.pm25 = min(merged.pm25, t.pm25)
            merged.pm10 = min(merged.pm10, t.pm10)
            merged.o3 = min(merged.o3, t.o3)
            merged.no2 = min(merged.no2, t.no2)
            merged.humidity_max = min(merged.humidity_max, t.humidity_max)
            merged.humidity_min = max(merged.humidity_min, t.humidity_min)
            merged.temp_max = min(merged.temp_max, t.temp_max)
            merged.temp_min = max(merged.temp_min, t.temp_min)
        return merged


@dataclass
class EnvironmentalSnapshot:
    """Current environmental conditions at a location."""
    aqi: Optional[float] = None
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    o3: Optional[float] = None
    no2: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[str] = None
    source_timestamp: Optional[datetime] = None
    reading_count: int = 0
    grid_id: Optional[str] = None


def _pollutant_score(value: Optional[float], threshold: float) -> float:
    """Map a pollutant concentration to a 0-100 safety score.

    Piecewise-linear against the patient's threshold:
        <= 0.5x threshold -> 100 (safe)
        <= 1.0x threshold -> 100..70
        <= 2.0x threshold -> 70..40
        <= 3.0x threshold -> 40..10
        >  3.0x threshold -> 10 (dangerous)
    """
    if value is None or threshold <= 0:
        return 100.0
    ratio = value / threshold
    if ratio <= 0.5:
        return 100.0
    if ratio <= 1.0:
        return 70.0 + 30.0 * (1.0 - (ratio - 0.5) / 0.5)
    if ratio <= 2.0:
        return 40.0 + 30.0 * (1.0 - (ratio - 1.0) / 1.0)
    if ratio <= 3.0:
        return 10.0 + 30.0 * (1.0 - (ratio - 2.0) / 1.0)
    return 10.0


def _aqi_score(aqi: Optional[float]) -> float:
    """Map a raw AQI to a 0-100 safety score using EPA bands."""
    if aqi is None:
        return 100.0
    if aqi <= 50:
        return 100.0
    if aqi <= 100:
        return 90.0
    if aqi <= 150:
        return 70.0
    if aqi <= 200:
        return 45.0
    if aqi <= 300:
        return 25.0
    return 10.0


class RiskScorer:
    """Computes the weighted multi-factor safety score for a patient."""

    def __init__(self, context: PatientContext):
        self.context = context
        self.thresholds = context.thresholds()

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------
    def score_aqi_component(
        self, snapshot: EnvironmentalSnapshot
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Score the pollutant/AQI component (weight 0.40)."""
        factors: List[Dict[str, Any]] = []
        checks = [
            ("PM2.5", snapshot.pm25, self.thresholds.pm25, "µg/m³"),
            ("PM10", snapshot.pm10, self.thresholds.pm10, "µg/m³"),
            ("Ozone (O₃)", snapshot.o3, self.thresholds.o3, "µg/m³"),
            ("Nitrogen dioxide (NO₂)", snapshot.no2, self.thresholds.no2, "µg/m³"),
        ]
        scores: List[float] = []
        for name, value, threshold, unit in checks:
            s = _pollutant_score(value, threshold)
            scores.append(s)
            if value is not None and s < 100:
                factors.append({
                    "factor": name,
                    "value": round(float(value), 1),
                    "threshold": threshold,
                    "unit": unit,
                    "detail": (
                        f"{name} is {value:.1f} {unit} — above the safe "
                        f"threshold of {threshold:.0f} {unit} for your condition"
                    ),
                })
        aqi_s = _aqi_score(snapshot.aqi)
        scores.append(aqi_s)
        if snapshot.aqi is not None and aqi_s < 100:
            factors.append({
                "factor": "Air Quality Index",
                "value": round(float(snapshot.aqi), 0),
                "threshold": self.context.aqi_threshold,
                "unit": "index",
                "detail": (
                    f"Air Quality Index is {snapshot.aqi:.0f} — elevated "
                    f"for your condition (threshold {self.context.aqi_threshold:.0f})"
                ),
            })
        component = min(scores) if scores else 100.0
        return component, factors

    def score_weather_component(
        self, snapshot: EnvironmentalSnapshot
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Score the weather component (weight 0.20)."""
        factors: List[Dict[str, Any]] = []
        penalty = 0.0
        t = self.thresholds

        if snapshot.temperature is not None:
            if snapshot.temperature < t.temp_min or snapshot.temperature > t.temp_max:
                penalty = max(penalty, 20.0)
                factors.append({
                    "factor": "Temperature",
                    "value": round(float(snapshot.temperature), 1),
                    "threshold": f"{t.temp_min:.0f}–{t.temp_max:.0f}",
                    "unit": "°C",
                    "detail": (
                        f"Temperature of {snapshot.temperature:.0f}°C is outside "
                        f"the comfortable range ({t.temp_min:.0f}–{t.temp_max:.0f}°C) "
                        f"for your condition"
                    ),
                })

        if snapshot.humidity is not None:
            if snapshot.humidity > t.humidity_max or snapshot.humidity < t.humidity_min:
                penalty = max(penalty, 25.0)
                factors.append({
                    "factor": "Humidity",
                    "value": round(float(snapshot.humidity), 1),
                    "threshold": f"{t.humidity_min:.0f}–{t.humidity_max:.0f}",
                    "unit": "%",
                    "detail": (
                        f"Humidity of {snapshot.humidity:.0f}% can aggravate "
                        f"your condition (comfortable range "
                        f"{t.humidity_min:.0f}–{t.humidity_max:.0f}%)"
                    ),
                })

        # Stagnant air traps pollutants close to the ground
        if snapshot.wind_speed is not None and snapshot.wind_speed < 5.0:
            if (snapshot.aqi or 0) >= 100:
                penalty = max(penalty, 10.0)
                factors.append({
                    "factor": "Stagnant air",
                    "value": round(float(snapshot.wind_speed), 1),
                    "threshold": 5.0,
                    "unit": "km/h",
                    "detail": (
                        f"Low wind speed ({snapshot.wind_speed:.0f} km/h) is "
                        f"trapping pollutants near ground level"
                    ),
                })

        component = max(100.0 - penalty, 0.0)
        return component, factors

    def score_news_component(
        self, news_events: List[Dict[str, Any]]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Score the news-events component (weight 0.25).

        Each event carries: severity (0-100), respiratory_relevance
        (0-100), distance_km, title, category. The most impactful event
        dominates this component.
        """
        if not news_events:
            return 100.0, []
        factors: List[Dict[str, Any]] = []
        worst = 100.0
        radius = max(self.context.alert_radius_km, 1.0)
        for event in news_events:
            severity = float(event.get("severity", 50))
            relevance = float(event.get("respiratory_relevance", 50))
            distance = float(event.get("distance_km", 0.0))
            proximity = max(0.2, 1.0 - (distance / radius))
            impact = severity * relevance / 100.0 * proximity
            score = max(20.0, 100.0 - impact)
            worst = min(worst, score)
            factors.append({
                "factor": "News event",
                "value": event.get("title", "Air quality event"),
                "threshold": None,
                "unit": None,
                "detail": (
                    f"{event.get('title', 'Air quality event')} "
                    f"({event.get('category', 'event')}) reported "
                    f"{distance:.1f} km away"
                ),
            })
        return worst, factors

    def score_history_component(
        self, symptoms: List[Dict[str, Any]]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Score the personal-history component (weight 0.15).

        Phase 1 returns neutral (no history engine yet). Phase 3 adds
        symptom/trigger correlation.
        """
        if not symptoms:
            return 100.0, []
        # Reserved for Phase 3 trigger-learning; conservative neutral now.
        return 100.0, []

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def assess(
        self,
        snapshot: EnvironmentalSnapshot,
        news_events: Optional[List[Dict[str, Any]]] = None,
        symptoms: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compute the full risk assessment for a patient at a location."""
        news_events = news_events or []
        symptoms = symptoms or []

        aqi_s, aqi_factors = self.score_aqi_component(snapshot)
        weather_s, weather_factors = self.score_weather_component(snapshot)
        news_s, news_factors = self.score_news_component(news_events)
        history_s, history_factors = self.score_history_component(symptoms)

        component_scores = {
            "aqi": round(aqi_s, 1),
            "weather": round(weather_s, 1),
            "news": round(news_s, 1),
            "history": round(history_s, 1),
        }

        raw_score = (
            WEIGHT_AQI * aqi_s
            + WEIGHT_WEATHER * weather_s
            + WEIGHT_NEWS * news_s
            + WEIGHT_HISTORY * history_s
        )
        safety_score = int(round(max(0.0, min(100.0, raw_score))))

        # Clinical safety cap: dangerous pollutant levels dominate the
        # assessment regardless of favorable weather/news conditions.
        #   pollutant ratio > 3x threshold  OR  AQI > 300  -> cap at 39
        #   pollutant ratio 2-3x threshold  OR  AQI 200-300 -> cap at 59
        if aqi_s <= 10.0:
            safety_score = min(safety_score, 39)
        elif aqi_s <= 25.0:
            safety_score = min(safety_score, 59)

        # Contribution percentages are proportional to each component's
        # deficit from perfect safety (100).
        deficits = {
            k: max(0.0, 100.0 - v) for k, v in component_scores.items()
        }
        total_deficit = sum(deficits.values())
        contributions = {}
        if total_deficit > 0:
            contributions = {
                k: int(round(v / total_deficit * 100.0))
                for k, v in deficits.items()
            }
            # Normalize rounding drift so contributions sum to ~100
            diff = 100 - sum(contributions.values())
            if contributions and diff:
                key = max(contributions, key=contributions.get)
                contributions[key] += diff

        contributing_factors = (
            aqi_factors + weather_factors + news_factors + history_factors
        )

        return {
            "safety_score": safety_score,
            "risk_level": risk_level_from_score(safety_score),
            "component_scores": component_scores,
            "contributions": contributions,
            "contributing_factors": contributing_factors,
        }
