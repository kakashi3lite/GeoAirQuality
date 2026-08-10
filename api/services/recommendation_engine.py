"""Rule-based Recommendation Engine.

Generates actionable, condition-specific guidance from a risk assessment.
Instead of reporting raw numbers ("AQI is 150"), the engine tells the
patient what to DO: "Wait until 4pm", "Avoid downtown", "Wear an N95 mask".

Recommendation types:
    * general    - overall guidance based on risk level
    * precaution - protective actions (masks, medication, windows)
    * location   - areas to avoid
    * timing     - safe time windows
    * route      - route-specific guidance (when destination is provided)

Rule-based by design: deterministic, auditable, and safe for health
guidance (no LLM hallucination risk).
"""

import logging
from typing import Any, Dict, List, Optional

from .safety_engine import EnvironmentalSnapshot, PatientContext

logger = logging.getLogger(__name__)

# Human-readable labels per condition
_CONDITION_LABELS = {
    "asthma": "asthma",
    "copd": "COPD",
    "bronchitis": "bronchitis",
    "allergy": "allergies",
}


class RecommendationEngine:
    """Generates condition-specific recommendations from an assessment."""

    def __init__(self, context: PatientContext):
        self.context = context
        self.thresholds = context.thresholds()

    def _condition_label(self) -> str:
        labels = [
            _CONDITION_LABELS.get(c.lower(), c)
            for c in self.context.conditions
        ]
        if len(labels) == 1:
            return labels[0]
        return "your conditions"

    def generate(
        self,
        assessment: Dict[str, Any],
        snapshot: EnvironmentalSnapshot,
        news_events: Optional[List[Dict[str, Any]]] = None,
        route_risk: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Build the ordered recommendation list for a patient."""
        recommendations: List[Dict[str, str]] = []
        score = assessment["safety_score"]
        label = self._condition_label()

        # --- Base guidance by risk level -------------------------------
        if score >= 80:
            recommendations.append({
                "type": "general",
                "text": (
                    f"Conditions are favorable for {label}. "
                    f"Enjoy your outdoor activities."
                ),
            })
        elif score >= 60:
            recommendations.append({
                "type": "precaution",
                "text": (
                    f"Limit prolonged outdoor exertion. Keep your rescue "
                    f"medication accessible if you step outside."
                ),
            })
        elif score >= 40:
            recommendations.append({
                "type": "precaution",
                "text": (
                    "Avoid strenuous outdoor activity. Consider staying "
                    "indoors where possible and keep windows closed."
                ),
            })
        else:
            recommendations.append({
                "type": "precaution",
                "text": (
                    "Stay indoors with windows closed. Use air purification "
                    "if available. Avoid outdoor activity entirely."
                ),
            })

        # --- Pollutant-specific guidance ------------------------------
        if snapshot.pm25 is not None and snapshot.pm25 > self.thresholds.pm25:
            recommendations.append({
                "type": "precaution",
                "text": (
                    f"PM2.5 is {snapshot.pm25:.0f} µg/m³ — above your "
                    f"{label} threshold of {self.thresholds.pm25:.0f} µg/m³. "
                    f"Wear an N95 mask if you must go outside and limit "
                    f"exposure to under 30 minutes."
                ),
            })
        if snapshot.o3 is not None and snapshot.o3 > self.thresholds.o3:
            recommendations.append({
                "type": "timing",
                "text": (
                    "Ozone levels are elevated. Ozone typically peaks in "
                    "the afternoon — plan outdoor activity for the morning."
                ),
            })
        if snapshot.no2 is not None and snapshot.no2 > self.thresholds.no2:
            recommendations.append({
                "type": "location",
                "text": (
                    "Nitrogen dioxide is elevated, often near busy roads. "
                    "Choose routes away from major traffic corridors."
                ),
            })

        # --- Weather-specific guidance --------------------------------
        if snapshot.humidity is not None and snapshot.humidity > self.thresholds.humidity_max:
            recommendations.append({
                "type": "precaution",
                "text": (
                    f"High humidity ({snapshot.humidity:.0f}%) can trigger "
                    f"{label}. Consider using your preventive inhaler "
                    f"before going out."
                ),
            })
        if snapshot.temperature is not None and snapshot.temperature < self.thresholds.temp_min:
            recommendations.append({
                "type": "precaution",
                "text": (
                    "Cold air can constrict airways. Cover your nose and "
                    "mouth with a scarf and warm up gradually before exertion."
                ),
            })

        # --- News/event-specific guidance -----------------------------
        for event in (news_events or []):
            category = (event.get("category") or "").lower()
            title = event.get("title", "Air quality event")
            distance = float(event.get("distance_km", 0.0))
            if category == "wildfire":
                recommendations.append({
                    "type": "location",
                    "text": (
                        f"Active wildfire smoke reported {distance:.0f} km "
                        f"away ({title}). Avoid that direction and keep "
                        f"windows closed."
                    ),
                })
            elif category in ("industrial", "traffic"):
                recommendations.append({
                    "type": "location",
                    "text": (
                        f"Localized pollution reported {distance:.0f} km "
                        f"away ({title}). Consider alternate routes that "
                        f"avoid the affected area."
                    ),
                })
            elif category == "health_advisory":
                recommendations.append({
                    "type": "precaution",
                    "text": (
                        f"A health advisory affecting {label} patients is "
                        f"active ({title}). Limit outdoor exposure until it "
                        f"is lifted."
                    ),
                })

        # --- Route-specific guidance -----------------------------------
        if route_risk is not None:
            route_score = int(route_risk.get("route_risk_score", score))
            diff = route_score - score
            if diff >= 10:
                recommendations.append({
                    "type": "route",
                    "text": (
                        f"The route to your destination is notably safer "
                        f"than your current area (route safety {route_score}/100 "
                        f"vs {score}/100). Consider heading out now."
                    ),
                })
            elif diff <= -10:
                recommendations.append({
                    "type": "route",
                    "text": (
                        f"The route to your destination is riskier than your "
                        f"current area (route safety {route_score}/100). "
                        f"Consider delaying travel or choosing an alternate path."
                    ),
                })
            else:
                recommendations.append({
                    "type": "route",
                    "text": (
                        f"Route safety to your destination is {route_score}/100 "
                        f"— similar to your current area. Proceed with the "
                        f"general precautions above."
                    ),
                })

        return recommendations
