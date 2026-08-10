"""Rule-based news enrichment: location, category, severity, relevance.

Deterministic NLP-lite processing — no ML models, no external geocoding
by default. Designed to be fast, auditable, and unit-testable.

Pipeline for each raw article:
  1. extract_location  -> (lat, lon) from source coords or a bundled
                          cities gazetteer (optionally Nominatim)
  2. categorize_event  -> wildfire | industrial | traffic | pollen |
                          dust_storm | regulatory | health_advisory | general
  3. score_severity    -> 0-100 (source base + urgency keywords)
  4. score_relevance   -> 0-100 (category base + disease keywords)
"""

import csv
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Keyword tables --------------------------------------------------------------
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "wildfire": [
        "wildfire",
        "forest fire",
        "brush fire",
        "bushfire",
        "smoke plume",
        "prescribed burn",
        "burn ban",
        "smoke advisory",
    ],
    "industrial": [
        "factory",
        "chemical leak",
        "industrial",
        "refinery",
        "emissions violation",
        "plant fire",
        "stack emission",
        "smelter",
    ],
    "traffic": [
        "traffic",
        "congestion",
        "highway",
        "diesel",
        "vehicle emissions",
        "idling",
        "tailpipe",
        "rush hour",
    ],
    "pollen": [
        "pollen",
        "allergy",
        "ragweed",
        "grass pollen",
        "tree pollen",
        "hay fever",
        "allergen",
    ],
    "dust_storm": [
        "dust storm",
        "sandstorm",
        "haboob",
        "saharan dust",
        "particulate matter",
        "dust advisory",
    ],
    "regulatory": [
        "epa",
        "violation",
        "fine",
        "regulation",
        "consent decree",
        "clean air act",
        "lawsuit",
        "sued",
    ],
    "health_advisory": [
        "health advisory",
        "asthma",
        "respiratory",
        "sensitive groups",
        "vulnerable",
        "health warning",
        "air quality alert",
        "action day",
        "ozone action",
    ],
}

_SEVERITY_KEYWORDS = {
    "emergency": 25,
    "evacuate": 25,
    "warning": 20,
    "hazardous": 20,
    "severe": 15,
    "critical": 15,
    "urgent": 15,
    "dangerous": 15,
    "extreme": 10,
    "unhealthy": 10,
    "advisory": 5,
    "alert": 5,
}

_RELEVANCE_KEYWORDS = {
    "asthma": 15,
    "copd": 15,
    "lung": 10,
    "breathing": 10,
    "respiratory": 10,
    "bronchitis": 10,
    "heart": 5,
}

# Category base relevance for respiratory patients (0-100)
_CATEGORY_RELEVANCE_BASE = {
    "wildfire": 90,
    "health_advisory": 85,
    "dust_storm": 75,
    "industrial": 70,
    "traffic": 50,
    "pollen": 40,
    "regulatory": 30,
    "general": 20,
}

# Source-type severity base
_SOURCE_SEVERITY_BASE = {"official_alert": 70, "news_media": 40}


# Gazetteer -------------------------------------------------------------------
_gazetteer: Optional[List[Dict[str, Any]]] = None


def load_gazetteer(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load the cities gazetteer once (cached)."""
    global _gazetteer
    if _gazetteer is not None:
        return _gazetteer
    path = path or os.path.join(
        os.path.dirname(__file__), "..", "data", "cities_gazetteer.csv"
    )
    cities = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cities.append(
                {
                    "name": row["name"].strip(),
                    "region": row["region"].strip(),
                    "country": row["country"].strip(),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                }
            )
    # Longer names first so "San Jose" matches before "San" style false hits
    cities.sort(key=lambda c: len(c["name"]), reverse=True)
    _gazetteer = cities
    return _gazetteer


def _gazetteer_regex() -> re.Pattern:
    """Compile a word-boundary regex over all gazetteer city names."""
    names = [re.escape(c["name"]) for c in load_gazetteer()]
    return re.compile(r"\b(" + "|".join(names) + r")\b", re.IGNORECASE)


# Location --------------------------------------------------------------------
def extract_location(
    title: str,
    summary: str = "",
    source_lat: Optional[float] = None,
    source_lon: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve article coordinates.

    Priority: (1) source-reported coords, (2) gazetteer city mention in
    title/summary, (3) None (Nominatim is opt-in via the caller).
    """
    if source_lat is not None and source_lon is not None:
        if -90 <= source_lat <= 90 and -180 <= source_lon <= 180:
            return float(source_lat), float(source_lon)

    text = f"{title} {summary}"
    match = _gazetteer_regex().search(text)
    if match:
        name = match.group(1)
        for city in load_gazetteer():
            if city["name"].lower() == name.lower():
                return city["lat"], city["lon"]
    return None, None


# Category --------------------------------------------------------------------
def categorize_event(title: str, summary: str = "") -> str:
    """Classify the event using curated keyword lists (first match wins)."""
    text = f"{title} {summary}".lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return category
    return "general"


# Severity --------------------------------------------------------------------
def score_severity(
    title: str,
    summary: str = "",
    source_type: str = "news_media",
) -> int:
    """0-100 severity: source base + urgency keyword boosts, capped."""
    text = f"{title} {summary}".lower()
    score = _SOURCE_SEVERITY_BASE.get(source_type, 40)
    for kw, boost in _SEVERITY_KEYWORDS.items():
        if kw in text:
            score += boost
    return max(0, min(100, score))


# Relevance -------------------------------------------------------------------
def score_respiratory_relevance(
    title: str,
    summary: str = "",
    category: str = "general",
) -> int:
    """0-100 relevance for respiratory patients: category base + boosts."""
    text = f"{title} {summary}".lower()
    score = _CATEGORY_RELEVANCE_BASE.get(category, 20)
    for kw, boost in _RELEVANCE_KEYWORDS.items():
        if kw in text:
            score += boost
    return max(0, min(100, score))


# Full processing -------------------------------------------------------------
def process_article(
    external_id: str,
    source_name: str,
    source_type: str,
    title: str,
    summary: str = "",
    url: Optional[str] = None,
    published_at: Any = None,
    source_lat: Optional[float] = None,
    source_lon: Optional[float] = None,
    raw_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Enrich a raw article into a dict ready for the news store."""
    lat, lon = extract_location(title, summary, source_lat, source_lon)
    category = categorize_event(title, summary)
    severity = score_severity(title, summary, source_type)
    relevance = score_respiratory_relevance(title, summary, category)

    return {
        "external_id": external_id,
        "source_name": source_name,
        "source_type": source_type,
        "title": title,
        "summary": summary,
        "url": url,
        "published_at": published_at,
        "latitude": lat,
        "longitude": lon,
        "event_category": category,
        "severity": severity,
        "respiratory_relevance": relevance,
        "raw_metadata": raw_data or {},
    }


# Title-similarity dedup helper (used by the aggregator)
def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
