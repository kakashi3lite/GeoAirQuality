"""News article persistence and spatial queries.

Uses the existing sync SessionLocal pattern. The upsert is idempotent
(ON CONFLICT DO UPDATE on external_id) so the scheduler can re-run
safely. Distances are computed with a dependency-free haversine so the
ORM rows stay plain — no extra SQL columns to map.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from models import NewsArticle

logger = logging.getLogger(__name__)

_STALE_HOURS_DEFAULT = 168  # 7 days


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (accurate enough for proximity decay)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def upsert_articles(db: Session, articles: List[Dict[str, Any]]) -> int:
    """Insert or update articles (idempotent on external_id)."""
    if not articles:
        return 0
    stmt = insert(NewsArticle).values(
        [
            {
                "external_id": a["external_id"],
                "source_name": a["source_name"],
                "source_type": a["source_type"],
                "title": a["title"],
                "summary": a.get("summary", ""),
                "url": a.get("url"),
                "published_at": a.get("published_at") or datetime.utcnow(),
                "latitude": a.get("latitude"),
                "longitude": a.get("longitude"),
                # Geo-tagged geometry when coordinates resolved
                "location": (
                    f"SRID=4326;POINT({a['longitude']} {a['latitude']})"
                    if a.get("latitude") is not None and a.get("longitude") is not None
                    else None
                ),
                "event_category": a.get("event_category", "general"),
                "severity": a.get("severity", 0),
                "respiratory_relevance": a.get("respiratory_relevance", 0),
                "raw_metadata": a.get("raw_metadata", {}),
                "is_active": True,
            }
            for a in articles
        ]
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[NewsArticle.external_id],
        set_={
            "title": stmt.excluded.title,
            "summary": stmt.excluded.summary,
            "url": stmt.excluded.url,
            "published_at": stmt.excluded.published_at,
            "latitude": stmt.excluded.latitude,
            "longitude": stmt.excluded.longitude,
            "location": stmt.excluded.location,
            "event_category": stmt.excluded.event_category,
            "severity": stmt.excluded.severity,
            "respiratory_relevance": stmt.excluded.respiratory_relevance,
            "raw_metadata": stmt.excluded.raw_metadata,
            "is_active": True,
        },
    )
    result = db.execute(stmt)
    db.commit()
    logger.info(f"upserted {len(articles)} news articles")
    return result.rowcount or len(articles)


def get_active_articles_nearby(
    db: Session,
    lat: float,
    lon: float,
    radius_km: float,
    hours: int = 72,
    limit: int = 50,
    min_relevance: int = 0,
) -> List[Dict[str, Any]]:
    """Active located articles within radius, enriched with distance_km.

    Returns dicts (not ORM rows) so callers never touch detached-object
    state: {article, distance_km}.
    """
    time_threshold = datetime.utcnow() - timedelta(hours=hours)
    rows = (
        db.execute(
            select(NewsArticle)
            .where(
                NewsArticle.is_active.is_(True),
                NewsArticle.latitude.is_not(None),
                NewsArticle.longitude.is_not(None),
                NewsArticle.published_at >= time_threshold,
                NewsArticle.respiratory_relevance >= min_relevance,
            )
            .order_by(
                NewsArticle.respiratory_relevance.desc(),
                NewsArticle.published_at.desc(),
            )
            .limit(limit * 3)  # pre-filter, then keep those actually in radius
        )
        .scalars()
        .all()
    )

    # NOTE: the SQL filter above uses the GiST index-free columns; the
    # radius check is applied in Python (haversine) which is exact for a
    # point/point distance. Indexed spatial filtering is applied in the
    # API-layer variant when a geometry column is available.
    results: List[Dict[str, Any]] = []
    for row in rows:
        distance = haversine_km(lat, lon, row.latitude, row.longitude)
        if distance > radius_km:
            continue
        results.append(
            {
                "article": row,
                "distance_km": round(distance, 2),
            }
        )
        if len(results) >= limit:
            break
    return results


def deactivate_stale_articles(
    db: Session, older_than_hours: int = _STALE_HOURS_DEFAULT
) -> int:
    """Mark articles older than N hours inactive (keeps table lean)."""
    cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
    result = db.execute(
        update(NewsArticle)
        .where(NewsArticle.published_at < cutoff, NewsArticle.is_active.is_(True))
        .values(is_active=False)
    )
    db.commit()
    count = result.rowcount or 0
    if count:
        logger.info(f"deactivated {count} stale news articles")
    return count
