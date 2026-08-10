"""Background news fetch scheduler.

Runs an infinite loop registered in the FastAPI lifespan. Every
NEWS_FETCH_INTERVAL_MINUTES (default 30) it:

  1. acquires a Redis lock (avoids duplicate fetches across replicas)
  2. fetches raw items from all sources (AirNow + NewsAPI)
  3. enriches them (location / category / severity / relevance)
  4. upserts into PostGIS (idempotent on external_id)
  5. deactivates articles older than NEWS_RETENTION_DAYS

External API keys are optional: without them fetch is a no-op and the
loop simply keeps the retention sweep running. All failures are logged
and swallowed so the API never breaks because news is down.
"""

import asyncio
import logging
import os
from typing import Optional

from prometheus_client import Counter, Gauge

from cache import get_cache
from .news_fetcher import NewsAggregator
from .news_processor import process_article
from .news_store import deactivate_stale_articles, upsert_articles

logger = logging.getLogger(__name__)

# Prometheus metrics
NEWS_FETCHES = Counter(
    "geoairquality_news_fetches_total", "Total news fetch cycles completed"
)
NEWS_FETCH_ERRORS = Counter(
    "geoairquality_news_fetch_errors_total", "News fetch cycles that failed"
)
NEWS_ARTICLES_ACTIVE = Gauge(
    "geoairquality_news_articles_active", "Active news articles currently stored"
)
NEWS_ARTICLES_PER_SOURCE = Gauge(
    "geoairquality_news_articles_per_source",
    "Active news articles per source",
    ["source"],
)
NEWS_LAST_FETCH = Gauge(
    "geoairquality_news_last_fetch_timestamp",
    "Unix timestamp of the last successful news fetch",
)

_LOCK_KEY = "news_fetch_lock"


class NewsScheduler:
    """Periodic news ingestion loop."""

    def __init__(
        self,
        interval_minutes: Optional[int] = None,
        retention_days: Optional[int] = None,
    ):
        self.interval = interval_minutes or int(
            os.environ.get("NEWS_FETCH_INTERVAL_MINUTES", "30")
        )
        self.retention_days = retention_days or int(
            os.environ.get("NEWS_RETENTION_DAYS", "7")
        )
        self.aggregator = NewsAggregator()
        self._session_local = None  # set from main at startup

    def bind_session_factory(self, session_local) -> "NewsScheduler":
        """Inject the app's SessionLocal so the scheduler shares the pool."""
        self._session_local = session_local
        return self

    async def run_forever(self) -> None:
        logger.info(
            f"news scheduler started (interval={self.interval}m, "
            f"retention={self.retention_days}d)"
        )
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                logger.info("news scheduler stopped")
                raise
            except Exception as exc:
                NEWS_FETCH_ERRORS.inc()
                logger.error(f"news scheduler tick failed: {exc}")
            await asyncio.sleep(self.interval * 60)

    async def tick(self) -> None:
        """One fetch + store + retention cycle."""
        # Redis lock to avoid concurrent cycles across replicas
        cache = await get_cache()
        if await cache.get(_LOCK_KEY):
            logger.debug("news fetch already in progress — skipping")
            return
        await cache.set(_LOCK_KEY, "1", ttl=self.interval * 60)

        try:
            raw_items = await self.aggregator.fetch_all()
            articles = [
                process_article(
                    external_id=item.external_id,
                    source_name=item.source_name,
                    source_type=item.source_type,
                    title=item.title,
                    summary=item.summary,
                    url=item.url,
                    published_at=item.published_at,
                    source_lat=item.lat,
                    source_lon=item.lon,
                    raw_data=item.raw_data,
                )
                for item in raw_items
            ]

            if articles and self._session_local is not None:

                def _persist():
                    db = self._session_local()
                    try:
                        upsert_articles(db, articles)
                        return deactivate_stale_articles(
                            db, older_than_hours=self.retention_days * 24
                        )
                    finally:
                        db.close()

                await asyncio.to_thread(_persist)

            # Keep stored-article metrics fresh (counts + per source)
            await asyncio.to_thread(self._refresh_metrics)

            # New/updated articles can change safety scores — invalidate
            # the per-user assessment caches and the news feed cache so the
            # next request reflects the freshest events.
            await cache.delete_pattern("safety_assessment:*")
            await cache.delete_pattern("news_nearby:*")

            NEWS_FETCHES.inc()
            NEWS_LAST_FETCH.set_to_current_time()
            logger.info(
                f"news cycle complete: {len(raw_items)} raw, "
                f"{len(articles)} enriched"
            )
        finally:
            # Release the lock for the next cycle
            await cache.delete(_LOCK_KEY)

    def _refresh_metrics(self) -> None:
        """Query the DB for active-article counts and update gauges."""
        if self._session_local is None:
            return
        from sqlalchemy import func, select

        from models import NewsArticle

        db = self._session_local()
        try:
            total = db.execute(
                select(func.count())
                .select_from(NewsArticle)
                .where(NewsArticle.is_active.is_(True))
            ).scalar()
            NEWS_ARTICLES_ACTIVE.set(total or 0)
            rows = db.execute(
                select(NewsArticle.source_name, func.count())
                .where(NewsArticle.is_active.is_(True))
                .group_by(NewsArticle.source_name)
            ).all()
            for source, count in rows:
                NEWS_ARTICLES_PER_SOURCE.labels(source=source).set(count)
        finally:
            db.close()
