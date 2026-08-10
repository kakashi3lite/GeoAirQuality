"""News fetching service — AirNow official alerts + NewsAPI.org media.

Lightweight, no ML. Each fetcher is async and accepts an injected
httpx.AsyncClient so tests can mock responses deterministically.

Sources:
  * AirNow API   — official EPA air quality observations; unhealthy
                   readings (AQI > 100) are converted into advisories.
                   Free (requires a free API key; skipped when unset).
  * NewsAPI.org  — general media articles matching air-quality keywords.
                   Free tier: 100 requests/day (skipped when unset).

Both sources are global queries (bbox / keyword), so a single fetch
round covers the whole coverage area — no per-city fan-out needed.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import httpx

from .news_processor import title_similarity

logger = logging.getLogger(__name__)

USER_AGENT = "GeoAirQuality/1.0 (health dashboard)"


@dataclass
class RawNewsItem:
    """Normalized article/alert before rule-based enrichment."""

    external_id: str
    source_name: str
    source_type: str  # official_alert | news_media
    title: str
    summary: str = ""
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    raw_data: Dict = field(default_factory=dict)


class AirNowFetcher:
    """EPA AirNow — derives health advisories from unhealthy readings."""

    BASE_URL = "https://airnowapi.org/aq/data/"
    # Continental US bounding box (minlon,minlat,maxlon,maxlat)
    DEFAULT_BBOX = "-125,25,-65,50"

    def __init__(
        self,
        api_key: Optional[str] = None,
        bbox: str = DEFAULT_BBOX,
        unhealthy_threshold: int = 101,
        max_items: int = 20,
    ):
        self.api_key = api_key or os.environ.get("AIRNOW_API_KEY", "")
        self.bbox = bbox
        self.threshold = unhealthy_threshold
        self.max_items = max_items

    async def fetch(self, client: httpx.AsyncClient) -> List[RawNewsItem]:
        """Fetch current PM2.5/O3 observations; emit advisories where unhealthy."""
        if not self.api_key:
            logger.info("AIRNOW_API_KEY not set — skipping AirNow fetcher")
            return []

        now = datetime.utcnow()
        params = {
            "startDate": now.strftime("%Y-%m-%dT00"),
            "endDate": now.strftime("%Y-%m-%dT23"),
            "parameters": "PM25,OZONE",
            "bbox": self.bbox,
            "format": "application/json",
            "api_key": self.api_key,
        }
        resp = await client.get(self.BASE_URL, params=params)
        resp.raise_for_status()
        rows = resp.json() or []

        items: List[RawNewsItem] = []
        for row in rows:
            try:
                aqi = int(row.get("AQI", 0))
                if aqi < self.threshold:
                    continue
                param = row.get("ParameterName", "PM2.5")
                area = row.get("ReportingArea") or row.get("SiteName") or "the area"
                lat = row.get("Latitude")
                lon = row.get("Longitude")
                category = (row.get("Category") or {}).get("Name", "Unhealthy")
                ext_id = (
                    f"airnow:{row.get('Latitude')},{row.get('Longitude')}"
                    f":{param}:{row.get('DateObserved')}:{row.get('HourObserved')}"
                )
                items.append(
                    RawNewsItem(
                        external_id=ext_id,
                        source_name="EPA AirNow",
                        source_type="official_alert",
                        title=(
                            f"Unhealthy air quality ({category}) reported near "
                            f"{area} — {param} at AQI {aqi}"
                        ),
                        summary=(
                            f"Official EPA monitoring reports {param} at AQI {aqi} "
                            f"({category}). Sensitive groups including people with "
                            f"asthma or COPD should reduce outdoor exposure."
                        ),
                        url=None,
                        published_at=now,
                        lat=float(lat) if lat is not None else None,
                        lon=float(lon) if lon is not None else None,
                        raw_data=row,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug(f"skipped malformed AirNow row: {exc}")
            if len(items) >= self.max_items:
                break
        return items


class NewsAPIFetcher:
    """NewsAPI.org — media articles on air-quality topics."""

    BASE_URL = "https://newsapi.org/v2/everything"
    QUERY = (
        "air pollution OR air quality OR wildfire smoke OR smog "
        "OR asthma alert OR health advisory"
    )

    def __init__(self, api_key: Optional[str] = None, max_items: int = 25):
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY", "")
        self.max_items = max_items

    async def fetch(self, client: httpx.AsyncClient) -> List[RawNewsItem]:
        if not self.api_key:
            logger.info("NEWSAPI_KEY not set — skipping NewsAPI fetcher")
            return []

        params = {
            "q": self.QUERY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(self.max_items, 100),
        }
        headers = {"X-Api-Key": self.api_key}
        resp = await client.get(self.BASE_URL, params=params, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "ok":
            logger.warning(f"NewsAPI returned status={payload.get('status')}")
            return []

        items: List[RawNewsItem] = []
        for art in payload.get("articles", []):
            url = art.get("url")
            if not url:
                continue
            published = None
            if art.get("publishedAt"):
                try:
                    published = datetime.fromisoformat(
                        art["publishedAt"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except ValueError:
                    published = None
            items.append(
                RawNewsItem(
                    external_id=f"newsapi:{url}",
                    source_name=(art.get("source") or {}).get("name") or "News",
                    source_type="news_media",
                    title=art.get("title") or "",
                    summary=art.get("description") or "",
                    url=url,
                    published_at=published or datetime.utcnow(),
                    lat=None,
                    lon=None,
                    raw_data=art,
                )
            )
            if len(items) >= self.max_items:
                break
        return items


class NewsAggregator:
    """Runs all fetchers in parallel, rate-limits, and de-duplicates."""

    def __init__(self, fetchers: Optional[List] = None, dedup_threshold: float = 0.85):
        self.fetchers = fetchers or [AirNowFetcher(), NewsAPIFetcher()]
        self.dedup_threshold = dedup_threshold

    async def fetch_all(self, timeout: float = 20.0) -> List[RawNewsItem]:
        """Fetch from all sources concurrently; failures are logged, not fatal."""
        async with httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": USER_AGENT}
        ) as client:
            results = await asyncio.gather(
                *[f.fetch(client) for f in self.fetchers],
                return_exceptions=True,
            )

        items: List[RawNewsItem] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"news fetcher failed: {result}")
                continue
            items.extend(result)
        logger.info(f"fetched {len(items)} raw items from {len(self.fetchers)} sources")
        return self._dedupe(items)

    def _dedupe(self, items: List[RawNewsItem]) -> List[RawNewsItem]:
        """Drop duplicate URLs and near-identical titles (keep newest)."""
        ordered = sorted(
            items,
            key=lambda it: it.published_at or datetime.min,
            reverse=True,
        )
        seen_urls = set()
        deduped: List[RawNewsItem] = []
        for item in ordered:
            if item.url and item.url in seen_urls:
                continue
            if any(
                title_similarity(item.title, other.title) > self.dedup_threshold
                for other in deduped
                if other.title
            ):
                continue
            if item.url:
                seen_urls.add(item.url)
            deduped.append(item)
        return deduped


async def fetch_once() -> List[RawNewsItem]:
    """Convenience entrypoint (also usable from CLI / scheduler)."""
    return await NewsAggregator().fetch_all()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetched = asyncio.run(fetch_once())
    for item in fetched:
        print(f"[{item.source_type:14s}] {item.title[:90]}")
