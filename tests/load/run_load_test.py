#!/usr/bin/env python3
"""Concurrency load test for the GeoAirQuality API.

Hammers every public endpoint with concurrent workers and reports
latency percentiles (p50/p95/p99), throughput and error rate.

Usage:
    python tests/load/run_load_test.py --workers 50 --duration 15 --base http://localhost:8000
"""

import argparse
import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import httpx

ENDPOINTS = [
    ("GET", "/health", {}),
    ("GET", "/ready", {}),
    (
        "GET",
        "/api/v1/air-quality/readings",
        {"lat": 40.7128, "lon": -74.006, "radius_km": 25, "hours": 24, "limit": 50},
    ),
    ("GET", "/api/v1/air-quality/grid/grid_1deg_40_-74", {"hours": 24}),
    (
        "GET",
        "/api/v1/weather/readings",
        {"lat": 40.7128, "lon": -74.006, "radius_km": 25, "hours": 24, "limit": 50},
    ),
    ("GET", "/api/v1/aggregated/grid/grid_1deg_40_-74", {"level": "hourly", "days": 7}),
    (
        "GET",
        "/api/v1/patients/load-user/safety-assessment",
        {"lat": 40.7128, "lon": -74.006},
    ),
    (
        "GET",
        "/api/v1/patients/load-user/safety-assessment",
        {"lat": 40.7128, "lon": -74.006, "dest_lat": 40.7580, "dest_lon": -73.9855},
    ),
    ("GET", "/api/v1/news/nearby", {"lat": 40.7128, "lon": -74.006}),
    ("GET", "/api/v1/patients/load-user/insights", {}),
    ("GET", "/metrics", {}),
]

SYMPTOM_PAYLOAD = {
    "symptom_type": "shortness_of_breath",
    "severity": 3,
    "lat": 40.7128,
    "lon": -74.006,
    "notes": "load test",
}


@dataclass
class Stats:
    name: str
    latencies: List[float] = field(default_factory=list)
    errors: int = 0
    status_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def report(self) -> str:
        if not self.latencies:
            return f"{self.name:45s} n=0 ERRORS={self.errors}"
        lat = sorted(self.latencies)

        def pct(p: float) -> float:
            idx = min(len(lat) - 1, int(len(lat) * p))
            return lat[idx] * 1000  # ms

        return (
            f"{self.name:45s} n={len(lat):6d} req/s={len(lat)/float(args.duration):7.1f} "
            f"p50={pct(0.50):7.1f}ms p95={pct(0.95):7.1f}ms p99={pct(0.99):7.1f}ms "
            f"max={lat[-1]*1000:7.1f}ms errors={self.errors}"
        )


args: argparse.Namespace


async def worker(
    client: httpx.AsyncClient,
    stats_by_name: Dict[str, Stats],
    stop: asyncio.Event,
    duration: float,
    include_writes: bool,
) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        for method, path, params in ENDPOINTS:
            if method == "POST" and not include_writes:
                continue
            name = (
                f"{method} {path.split('/api/v1')[-1] if '/api/v1' in path else path}"
            )
            stats = stats_by_name[name]
            start = time.perf_counter()
            try:
                if method == "GET":
                    resp = await client.get(f"{args.base}{path}", params=params)
                else:
                    resp = await client.post(f"{args.base}{path}", json=SYMPTOM_PAYLOAD)
                stats.latencies.append(time.perf_counter() - start)
                stats.status_counts[resp.status_code] += 1
                if resp.status_code >= 500:
                    stats.errors += 1
            except httpx.HTTPError:
                stats.latencies.append(time.perf_counter() - start)
                stats.errors += 1
        if stop.is_set():
            break


async def main() -> None:
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--duration", type=int, default=15)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument(
        "--writes", action="store_true", help="include POST symptom endpoint"
    )
    args = ap.parse_args()

    stats_by_name: Dict[str, Stats] = {}
    for method, path, _ in ENDPOINTS:
        name = f"{method} {path.split('/api/v1')[-1] if '/api/v1' in path else path}"
        stats_by_name[name] = Stats(name=name)

    limits = httpx.Limits(
        max_connections=args.workers * 2, max_keepalive_connections=args.workers
    )
    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
        stop = asyncio.Event()
        t0 = time.monotonic()
        tasks = [
            asyncio.create_task(
                worker(client, stats_by_name, stop, args.duration, args.writes)
            )
            for _ in range(args.workers)
        ]
        await asyncio.sleep(args.duration)
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - t0

    print(
        f"\nLoad test: workers={args.workers} duration={elapsed:.1f}s base={args.base}\n"
    )
    total = 0
    total_errors = 0
    for stats in stats_by_name.values():
        print(stats.report())
        total += len(stats.latencies)
        total_errors += stats.errors
    print(
        f"\nTOTAL requests={total} errors={total_errors} "
        f"error_rate={total_errors / max(total, 1) * 100:.2f}% "
        f"throughput={total / elapsed:.1f} req/s"
    )


if __name__ == "__main__":
    asyncio.run(main())
