"""GeoAirQuality FastAPI application with Redis caching.

Main application entry point with spatial air quality API endpoints.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge
from sqlalchemy import create_engine, text, select, func
from sqlalchemy.orm import sessionmaker, Session
from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_DWithin, ST_Point, ST_Distance

from models import (
    SpatialGrid, AirQualityReading, WeatherReading,
    AggregatedData, DataSource, PatientProfile, SymptomLog, Base
)
from cache import get_cache, cached, cache_health_status, CacheSettings
from services.safety_engine import (
    PatientContext, RiskScorer, EnvironmentalSnapshot, RiskLevel
)
from services.recommendation_engine import RecommendationEngine

# Prometheus metrics for the patient safety engine
SAFETY_ASSESSMENTS = Counter(
    "geoairquality_safety_assessments_total",
    "Total safety assessments computed"
)
SAFETY_AVG_SCORE = Gauge(
    "geoairquality_safety_avg_score",
    "Most recent safety assessment score (0-100)"
)
SAFETY_CACHE_HITS = Counter(
    "geoairquality_safety_cache_hits_total",
    "Safety assessments served from cache"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = "postgresql://geoair_user:geoair_pass@postgres:5432/geoairquality"

# Create sync engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Create session factory
SessionLocal = sessionmaker(
    bind=engine, 
    autocommit=False,
    autoflush=False
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting GeoAirQuality API")
    
    # Initialize database
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
    
    # Initialize cache
    try:
        cache = await get_cache()
        logger.info("Cache initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GeoAirQuality API")
    
    # Close cache connections
    try:
        cache = await get_cache()
        await cache.close()
    except Exception as e:
        logger.error(f"Error closing cache: {e}")
    
    # Close database connections
    engine.dispose()


# Create FastAPI app
app = FastAPI(
    title="GeoAirQuality API",
    description="Real-time air quality monitoring with spatial analytics",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Pydantic models
class LocationModel(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class AirQualityResponse(BaseModel):
    id: int
    location: Dict[str, float]
    timestamp: datetime
    pm25: Optional[float]
    pm10: Optional[float]
    no2: Optional[float]
    o3: Optional[float]
    co: Optional[float]
    so2: Optional[float]
    aqi: Optional[int]
    grid_id: Optional[str]


class WeatherResponse(BaseModel):
    id: int
    location: Dict[str, float]
    timestamp: datetime
    temperature: Optional[float]
    humidity: Optional[float]
    pressure: Optional[float]
    wind_speed: Optional[float]
    wind_direction: Optional[float]
    precipitation: Optional[float]
    grid_id: Optional[str]


class AggregatedResponse(BaseModel):
    grid_id: str
    time_bucket: datetime
    aggregation_level: str
    avg_pm25: Optional[float]
    max_pm25: Optional[float]
    min_pm25: Optional[float]
    avg_aqi: Optional[int]
    max_aqi: Optional[int]
    reading_count: int
    avg_temperature: Optional[float]
    avg_humidity: Optional[float]


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    database: Dict[str, Any]
    cache: Dict[str, Any]
    version: str


class SafetyAssessmentResponse(BaseModel):
    """Personalized safety assessment for a respiratory patient."""
    user_id: str
    safety_score: int = Field(..., ge=0, le=100)
    risk_level: str
    summary: str
    component_scores: Dict[str, float]
    contributions: Dict[str, int]
    contributing_factors: List[Dict[str, Any]]
    current_conditions: Dict[str, Any]
    recommendations: List[Dict[str, str]]
    nearby_events: List[Dict[str, Any]] = []
    forecast_window: Optional[Dict[str, Any]] = None
    route_risk: Optional[Dict[str, Any]] = None
    generated_at: datetime


# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Comprehensive health check."""
    try:
        # Test database connection
        result = db.execute(text("SELECT 1"))
        db_healthy = result.scalar() == 1
        
        # Test PostGIS extension
        postgis_result = db.execute(text("SELECT PostGIS_Version()"))
        postgis_version = postgis_result.scalar()
        
        # Get cache health
        cache_status = await cache_health_status()
        
        return HealthResponse(
            status="healthy" if db_healthy and cache_status.get("status") == "healthy" else "degraded",
            timestamp=datetime.utcnow(),
            database={
                "healthy": db_healthy,
                "postgis_version": postgis_version
            },
            cache=cache_status,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes."""
    return {"status": "ready", "timestamp": datetime.utcnow()}


# Air quality endpoints
@app.get("/api/v1/air-quality/readings", response_model=List[AirQualityResponse])
@cached(prefix="air_quality_readings", ttl=300)  # 5 minutes cache
async def get_air_quality_readings(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    radius_km: float = Query(10, ge=0.1, le=100, description="Search radius in kilometers"),
    hours: int = Query(24, ge=1, le=168, description="Hours of data to retrieve"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    db: Session = Depends(get_db)
):
    """Get air quality readings within radius of a location."""
    try:
        # Calculate time threshold
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        # Create point geometry
        point = ST_Point(lon, lat)
        
        # Query air quality readings
        query = select(AirQualityReading).where(
            ST_DWithin(
                AirQualityReading.location,
                point,
                radius_km * 1000  # Convert km to meters
            ),
            AirQualityReading.timestamp >= time_threshold
        ).order_by(
            AirQualityReading.timestamp.desc()
        ).limit(limit)
        
        result = db.execute(query)
        readings = result.scalars().all()
        
        # Convert to response format
        response_data = []
        for reading in readings:
            # Extract coordinates from geometry
            coords_result = db.execute(
                text("SELECT ST_X(:geom) as lon, ST_Y(:geom) as lat").bindparam(
                    geom=reading.location
                )
            )
            coords = coords_result.first()
            
            response_data.append(AirQualityResponse(
                id=reading.id,
                location={"latitude": coords.lat, "longitude": coords.lon},
                timestamp=reading.timestamp,
                pm25=reading.pm25,
                pm10=reading.pm10,
                no2=reading.no2,
                o3=reading.o3,
                co=reading.co,
                so2=reading.so2,
                aqi=reading.aqi,
                grid_id=reading.grid_id
            ))
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error fetching air quality readings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/air-quality/grid/{grid_id}", response_model=List[AirQualityResponse])
@cached(prefix="grid_air_quality", ttl=600)  # 10 minutes cache
async def get_grid_air_quality(
    grid_id: str = Path(..., description="Grid cell identifier"),
    hours: int = Query(24, ge=1, le=168, description="Hours of data to retrieve"),
    db: Session = Depends(get_db)
):
    """Get air quality readings for a specific grid cell."""
    try:
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        query = select(AirQualityReading).where(
            AirQualityReading.grid_id == grid_id,
            AirQualityReading.timestamp >= time_threshold
        ).order_by(AirQualityReading.timestamp.desc())
        
        result = db.execute(query)
        readings = result.scalars().all()
        
        response_data = []
        for reading in readings:
            coords_result = db.execute(
                text("SELECT ST_X(:geom) as lon, ST_Y(:geom) as lat").bindparam(
                    geom=reading.location
                )
            )
            coords = coords_result.first()
            
            response_data.append(AirQualityResponse(
                id=reading.id,
                location={"latitude": coords.lat, "longitude": coords.lon},
                timestamp=reading.timestamp,
                pm25=reading.pm25,
                pm10=reading.pm10,
                no2=reading.no2,
                o3=reading.o3,
                co=reading.co,
                so2=reading.so2,
                aqi=reading.aqi,
                grid_id=reading.grid_id
            ))
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error fetching grid air quality: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Weather endpoints
@app.get("/api/v1/weather/readings", response_model=List[WeatherResponse])
@cached(prefix="weather_readings", ttl=300)  # 5 minutes cache
async def get_weather_readings(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    radius_km: float = Query(10, ge=0.1, le=100, description="Search radius in kilometers"),
    hours: int = Query(24, ge=1, le=168, description="Hours of data to retrieve"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    db: Session = Depends(get_db)
):
    """Get weather readings within radius of a location."""
    try:
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        point = ST_Point(lon, lat)
        
        query = select(WeatherReading).where(
            ST_DWithin(
                WeatherReading.location,
                point,
                radius_km * 1000
            ),
            WeatherReading.timestamp >= time_threshold
        ).order_by(
            WeatherReading.timestamp.desc()
        ).limit(limit)
        
        result = db.execute(query)
        readings = result.scalars().all()
        
        response_data = []
        for reading in readings:
            coords_result = db.execute(
                text("SELECT ST_X(:geom) as lon, ST_Y(:geom) as lat").bindparam(
                    geom=reading.location
                )
            )
            coords = coords_result.first()
            
            response_data.append(WeatherResponse(
                id=reading.id,
                location={"latitude": coords.lat, "longitude": coords.lon},
                timestamp=reading.timestamp,
                temperature=reading.temperature,
                humidity=reading.humidity,
                pressure=reading.pressure,
                wind_speed=reading.wind_speed,
                wind_direction=reading.wind_direction,
                precipitation=reading.precipitation,
                grid_id=reading.grid_id
            ))
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error fetching weather readings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Aggregated data endpoints
@app.get("/api/v1/aggregated/grid/{grid_id}", response_model=List[AggregatedResponse])
@cached(prefix="aggregated_grid", ttl=900)  # 15 minutes cache
async def get_aggregated_data(
    grid_id: str = Path(..., description="Grid cell identifier"),
    level: str = Query("hourly", regex="^(hourly|daily|weekly)$", description="Aggregation level"),
    days: int = Query(7, ge=1, le=30, description="Days of data to retrieve"),
    db: Session = Depends(get_db)
):
    """Get aggregated data for a grid cell."""
    try:
        time_threshold = datetime.utcnow() - timedelta(days=days)
        
        query = select(AggregatedData).where(
            AggregatedData.grid_id == grid_id,
            AggregatedData.aggregation_level == level,
            AggregatedData.time_bucket >= time_threshold
        ).order_by(AggregatedData.time_bucket.desc())
        
        result = db.execute(query)
        aggregated = result.scalars().all()
        
        return [
            AggregatedResponse(
                grid_id=agg.grid_id,
                time_bucket=agg.time_bucket,
                aggregation_level=agg.aggregation_level,
                avg_pm25=agg.avg_pm25,
                max_pm25=agg.max_pm25,
                min_pm25=agg.min_pm25,
                avg_aqi=agg.avg_aqi,
                max_aqi=agg.max_aqi,
                reading_count=agg.reading_count,
                avg_temperature=agg.avg_temperature,
                avg_humidity=agg.avg_humidity
            )
            for agg in aggregated
        ]
        
    except Exception as e:
        logger.error(f"Error fetching aggregated data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ----------------------------------------------------------------------
# Personalized Safety Assessment (Patient Safety Engine)
# ----------------------------------------------------------------------

def _condition_label(context: PatientContext) -> str:
    """Human-readable label for the patient's condition(s)."""
    labels = {
        "asthma": "asthma",
        "copd": "COPD",
        "bronchitis": "bronchitis",
        "allergy": "allergies",
    }
    names = [labels.get(c.lower(), c) for c in context.conditions]
    if len(names) == 1:
        return names[0]
    return "your conditions"


def _build_environmental_snapshot(
    db: Session,
    lat: float,
    lon: float,
    radius_km: float,
    hours: int = 24,
    limit: int = 5,
) -> EnvironmentalSnapshot:
    """Build a snapshot from the most recent weather + AQ readings."""
    time_threshold = datetime.utcnow() - timedelta(hours=hours)
    point = ST_Point(lon, lat)
    radius_m = radius_km * 1000
    snapshot = EnvironmentalSnapshot()

    weather_rows = db.execute(
        select(WeatherReading).where(
            ST_DWithin(WeatherReading.location, point, radius_m),
            WeatherReading.timestamp >= time_threshold,
        ).order_by(WeatherReading.timestamp.desc()).limit(limit)
    ).scalars().all()

    if weather_rows:
        w = weather_rows[0]
        snapshot.pm25 = w.pm2_5
        snapshot.pm10 = w.pm10
        snapshot.o3 = w.ozone
        snapshot.no2 = w.nitrogen_dioxide
        snapshot.aqi = w.aqi
        snapshot.temperature = w.temperature_celsius
        snapshot.humidity = w.humidity
        snapshot.wind_speed = w.wind_kph
        snapshot.wind_direction = w.wind_direction
        snapshot.source_timestamp = w.timestamp
        snapshot.reading_count = len(weather_rows)
        if w.grid_cell_id is not None:
            snapshot.grid_id = str(w.grid_cell_id)

    # Augment / fall back using air quality readings
    aq_rows = db.execute(
        select(AirQualityReading).where(
            ST_DWithin(AirQualityReading.location, point, radius_m),
            AirQualityReading.timestamp >= time_threshold,
        ).order_by(AirQualityReading.timestamp.desc()).limit(limit)
    ).scalars().all()

    if aq_rows:
        a = aq_rows[0]
        snapshot.aqi = snapshot.aqi if snapshot.aqi is not None else a.aqi
        snapshot.temperature = (
            snapshot.temperature if snapshot.temperature is not None else a.temperature
        )
        snapshot.humidity = (
            snapshot.humidity if snapshot.humidity is not None else a.relative_humidity
        )
        snapshot.no2 = snapshot.no2 if snapshot.no2 is not None else a.no2_gt
        snapshot.o3 = snapshot.o3 if snapshot.o3 is not None else a.pt08_s5_o3
        if snapshot.reading_count == 0:
            snapshot.source_timestamp = a.timestamp
            snapshot.reading_count = len(aq_rows)

    return snapshot


def _compute_route_risk(
    db: Session,
    context: PatientContext,
    scorer: RiskScorer,
    lat: float,
    lon: float,
    dest_lat: float,
    dest_lon: float,
) -> Dict[str, Any]:
    """Sample environmental conditions along the origin->destination corridor."""
    mid_lat = (lat + dest_lat) / 2.0
    mid_lon = (lon + dest_lon) / 2.0
    radius_km = max(context.alert_radius_km, 5.0)

    samples = [
        ("origin", lat, lon),
        ("midpoint", mid_lat, mid_lon),
        ("destination", dest_lat, dest_lon),
    ]
    segments = []
    scores = []
    for label, slat, slon in samples:
        snap = _build_environmental_snapshot(db, slat, slon, radius_km, hours=24, limit=3)
        assessment = scorer.assess(snap)
        scores.append(assessment["safety_score"])
        segments.append({
            "point": label,
            "safety_score": assessment["safety_score"],
            "aqi": snap.aqi,
            "pm25": snap.pm25,
        })

    route_score = int(round(min(scores)))  # worst segment dominates
    worst_idx = scores.index(min(scores))
    return {
        "route_risk_score": route_score,
        "worst_segment": segments[worst_idx]["point"],
        "segments": segments,
    }


def _build_summary(assessment: Dict[str, Any], context: PatientContext) -> str:
    """One-line plain-language summary of the assessment."""
    label = _condition_label(context)
    level = assessment["risk_level"]
    summaries = {
        RiskLevel.LOW.value: (
            f"Low risk for {label}. It is safe to be outdoors with normal precautions."
        ),
        RiskLevel.MODERATE.value: (
            f"Moderate risk for {label}. Be cautious and limit prolonged exposure."
        ),
        RiskLevel.HIGH.value: (
            f"High risk for {label}. Avoid outdoor activity where possible."
        ),
        RiskLevel.VERY_HIGH.value: (
            f"Very high risk for {label}. Stay indoors and avoid outdoor activity."
        ),
    }
    return summaries.get(level, f"Risk level: {level} for {label}.")


@app.get("/api/v1/patients/{user_id}/safety-assessment", response_model=SafetyAssessmentResponse)
async def get_patient_safety_assessment(
    user_id: str = Path(..., description="Patient identifier"),
    lat: float = Query(..., ge=-90, le=90, description="Current latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Current longitude"),
    dest_lat: Optional[float] = Query(
        None, ge=-90, le=90, description="Destination latitude (optional)"
    ),
    dest_lon: Optional[float] = Query(
        None, ge=-180, le=180, description="Destination longitude (optional)"
    ),
    db: Session = Depends(get_db)
):
    """Personalized safety assessment: is it safe for THIS patient right now?

    Combines local AQI/pollutants, weather, and (in later phases) news
    events and personal history into a single 0-100 safety score with
    actionable recommendations. The patient profile auto-registers on
    first use.
    """
    try:
        # Explicit caching on (user_id, lat, lon, dest) — avoids serializing
        # the DB session into the cache key.
        cache = await get_cache()
        cache_key = (
            f"safety_assessment:{user_id}:{lat:.4f}:{lon:.4f}:"
            f"{dest_lat if dest_lat is not None else 0.0:.4f}:"
            f"{dest_lon if dest_lon is not None else 0.0:.4f}"
        )
        cached_result = await cache.get(cache_key)
        if cached_result is not None:
            SAFETY_ASSESSMENTS.inc()
            SAFETY_CACHE_HITS.inc()
            return SafetyAssessmentResponse(**cached_result)

        # Load or auto-register the patient profile
        profile = db.execute(
            select(PatientProfile).where(PatientProfile.user_id == user_id)
        ).scalar_one_or_none()
        if profile is None:
            profile = PatientProfile(
                user_id=user_id,
                home_lat=lat,
                home_lon=lon,
                home_location=f"SRID=4326;POINT({lon} {lat})",
            )
            db.add(profile)
            db.commit()
            db.refresh(profile)
            logger.info(f"Auto-registered patient profile for user_id={user_id}")

        context = PatientContext.from_profile(profile)
        radius_km = context.alert_radius_km

        snapshot = _build_environmental_snapshot(db, lat, lon, radius_km)

        # Phase 1: news + history inputs are empty; wired in Phases 2 & 3
        news_events: List[Dict[str, Any]] = []
        symptoms: List[Dict[str, Any]] = []

        scorer = RiskScorer(context)
        assessment = scorer.assess(snapshot, news_events, symptoms)

        route_risk = None
        if dest_lat is not None and dest_lon is not None:
            route_risk = _compute_route_risk(
                db, context, scorer, lat, lon, dest_lat, dest_lon
            )

        engine = RecommendationEngine(context)
        recommendations = engine.generate(
            assessment, snapshot, news_events, route_risk
        )

        response = SafetyAssessmentResponse(
            user_id=user_id,
            safety_score=assessment["safety_score"],
            risk_level=assessment["risk_level"],
            summary=_build_summary(assessment, context),
            component_scores=assessment["component_scores"],
            contributions=assessment["contributions"],
            contributing_factors=assessment["contributing_factors"],
            current_conditions={
                "aqi": snapshot.aqi,
                "pm25": snapshot.pm25,
                "pm10": snapshot.pm10,
                "o3": snapshot.o3,
                "no2": snapshot.no2,
                "temperature": snapshot.temperature,
                "humidity": snapshot.humidity,
                "wind_speed": snapshot.wind_speed,
                "wind_direction": snapshot.wind_direction,
                "reading_count": snapshot.reading_count,
                "source_timestamp": snapshot.source_timestamp,
                "grid_id": snapshot.grid_id,
            },
            recommendations=recommendations,
            nearby_events=news_events,
            route_risk=route_risk,
            generated_at=datetime.utcnow(),
        )

        # Version-agnostic serialization (pydantic v1: .json(); v2: model_dump_json)
        if hasattr(response, "model_dump_json"):
            serialized = json.loads(response.model_dump_json())
        else:
            serialized = json.loads(response.json())
        await cache.set(cache_key, serialized, ttl=300)
        SAFETY_ASSESSMENTS.inc()
        SAFETY_AVG_SCORE.set(response.safety_score)
        return response

    except Exception as e:
        logger.error(f"Error computing safety assessment: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    # This would integrate with prometheus_client
    # For now, return basic metrics
    cache_status = await cache_health_status()
    
    metrics_text = f"""
# HELP geoairquality_cache_hits_total Total cache hits
# TYPE geoairquality_cache_hits_total counter
geoairquality_cache_hits_total {cache_status.get('cache_metrics', {}).get('hits', 0)}

# HELP geoairquality_cache_misses_total Total cache misses
# TYPE geoairquality_cache_misses_total counter
geoairquality_cache_misses_total {cache_status.get('cache_metrics', {}).get('misses', 0)}

# HELP geoairquality_cache_hit_rate Cache hit rate
# TYPE geoairquality_cache_hit_rate gauge
geoairquality_cache_hit_rate {cache_status.get('cache_metrics', {}).get('hit_rate', 0)}

# HELP geoairquality_api_status API health status (1=healthy, 0=unhealthy)
# TYPE geoairquality_api_status gauge
geoairquality_api_status {1 if cache_status.get('status') == 'healthy' else 0}

# HELP geoairquality_safety_assessments_total Total safety assessments computed
# TYPE geoairquality_safety_assessments_total counter
geoairquality_safety_assessments_total {SAFETY_ASSESSMENTS._value.get()}

# HELP geoairquality_safety_cache_hits_total Safety assessments served from cache
# TYPE geoairquality_safety_cache_hits_total counter
geoairquality_safety_cache_hits_total {SAFETY_CACHE_HITS._value.get()}

# HELP geoairquality_safety_avg_score Most recent safety score (0-100)
# TYPE geoairquality_safety_avg_score gauge
geoairquality_safety_avg_score {SAFETY_AVG_SCORE._value.get()}
"""
    
    return JSONResponse(
        content=metrics_text,
        media_type="text/plain"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
