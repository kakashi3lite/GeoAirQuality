"""GeoAirQuality FastAPI application with Redis caching.

Main application entry point with spatial air quality API endpoints.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select, func
from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_DWithin, ST_Point, ST_Distance

from models import (
    SpatialGrid, AirQualityReading, WeatherReading, 
    AggregatedData, DataSource, Base
)
from cache import get_cache, cached, cache_health_status, CacheSettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = "postgresql+asyncpg://geoair_user:geoair_pass@localhost:5432/geoairquality"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting GeoAirQuality API")
    
    # Initialize database
    async with engine.begin() as conn:
        # Create tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)
    
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
    await engine.dispose()


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
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


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


# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Comprehensive health check."""
    try:
        # Test database connection
        result = await db.execute(text("SELECT 1"))
        db_healthy = result.scalar() == 1
        
        # Test PostGIS extension
        postgis_result = await db.execute(text("SELECT PostGIS_Version()"))
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
    db: AsyncSession = Depends(get_db)
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
        
        result = await db.execute(query)
        readings = result.scalars().all()
        
        # Convert to response format
        response_data = []
        for reading in readings:
            # Extract coordinates from geometry
            coords_result = await db.execute(
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
    db: AsyncSession = Depends(get_db)
):
    """Get air quality readings for a specific grid cell."""
    try:
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        query = select(AirQualityReading).where(
            AirQualityReading.grid_id == grid_id,
            AirQualityReading.timestamp >= time_threshold
        ).order_by(AirQualityReading.timestamp.desc())
        
        result = await db.execute(query)
        readings = result.scalars().all()
        
        response_data = []
        for reading in readings:
            coords_result = await db.execute(
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
    db: AsyncSession = Depends(get_db)
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
        
        result = await db.execute(query)
        readings = result.scalars().all()
        
        response_data = []
        for reading in readings:
            coords_result = await db.execute(
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
    db: AsyncSession = Depends(get_db)
):
    """Get aggregated data for a grid cell."""
    try:
        time_threshold = datetime.utcnow() - timedelta(days=days)
        
        query = select(AggregatedData).where(
            AggregatedData.grid_id == grid_id,
            AggregatedData.aggregation_level == level,
            AggregatedData.time_bucket >= time_threshold
        ).order_by(AggregatedData.time_bucket.desc())
        
        result = await db.execute(query)
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