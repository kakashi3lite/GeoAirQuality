#!/usr/bin/env python3
"""
SQLAlchemy Models for GeoAirQuality PostGIS Database
Author: Captain Aurelia "Skyforge" Stratos

Defines spatial database models with optimized GiST indexes for
high-performance geospatial queries and time-series analysis.
"""

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean, Text,
    Index, ForeignKey, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
from geoalchemy2.types import Geography
import datetime

Base = declarative_base()

class SpatialGrid(Base):
    """Spatial grid cells for aggregating sensor data.
    
    Uses a hierarchical grid system with multiple resolution levels
    for efficient spatial queries and data aggregation.
    """
    __tablename__ = 'spatial_grids'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    grid_id = Column(String(50), unique=True, nullable=False, index=True)
    resolution = Column(Float, nullable=False)  # Grid resolution in degrees
    center_lat = Column(Float, nullable=False)
    center_lon = Column(Float, nullable=False)
    
    # PostGIS geometry column with SRID 4326 (WGS84)
    geometry = Column(
        Geometry('POLYGON', srid=4326, spatial_index=True),
        nullable=False
    )
    
    # Geography column for accurate distance calculations
    geography = Column(
        Geography('POLYGON', srid=4326, spatial_index=True),
        nullable=False
    )
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    air_quality_readings = relationship("AirQualityReading", back_populates="grid_cell")
    weather_readings = relationship("WeatherReading", back_populates="grid_cell")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('resolution > 0', name='positive_resolution'),
        CheckConstraint('center_lat >= -90 AND center_lat <= 90', name='valid_latitude'),
        CheckConstraint('center_lon >= -180 AND center_lon <= 180', name='valid_longitude'),
        Index('idx_spatial_grid_resolution', 'resolution'),
        Index('idx_spatial_grid_center', 'center_lat', 'center_lon'),
        # GiST spatial indexes are automatically created for geometry/geography columns
    )

class AirQualityReading(Base):
    """Air quality sensor readings with spatial and temporal indexing.
    
    Stores measurements from various air quality sensors with
    optimized indexes for time-series and spatial queries.
    """
    __tablename__ = 'air_quality_readings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    sensor_id = Column(String(100), nullable=False, index=True)
    
    # Spatial information
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    location = Column(
        Geometry('POINT', srid=4326, spatial_index=True),
        nullable=False
    )
    
    # Grid cell reference for spatial aggregation
    grid_cell_id = Column(Integer, ForeignKey('spatial_grids.id'), index=True)
    grid_cell = relationship("SpatialGrid", back_populates="air_quality_readings")
    
    # Temporal information
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Air quality measurements
    co_gt = Column(Float)  # Carbon Monoxide (mg/m^3)
    pt08_s1_co = Column(Float)  # CO sensor response
    nmhc_gt = Column(Float)  # Non-methane hydrocarbons (microg/m^3)
    c6h6_gt = Column(Float)  # Benzene (microg/m^3)
    pt08_s2_nmhc = Column(Float)  # NMHC sensor response
    nox_gt = Column(Float)  # Nitrogen oxides (ppb)
    pt08_s3_nox = Column(Float)  # NOx sensor response
    no2_gt = Column(Float)  # Nitrogen dioxide (microg/m^3)
    pt08_s4_no2 = Column(Float)  # NO2 sensor response
    pt08_s5_o3 = Column(Float)  # Ozone sensor response
    
    # Environmental conditions
    temperature = Column(Float)  # Temperature (Celsius)
    relative_humidity = Column(Float)  # Relative humidity (%)
    absolute_humidity = Column(Float)  # Absolute humidity
    
    # Computed air quality index
    aqi = Column(Float, index=True)  # Air Quality Index
    aqi_category = Column(String(20))  # Good, Moderate, Unhealthy, etc.
    
    # Data quality flags
    is_validated = Column(Boolean, default=False, index=True)
    quality_score = Column(Float)  # Data quality score (0-1)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('latitude >= -90 AND latitude <= 90', name='valid_latitude'),
        CheckConstraint('longitude >= -180 AND longitude <= 180', name='valid_longitude'),
        CheckConstraint('relative_humidity >= 0 AND relative_humidity <= 100', name='valid_humidity'),
        CheckConstraint('aqi >= 0', name='positive_aqi'),
        CheckConstraint('quality_score >= 0 AND quality_score <= 1', name='valid_quality_score'),
        
        # Composite indexes for common query patterns
        Index('idx_aq_sensor_time', 'sensor_id', 'timestamp'),
        Index('idx_aq_time_aqi', 'timestamp', 'aqi'),
        Index('idx_aq_location_time', 'latitude', 'longitude', 'timestamp'),
        Index('idx_aq_grid_time', 'grid_cell_id', 'timestamp'),
        Index('idx_aq_validated_time', 'is_validated', 'timestamp'),
        
        # Partial index for high-quality recent data
        Index('idx_aq_recent_quality', 'timestamp', 'aqi',
              postgresql_where='is_validated = true AND timestamp > NOW() - INTERVAL \'7 days\''),
    )

class WeatherReading(Base):
    """Weather station readings with spatial correlation to air quality.
    
    Stores meteorological data that influences air quality patterns
    with optimized spatial and temporal indexing.
    """
    __tablename__ = 'weather_readings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    station_id = Column(String(100), nullable=False, index=True)
    
    # Location information
    country = Column(String(100), nullable=False, index=True)
    location_name = Column(String(200), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    location = Column(
        Geometry('POINT', srid=4326, spatial_index=True),
        nullable=False
    )
    
    # Grid cell reference
    grid_cell_id = Column(Integer, ForeignKey('spatial_grids.id'), index=True)
    grid_cell = relationship("SpatialGrid", back_populates="weather_readings")
    
    # Temporal information
    timestamp = Column(DateTime, nullable=False, index=True)
    last_updated = Column(DateTime, nullable=False)
    
    # Weather measurements
    temperature_celsius = Column(Float)
    temperature_fahrenheit = Column(Float)
    humidity = Column(Float)  # Relative humidity (%)
    pressure_mb = Column(Float)  # Atmospheric pressure (millibars)
    pressure_in = Column(Float)  # Atmospheric pressure (inches)
    wind_kph = Column(Float)  # Wind speed (km/h)
    wind_mph = Column(Float)  # Wind speed (mph)
    wind_degree = Column(Float)  # Wind direction (degrees)
    wind_direction = Column(String(10))  # Wind direction (N, NE, etc.)
    
    # Air quality measurements from weather stations
    pm2_5 = Column(Float)  # PM2.5 (μg/m³)
    pm10 = Column(Float)  # PM10 (μg/m³)
    carbon_monoxide = Column(Float)  # CO (μg/m³)
    ozone = Column(Float)  # O3 (μg/m³)
    nitrogen_dioxide = Column(Float)  # NO2 (μg/m³)
    sulphur_dioxide = Column(Float)  # SO2 (μg/m³)
    
    # Computed values
    aqi = Column(Float, index=True)
    aqi_category = Column(String(20))
    
    # Data quality
    is_validated = Column(Boolean, default=False, index=True)
    quality_score = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('latitude >= -90 AND latitude <= 90', name='valid_latitude'),
        CheckConstraint('longitude >= -180 AND longitude <= 180', name='valid_longitude'),
        CheckConstraint('humidity >= 0 AND humidity <= 100', name='valid_humidity'),
        CheckConstraint('pressure_mb > 0', name='positive_pressure'),
        CheckConstraint('wind_kph >= 0', name='positive_wind_speed'),
        CheckConstraint('wind_degree >= 0 AND wind_degree < 360', name='valid_wind_direction'),
        
        # Composite indexes for weather queries
        Index('idx_weather_station_time', 'station_id', 'timestamp'),
        Index('idx_weather_country_time', 'country', 'timestamp'),
        Index('idx_weather_location_time', 'latitude', 'longitude', 'timestamp'),
        Index('idx_weather_grid_time', 'grid_cell_id', 'timestamp'),
        
        # Index for air quality correlation queries
        Index('idx_weather_aq_time', 'timestamp', 'pm2_5', 'pm10'),
    )

class AggregatedData(Base):
    """Pre-aggregated data for fast API responses.
    
    Stores hourly, daily, and monthly aggregations of air quality
    and weather data by spatial grid for optimized query performance.
    """
    __tablename__ = 'aggregated_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Spatial reference
    grid_cell_id = Column(Integer, ForeignKey('spatial_grids.id'), nullable=False, index=True)
    grid_cell = relationship("SpatialGrid")
    
    # Temporal aggregation
    aggregation_level = Column(String(10), nullable=False)  # 'hour', 'day', 'month'
    time_bucket = Column(DateTime, nullable=False, index=True)
    
    # Aggregated air quality metrics
    avg_aqi = Column(Float)
    max_aqi = Column(Float)
    min_aqi = Column(Float)
    avg_pm2_5 = Column(Float)
    avg_pm10 = Column(Float)
    avg_co = Column(Float)
    avg_no2 = Column(Float)
    avg_o3 = Column(Float)
    
    # Aggregated weather metrics
    avg_temperature = Column(Float)
    avg_humidity = Column(Float)
    avg_pressure = Column(Float)
    avg_wind_speed = Column(Float)
    
    # Statistical measures
    data_points_count = Column(Integer, nullable=False)
    quality_score = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('grid_cell_id', 'aggregation_level', 'time_bucket',
                        name='unique_aggregation'),
        CheckConstraint('data_points_count > 0', name='positive_data_points'),
        CheckConstraint('aggregation_level IN (\'hour\', \'day\', \'month\')',
                       name='valid_aggregation_level'),
        
        # Indexes for time-series queries
        Index('idx_agg_level_time', 'aggregation_level', 'time_bucket'),
        Index('idx_agg_grid_level_time', 'grid_cell_id', 'aggregation_level', 'time_bucket'),
    )

class DataSource(Base):
    """Metadata about data sources and ingestion status.
    
    Tracks data lineage, ingestion status, and quality metrics
    for monitoring and debugging the data pipeline.
    """
    __tablename__ = 'data_sources'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_name = Column(String(100), nullable=False, unique=True)
    source_type = Column(String(50), nullable=False)  # 'sensor', 'api', 'file', 'stream'
    description = Column(Text)
    
    # Connection details (encrypted)
    connection_string = Column(Text)
    api_endpoint = Column(String(500))
    
    # Status tracking
    is_active = Column(Boolean, default=True, index=True)
    last_ingestion = Column(DateTime, index=True)
    next_scheduled_ingestion = Column(DateTime, index=True)
    ingestion_frequency = Column(String(50))  # 'realtime', 'hourly', 'daily'
    
    # Quality metrics
    total_records_ingested = Column(Integer, default=0)
    failed_ingestions = Column(Integer, default=0)
    average_quality_score = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    __table_args__ = (
        CheckConstraint('total_records_ingested >= 0', name='positive_records'),
        CheckConstraint('failed_ingestions >= 0', name='positive_failures'),
        Index('idx_source_active_schedule', 'is_active', 'next_scheduled_ingestion'),
    )

class PatientProfile(Base):
    """Patient health profile for personalized safety assessments.
    
    Stores respiratory conditions, personalized pollutant thresholds,
    and home location. Used by the safety engine to calibrate risk
    scoring and recommendations to each patient's specific condition.
    """
    __tablename__ = 'patient_profiles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, unique=True, index=True)

    # Health conditions: asthma, copd, bronchitis, allergy
    conditions = Column(
        postgresql.ARRAY(String(50)),
        nullable=False,
        default=['asthma']
    )

    # Personalized pollutant thresholds
    aqi_threshold = Column(Integer, nullable=False, default=100)
    pm25_threshold = Column(Float, nullable=False, default=35.4)
    o3_threshold = Column(Float, nullable=False, default=70.0)
    no2_threshold = Column(Float, nullable=False, default=100.0)
    alert_radius_km = Column(Float, nullable=False, default=25.0)

    # Home location
    home_lat = Column(Float, nullable=False, default=40.7128)
    home_lon = Column(Float, nullable=False, default=-74.0060)
    home_location = Column(
        Geometry('POINT', srid=4326, spatial_index=True),
        nullable=True
    )

    # Reserved for future push/email notification delivery
    notification_preferences = Column(postgresql.JSONB, default=dict)

    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    symptoms = relationship(
        "SymptomLog",
        back_populates="patient",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint('aqi_threshold >= 0', name='positive_aqi_threshold'),
        CheckConstraint('pm25_threshold >= 0', name='positive_pm25_threshold'),
        CheckConstraint('o3_threshold >= 0', name='positive_o3_threshold'),
        CheckConstraint('no2_threshold >= 0', name='positive_no2_threshold'),
        CheckConstraint('alert_radius_km > 0 AND alert_radius_km <= 200',
                       name='valid_alert_radius'),
        CheckConstraint('home_lat >= -90 AND home_lat <= 90', name='valid_home_lat'),
        CheckConstraint('home_lon >= -180 AND home_lon <= 180', name='valid_home_lon'),
        Index('idx_patient_conditions', 'conditions'),
    )


class SymptomLog(Base):
    """Patient symptom log for trigger correlation and personalization.
    
    Stores symptom events with a snapshot of environmental conditions
    at the time of logging. Enables the safety engine to learn personal
    triggers (e.g., 'symptoms consistently appear when PM2.5 > 40').
    """
    __tablename__ = 'symptom_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(
        Integer,
        ForeignKey('patient_profiles.id'),
        nullable=False,
        index=True
    )
    patient = relationship("PatientProfile", back_populates="symptoms")

    symptom_type = Column(String(50), nullable=False)
    severity = Column(Integer, nullable=False)  # 1-5 scale

    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    location = Column(
        Geometry('POINT', srid=4326, spatial_index=True),
        nullable=True
    )

    # Snapshot of conditions at the time of symptom (AQI, temp, humidity...)
    weather_snapshot = Column(postgresql.JSONB, default=dict)

    logged_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint('severity >= 1 AND severity <= 5', name='valid_severity'),
        CheckConstraint('lat >= -90 AND lat <= 90', name='valid_lat'),
        CheckConstraint('lon >= -180 AND lon <= 180', name='valid_lon'),
        Index('idx_symptom_patient_time', 'patient_id', 'logged_at'),
        Index('idx_symptom_type_time', 'symptom_type', 'logged_at'),
    )


class NewsArticle(Base):
    """Curated air-quality news events and official alerts.

    Stores official alerts (EPA AirNow) and news-media articles that
    affect respiratory patients, enriched and geo-tagged to the spatial
    grid. Feeds the safety engine's news component (severity ×
    respiratory relevance × proximity) and the /news/nearby endpoint.
    """
    __tablename__ = 'news_articles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(200), nullable=False, unique=True)
    source_name = Column(String(100), nullable=False)
    # official_alert | news_media
    source_type = Column(String(20), nullable=False)
    title = Column(String(500), nullable=False)
    summary = Column(Text)
    url = Column(String(1000))

    published_at = Column(DateTime, nullable=False, index=True)
    fetched_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    # Geo-tagged location (nullable if it could not be resolved)
    latitude = Column(Float)
    longitude = Column(Float)
    location = Column(
        Geometry('POINT', srid=4326, spatial_index=True),
        nullable=True
    )
    grid_cell_id = Column(Integer, ForeignKey('spatial_grids.id'), index=True)

    # Rule-based enrichment
    event_category = Column(String(30), nullable=False, default='general')
    severity = Column(Integer, nullable=False, default=0)       # 0-100
    respiratory_relevance = Column(Integer, nullable=False, default=0)  # 0-100
    is_active = Column(Boolean, default=True, index=True)
    raw_metadata = Column(postgresql.JSONB, default=dict)

    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint('severity >= 0 AND severity <= 100', name='valid_severity'),
        CheckConstraint(
            'respiratory_relevance >= 0 AND respiratory_relevance <= 100',
            name='valid_respiratory_relevance'
        ),
        CheckConstraint(
            'latitude IS NULL OR (latitude >= -90 AND latitude <= 90)',
            name='valid_latitude'
        ),
        CheckConstraint(
            'longitude IS NULL OR (longitude >= -180 AND longitude <= 180)',
            name='valid_longitude'
        ),
        Index('idx_news_published', 'published_at'),
        Index('idx_news_active_published', 'is_active', 'published_at'),
        Index('idx_news_grid_published', 'grid_cell_id', 'published_at'),
        Index('idx_news_metadata_gin', 'raw_metadata', postgresql_using='gin'),
    )


# Create database functions for common operations
class DatabaseFunctions:
    """SQL functions for common geospatial and time-series operations."""
    
    @staticmethod
    def create_functions(engine):
        """Create custom PostgreSQL functions for optimized queries."""
        
        # Function to calculate distance between points
        distance_function = """
        CREATE OR REPLACE FUNCTION calculate_distance(
            lat1 FLOAT, lon1 FLOAT, lat2 FLOAT, lon2 FLOAT
        ) RETURNS FLOAT AS $$
        BEGIN
            RETURN ST_Distance(
                ST_GeogFromText('POINT(' || lon1 || ' ' || lat1 || ')'),
                ST_GeogFromText('POINT(' || lon2 || ' ' || lat2 || ')')
            );
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
        """
        
        # Function to find nearest grid cell
        nearest_grid_function = """
        CREATE OR REPLACE FUNCTION find_nearest_grid(
            input_lat FLOAT, input_lon FLOAT, resolution FLOAT
        ) RETURNS INTEGER AS $$
        DECLARE
            grid_id INTEGER;
        BEGIN
            SELECT id INTO grid_id
            FROM spatial_grids
            WHERE resolution = $3
            ORDER BY ST_Distance(
                geography,
                ST_GeogFromText('POINT(' || input_lon || ' ' || input_lat || ')')
            )
            LIMIT 1;
            
            RETURN grid_id;
        END;
        $$ LANGUAGE plpgsql STABLE;
        """
        
        # Function to calculate AQI from pollutant concentrations
        aqi_calculation_function = """
        CREATE OR REPLACE FUNCTION calculate_aqi(
            pm25 FLOAT, pm10 FLOAT, co FLOAT, no2 FLOAT, o3 FLOAT
        ) RETURNS FLOAT AS $$
        DECLARE
            aqi_pm25 FLOAT := 0;
            aqi_pm10 FLOAT := 0;
            aqi_co FLOAT := 0;
            aqi_no2 FLOAT := 0;
            aqi_o3 FLOAT := 0;
            max_aqi FLOAT := 0;
        BEGIN
            -- PM2.5 AQI calculation (simplified)
            IF pm25 IS NOT NULL THEN
                aqi_pm25 := (pm25 / 35.4) * 100;
            END IF;
            
            -- PM10 AQI calculation
            IF pm10 IS NOT NULL THEN
                aqi_pm10 := (pm10 / 154) * 100;
            END IF;
            
            -- CO AQI calculation
            IF co IS NOT NULL THEN
                aqi_co := (co / 30000) * 100;
            END IF;
            
            -- NO2 AQI calculation
            IF no2 IS NOT NULL THEN
                aqi_no2 := (no2 / 100) * 100;
            END IF;
            
            -- O3 AQI calculation
            IF o3 IS NOT NULL THEN
                aqi_o3 := (o3 / 160) * 100;
            END IF;
            
            -- Return the maximum AQI value
            max_aqi := GREATEST(
                COALESCE(aqi_pm25, 0),
                COALESCE(aqi_pm10, 0),
                COALESCE(aqi_co, 0),
                COALESCE(aqi_no2, 0),
                COALESCE(aqi_o3, 0)
            );
            
            RETURN max_aqi;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
        """
        
        with engine.connect() as conn:
            conn.execute(distance_function)
            conn.execute(nearest_grid_function)
            conn.execute(aqi_calculation_function)
            conn.commit()