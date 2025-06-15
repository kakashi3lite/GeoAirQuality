#!/usr/bin/env python3
"""
Initial PostGIS Schema Migration
Author: Captain Aurelia "Skyforge" Stratos

Revision ID: 001_initial_postgis_schema
Revises: 
Create Date: 2024-05-16 14:30:00.000000

Sets up the complete PostGIS database schema with optimized spatial indexes,
constraints, and custom functions for high-performance geospatial queries.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from geoalchemy2 import Geometry, Geography
import datetime

# revision identifiers
revision = '001_initial_postgis_schema'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    """Create the initial PostGIS schema with all tables and indexes."""
    
    # Enable PostGIS extension
    op.execute('CREATE EXTENSION IF NOT EXISTS postgis;')
    op.execute('CREATE EXTENSION IF NOT EXISTS postgis_topology;')
    
    # Create spatial_grids table
    op.create_table(
        'spatial_grids',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('grid_id', sa.String(length=50), nullable=False),
        sa.Column('resolution', sa.Float(), nullable=False),
        sa.Column('center_lat', sa.Float(), nullable=False),
        sa.Column('center_lon', sa.Float(), nullable=False),
        sa.Column('geometry', Geometry('POLYGON', srid=4326, spatial_index=True), nullable=False),
        sa.Column('geography', Geography('POLYGON', srid=4326, spatial_index=True), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.datetime.utcnow),
        sa.CheckConstraint('resolution > 0', name='positive_resolution'),
        sa.CheckConstraint('center_lat >= -90 AND center_lat <= 90', name='valid_latitude'),
        sa.CheckConstraint('center_lon >= -180 AND center_lon <= 180', name='valid_longitude'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('grid_id')
    )
    
    # Create indexes for spatial_grids
    op.create_index('idx_spatial_grid_resolution', 'spatial_grids', ['resolution'])
    op.create_index('idx_spatial_grid_center', 'spatial_grids', ['center_lat', 'center_lon'])
    op.create_index('idx_spatial_grids_grid_id', 'spatial_grids', ['grid_id'])
    
    # Create air_quality_readings table
    op.create_table(
        'air_quality_readings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('sensor_id', sa.String(length=100), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=False),
        sa.Column('longitude', sa.Float(), nullable=False),
        sa.Column('location', Geometry('POINT', srid=4326, spatial_index=True), nullable=False),
        sa.Column('grid_cell_id', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('co_gt', sa.Float(), nullable=True),
        sa.Column('pt08_s1_co', sa.Float(), nullable=True),
        sa.Column('nmhc_gt', sa.Float(), nullable=True),
        sa.Column('c6h6_gt', sa.Float(), nullable=True),
        sa.Column('pt08_s2_nmhc', sa.Float(), nullable=True),
        sa.Column('nox_gt', sa.Float(), nullable=True),
        sa.Column('pt08_s3_nox', sa.Float(), nullable=True),
        sa.Column('no2_gt', sa.Float(), nullable=True),
        sa.Column('pt08_s4_no2', sa.Float(), nullable=True),
        sa.Column('pt08_s5_o3', sa.Float(), nullable=True),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('relative_humidity', sa.Float(), nullable=True),
        sa.Column('absolute_humidity', sa.Float(), nullable=True),
        sa.Column('aqi', sa.Float(), nullable=True),
        sa.Column('aqi_category', sa.String(length=20), nullable=True),
        sa.Column('is_validated', sa.Boolean(), nullable=True, default=False),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.datetime.utcnow),
        sa.CheckConstraint('latitude >= -90 AND latitude <= 90', name='valid_latitude'),
        sa.CheckConstraint('longitude >= -180 AND longitude <= 180', name='valid_longitude'),
        sa.CheckConstraint('relative_humidity >= 0 AND relative_humidity <= 100', name='valid_humidity'),
        sa.CheckConstraint('aqi >= 0', name='positive_aqi'),
        sa.CheckConstraint('quality_score >= 0 AND quality_score <= 1', name='valid_quality_score'),
        sa.ForeignKeyConstraint(['grid_cell_id'], ['spatial_grids.id']),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for air_quality_readings
    op.create_index('idx_air_quality_readings_sensor_id', 'air_quality_readings', ['sensor_id'])
    op.create_index('idx_air_quality_readings_grid_cell_id', 'air_quality_readings', ['grid_cell_id'])
    op.create_index('idx_air_quality_readings_timestamp', 'air_quality_readings', ['timestamp'])
    op.create_index('idx_air_quality_readings_aqi', 'air_quality_readings', ['aqi'])
    op.create_index('idx_air_quality_readings_is_validated', 'air_quality_readings', ['is_validated'])
    op.create_index('idx_aq_sensor_time', 'air_quality_readings', ['sensor_id', 'timestamp'])
    op.create_index('idx_aq_time_aqi', 'air_quality_readings', ['timestamp', 'aqi'])
    op.create_index('idx_aq_location_time', 'air_quality_readings', ['latitude', 'longitude', 'timestamp'])
    op.create_index('idx_aq_grid_time', 'air_quality_readings', ['grid_cell_id', 'timestamp'])
    op.create_index('idx_aq_validated_time', 'air_quality_readings', ['is_validated', 'timestamp'])
    
    # Create partial index for high-quality recent data
    op.execute("""
        CREATE INDEX idx_aq_recent_quality 
        ON air_quality_readings (timestamp, aqi) 
        WHERE is_validated = true AND timestamp > NOW() - INTERVAL '7 days'
    """)
    
    # Create weather_readings table
    op.create_table(
        'weather_readings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('station_id', sa.String(length=100), nullable=False),
        sa.Column('country', sa.String(length=100), nullable=False),
        sa.Column('location_name', sa.String(length=200), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=False),
        sa.Column('longitude', sa.Float(), nullable=False),
        sa.Column('location', Geometry('POINT', srid=4326, spatial_index=True), nullable=False),
        sa.Column('grid_cell_id', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('last_updated', sa.DateTime(), nullable=False),
        sa.Column('temperature_celsius', sa.Float(), nullable=True),
        sa.Column('temperature_fahrenheit', sa.Float(), nullable=True),
        sa.Column('humidity', sa.Float(), nullable=True),
        sa.Column('pressure_mb', sa.Float(), nullable=True),
        sa.Column('pressure_in', sa.Float(), nullable=True),
        sa.Column('wind_kph', sa.Float(), nullable=True),
        sa.Column('wind_mph', sa.Float(), nullable=True),
        sa.Column('wind_degree', sa.Float(), nullable=True),
        sa.Column('wind_direction', sa.String(length=10), nullable=True),
        sa.Column('pm2_5', sa.Float(), nullable=True),
        sa.Column('pm10', sa.Float(), nullable=True),
        sa.Column('carbon_monoxide', sa.Float(), nullable=True),
        sa.Column('ozone', sa.Float(), nullable=True),
        sa.Column('nitrogen_dioxide', sa.Float(), nullable=True),
        sa.Column('sulphur_dioxide', sa.Float(), nullable=True),
        sa.Column('aqi', sa.Float(), nullable=True),
        sa.Column('aqi_category', sa.String(length=20), nullable=True),
        sa.Column('is_validated', sa.Boolean(), nullable=True, default=False),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.datetime.utcnow),
        sa.CheckConstraint('latitude >= -90 AND latitude <= 90', name='valid_latitude'),
        sa.CheckConstraint('longitude >= -180 AND longitude <= 180', name='valid_longitude'),
        sa.CheckConstraint('humidity >= 0 AND humidity <= 100', name='valid_humidity'),
        sa.CheckConstraint('pressure_mb > 0', name='positive_pressure'),
        sa.CheckConstraint('wind_kph >= 0', name='positive_wind_speed'),
        sa.CheckConstraint('wind_degree >= 0 AND wind_degree < 360', name='valid_wind_direction'),
        sa.ForeignKeyConstraint(['grid_cell_id'], ['spatial_grids.id']),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for weather_readings
    op.create_index('idx_weather_readings_station_id', 'weather_readings', ['station_id'])
    op.create_index('idx_weather_readings_country', 'weather_readings', ['country'])
    op.create_index('idx_weather_readings_grid_cell_id', 'weather_readings', ['grid_cell_id'])
    op.create_index('idx_weather_readings_timestamp', 'weather_readings', ['timestamp'])
    op.create_index('idx_weather_readings_aqi', 'weather_readings', ['aqi'])
    op.create_index('idx_weather_readings_is_validated', 'weather_readings', ['is_validated'])
    op.create_index('idx_weather_station_time', 'weather_readings', ['station_id', 'timestamp'])
    op.create_index('idx_weather_country_time', 'weather_readings', ['country', 'timestamp'])
    op.create_index('idx_weather_location_time', 'weather_readings', ['latitude', 'longitude', 'timestamp'])
    op.create_index('idx_weather_grid_time', 'weather_readings', ['grid_cell_id', 'timestamp'])
    op.create_index('idx_weather_aq_time', 'weather_readings', ['timestamp', 'pm2_5', 'pm10'])
    
    # Create aggregated_data table
    op.create_table(
        'aggregated_data',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('grid_cell_id', sa.Integer(), nullable=False),
        sa.Column('aggregation_level', sa.String(length=10), nullable=False),
        sa.Column('time_bucket', sa.DateTime(), nullable=False),
        sa.Column('avg_aqi', sa.Float(), nullable=True),
        sa.Column('max_aqi', sa.Float(), nullable=True),
        sa.Column('min_aqi', sa.Float(), nullable=True),
        sa.Column('avg_pm2_5', sa.Float(), nullable=True),
        sa.Column('avg_pm10', sa.Float(), nullable=True),
        sa.Column('avg_co', sa.Float(), nullable=True),
        sa.Column('avg_no2', sa.Float(), nullable=True),
        sa.Column('avg_o3', sa.Float(), nullable=True),
        sa.Column('avg_temperature', sa.Float(), nullable=True),
        sa.Column('avg_humidity', sa.Float(), nullable=True),
        sa.Column('avg_pressure', sa.Float(), nullable=True),
        sa.Column('avg_wind_speed', sa.Float(), nullable=True),
        sa.Column('data_points_count', sa.Integer(), nullable=False),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.datetime.utcnow),
        sa.CheckConstraint('data_points_count > 0', name='positive_data_points'),
        sa.CheckConstraint("aggregation_level IN ('hour', 'day', 'month')", name='valid_aggregation_level'),
        sa.ForeignKeyConstraint(['grid_cell_id'], ['spatial_grids.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('grid_cell_id', 'aggregation_level', 'time_bucket', name='unique_aggregation')
    )
    
    # Create indexes for aggregated_data
    op.create_index('idx_aggregated_data_grid_cell_id', 'aggregated_data', ['grid_cell_id'])
    op.create_index('idx_aggregated_data_time_bucket', 'aggregated_data', ['time_bucket'])
    op.create_index('idx_agg_level_time', 'aggregated_data', ['aggregation_level', 'time_bucket'])
    op.create_index('idx_agg_grid_level_time', 'aggregated_data', ['grid_cell_id', 'aggregation_level', 'time_bucket'])
    
    # Create data_sources table
    op.create_table(
        'data_sources',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('source_name', sa.String(length=100), nullable=False),
        sa.Column('source_type', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('connection_string', sa.Text(), nullable=True),
        sa.Column('api_endpoint', sa.String(length=500), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
        sa.Column('last_ingestion', sa.DateTime(), nullable=True),
        sa.Column('next_scheduled_ingestion', sa.DateTime(), nullable=True),
        sa.Column('ingestion_frequency', sa.String(length=50), nullable=True),
        sa.Column('total_records_ingested', sa.Integer(), nullable=True, default=0),
        sa.Column('failed_ingestions', sa.Integer(), nullable=True, default=0),
        sa.Column('average_quality_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.datetime.utcnow),
        sa.CheckConstraint('total_records_ingested >= 0', name='positive_records'),
        sa.CheckConstraint('failed_ingestions >= 0', name='positive_failures'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_name')
    )
    
    # Create indexes for data_sources
    op.create_index('idx_data_sources_is_active', 'data_sources', ['is_active'])
    op.create_index('idx_data_sources_last_ingestion', 'data_sources', ['last_ingestion'])
    op.create_index('idx_data_sources_next_scheduled_ingestion', 'data_sources', ['next_scheduled_ingestion'])
    op.create_index('idx_source_active_schedule', 'data_sources', ['is_active', 'next_scheduled_ingestion'])
    
    # Create custom PostgreSQL functions
    create_custom_functions()
    
    # Insert initial spatial grids
    create_initial_spatial_grids()
    
    # Insert sample data sources
    create_sample_data_sources()

def create_custom_functions():
    """Create custom PostgreSQL functions for optimized queries."""
    
    # Function to calculate distance between points
    op.execute("""
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
    """)
    
    # Function to find nearest grid cell
    op.execute("""
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
    """)
    
    # Function to calculate AQI from pollutant concentrations
    op.execute("""
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
    """)
    
    # Function to update aggregated data
    op.execute("""
        CREATE OR REPLACE FUNCTION update_aggregated_data(
            grid_id INTEGER, agg_level TEXT, bucket_time TIMESTAMP
        ) RETURNS VOID AS $$
        BEGIN
            INSERT INTO aggregated_data (
                grid_cell_id, aggregation_level, time_bucket,
                avg_aqi, max_aqi, min_aqi, avg_pm2_5, avg_pm10,
                avg_co, avg_no2, avg_o3, avg_temperature,
                avg_humidity, avg_pressure, avg_wind_speed,
                data_points_count, quality_score
            )
            SELECT 
                grid_id,
                agg_level,
                bucket_time,
                AVG(aqi) as avg_aqi,
                MAX(aqi) as max_aqi,
                MIN(aqi) as min_aqi,
                AVG(pm2_5) as avg_pm2_5,
                AVG(pm10) as avg_pm10,
                AVG(carbon_monoxide) as avg_co,
                AVG(nitrogen_dioxide) as avg_no2,
                AVG(ozone) as avg_o3,
                AVG(temperature_celsius) as avg_temperature,
                AVG(humidity) as avg_humidity,
                AVG(pressure_mb) as avg_pressure,
                AVG(wind_kph) as avg_wind_speed,
                COUNT(*) as data_points_count,
                AVG(quality_score) as quality_score
            FROM weather_readings
            WHERE grid_cell_id = grid_id
                AND timestamp >= bucket_time
                AND timestamp < bucket_time + INTERVAL '1 ' || agg_level
            ON CONFLICT (grid_cell_id, aggregation_level, time_bucket)
            DO UPDATE SET
                avg_aqi = EXCLUDED.avg_aqi,
                max_aqi = EXCLUDED.max_aqi,
                min_aqi = EXCLUDED.min_aqi,
                avg_pm2_5 = EXCLUDED.avg_pm2_5,
                avg_pm10 = EXCLUDED.avg_pm10,
                avg_co = EXCLUDED.avg_co,
                avg_no2 = EXCLUDED.avg_no2,
                avg_o3 = EXCLUDED.avg_o3,
                avg_temperature = EXCLUDED.avg_temperature,
                avg_humidity = EXCLUDED.avg_humidity,
                avg_pressure = EXCLUDED.avg_pressure,
                avg_wind_speed = EXCLUDED.avg_wind_speed,
                data_points_count = EXCLUDED.data_points_count,
                quality_score = EXCLUDED.quality_score,
                updated_at = NOW();
        END;
        $$ LANGUAGE plpgsql;
    """)

def create_initial_spatial_grids():
    """Create initial spatial grid cells for common resolutions."""
    
    # Create 1-degree resolution grid for global coverage
    op.execute("""
        INSERT INTO spatial_grids (grid_id, resolution, center_lat, center_lon, geometry, geography)
        SELECT 
            'grid_1deg_' || lat || '_' || lon as grid_id,
            1.0 as resolution,
            lat as center_lat,
            lon as center_lon,
            ST_GeomFromText(
                'POLYGON((' || 
                (lon - 0.5) || ' ' || (lat - 0.5) || ',' ||
                (lon + 0.5) || ' ' || (lat - 0.5) || ',' ||
                (lon + 0.5) || ' ' || (lat + 0.5) || ',' ||
                (lon - 0.5) || ' ' || (lat + 0.5) || ',' ||
                (lon - 0.5) || ' ' || (lat - 0.5) || '))', 
                4326
            ) as geometry,
            ST_GeogFromText(
                'POLYGON((' || 
                (lon - 0.5) || ' ' || (lat - 0.5) || ',' ||
                (lon + 0.5) || ' ' || (lat - 0.5) || ',' ||
                (lon + 0.5) || ' ' || (lat + 0.5) || ',' ||
                (lon - 0.5) || ' ' || (lat + 0.5) || ',' ||
                (lon - 0.5) || ' ' || (lat - 0.5) || '))', 
                4326
            ) as geography
        FROM generate_series(-89, 89, 1) as lat,
             generate_series(-179, 179, 1) as lon;
    """)
    
    # Create 0.1-degree resolution grid for high-density areas
    op.execute("""
        INSERT INTO spatial_grids (grid_id, resolution, center_lat, center_lon, geometry, geography)
        SELECT 
            'grid_01deg_' || ROUND(lat::numeric, 1) || '_' || ROUND(lon::numeric, 1) as grid_id,
            0.1 as resolution,
            ROUND(lat::numeric, 1) as center_lat,
            ROUND(lon::numeric, 1) as center_lon,
            ST_GeomFromText(
                'POLYGON((' || 
                (ROUND(lon::numeric, 1) - 0.05) || ' ' || (ROUND(lat::numeric, 1) - 0.05) || ',' ||
                (ROUND(lon::numeric, 1) + 0.05) || ' ' || (ROUND(lat::numeric, 1) - 0.05) || ',' ||
                (ROUND(lon::numeric, 1) + 0.05) || ' ' || (ROUND(lat::numeric, 1) + 0.05) || ',' ||
                (ROUND(lon::numeric, 1) - 0.05) || ' ' || (ROUND(lat::numeric, 1) + 0.05) || ',' ||
                (ROUND(lon::numeric, 1) - 0.05) || ' ' || (ROUND(lat::numeric, 1) - 0.05) || '))', 
                4326
            ) as geometry,
            ST_GeogFromText(
                'POLYGON((' || 
                (ROUND(lon::numeric, 1) - 0.05) || ' ' || (ROUND(lat::numeric, 1) - 0.05) || ',' ||
                (ROUND(lon::numeric, 1) + 0.05) || ' ' || (ROUND(lat::numeric, 1) - 0.05) || ',' ||
                (ROUND(lon::numeric, 1) + 0.05) || ' ' || (ROUND(lat::numeric, 1) + 0.05) || ',' ||
                (ROUND(lon::numeric, 1) - 0.05) || ' ' || (ROUND(lat::numeric, 1) + 0.05) || ',' ||
                (ROUND(lon::numeric, 1) - 0.05) || ' ' || (ROUND(lat::numeric, 1) - 0.05) || '))', 
                4326
            ) as geography
        FROM generate_series(35.0, 45.0, 0.1) as lat,
             generate_series(-125.0, -65.0, 0.1) as lon
        WHERE lat BETWEEN 35.0 AND 45.0 AND lon BETWEEN -125.0 AND -65.0;
    """)

def create_sample_data_sources():
    """Insert sample data source configurations."""
    
    op.execute("""
        INSERT INTO data_sources (
            source_name, source_type, description, 
            is_active, ingestion_frequency
        ) VALUES 
        (
            'AirQuality_CSV', 'file', 
            'Local air quality sensor data from CSV files',
            true, 'daily'
        ),
        (
            'GlobalWeather_API', 'api',
            'Global weather repository API with air quality data',
            true, 'hourly'
        ),
        (
            'IoT_Sensors_Stream', 'stream',
            'Real-time IoT sensor data stream',
            true, 'realtime'
        );
    """)

def downgrade() -> None:
    """Drop all tables and extensions."""
    
    # Drop custom functions
    op.execute('DROP FUNCTION IF EXISTS calculate_distance(FLOAT, FLOAT, FLOAT, FLOAT);')
    op.execute('DROP FUNCTION IF EXISTS find_nearest_grid(FLOAT, FLOAT, FLOAT);')
    op.execute('DROP FUNCTION IF EXISTS calculate_aqi(FLOAT, FLOAT, FLOAT, FLOAT, FLOAT);')
    op.execute('DROP FUNCTION IF EXISTS update_aggregated_data(INTEGER, TEXT, TIMESTAMP);')
    
    # Drop tables in reverse order
    op.drop_table('data_sources')
    op.drop_table('aggregated_data')
    op.drop_table('weather_readings')
    op.drop_table('air_quality_readings')
    op.drop_table('spatial_grids')
    
    # Note: We don't drop PostGIS extension as it might be used by other databases