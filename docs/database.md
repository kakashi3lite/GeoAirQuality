# Database Architecture

## Overview

The GeoAirQuality database uses PostgreSQL with PostGIS extension for spatial data handling. The schema is designed for high-performance spatial queries and time-series data storage.

## Schema Design

### Core Tables

#### 1. spatial_grids
Stores the spatial grid system for data aggregation.

```sql
CREATE TABLE spatial_grids (
    id SERIAL PRIMARY KEY,
    grid_id VARCHAR(50) UNIQUE NOT NULL,
    geometry GEOMETRY(POLYGON, 4326) NOT NULL,
    center_point GEOMETRY(POINT, 4326) NOT NULL,
    grid_size_km FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes:**
- `idx_spatial_grids_geometry` (GiST): Enables fast spatial queries
- `idx_spatial_grids_center` (GiST): Optimizes nearest neighbor searches
- `idx_spatial_grids_grid_id` (B-tree): Fast lookups by grid identifier

#### 2. air_quality_readings
Stores raw air quality sensor data with spatial and temporal dimensions.

```sql
CREATE TABLE air_quality_readings (
    id BIGSERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    pm25 FLOAT,
    pm10 FLOAT,
    no2 FLOAT,
    o3 FLOAT,
    co FLOAT,
    so2 FLOAT,
    aqi INTEGER,
    data_source_id INTEGER REFERENCES data_sources(id),
    grid_id VARCHAR(50) REFERENCES spatial_grids(grid_id)
);
```

**Indexes:**
- `idx_air_quality_location` (GiST): Spatial queries
- `idx_air_quality_timestamp` (B-tree): Time-based filtering
- `idx_air_quality_grid_timestamp` (B-tree): Grid-based time series
- `idx_air_quality_aqi` (B-tree): AQI range queries

#### 3. weather_readings
Stores meteorological data correlated with air quality.

```sql
CREATE TABLE weather_readings (
    id BIGSERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    pressure FLOAT,
    wind_speed FLOAT,
    wind_direction FLOAT,
    precipitation FLOAT,
    data_source_id INTEGER REFERENCES data_sources(id),
    grid_id VARCHAR(50) REFERENCES spatial_grids(grid_id)
);
```

**Indexes:**
- `idx_weather_location` (GiST): Spatial queries
- `idx_weather_timestamp` (B-tree): Time-based filtering
- `idx_weather_grid_timestamp` (B-tree): Grid-based time series

#### 4. aggregated_data
Pre-computed aggregations for fast API responses.

```sql
CREATE TABLE aggregated_data (
    id BIGSERIAL PRIMARY KEY,
    grid_id VARCHAR(50) REFERENCES spatial_grids(grid_id),
    time_bucket TIMESTAMP NOT NULL,
    aggregation_level VARCHAR(20) NOT NULL, -- 'hourly', 'daily', 'weekly'
    avg_pm25 FLOAT,
    max_pm25 FLOAT,
    min_pm25 FLOAT,
    avg_aqi INTEGER,
    max_aqi INTEGER,
    reading_count INTEGER,
    avg_temperature FLOAT,
    avg_humidity FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes:**
- `idx_aggregated_grid_time` (B-tree): Fast time-series queries
- `idx_aggregated_level_time` (B-tree): Aggregation level filtering

#### 5. data_sources
Metadata about data providers and sensor networks.

```sql
CREATE TABLE data_sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    api_endpoint VARCHAR(255),
    update_frequency_minutes INTEGER,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Spatial Indexing Strategy

### GiST Indexes
PostGIS uses Generalized Search Tree (GiST) indexes for spatial data:

1. **Geometry Columns**: All geometry columns have GiST indexes
2. **Spatial Queries**: Enable fast bounding box and intersection queries
3. **Nearest Neighbor**: Support KNN queries for finding closest sensors

### Index Rationale

#### Performance Considerations
- **Spatial Grids**: GiST indexes on geometry enable sub-second spatial joins
- **Time Series**: B-tree indexes on timestamps support efficient time range queries
- **Composite Indexes**: grid_id + timestamp indexes optimize common query patterns

#### Query Optimization
Common query patterns and their optimized indexes:

```sql
-- Pattern 1: Spatial range queries
SELECT * FROM air_quality_readings 
WHERE ST_DWithin(location, ST_Point(-74.006, 40.7128), 1000);
-- Optimized by: idx_air_quality_location

-- Pattern 2: Time series for specific grid
SELECT * FROM air_quality_readings 
WHERE grid_id = 'grid_123' AND timestamp >= '2024-01-01';
-- Optimized by: idx_air_quality_grid_timestamp

-- Pattern 3: AQI threshold queries
SELECT * FROM air_quality_readings 
WHERE aqi > 100 AND timestamp >= NOW() - INTERVAL '24 hours';
-- Optimized by: idx_air_quality_aqi, idx_air_quality_timestamp
```

## Custom Functions

### calculate_distance_km
Calculates great circle distance between two points:

```sql
CREATE OR REPLACE FUNCTION calculate_distance_km(
    lat1 FLOAT, lon1 FLOAT, lat2 FLOAT, lon2 FLOAT
) RETURNS FLOAT AS $$
BEGIN
    RETURN ST_Distance(
        ST_Transform(ST_SetSRID(ST_Point(lon1, lat1), 4326), 3857),
        ST_Transform(ST_SetSRID(ST_Point(lon2, lat2), 4326), 3857)
    ) / 1000.0;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

### find_nearest_grid
Finds the closest grid cell for a given point:

```sql
CREATE OR REPLACE FUNCTION find_nearest_grid(
    input_lat FLOAT, input_lon FLOAT
) RETURNS VARCHAR(50) AS $$
DECLARE
    nearest_grid VARCHAR(50);
BEGIN
    SELECT grid_id INTO nearest_grid
    FROM spatial_grids
    ORDER BY center_point <-> ST_SetSRID(ST_Point(input_lon, input_lat), 4326)
    LIMIT 1;
    
    RETURN nearest_grid;
END;
$$ LANGUAGE plpgsql STABLE;
```

### calculate_aqi
Calculates Air Quality Index from pollutant concentrations:

```sql
CREATE OR REPLACE FUNCTION calculate_aqi(
    pm25_val FLOAT, pm10_val FLOAT, no2_val FLOAT, o3_val FLOAT
) RETURNS INTEGER AS $$
DECLARE
    aqi_pm25 INTEGER := 0;
    aqi_pm10 INTEGER := 0;
    max_aqi INTEGER := 0;
BEGIN
    -- PM2.5 AQI calculation (simplified)
    IF pm25_val IS NOT NULL THEN
        CASE 
            WHEN pm25_val <= 12.0 THEN aqi_pm25 := ROUND(50 * pm25_val / 12.0);
            WHEN pm25_val <= 35.4 THEN aqi_pm25 := ROUND(50 + (100-50) * (pm25_val - 12.0) / (35.4 - 12.0));
            WHEN pm25_val <= 55.4 THEN aqi_pm25 := ROUND(100 + (150-100) * (pm25_val - 35.4) / (55.4 - 35.4));
            WHEN pm25_val <= 150.4 THEN aqi_pm25 := ROUND(150 + (200-150) * (pm25_val - 55.4) / (150.4 - 55.4));
            ELSE aqi_pm25 := 300;
        END CASE;
    END IF;
    
    -- PM10 AQI calculation (simplified)
    IF pm10_val IS NOT NULL THEN
        CASE 
            WHEN pm10_val <= 54 THEN aqi_pm10 := ROUND(50 * pm10_val / 54);
            WHEN pm10_val <= 154 THEN aqi_pm10 := ROUND(50 + (100-50) * (pm10_val - 54) / (154 - 54));
            WHEN pm10_val <= 254 THEN aqi_pm10 := ROUND(100 + (150-100) * (pm10_val - 154) / (254 - 154));
            ELSE aqi_pm10 := 300;
        END CASE;
    END IF;
    
    max_aqi := GREATEST(aqi_pm25, aqi_pm10);
    RETURN COALESCE(max_aqi, 0);
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

## Performance Tuning

### Connection Pooling
- Use pgbouncer for connection pooling
- Recommended pool size: 20-50 connections

### Query Optimization
- Use EXPLAIN ANALYZE for query planning
- Monitor slow queries with pg_stat_statements
- Regular VACUUM and ANALYZE operations

### Partitioning Strategy
For large datasets, consider partitioning by time:

```sql
-- Example: Monthly partitioning for air_quality_readings
CREATE TABLE air_quality_readings_y2024m01 
PARTITION OF air_quality_readings 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

## Backup and Recovery

### Backup Strategy
- Daily full backups using pg_dump
- Continuous WAL archiving for point-in-time recovery
- Test restore procedures monthly

### Monitoring
- Track database size growth
- Monitor index usage statistics
- Alert on connection pool exhaustion
- Track query performance metrics

## Security

### Access Control
- Separate read/write roles
- Row-level security for multi-tenant scenarios
- SSL/TLS encryption for connections

### Data Privacy
- Anonymize location data when required
- Implement data retention policies
- Regular security audits