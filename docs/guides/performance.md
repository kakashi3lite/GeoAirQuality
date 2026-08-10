# Performance Optimization Guide

## Overview

This document outlines performance optimization strategies for the GeoAirQuality platform, focusing on caching patterns, database optimization, and system-wide performance tuning.

## Redis Caching Architecture

### Cache Strategy

The GeoAirQuality API implements a **read-through caching pattern** with Redis to minimize database load and improve response times.

#### Cache Layers

1. **Application Cache** (Redis)
   - TTL-based expiration
   - Automatic cache warming
   - Pattern-based invalidation
   - Health monitoring

2. **Database Query Cache** (PostgreSQL)
   - Query result caching
   - Prepared statement caching
   - Connection pooling

### Cache Patterns

#### Read-Through Pattern

```python
@cached(prefix="air_quality_readings", ttl=300)
async def get_air_quality_readings(lat, lon, radius_km, hours, limit):
    # Cache miss triggers database query
    # Cache hit returns cached result
    pass
```

**Benefits:**
- Automatic cache population
- Consistent data access pattern
- Reduced database load

**TTL Configuration:**
- Air quality readings: 5 minutes (300s)
- Weather readings: 5 minutes (300s)
- Grid data: 10 minutes (600s)
- Aggregated data: 15 minutes (900s)

#### Cache Invalidation

```python
# Pattern-based invalidation
await cache.delete_pattern("air_quality_readings:*")

# Specific key invalidation
await cache.delete("grid_air_quality:grid_123")
```

### Cache Key Strategy

#### Hierarchical Keys

```
air_quality_readings:{lat}:{lon}:{radius}:{hours}:{limit}
grid_air_quality:{grid_id}:{hours}
weather_readings:{lat}:{lon}:{radius}:{hours}:{limit}
aggregated_grid:{grid_id}:{level}:{days}
```

#### Benefits
- Predictable key structure
- Pattern-based operations
- Easy debugging
- Efficient memory usage

## Database Performance

### Spatial Indexing

#### GiST Indexes

```sql
-- Spatial indexes for geometry columns
CREATE INDEX idx_air_quality_location_gist 
ON air_quality_readings USING GIST (location);

CREATE INDEX idx_weather_location_gist 
ON weather_readings USING GIST (location);

CREATE INDEX idx_spatial_grids_geom_gist 
ON spatial_grids USING GIST (geometry);
```

#### B-tree Indexes

```sql
-- Temporal indexes
CREATE INDEX idx_air_quality_timestamp 
ON air_quality_readings (timestamp DESC);

CREATE INDEX idx_weather_timestamp 
ON weather_readings (timestamp DESC);

-- Composite indexes
CREATE INDEX idx_air_quality_grid_time 
ON air_quality_readings (grid_id, timestamp DESC);
```

### Query Optimization

#### Spatial Queries

```sql
-- Optimized radius search
SELECT * FROM air_quality_readings 
WHERE ST_DWithin(location, ST_Point($1, $2), $3)
AND timestamp >= $4
ORDER BY timestamp DESC
LIMIT $5;
```

**Optimization Tips:**
- Use `ST_DWithin` instead of `ST_Distance`
- Always include temporal filters
- Limit result sets appropriately
- Use prepared statements

#### Connection Pooling

```python
# SQLAlchemy async engine configuration
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,        # Base connections
    max_overflow=30,     # Additional connections
    pool_pre_ping=True,  # Validate connections
    pool_recycle=3600    # Recycle after 1 hour
)
```

### Database Partitioning

#### Time-based Partitioning

```sql
-- Partition air quality readings by month
CREATE TABLE air_quality_readings_y2024m01 
PARTITION OF air_quality_readings 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE air_quality_readings_y2024m02 
PARTITION OF air_quality_readings 
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
```

**Benefits:**
- Faster queries on recent data
- Efficient data archival
- Parallel query execution
- Reduced index size

## API Performance

### Response Time Targets

| Endpoint Type | Target Response Time | Cache TTL |
|---------------|---------------------|----------|
| Health checks | < 50ms | No cache |
| Real-time data | < 200ms | 5 minutes |
| Historical data | < 500ms | 15 minutes |
| Aggregated data | < 1000ms | 30 minutes |

### Async Processing

#### Database Sessions

```python
# Async session management
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

#### Concurrent Requests

```python
# FastAPI automatic concurrency
# Uses asyncio event loop for I/O operations
# Supports thousands of concurrent connections
```

### Request Optimization

#### Pagination

```python
# Limit result sets
limit: int = Query(100, ge=1, le=1000)

# Use cursor-based pagination for large datasets
after_id: Optional[int] = Query(None)
```

#### Field Selection

```python
# Allow clients to specify required fields
fields: Optional[str] = Query(None, description="Comma-separated field list")
```

## Monitoring and Metrics

### Cache Metrics

```python
# Cache performance indicators
cache_metrics = {
    "hits": cache.hits,
    "misses": cache.misses,
    "hit_rate": cache.hits / (cache.hits + cache.misses),
    "memory_usage": cache.memory_usage,
    "key_count": cache.key_count
}
```

### Database Metrics

```sql
-- Query performance
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements 
ORDER BY total_time DESC;

-- Index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes;
```

### Application Metrics

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

request_count = Counter('api_requests_total', 'Total API requests')
request_duration = Histogram('api_request_duration_seconds', 'Request duration')
cache_hit_rate = Gauge('cache_hit_rate', 'Cache hit rate')
```

## Performance Tuning

### Redis Configuration

```redis
# redis.conf optimizations
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000

# Disable expensive operations in production
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""
```

### PostgreSQL Configuration

```postgresql
# postgresql.conf optimizations
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Connection settings
max_connections = 100

# Query planner
random_page_cost = 1.1
effective_io_concurrency = 200

# WAL settings
wal_buffers = 16MB
checkpoint_completion_target = 0.9
```

### Application Tuning

#### Uvicorn Configuration

```python
# Production server settings
uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=8000,
    workers=4,              # CPU cores
    worker_class="uvicorn.workers.UvicornWorker",
    access_log=False,       # Disable for performance
    loop="uvloop",          # Faster event loop
    http="httptools"        # Faster HTTP parser
)
```

#### Memory Management

```python
# Limit memory usage
import resource

# Set memory limit (1GB)
resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024, -1))

# Monitor memory usage
import psutil
process = psutil.Process()
memory_percent = process.memory_percent()
```

## Load Testing

### Test Scenarios

```bash
# Basic load test with Apache Bench
ab -n 1000 -c 10 http://localhost:8000/api/v1/air-quality/readings?lat=40.7128&lon=-74.0060

# Stress test with wrk
wrk -t12 -c400 -d30s http://localhost:8000/health

# Complex scenario with Artillery
artillery run load-test.yml
```

### Performance Benchmarks

| Scenario | Requests/sec | Avg Response Time | 95th Percentile |
|----------|-------------|------------------|----------------|
| Health check | 5000+ | 10ms | 25ms |
| Cached queries | 1000+ | 50ms | 100ms |
| Database queries | 200+ | 200ms | 500ms |
| Spatial queries | 100+ | 300ms | 800ms |

## Optimization Checklist

### Database
- [ ] Spatial indexes on geometry columns
- [ ] Temporal indexes on timestamp columns
- [ ] Composite indexes for common query patterns
- [ ] Connection pooling configured
- [ ] Query performance monitoring enabled
- [ ] Partitioning for large tables

### Caching
- [ ] Redis cluster for high availability
- [ ] Appropriate TTL values set
- [ ] Cache hit rate monitoring
- [ ] Memory usage monitoring
- [ ] Cache invalidation strategy

### Application
- [ ] Async/await patterns used
- [ ] Connection pooling enabled
- [ ] Request/response compression
- [ ] Pagination implemented
- [ ] Rate limiting configured

### Infrastructure
- [ ] Load balancer configured
- [ ] Auto-scaling enabled
- [ ] Resource limits set
- [ ] Health checks implemented
- [ ] Monitoring and alerting active

## Troubleshooting

### Common Performance Issues

#### Slow Queries

```sql
-- Identify slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
WHERE mean_time > 1000 
ORDER BY mean_time DESC;
```

**Solutions:**
- Add missing indexes
- Optimize query structure
- Increase work_mem
- Consider query rewriting

#### Cache Misses

```python
# Monitor cache performance
cache_stats = await cache.get_stats()
if cache_stats['hit_rate'] < 0.8:
    # Investigate cache configuration
    # Check TTL values
    # Review cache key patterns
```

#### Memory Issues

```bash
# Monitor memory usage
free -h
top -p $(pgrep -f "uvicorn")

# Check Redis memory
redis-cli info memory
```

**Solutions:**
- Increase available memory
- Optimize cache eviction policy
- Reduce cache TTL values
- Implement memory limits

## Best Practices

1. **Cache Early, Cache Often**
   - Cache at multiple layers
   - Use appropriate TTL values
   - Monitor cache performance

2. **Database Optimization**
   - Index frequently queried columns
   - Use spatial indexes for geometry
   - Implement connection pooling

3. **Async Programming**
   - Use async/await patterns
   - Avoid blocking operations
   - Pool database connections

4. **Monitoring**
   - Track key performance metrics
   - Set up alerting
   - Regular performance reviews

5. **Testing**
   - Load test regularly
   - Profile application performance
   - Test cache invalidation

By following these performance optimization strategies, the GeoAirQuality platform can efficiently handle high-volume spatial queries while maintaining sub-second response times for most operations.