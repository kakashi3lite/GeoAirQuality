"""Redis caching module for GeoAirQuality API.

Provides read-through caching for hot queries with health monitoring.
"""

import json
import logging
from typing import Any, Optional, Union, Dict, List
from functools import wraps
import hashlib
import asyncio
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError
from pydantic import BaseSettings

logger = logging.getLogger(__name__)


class CacheSettings(BaseSettings):
    """Redis cache configuration."""
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_ssl_cert_reqs: str = "required"
    
    # Cache settings
    default_ttl: int = 3600  # 1 hour
    max_connections: int = 20
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    
    # Health check settings
    health_check_interval: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    class Config:
        env_prefix = "CACHE_"


class RedisCache:
    """Redis cache manager with health monitoring."""
    
    def __init__(self, settings: CacheSettings = None):
        self.settings = settings or CacheSettings()
        self.redis_client: Optional[redis.Redis] = None
        self._healthy = False
        self._last_health_check = datetime.min
        self._connection_pool = None
        
    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        try:
            self._connection_pool = redis.ConnectionPool.from_url(
                self.settings.redis_url,
                password=self.settings.redis_password,
                ssl=self.settings.redis_ssl,
                ssl_cert_reqs=self.settings.redis_ssl_cert_reqs,
                max_connections=self.settings.max_connections,
                socket_timeout=self.settings.socket_timeout,
                socket_connect_timeout=self.settings.socket_connect_timeout,
                decode_responses=True
            )
            
            self.redis_client = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            self._healthy = True
            self._last_health_check = datetime.utcnow()
            
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self._healthy = False
            raise
    
    async def close(self) -> None:
        """Close Redis connections."""
        if self.redis_client:
            await self.redis_client.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()
        logger.info("Redis cache connections closed")
    
    async def health_check(self) -> bool:
        """Check Redis health status."""
        now = datetime.utcnow()
        
        # Skip if recently checked
        if (now - self._last_health_check).seconds < self.settings.health_check_interval:
            return self._healthy
        
        try:
            if self.redis_client:
                await self.redis_client.ping()
                self._healthy = True
            else:
                self._healthy = False
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            self._healthy = False
        
        self._last_health_check = now
        return self._healthy
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        # Create a deterministic hash of arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_hash = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not await self.health_check():
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not await self.health_check():
            return False
        
        try:
            ttl = ttl or self.settings.default_ttl
            serialized_value = json.dumps(value, default=str)
            await self.redis_client.setex(key, ttl, serialized_value)
            return True
        except (RedisError, TypeError) as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not await self.health_check():
            return False
        
        try:
            await self.redis_client.delete(key)
            return True
        except RedisError as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        if not await self.health_check():
            return 0
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        except RedisError as e:
            logger.warning(f"Cache pattern delete failed for pattern {pattern}: {e}")
            return 0
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """Increment counter in cache."""
        if not await self.health_check():
            return None
        
        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(key, amount)
            if ttl:
                pipe.expire(key, ttl)
            results = await pipe.execute()
            return results[0]
        except RedisError as e:
            logger.warning(f"Cache increment failed for key {key}: {e}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not await self.health_check():
            return {"healthy": False, "error": "Redis unavailable"}
        
        try:
            info = await self.redis_client.info()
            return {
                "healthy": True,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
        except RedisError as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"healthy": False, "error": str(e)}


# Global cache instance
cache_instance: Optional[RedisCache] = None


async def get_cache() -> RedisCache:
    """Get or create cache instance."""
    global cache_instance
    if cache_instance is None:
        cache_instance = RedisCache()
        await cache_instance.initialize()
    return cache_instance


def cached(prefix: str, ttl: Optional[int] = None, skip_cache: bool = False):
    """Decorator for caching function results.
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
        skip_cache: Skip cache for testing
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if skip_cache:
                return await func(*args, **kwargs)
            
            cache = await get_cache()
            cache_key = cache._generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for key: {cache_key}")
            result = await func(*args, **kwargs)
            
            # Cache the result
            await cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


def cache_invalidate(prefix: str):
    """Decorator to invalidate cache after function execution.
    
    Args:
        prefix: Cache key prefix pattern to invalidate
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache
            cache = await get_cache()
            pattern = f"{prefix}:*"
            deleted_count = await cache.delete_pattern(pattern)
            logger.debug(f"Invalidated {deleted_count} cache entries with pattern: {pattern}")
            
            return result
        return wrapper
    return decorator


class CacheMetrics:
    """Cache metrics collector."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.total_requests = 0
    
    def record_hit(self):
        self.hits += 1
        self.total_requests += 1
    
    def record_miss(self):
        self.misses += 1
        self.total_requests += 1
    
    def record_error(self):
        self.errors += 1
        self.total_requests += 1
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.misses / self.total_requests
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.errors / self.total_requests
    
    def reset(self):
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.total_requests = 0
    
    def to_dict(self) -> Dict[str, Union[int, float]]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "error_rate": self.error_rate
        }


# Global metrics instance
metrics = CacheMetrics()


# Example usage patterns
class AirQualityCache:
    """Air quality specific cache operations."""
    
    @staticmethod
    @cached(prefix="aqi_grid", ttl=300)  # 5 minutes
    async def get_grid_aqi(grid_id: str, timestamp: datetime) -> Optional[Dict]:
        """Cache AQI data for grid cells."""
        # This would be implemented in the actual service
        pass
    
    @staticmethod
    @cached(prefix="spatial_query", ttl=600)  # 10 minutes
    async def get_nearby_readings(lat: float, lon: float, radius_km: float) -> List[Dict]:
        """Cache spatial query results."""
        # This would be implemented in the actual service
        pass
    
    @staticmethod
    @cache_invalidate(prefix="aqi_grid")
    async def update_grid_data(grid_id: str, data: Dict) -> bool:
        """Update grid data and invalidate cache."""
        # This would be implemented in the actual service
        pass


# Health check endpoint helper
async def cache_health_status() -> Dict[str, Any]:
    """Get comprehensive cache health status."""
    try:
        cache = await get_cache()
        stats = await cache.get_stats()
        
        return {
            "status": "healthy" if stats.get("healthy") else "unhealthy",
            "redis_stats": stats,
            "cache_metrics": metrics.to_dict(),
            "last_health_check": cache._last_health_check.isoformat(),
            "settings": {
                "default_ttl": cache.settings.default_ttl,
                "max_connections": cache.settings.max_connections,
                "health_check_interval": cache.settings.health_check_interval
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "cache_metrics": metrics.to_dict()
        }