"""Prometheus metrics configuration."""

import time
from typing import Callable
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# Request metrics
REQUEST_DURATION = Histogram(
    "request_duration_ms",
    "Request duration in milliseconds",
    ["route", "method", "code"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
)

REQUEST_COUNT = Counter(
    "request_total",
    "Total number of requests",
    ["route", "method", "code"]
)

# ML-specific metrics
FAISS_LATENCY = Histogram(
    "faiss_latency_ms",
    "FAISS search latency in milliseconds",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

RERANK_LATENCY = Histogram(
    "rerank_latency_ms",
    "Reranking latency in milliseconds",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500]
)

CACHE_HIT_RATIO = Gauge(
    "cache_hit_ratio",
    "Cache hit ratio (0-1)"
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Total cache hits"
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Total cache misses"
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Extract route template (e.g., "/rec/users/{user_id}" instead of "/rec/users/123")
        route_template = self._get_route_template(request)
        
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        REQUEST_DURATION.labels(
            route=route_template,
            method=request.method,
            code=response.status_code
        ).observe(duration_ms)
        
        REQUEST_COUNT.labels(
            route=route_template,
            method=request.method,
            code=response.status_code
        ).inc()
        
        return response

    def _get_route_template(self, request: Request) -> str:
        """Extract route template from request."""
        if hasattr(request, "route") and request.route:
            return request.route.path
        return request.url.path


def update_cache_metrics(hit: bool) -> None:
    """Update cache hit/miss metrics."""
    if hit:
        CACHE_HITS.inc()
    else:
        CACHE_MISSES.inc()
    
    # Update hit ratio
    total = CACHE_HITS._value._value + CACHE_MISSES._value._value
    if total > 0:
        CACHE_HIT_RATIO.set(CACHE_HITS._value._value / total)


def get_metrics() -> str:
    """Get Prometheus metrics in text format."""
    return generate_latest(REGISTRY).decode('utf-8')

