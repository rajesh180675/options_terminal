# ═══════════════════════════════════════════════════════════════
# FILE: rate_limiter.py
# ═══════════════════════════════════════════════════════════════
"""
Sliding-window rate limiter that enforces the Breeze 100-calls/min ceiling.
Used as a decorator on every outbound API method.
"""

import time
import threading
import functools
from collections import deque
from utils import LOG
from config import Config


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window limiter."""

    def __init__(self, max_calls: int, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a slot is available.  Returns wait time in seconds."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Purge timestamps older than the window
                while self._timestamps and self._timestamps[0] <= now - self.window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return 0.0
                # Must wait until the oldest timestamp expires
                wait = self._timestamps[0] + self.window - now + 0.01
            LOG.debug(f"Rate-limiter: sleeping {wait:.2f}s")
            time.sleep(wait)

    @property
    def remaining(self) -> int:
        with self._lock:
            now = time.monotonic()
            while self._timestamps and self._timestamps[0] <= now - self.window:
                self._timestamps.popleft()
            return self.max_calls - len(self._timestamps)


# Module-level singleton
_GLOBAL_LIMITER = SlidingWindowRateLimiter(
    max_calls=Config.API_RATE_LIMIT, window_seconds=60.0
)


def rate_limited(func):
    """Decorator: blocks until the sliding window has capacity."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wait = _GLOBAL_LIMITER.acquire()
        if wait > 0:
            LOG.warning(f"Rate-limit wait {wait:.1f}s before {func.__name__}")
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - t0) * 1000
        if elapsed_ms > Config.MAX_API_LATENCY_MS:
            LOG.warning(
                f"Latency circuit-breaker: {func.__name__} took {elapsed_ms:.0f}ms"
            )
        return result
    return wrapper


def get_rate_limiter() -> SlidingWindowRateLimiter:
    return _GLOBAL_LIMITER
