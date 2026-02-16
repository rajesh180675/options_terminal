# ═══════════════════════════════════════════════════════════════
# FILE: rate_limiter.py
# ═══════════════════════════════════════════════════════════════
"""
Sliding-window rate limiter.
Enforces Breeze's 100-calls/minute ceiling.
We target 85/min to leave headroom for the chase algorithm.
"""

import time
import threading
import functools
from collections import deque
from config import Config


class SlidingWindowRateLimiter:
    def __init__(self, max_calls: int, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and self._timestamps[0] <= now - self.window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return 0.0
                wait = self._timestamps[0] + self.window - now + 0.02
            time.sleep(wait)

    @property
    def remaining(self) -> int:
        with self._lock:
            now = time.monotonic()
            while self._timestamps and self._timestamps[0] <= now - self.window:
                self._timestamps.popleft()
            return self.max_calls - len(self._timestamps)


_GLOBAL_LIMITER = SlidingWindowRateLimiter(
    max_calls=Config.API_RATE_LIMIT, window_seconds=60.0
)


def rate_limited(func):
    """Decorator: blocks until sliding window has capacity."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from utils import LOG
        wait = _GLOBAL_LIMITER.acquire()
        if wait > 0:
            LOG.warning(f"Rate-limit wait {wait:.1f}s before {func.__name__}")
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - t0) * 1000
        if elapsed_ms > Config.MAX_API_LATENCY_MS:
            LOG.warning(
                f"Latency alert: {func.__name__} took {elapsed_ms:.0f}ms"
            )
        return result
    return wrapper


def get_rate_limiter() -> SlidingWindowRateLimiter:
    return _GLOBAL_LIMITER
