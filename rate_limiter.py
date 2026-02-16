# ═══════════════════════════════════════════════════════════════
# FILE: rate_limiter.py
# ═══════════════════════════════════════════════════════════════
"""
Sliding-window rate limiter. Import from app_config (not config).
"""

import time
import threading
import functools
from collections import deque
from app_config import Config


class SlidingWindowRateLimiter:
    def __init__(self, max_calls: int, window: float = 60.0):
        self.max_calls = max_calls
        self.window = window
        self._ts: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._ts and self._ts[0] <= now - self.window:
                    self._ts.popleft()
                if len(self._ts) < self.max_calls:
                    self._ts.append(now)
                    return 0.0
                wait = self._ts[0] + self.window - now + 0.02
            time.sleep(wait)

    @property
    def remaining(self) -> int:
        with self._lock:
            now = time.monotonic()
            while self._ts and self._ts[0] <= now - self.window:
                self._ts.popleft()
            return self.max_calls - len(self._ts)


_LIMITER = SlidingWindowRateLimiter(Config.API_RATE_LIMIT, 60.0)


def rate_limited(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _LIMITER.acquire()
        return func(*args, **kwargs)
    return wrapper


def get_rate_limiter():
    return _LIMITER
