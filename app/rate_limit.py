"""Per-IP rate limiting for the LLM-backed endpoint.

With the API key gate off, /v1/query is open to anyone with the URL, and each
call spends three Groq completions. This caps how fast a single caller can burn
that quota.

Scope honestly stated: the counters live in one instance's memory, so the real
ceiling is roughly `limit x instances` rather than `limit`. That is a genuine
weakness under heavy distributed load, and the fix would be shared state
(Vercel KV, Upstash). For a demo on a free tier the tradeoff is deliberate —
this stops a loop hammering the endpoint, which is the realistic failure mode,
without adding a paid dependency.
"""

import os
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request, status

WINDOW_SECONDS = 3600
DEFAULT_LIMIT = 30

# Above this many tracked IPs, drop the ones whose windows have fully expired.
# Without a sweep the dict grows unbounded across an instance's lifetime.
_MAX_TRACKED_IPS = 10_000

_lock = Lock()
_hits: dict[str, deque[float]] = defaultdict(deque)


def _limit() -> int:
    """Requests per IP per hour. Set RATE_LIMIT_PER_HOUR=0 to disable."""
    try:
        return int(os.environ.get("RATE_LIMIT_PER_HOUR", DEFAULT_LIMIT))
    except ValueError:
        return DEFAULT_LIMIT


def _client_ip(request: Request) -> str:
    # Vercel terminates TLS upstream, so request.client is the proxy. The real
    # caller is the first entry in x-forwarded-for.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _sweep(cutoff: float) -> None:
    """Caller must hold the lock."""
    if len(_hits) <= _MAX_TRACKED_IPS:
        return
    stale = [ip for ip, q in _hits.items() if not q or q[-1] < cutoff]
    for ip in stale:
        del _hits[ip]


def enforce_rate_limit(request: Request) -> None:
    limit = _limit()
    if limit <= 0:
        return

    ip = _client_ip(request)
    now = time.monotonic()
    cutoff = now - WINDOW_SECONDS

    with _lock:
        window = _hits[ip]

        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= limit:
            retry_after = int(window[0] + WINDOW_SECONDS - now) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit reached ({limit} queries/hour). "
                    f"Try again in {retry_after // 60 + 1} minute(s)."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)
        _sweep(cutoff)
