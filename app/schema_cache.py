"""TTL cache in front of schema introspection.

introspect_schema issues one round trip for table names and then three more per
table (columns, primary key, foreign keys). Against a local Postgres that was
free; against a managed database reached over the network — with NullPool
opening a fresh connection each time — it is the single largest cost in a
request, and it re-fetches a schema that changes approximately never.

The cache is per-instance and in-memory, so a schema migration takes effect
within TTL seconds on each running instance rather than instantly. That is the
right trade for a read-only analytics API.
"""

import os
import time
from threading import Lock

from sqlalchemy.engine import Engine

from schema_introspection import TableInfo, introspect_schema

DEFAULT_TTL_SECONDS = 300

_lock = Lock()
_cached: list[TableInfo] | None = None
_cached_at: float = 0.0


def _ttl() -> int:
    return int(os.environ.get("SCHEMA_CACHE_TTL_SECONDS", DEFAULT_TTL_SECONDS))


def get_schema(engine: Engine) -> list[TableInfo]:
    global _cached, _cached_at

    ttl = _ttl()
    now = time.monotonic()

    with _lock:
        if _cached is not None and (now - _cached_at) < ttl:
            return _cached

    # Introspect outside the lock: it is slow and network-bound, and holding
    # the lock across it would serialise every concurrent request behind one
    # refresh. A race here just means two requests both refresh, which is
    # wasteful but correct — they compute identical results.
    fresh = introspect_schema(engine)

    with _lock:
        _cached = fresh
        _cached_at = time.monotonic()

    return fresh


def invalidate() -> None:
    global _cached, _cached_at

    with _lock:
        _cached = None
        _cached_at = 0.0
