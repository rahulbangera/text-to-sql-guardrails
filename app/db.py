import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool


def _normalize_url(url: str) -> str:
    """Force the psycopg3 driver.

    Managed Postgres providers hand out plain `postgresql://` (and Heroku-style
    `postgres://`) URLs. SQLAlchemy would resolve those to psycopg2, which
    isn't installed. Rewriting here means a connection string can be pasted
    straight from the provider's dashboard without editing.
    """
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://"):
        if url.startswith(prefix):
            return url

    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]

    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Build the engine on first use, not at import time.

    Importing this module used to raise KeyError on a missing DATABASE_URL,
    which in a serverless deployment surfaces as an opaque function crash
    before any handler runs. Deferring the lookup means a misconfigured
    environment produces a readable error instead.
    """
    url = os.environ.get("DATABASE_URL")

    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Point it at your Postgres instance, e.g. "
            "postgresql://user:pass@host/dbname?sslmode=require"
        )

    return create_engine(
        _normalize_url(url),
        # NullPool, not a sized pool. Each serverless instance would otherwise
        # hold pool_size + max_overflow connections open for its whole lifetime,
        # and enough concurrent instances will exhaust the database's connection
        # limit. Connection reuse belongs to the provider's pooler here, not to
        # this process.
        poolclass=NullPool,
        connect_args={"connect_timeout": 10},
    )
