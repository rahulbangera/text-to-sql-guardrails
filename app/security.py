import hmac
import os

from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Gate endpoints behind a shared secret.

    This fails closed. /v1/query runs LLM-generated SQL against a real database
    and spends LLM quota per call, so an unset API_KEY is treated as a
    misconfiguration (503) rather than as "auth disabled" — the alternative
    silently serves an open endpoint the moment the variable goes missing.
    """
    expected = os.environ.get("API_KEY")

    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API_KEY is not configured on the server.",
        )

    # compare_digest rather than == so the comparison doesn't leak the key's
    # length or matching prefix through timing.
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
