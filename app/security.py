import hmac
import os

from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Optional shared-secret gate, driven entirely by the environment.

    Set API_KEY and every gated endpoint demands a matching X-API-Key header.
    Leave it unset and the endpoints are open.

    This is deliberately opt-in rather than fail-closed. A browser cannot hold a
    secret — anything the server sends to the page is readable in view-source —
    so a public demo UI and a key-protected API are mutually exclusive. Rather
    than fake it by shipping the key to the client, the deployment picks one:
    unset API_KEY for an open demo, set it for a genuinely protected API.

    The database is not what this protects. Read-only enforcement lives in the
    guardrail and executor layers and applies either way; API_KEY only controls
    who may spend LLM quota.
    """
    expected = os.environ.get("API_KEY")

    if not expected:
        return

    # compare_digest rather than == so the comparison doesn't leak the key's
    # length or matching prefix through timing.
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
