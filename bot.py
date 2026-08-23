import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

APP_VERSION = "0.1.1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("vera_bot")


class FlexibleModel(BaseModel):
    """
    Base model that allows extra fields.
    This ensures we don't lose category-specific data (like 'delivery_orders_30d')
    when dumping the model back to a dict for the LLM.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# Flexible Context Schemas
# --------------------------------------------------------------------------- #


class CategoryContext(FlexibleModel):
    slug: str
    offer_catalog: list[dict[str, Any]] = Field(default_factory=list)
    voice: dict[str, Any] = Field(default_factory=dict)
    peer_stats: dict[str, Any] = Field(default_factory=dict)
    digest: list[dict[str, Any]] = Field(default_factory=list)


class MerchantContext(FlexibleModel):
    merchant_id: str
    category_slug: str
    identity: dict[str, Any]
    subscription: dict[str, Any]
    performance: dict[str, Any]
    offers: list[dict[str, Any]] = Field(default_factory=list)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    customer_aggregate: dict[str, Any] = Field(default_factory=dict)
    signals: list[str] = Field(default_factory=list)


class TriggerContext(FlexibleModel):
    id: str
    scope: Literal["merchant", "customer"]
    kind: str
    source: Literal["external", "internal"]
    merchant_id: str
    customer_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    urgency: int
    suppression_key: str
    expires_at: str


class CustomerContext(FlexibleModel):
    customer_id: str
    merchant_id: str
    identity: dict[str, Any]
    relationship: dict[str, Any]
    state: str
    preferences: dict[str, Any]
    consent: dict[str, Any]


SCOPE_MODELS: dict[str, type[FlexibleModel]] = {
    "category": CategoryContext,
    "merchant": MerchantContext,
    "trigger": TriggerContext,
    "customer": CustomerContext,
}


# --------------------------------------------------------------------------- #
# Request / Response Schemas
# --------------------------------------------------------------------------- #


class ContextPushRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str | None = None


class ContextAcceptedResponse(BaseModel):
    accepted: Literal[True] = True
    ack_id: str
    stored_at: str


class ContextRejectedResponse(BaseModel):
    accepted: Literal[False] = False
    reason: str
    current_version: int | None = None
    details: str | None = None


# --------------------------------------------------------------------------- #
# App + State Management
# --------------------------------------------------------------------------- #

app = FastAPI(title="Vera Challenge Bot", version=APP_VERSION)

_START_TIME = time.monotonic()
CONTEXT_STORE: dict[tuple[str, str], dict[str, Any]] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@app.get("/v1/healthz")
async def healthz() -> dict[str, Any]:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for scope, _context_id in CONTEXT_STORE:
        counts[scope] = counts.get(scope, 0) + 1

    return {
        "status": "ok",
        "uptime_seconds": int(time.monotonic() - _START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata() -> dict[str, Any]:
    return {
        "team_name": "Team Sriyash",
        "team_members": ["Sriyash"],
        "model": "qwen2:7b (via Ollama)",
        "approach": "Adapter-first Phase 1 with highly flexible schema validation.",
        "version": app.version,
        "submitted_at": _utc_now_iso(),
    }


@app.post("/v1/context")
async def push_context(request: ContextPushRequest) -> JSONResponse:
    key = (request.scope, request.context_id)
    existing = CONTEXT_STORE.get(key)

    # 1. Idempotency Check: Reject stale versions
    if existing is not None and request.version < existing["version"]:
        body = ContextRejectedResponse(
            reason="stale_version",
            current_version=existing["version"],
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=body.model_dump(exclude_none=True),
        )

    # 2. Scope Validation
    model_cls = SCOPE_MODELS.get(request.scope)
    if not model_cls:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": "invalid_scope"},
        )

    # 3. Payload Validation
    try:
        validated = model_cls.model_validate(request.payload)
    except ValidationError as exc:
        logger.warning(
            "context push rejected (invalid_payload) | scope=%s context_id=%s",
            request.scope,
            request.context_id,
        )
        body = ContextRejectedResponse(
            reason="invalid_payload",
            details=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=body.model_dump(exclude_none=True),
        )

    # 4. Idempotency Check: Same version is a no-op 200 OK
    if existing is not None and request.version == existing["version"]:
        ack_id = f"ack_{request.context_id}_v{request.version}_noop"
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"accepted": True, "ack_id": ack_id, "stored_at": _utc_now_iso()},
        )

    # 5. Store Data
    CONTEXT_STORE[key] = {
        "version": request.version,
        "payload": validated.model_dump(by_alias=True),
    }

    ack_id = f"ack_{request.context_id}_v{request.version}_{uuid.uuid4().hex[:8]}"
    body = ContextAcceptedResponse(ack_id=ack_id, stored_at=_utc_now_iso())

    return JSONResponse(status_code=status.HTTP_200_OK, content=body.model_dump())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
