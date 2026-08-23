"""
bot.py — magicpin Vera AI Challenge submission (Phase 1 scaffolding)

Implements:
  - Strict Pydantic v2 schemas for the 4-context framework
    (CategoryContext, MerchantContext, TriggerContext, CustomerContext)
  - FastAPI app + in-memory context store
  - GET  /v1/healthz
  - GET  /v1/metadata
  - POST /v1/context   (idempotent context ingestion)

Deliberately NOT implemented yet (per spec): /v1/tick, /v1/reply.

Run locally:
    uvicorn bot:app --host 0.0.0.0 --port 8080

Reference:
    challenge-brief.md          §4  (the 4-context framework)
    challenge-testing-brief.md  §2.1, §3 (wire schemas + ingestion contract)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("vera_bot")


# --------------------------------------------------------------------------- #
# Base model config
# --------------------------------------------------------------------------- #


class StrictModel(BaseModel):
    """
    Shared base for every context schema.

    - `extra="forbid"` catches typos / schema drift from the judge harness
      (or from our own composer code) immediately, rather than silently
      dropping fields.
    - `str_strip_whitespace=True` keeps ids/keys clean for dict lookups
      (e.g. suppression_key, context_id) without surprising callers.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# 1. CategoryContext  (challenge-testing-brief.md §3.1)
# --------------------------------------------------------------------------- #


class OfferCatalogItem(StrictModel):
    title: str
    value: str
    audience: str


class VoiceProfile(StrictModel):
    tone: str
    vocab_allowed: list[str] = Field(default_factory=list)
    taboos: list[str] = Field(default_factory=list)


class PeerStats(StrictModel):
    avg_rating: float
    avg_reviews: float
    avg_ctr: float
    scope: str


class DigestItem(StrictModel):
    id: str
    kind: str
    title: str
    source: str
    trial_n: int | None = None
    patient_segment: str | None = None
    summary: str


class PatientContentItem(StrictModel):
    id: str
    title: str
    channel: str
    body: str


class SeasonalBeat(StrictModel):
    month_range: str
    note: str


class TrendSignal(StrictModel):
    query: str
    delta_yoy: float
    segment_age: str


class CategoryContext(StrictModel):
    """Slow-changing knowledge pack shared across all merchants in a vertical."""

    slug: str
    offer_catalog: list[OfferCatalogItem] = Field(default_factory=list)
    voice: VoiceProfile
    peer_stats: PeerStats
    digest: list[DigestItem] = Field(default_factory=list)
    patient_content_library: list[PatientContentItem] = Field(default_factory=list)
    seasonal_beats: list[SeasonalBeat] = Field(default_factory=list)
    trend_signals: list[TrendSignal] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 2. MerchantContext  (challenge-testing-brief.md §3.2)
# --------------------------------------------------------------------------- #


class Identity(StrictModel):
    name: str
    city: str
    locality: str
    place_id: str | None = None
    verified: bool
    languages: list[str] = Field(default_factory=list)


class Subscription(StrictModel):
    status: str
    plan: str
    days_remaining: int


class Delta7d(StrictModel):
    views_pct: float
    calls_pct: float


class PerformanceSnapshot(StrictModel):
    window_days: int
    views: int
    calls: int
    directions: int
    ctr: float
    delta_7d: Delta7d


class MerchantOffer(StrictModel):
    id: str
    title: str
    status: str


class ConversationHistoryItem(StrictModel):
    ts: str
    from_: str = Field(alias="from")
    body: str
    engagement: str | None = None

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, populate_by_name=True
    )


class CustomerAggregate(StrictModel):
    total_unique_ytd: int
    lapsed_180d_plus: int
    retention_6mo_pct: float


class MerchantContext(StrictModel):
    """The specific business's current state."""

    merchant_id: str
    category_slug: str
    identity: Identity
    subscription: Subscription
    performance: PerformanceSnapshot
    offers: list[MerchantOffer] = Field(default_factory=list)
    conversation_history: list[ConversationHistoryItem] = Field(default_factory=list)
    customer_aggregate: CustomerAggregate
    signals: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 3. TriggerContext  (challenge-testing-brief.md §3.4)
# --------------------------------------------------------------------------- #


class TriggerContext(StrictModel):
    """The event that prompts a message right now. Every message needs one."""

    id: str
    scope: Literal["merchant", "customer"]
    kind: str
    source: Literal["external", "internal"]
    merchant_id: str
    customer_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    urgency: int = Field(ge=1, le=5)
    suppression_key: str
    expires_at: str


# --------------------------------------------------------------------------- #
# 4. CustomerContext  (challenge-testing-brief.md §3.3)
# --------------------------------------------------------------------------- #


class CustomerIdentity(StrictModel):
    name: str
    phone_redacted: str
    language_pref: str


class Relationship(StrictModel):
    first_visit: str
    last_visit: str
    visits_total: int
    services_received: list[str] = Field(default_factory=list)


class CustomerPreferences(StrictModel):
    preferred_slots: str
    channel: str


class Consent(StrictModel):
    opted_in_at: str
    scope: list[str] = Field(default_factory=list)


class CustomerContext(StrictModel):
    """The customer asking about (or being engaged about) a merchant."""

    customer_id: str
    merchant_id: str
    identity: CustomerIdentity
    relationship: Relationship
    state: str
    preferences: CustomerPreferences
    consent: Consent


# --------------------------------------------------------------------------- #
# Scope -> schema registry
# --------------------------------------------------------------------------- #

SCOPE_MODELS: dict[str, type[StrictModel]] = {
    "category": CategoryContext,
    "merchant": MerchantContext,
    "trigger": TriggerContext,
    "customer": CustomerContext,
}


# --------------------------------------------------------------------------- #
# Wire-level request/response models for POST /v1/context
# --------------------------------------------------------------------------- #


class ContextPushRequest(BaseModel):
    """Envelope the judge harness posts to /v1/context (testing-brief §2.1)."""

    model_config = ConfigDict(extra="forbid")

    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int = Field(ge=0)
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
# App + in-memory state
# --------------------------------------------------------------------------- #

app = FastAPI(title="Vera Challenge Bot", version="0.1.0")

_START_TIME = time.monotonic()

# Keyed by (scope, context_id) -> {"version": int, "payload": dict}
# Values are the *validated* payload, dumped back to a plain dict, so
# downstream composer code (Phase 2+) can rely on it already being well-formed.
CONTEXT_STORE: dict[tuple[str, str], dict[str, Any]] = {}


def _utc_now_iso() -> str:
    """RFC3339 / ISO8601 UTC timestamp with millisecond precision, 'Z' suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# --------------------------------------------------------------------------- #
# GET /v1/healthz
# --------------------------------------------------------------------------- #


@app.get("/v1/healthz")
async def healthz() -> dict[str, Any]:
    """
    Liveness probe. Polled every 60s by the judge during the test window;
    three consecutive non-200s disqualify the bot for that slot, so this
    handler must stay dependency-free and never raise.
    """
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for scope, _context_id in CONTEXT_STORE:
        counts[scope] = counts.get(scope, 0) + 1

    uptime_seconds = int(time.monotonic() - _START_TIME)
    logger.debug("healthz check | uptime=%ss contexts=%s", uptime_seconds, counts)

    return {
        "status": "ok",
        "uptime_seconds": uptime_seconds,
        "contexts_loaded": counts,
    }


# --------------------------------------------------------------------------- #
# GET /v1/metadata
# --------------------------------------------------------------------------- #


@app.get("/v1/metadata")
async def metadata() -> dict[str, Any]:
    """
    Bot identity, returned as-is to the judge for the results report.
    Fill in real values before submitting.
    """
    return {
        "team_name": "Team Sriyash",
        "team_members": ["Sriyash"],
        "model": "claude-sonnet-5",
        "approach": (
            "Adapter-first Phase 1: strict typed contexts ingested via "
            "idempotent versioned pushes; composition (tick/reply) layered "
            "on top in Phase 2 with a single versioned prompt template per "
            "trigger kind."
        ),
        "contact_email": "team@example.com",
        "version": app.version,
        "submitted_at": _utc_now_iso(),
    }


# --------------------------------------------------------------------------- #
# POST /v1/context
# --------------------------------------------------------------------------- #


@app.post("/v1/context")
async def push_context(request: ContextPushRequest) -> JSONResponse:
    """
    Ingest a context push from the judge harness.

    Idempotency contract (testing-brief §2.1):
      - Keyed by (scope, context_id).
      - incoming.version <= stored.version  -> 409, stale_version, no mutation.
      - incoming.version >  stored.version  -> validate against the scope's
        schema, store, 200 with an ack.
      - payload fails schema validation      -> 400, invalid_payload.
    """
    key = (request.scope, request.context_id)
    existing = CONTEXT_STORE.get(key)

    if existing is not None and request.version <= existing["version"]:
        logger.info(
            "context push rejected (stale_version) | scope=%s context_id=%s "
            "incoming_version=%s current_version=%s",
            request.scope,
            request.context_id,
            request.version,
            existing["version"],
        )
        body = ContextRejectedResponse(
            reason="stale_version",
            current_version=existing["version"],
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=body.model_dump(exclude_none=True),
        )

    model_cls = SCOPE_MODELS[request.scope]
    try:
        validated = model_cls.model_validate(request.payload)
    except ValidationError as exc:
        logger.warning(
            "context push rejected (invalid_payload) | scope=%s context_id=%s "
            "version=%s errors=%s",
            request.scope,
            request.context_id,
            request.version,
            exc.errors(),
        )
        body = ContextRejectedResponse(
            reason="invalid_payload",
            details=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=body.model_dump(exclude_none=True),
        )

    CONTEXT_STORE[key] = {
        "version": request.version,
        "payload": validated.model_dump(by_alias=True),
    }

    ack_id = f"ack_{request.context_id}_v{request.version}_{uuid.uuid4().hex[:8]}"
    stored_at = _utc_now_iso()

    logger.info(
        "context push accepted | scope=%s context_id=%s version=%s ack_id=%s",
        request.scope,
        request.context_id,
        request.version,
        ack_id,
    )

    body = ContextAcceptedResponse(ack_id=ack_id, stored_at=stored_at)
    return JSONResponse(status_code=status.HTTP_200_OK, content=body.model_dump())


# --------------------------------------------------------------------------- #
# Entrypoint (for `python bot.py` convenience; prefer `uvicorn bot:app`)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
