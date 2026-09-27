"""
Vera 2.0 — Core Models
Pydantic models for contexts, MessagePlan, API contracts, and conversation state.
Every model maps 1:1 to the JSON shapes pushed by the judge harness.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
#  CATEGORY CONTEXT
# ═══════════════════════════════════════════════════════════════

class VoiceProfile(BaseModel):
    tone: str
    voice_register: str = Field("", alias="register")

    code_mix: str = ""
    vocab_allowed: list[str] = []
    vocab_taboo: list[str] = []
    salutation_examples: list[str] = []
    tone_examples: list[str] = []


class OfferTemplate(BaseModel):
    id: str
    title: str
    value: str = ""
    audience: str = ""
    type: str = ""


class PeerStats(BaseModel):
    scope: str = ""
    avg_rating: float = 0.0
    avg_review_count: int = 0
    avg_views_30d: int = 0
    avg_calls_30d: int = 0
    avg_directions_30d: int = 0
    avg_ctr: float = 0.0
    avg_photos: int = 0
    avg_post_freq_days: int = 0
    retention_6mo_pct: float = 0.0


class DigestItem(BaseModel):
    id: str
    kind: str                       # research | compliance | cde | trend | tech
    title: str
    source: str
    summary: str = ""
    actionable: str = ""
    trial_n: int | None = None
    patient_segment: str | None = None
    date: str | None = None
    credits: int | None = None


class ContentItem(BaseModel):
    id: str
    title: str
    channel: str = "whatsapp"
    length_seconds: int = 0
    body: str = ""


class SeasonalBeat(BaseModel):
    month_range: str
    note: str


class TrendSignal(BaseModel):
    query: str
    delta_yoy: float
    segment_age: str = ""
    skew: str = ""


class CategoryContext(BaseModel):
    slug: str
    display_name: str = ""
    voice: VoiceProfile
    offer_catalog: list[OfferTemplate] = []
    peer_stats: PeerStats = PeerStats()
    digest: list[DigestItem] = []
    patient_content_library: list[ContentItem] = []
    seasonal_beats: list[SeasonalBeat] = []
    trend_signals: list[TrendSignal] = []
    regulatory_authorities: list[str] = []
    professional_journals: list[str] = []


# ═══════════════════════════════════════════════════════════════
#  MERCHANT CONTEXT
# ═══════════════════════════════════════════════════════════════

class MerchantIdentity(BaseModel):
    name: str
    city: str = ""
    locality: str = ""
    place_id: str = ""
    verified: bool = False
    languages: list[str] = []
    owner_first_name: str = ""
    established_year: int | None = None


class Subscription(BaseModel):
    status: str             # active | expired | trial
    plan: str = ""
    days_remaining: int | None = None
    days_since_expiry: int | None = None
    renewed_at: str | None = None


class PerformanceSnapshot(BaseModel):
    window_days: int = 30
    views: int = 0
    calls: int = 0
    directions: int = 0
    ctr: float = 0.0
    leads: int = 0
    delta_7d: dict[str, float] = {}


class MerchantOffer(BaseModel):
    id: str
    title: str
    status: str             # active | expired | paused
    started: str | None = None
    ended: str | None = None


class ConversationTurn(BaseModel):
    ts: str
    from_: str = Field("", alias="from")
    body: str = ""
    engagement: str = ""

    model_config = {"populate_by_name": True}


class ReviewTheme(BaseModel):
    theme: str
    sentiment: str = ""
    occurrences_30d: int = 0
    common_quote: str = ""


class CustomerAggregate(BaseModel):
    total_unique_ytd: int = 0
    lapsed_180d_plus: int | None = None
    lapsed_90d_plus: int | None = None
    retention_6mo_pct: float | None = None
    retention_3mo_pct: float | None = None
    high_risk_adult_count: int | None = None


class MerchantContext(BaseModel):
    merchant_id: str
    category_slug: str
    identity: MerchantIdentity
    subscription: Subscription = Subscription(status="active")
    performance: PerformanceSnapshot = PerformanceSnapshot()
    offers: list[MerchantOffer] = []
    conversation_history: list[ConversationTurn] = []
    customer_aggregate: CustomerAggregate = CustomerAggregate()
    signals: list[str] = []
    review_themes: list[ReviewTheme] = []


# ═══════════════════════════════════════════════════════════════
#  CUSTOMER CONTEXT
# ═══════════════════════════════════════════════════════════════

class CustomerIdentity(BaseModel):
    name: str
    phone_redacted: str | None = None
    language_pref: str = "english"
    age_band: str = ""
    senior_citizen: bool = False


class Relationship(BaseModel):
    first_visit: str = ""
    last_visit: str = ""
    visits_total: int = 0
    services_received: list[str] = []
    lifetime_value: float = 0.0
    favourite_dish: str | None = None
    chronic_conditions: list[str] = []


class Preferences(BaseModel):
    preferred_slots: str = ""
    channel: str = "whatsapp"
    reminder_opt_in: bool = False
    delivery_address: str | None = None
    preferred_stylist: str | None = None
    wedding_date: str | None = None
    training_focus: str | None = None
    health_focus: str | None = None
    office_nearby: bool | None = None
    family_size: int | None = None
    household_size: int | None = None


class Consent(BaseModel):
    opted_in_at: str | None = None
    scope: list[str] = []


class CustomerContext(BaseModel):
    customer_id: str
    merchant_id: str
    identity: CustomerIdentity
    relationship: Relationship = Relationship()
    state: str = "new"              # new | active | lapsed_soft | lapsed_hard | churned
    preferences: Preferences = Preferences()
    consent: Consent = Consent()


# ═══════════════════════════════════════════════════════════════
#  TRIGGER CONTEXT
# ═══════════════════════════════════════════════════════════════

class TriggerContext(BaseModel):
    id: str
    scope: Literal["merchant", "customer"]
    kind: str
    source: Literal["external", "internal"]
    merchant_id: str
    customer_id: str | None = None
    payload: dict[str, Any] = {}
    urgency: int = 1                # 1-5
    suppression_key: str = ""
    expires_at: str = ""


# ═══════════════════════════════════════════════════════════════
#  MESSAGE PLAN — the contract between deterministic + generative
# ═══════════════════════════════════════════════════════════════

class MessagePlan(BaseModel):
    """
    The single most important object in the system.
    Every field maps to exactly one judge dimension:
      - why_now → Trigger Relevance
      - evidence → Specificity
      - angle + emotional_lever → Engagement Compulsion
      - voice → Category Fit
      - language → Merchant Fit
    """
    audience: Literal["merchant", "customer"]
    objective: str                          # e.g. "increase_gbp_engagement"
    why_now: str                            # trigger's literal justification
    evidence: list[str]                     # verifiable facts only
    angle: str                              # "diagnostic_insight", "peer_authority", ...
    emotional_lever: str                    # "curiosity", "loss_aversion", "social_proof"
    voice: str                              # from category policy
    language: Literal["hinglish", "hindi", "english"]
    cta: Literal["yes_no", "open_ended", "none"]
    required_facts: list[str]               # validator checks these appear verbatim
    merchant_name: str = ""                 # for template rendering
    owner_first_name: str = ""
    send_as: Literal["vera", "merchant_on_behalf"] = "vera"


# ═══════════════════════════════════════════════════════════════
#  COMPOSED MESSAGE — output of the pipeline
# ═══════════════════════════════════════════════════════════════

class ComposedMessage(BaseModel):
    body: str
    cta: str
    send_as: Literal["vera", "merchant_on_behalf"]
    suppression_key: str
    rationale: str


# ═══════════════════════════════════════════════════════════════
#  API CONTRACTS — request/response shapes for the HTTP layer
# ═══════════════════════════════════════════════════════════════

class ContextPushRequest(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str = ""


class ContextPushResponse(BaseModel):
    accepted: bool
    ack_id: str = ""
    stored_at: str = ""
    reason: str = ""
    details: str = ""
    current_version: int | None = None


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = []


class TickAction(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: str | None = None
    send_as: str
    trigger_id: str
    template_name: str = ""
    template_params: list[str] = []
    body: str
    cta: str
    suppression_key: str
    rationale: str


class TickResponse(BaseModel):
    actions: list[TickAction] = []


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str = ""
    turn_number: int = 0


class ReplyResponse(BaseModel):
    action: Literal["send", "wait", "end"]
    body: str = ""
    cta: str = ""
    rationale: str = ""
    wait_seconds: int | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    uptime_seconds: float = 0
    contexts_loaded: dict[str, int] = {}


class MetadataResponse(BaseModel):
    team_name: str
    team_members: list[str]
    model: str
    approach: str
    contact_email: str
    version: str
    submitted_at: str = ""


# ═══════════════════════════════════════════════════════════════
#  CONVERSATION STATE MACHINE
# ═══════════════════════════════════════════════════════════════

class ConversationPhase(str, Enum):
    IDLE = "idle"
    PROPOSED = "proposed"           # first message sent, waiting for reply
    ENGAGED = "engaged"             # merchant replied meaningfully
    EXECUTE = "execute"             # merchant said yes / gave intent
    BACKOFF = "backoff"             # merchant seems disengaged
    CLARIFY = "clarify"             # need more info from merchant
    AUTO_REPLY_DETECTED = "auto_reply_detected"
    END = "end"                     # conversation closed


class ConversationState(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: str | None = None
    phase: ConversationPhase = ConversationPhase.IDLE
    turns: list[dict[str, Any]] = []
    trigger_id: str = ""
    auto_reply_count: int = 0
    unanswered_nudge_count: int = 0
    created_at: str = ""
    last_activity_at: str = ""
