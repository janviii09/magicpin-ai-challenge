"""
Vera 2.0 — HTTP Server
FastAPI server exposing 5 endpoints for the judge harness:
  POST /v1/context   — receive versioned context pushes
  POST /v1/tick      — periodic wake-up; bot decides what to send
  POST /v1/reply     — handle merchant/customer replies (multi-turn)
  GET  /v1/healthz   — liveness probe
  GET  /v1/metadata  — bot identity

All wired to the deterministic pipeline through bot.py.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from bot import compose, compose_from_plan
from core.context_store import get_store
from core.decision_engine import decide, prioritize_triggers
from core.models import (
    CategoryContext,
    ContextPushRequest,
    ContextPushResponse,
    ConversationPhase,
    HealthResponse,
    MerchantContext,
    MetadataResponse,
    ReplyRequest,
    ReplyResponse,
    TickAction,
    TickRequest,
    TickResponse,
    TriggerContext,
)
from core.planner import build_message_plan
from core.suppression import get_suppression_engine
from core.trigger_engine import resolve as resolve_trigger
from conversations.state_machine import get_conversation_manager
from llm.composer import compose_with_llm

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("vera.server")

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="Vera 2.0", version=config.VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_start_time = time.time()


# ── Startup: load dataset ────────────────────────────────────
@app.on_event("startup")
async def startup():
    store = get_store()
    try:
        store.load_from_dataset(config.DATASET_DIR)
        logger.info("Startup complete. Contexts loaded: %s", store.counts())
    except Exception as e:
        logger.warning("Dataset load failed (will accept pushes): %s", e)


# ═══════════════════════════════════════════════════════════════
#  POST /v1/context — receive context push
# ═══════════════════════════════════════════════════════════════

@app.post("/v1/context", response_model=ContextPushResponse)
async def receive_context(req: ContextPushRequest):
    store = get_store()

    # Validate scope
    if req.scope not in ("category", "merchant", "customer", "trigger"):
        return ContextPushResponse(
            accepted=False,
            reason="invalid_scope",
            details=f"Unknown scope: {req.scope}",
        )

    try:
        accepted, current_version = store.upsert(
            scope=req.scope,
            context_id=req.context_id,
            version=req.version,
            payload=req.payload,
        )
    except Exception as e:
        logger.error("Context upsert failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    if accepted:
        return ContextPushResponse(
            accepted=True,
            ack_id=f"ack_{uuid.uuid4().hex[:12]}",
            stored_at=datetime.now(timezone.utc).isoformat(),
            current_version=current_version,
        )
    else:
        # Version conflict (409)
        return ContextPushResponse(
            accepted=False,
            reason="stale_version",
            current_version=current_version,
        )


# ═══════════════════════════════════════════════════════════════
#  POST /v1/tick — periodic wake-up
# ═══════════════════════════════════════════════════════════════

@app.post("/v1/tick", response_model=TickResponse)
async def tick(req: TickRequest):
    store = get_store()
    suppression = get_suppression_engine()
    now_str = req.now
    try:
        now_dt = datetime.fromisoformat(now_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        now_dt = datetime.now(timezone.utc)

    actions: list[TickAction] = []

    # Resolve each available trigger
    candidates = []
    for trigger_id in req.available_triggers:
        trigger = store.get("trigger", trigger_id)
        if trigger is None:
            logger.warning("Trigger %s not found in store", trigger_id)
            continue

        merchant = store.get("merchant", trigger.merchant_id)
        if merchant is None:
            logger.warning("Merchant %s not found for trigger %s", trigger.merchant_id, trigger_id)
            continue

        category = store.get_category_for_merchant(trigger.merchant_id)
        if category is None:
            logger.warning("Category not found for merchant %s", trigger.merchant_id)
            continue

        # Suppression check
        if suppression.is_suppressed(
            trigger.merchant_id, trigger.suppression_key, trigger.kind, now_dt,
        ):
            logger.info("Trigger %s suppressed for merchant %s", trigger_id, trigger.merchant_id)
            continue

        # Resolve trigger
        customer = store.get_customer_for_trigger(trigger_id)
        resolution = resolve_trigger(trigger, merchant, category, customer)

        # Decision check
        decision = decide(trigger, resolution, merchant, category, customer)
        if not decision.should_send:
            logger.info("Decision: skip trigger %s — %s", trigger_id, decision.reason)
            continue

        candidates.append((trigger, resolution, merchant, category, customer, decision))

    # Prioritize — max 1 per merchant, top N overall
    if candidates:
        # Build (trigger, resolution) pairs for prioritization
        trigger_resolutions = [(c[0], c[1]) for c in candidates]
        prioritized = prioritize_triggers(trigger_resolutions, config.MAX_ACTIONS_PER_TICK)

        # Map back to full candidate tuples
        prioritized_ids = {t.id for t, r in prioritized}
        for trigger, resolution, merchant, category, customer, decision in candidates:
            if trigger.id not in prioritized_ids:
                continue

            # Build plan and compose
            plan = build_message_plan(trigger, resolution, decision, merchant, category, customer)

            customer_name = customer.identity.name if customer else ""
            message = compose_with_llm(plan, category, customer_name)
            message.suppression_key = trigger.suppression_key

            # Generate conversation ID
            conv_id = f"conv_{uuid.uuid4().hex[:8]}"

            # Record in suppression
            suppression.record_sent(
                trigger.merchant_id, trigger.suppression_key, now_dt,
            )

            # Create conversation state
            conv_manager = get_conversation_manager()
            conv_manager.create_conversation(
                conv_id, trigger.merchant_id, trigger.id,
                customer_id=trigger.customer_id,
            )
            conv_manager.record_bot_message(conv_id, message.body)

            actions.append(TickAction(
                conversation_id=conv_id,
                merchant_id=trigger.merchant_id,
                customer_id=trigger.customer_id,
                send_as=message.send_as,
                trigger_id=trigger.id,
                template_name=f"vera_{trigger.kind}_v1",
                template_params=[],
                body=message.body,
                cta=message.cta,
                suppression_key=message.suppression_key,
                rationale=message.rationale,
            ))

    logger.info("Tick processed: %d triggers → %d actions", len(req.available_triggers), len(actions))
    return TickResponse(actions=actions)


# ═══════════════════════════════════════════════════════════════
#  POST /v1/reply — handle merchant/customer replies
# ═══════════════════════════════════════════════════════════════

@app.post("/v1/reply", response_model=ReplyResponse)
async def reply(req: ReplyRequest):
    store = get_store()
    conv_manager = get_conversation_manager()

    # Process through conversation state machine
    state, action, reason = conv_manager.process_reply(
        conversation_id=req.conversation_id,
        from_role=req.from_role,
        message=req.message,
        turn_number=req.turn_number,
    )

    logger.info(
        "Reply processed: conv=%s, phase=%s, action=%s, reason=%s",
        req.conversation_id, state.phase.value, action, reason,
    )

    # ── END action ────────────────────────────────────────────
    if action == "end":
        # Compose a graceful exit message
        exit_body = _compose_exit_message(state, req.message, reason)
        return ReplyResponse(
            action="end",
            body=exit_body,
            rationale=reason,
        )

    # ── WAIT action ───────────────────────────────────────────
    if action == "wait":
        return ReplyResponse(
            action="wait",
            wait_seconds=1800,
            rationale=reason,
        )

    # ── SEND action — compose a contextual reply ──────────────
    merchant = store.get("merchant", req.merchant_id) or store.get("merchant", state.merchant_id)
    if merchant is None:
        return ReplyResponse(
            action="send",
            body="Thanks for your message! I'll look into this and get back to you.",
            cta="open_ended",
            rationale="Merchant not found in store — generic acknowledgment",
        )

    category = store.get_category_for_merchant(merchant.merchant_id)

    # Compose reply based on conversation phase
    reply_body = _compose_reply(state, req.message, merchant, category, reason)

    conv_manager.record_bot_message(req.conversation_id, reply_body)

    return ReplyResponse(
        action="send",
        body=reply_body,
        cta="open_ended",
        rationale=reason,
    )


def _compose_exit_message(state: ConversationState, last_msg: str, reason: str) -> str:
    """Compose a graceful exit message."""
    if "auto-reply" in reason.lower():
        return "No worries — I'll check back when the owner/manager is available. Best wishes! 🙂"
    elif "stop" in reason.lower() or "no" in reason.lower():
        return "Understood. I won't follow up on this. If you need anything later, just message me anytime."
    else:
        return "Thanks for your time! Reach out whenever you need help. 🙂"


def _compose_reply(
    state: ConversationState,
    merchant_message: str,
    merchant: MerchantContext,
    category: CategoryContext | None,
    reason: str,
) -> str:
    """
    Compose a contextual reply based on conversation state + merchant message.
    Uses the LLM when available, falls back to templates.
    """
    phase = state.phase

    # ── AUTO_REPLY_DETECTED: redirect to real person ──────────
    if phase == ConversationPhase.AUTO_REPLY_DETECTED:
        name = merchant.identity.owner_first_name or merchant.identity.name
        return (
            f"Samajh gayi — automated reply lag raha hai. "
            f"Kya {name} ya koi team member directly reply kar sakte hain? "
            f"2 minute ka kaam hai. 🙂"
        )

    # ── EXECUTE: action mode ──────────────────────────────────
    if phase == ConversationPhase.EXECUTE:
        return _compose_execute_reply(state, merchant_message, merchant)

    # ── CLARIFY: answer the question ──────────────────────────
    if phase == ConversationPhase.CLARIFY:
        return _compose_clarify_reply(state, merchant_message, merchant, category)

    # ── ENGAGED: continue conversation ────────────────────────
    if phase == ConversationPhase.ENGAGED:
        return _compose_engaged_reply(state, merchant_message, merchant, category)

    # ── Default ───────────────────────────────────────────────
    return "Thanks for your message! How can I help you further?"


def _compose_execute_reply(
    state: ConversationState,
    merchant_message: str,
    merchant: MerchantContext,
) -> str:
    """Reply when merchant has confirmed an action."""
    name = merchant.identity.owner_first_name or merchant.identity.name
    return (
        f"Done, {name}! Proceeding with this right now. "
        f"I will confirm the update here shortly. "
        f"Next, is there anything else you would like me to handle?"
    )


def _compose_clarify_reply(
    state: ConversationState,
    merchant_message: str,
    merchant: MerchantContext,
    category: CategoryContext | None,
) -> str:
    """Reply to a merchant's question with available context."""
    name = merchant.identity.owner_first_name or merchant.identity.name

    # Try to use LLM for a contextual answer
    # For now, provide a structured acknowledgment
    return (
        f"Good question, {name}! "
        f"Let me pull the relevant details for you — "
        f"I'll have an answer in a moment."
    )


def _compose_engaged_reply(
    state: ConversationState,
    merchant_message: str,
    merchant: MerchantContext,
    category: CategoryContext | None,
) -> str:
    """Continue an engaged conversation."""
    name = merchant.identity.owner_first_name or merchant.identity.name
    return (
        f"Got it, {name}. "
        f"Shall I go ahead and set this up for you?"
    )


# ═══════════════════════════════════════════════════════════════
#  GET /v1/healthz — liveness
# ═══════════════════════════════════════════════════════════════

@app.get("/v1/healthz", response_model=HealthResponse)
async def healthz():
    store = get_store()
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
        contexts_loaded=store.counts(),
    )


# ═══════════════════════════════════════════════════════════════
#  GET /v1/metadata — bot identity
# ═══════════════════════════════════════════════════════════════

@app.get("/v1/metadata", response_model=MetadataResponse)
async def metadata():
    return MetadataResponse(
        team_name=config.TEAM_NAME,
        team_members=config.TEAM_MEMBERS,
        model=config.get_model(),
        approach=(
            "Deterministic pipeline: trigger_engine → suppression → decision_engine "
            "→ planner (MessagePlan) → LLM composer (grounded) → validator (retry ≤2x) "
            "→ fallback template. LLM only decides phrasing, not content."
        ),
        contact_email=config.CONTACT_EMAIL,
        version=config.VERSION,
        submitted_at=datetime.now(timezone.utc).isoformat(),
    )


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
