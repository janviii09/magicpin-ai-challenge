"""
Vera 2.0 — Bot Core
The public compose() function = the full pipeline.

Pipeline:
  Context Store → Trigger Engine → Suppression Check → Decision Engine
  → Fact Extraction → Message Planner → Composition Engine (LLM)
  → Validator (retry ≤2x) → Fallback Template → ComposedMessage
"""

from __future__ import annotations

import logging
from typing import Any

from core.models import (
    CategoryContext,
    ComposedMessage,
    CustomerContext,
    MerchantContext,
    MessagePlan,
    TriggerContext,
)
from core.trigger_engine import resolve as resolve_trigger
from core.decision_engine import decide, DecisionResult
from core.planner import build_message_plan
from core.validator import validate
from llm.composer import compose_with_llm

logger = logging.getLogger("vera.bot")


def compose(
    category: dict | CategoryContext,
    merchant: dict | MerchantContext,
    trigger: dict | TriggerContext,
    customer: dict | CustomerContext | None = None,
) -> dict:
    """
    Public API — matches the challenge contract.

    Inputs: dicts loaded from dataset JSON (or pre-parsed Pydantic models).
    Returns: dict with keys: body, cta, send_as, suppression_key, rationale.

    Must complete in < 30s.
    """

    # ── Parse inputs if raw dicts ─────────────────────────────
    cat_ctx = _ensure_model(category, CategoryContext)
    mer_ctx = _ensure_model(merchant, MerchantContext)
    trg_ctx = _ensure_model(trigger, TriggerContext)
    cus_ctx = _ensure_model(customer, CustomerContext) if customer else None

    # ── Step 1: Trigger Engine — interpret the trigger ────────
    resolution = resolve_trigger(trg_ctx, mer_ctx, cat_ctx, cus_ctx)
    logger.info(
        "Trigger resolved: kind=%s, objective=%s, angle=%s, priority=%d",
        trg_ctx.kind, resolution.objective, resolution.angle, resolution.priority,
    )

    # ── Step 2: Decision Engine — should we send? ─────────────
    decision = decide(trg_ctx, resolution, mer_ctx, cat_ctx, cus_ctx)
    if not decision.should_send:
        logger.info("Decision: DO NOT SEND — %s", decision.reason)
        return {
            "body": "",
            "cta": "none",
            "send_as": "vera",
            "suppression_key": trg_ctx.suppression_key,
            "rationale": f"Suppressed: {decision.reason}",
        }

    # ── Step 3: Build MessagePlan ─────────────────────────────
    plan = build_message_plan(
        trg_ctx, resolution, decision, mer_ctx, cat_ctx, cus_ctx,
    )

    # ── Step 4: Compose (LLM with retry + fallback) ───────────
    customer_name = ""
    if cus_ctx:
        customer_name = cus_ctx.identity.name

    message = compose_with_llm(plan, cat_ctx, customer_name)

    # ── Step 5: Set suppression key ───────────────────────────
    message.suppression_key = trg_ctx.suppression_key

    # ── Return as dict (challenge contract) ───────────────────
    return {
        "body": message.body,
        "cta": message.cta,
        "send_as": message.send_as,
        "suppression_key": message.suppression_key,
        "rationale": message.rationale,
    }


def compose_from_plan(
    plan: MessagePlan,
    category: CategoryContext,
    suppression_key: str,
    customer_name: str = "",
) -> ComposedMessage:
    """
    Lower-level API for when the server has already built the plan.
    Used by /v1/tick and /v1/reply handlers.
    """
    message = compose_with_llm(plan, category, customer_name)
    message.suppression_key = suppression_key
    return message


def _ensure_model(data: Any, model_class: type) -> Any:
    """Convert dict to Pydantic model if needed."""
    if isinstance(data, dict):
        return model_class.model_validate(data)
    return data
