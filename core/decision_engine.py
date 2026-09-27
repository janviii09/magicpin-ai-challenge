"""
Vera 2.0 — Decision Engine
Answers: should we send? To whom? In what language? What priority?

This layer sits between the trigger engine (which interprets what happened)
and the planner (which builds the MessagePlan). It applies business rules:
  - Don't message expired-subscription merchants about non-critical things
  - Don't message customers without consent
  - Pick the right language based on merchant/customer preferences
  - Prioritize when multiple triggers fire simultaneously
"""

from __future__ import annotations

import logging
from typing import Any

from core.models import (
    CategoryContext,
    CustomerContext,
    MerchantContext,
    TriggerContext,
)
from core.trigger_engine import TriggerResolution

logger = logging.getLogger("vera.decision_engine")


class DecisionResult:
    __slots__ = (
        "should_send", "reason", "audience", "language",
        "send_as", "priority",
    )

    def __init__(
        self,
        should_send: bool,
        reason: str = "",
        audience: str = "merchant",
        language: str = "hinglish",
        send_as: str = "vera",
        priority: int = 5,
    ):
        self.should_send = should_send
        self.reason = reason
        self.audience = audience
        self.language = language
        self.send_as = send_as
        self.priority = priority


def decide(
    trigger: TriggerContext,
    resolution: TriggerResolution,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None = None,
) -> DecisionResult:
    """
    Apply business rules to determine if we should send.
    Returns DecisionResult with should_send=False if blocked, else True with metadata.
    """

    # ── Rule 1: Don't message expired merchants on low-priority things ────
    if merchant.subscription.status == "expired":
        # Only allow high-urgency triggers (renewal, winback, critical alerts)
        allowed_for_expired = {
            "renewal_due", "winback_eligible", "supply_alert",
            "regulation_change", "dormant_with_vera",
        }
        if trigger.kind not in allowed_for_expired and resolution.priority < 7:
            return DecisionResult(
                should_send=False,
                reason=f"Subscription expired; trigger '{trigger.kind}' not critical enough",
            )

    # ── Rule 2: Customer consent check ────────────────────────
    if resolution.audience == "customer":
        if customer is None:
            return DecisionResult(
                should_send=False,
                reason="Customer-facing trigger but no customer context available",
            )
        if not customer.consent.opted_in_at:
            return DecisionResult(
                should_send=False,
                reason=f"Customer {customer.customer_id} has not opted in",
            )
        if not customer.preferences.reminder_opt_in:
            return DecisionResult(
                should_send=False,
                reason=f"Customer {customer.customer_id} has reminder opt-in disabled",
            )

    # ── Rule 3: Trigger expiry check ──────────────────────────
    # (We trust the judge to only send non-expired triggers, but check anyway)

    # ── Rule 4: Language detection ────────────────────────────
    language = _detect_language(merchant, customer, resolution.audience)

    # ── Rule 5: Send-as determination ─────────────────────────
    send_as = resolution.send_as  # from trigger engine

    return DecisionResult(
        should_send=True,
        audience=resolution.audience,
        language=language,
        send_as=send_as,
        priority=resolution.priority,
    )


def _detect_language(
    merchant: MerchantContext,
    customer: CustomerContext | None,
    audience: str,
) -> str:
    """
    Determine language based on audience's language preference.
    Returns: "hinglish" | "hindi" | "english"
    """
    if audience == "customer" and customer:
        pref = customer.identity.language_pref.lower()
    else:
        # Use merchant's language list
        langs = [l.lower() for l in merchant.identity.languages]
        if "hi" in langs and "en" in langs:
            pref = "hi-en mix"
        elif "hi" in langs:
            pref = "hi"
        elif "en" in langs:
            pref = "en"
        else:
            pref = "en"

    # Normalize to our three categories
    if "mix" in pref or ("hi" in pref and "en" in pref):
        return "hinglish"
    elif pref in ("hi", "hindi"):
        return "hindi"
    elif any(regional in pref for regional in ["te", "ta", "kn", "mr"]):
        # Regional languages → default to hinglish with regional flavor
        return "hinglish"
    else:
        return "english"


def prioritize_triggers(
    trigger_resolutions: list[tuple[TriggerContext, TriggerResolution]],
    max_actions: int = 3,
) -> list[tuple[TriggerContext, TriggerResolution]]:
    """
    Given multiple active triggers for the same tick, pick the top N by priority.
    Group by merchant_id to avoid bombarding a single merchant.
    """
    # Sort by priority (descending), then urgency
    sorted_triggers = sorted(
        trigger_resolutions,
        key=lambda x: (x[1].priority, x[0].urgency),
        reverse=True,
    )

    # Deduplicate: at most 1 trigger per merchant per tick
    seen_merchants: set[str] = set()
    result = []

    for trigger, resolution in sorted_triggers:
        if trigger.merchant_id in seen_merchants:
            continue
        seen_merchants.add(trigger.merchant_id)
        result.append((trigger, resolution))
        if len(result) >= max_actions:
            break

    return result
