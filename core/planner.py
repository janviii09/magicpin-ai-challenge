"""
Vera 2.0 — Message Planner
Builds the MessagePlan from TriggerResolution + DecisionResult + contexts.

The MessagePlan is THE contract between deterministic and generative code.
Everything in it is locked — the LLM only decides phrasing, not content.
"""

from __future__ import annotations

import logging

from core.models import (
    CategoryContext,
    CustomerContext,
    MerchantContext,
    MessagePlan,
    TriggerContext,
)
from core.trigger_engine import TriggerResolution
from core.decision_engine import DecisionResult

logger = logging.getLogger("vera.planner")


def build_message_plan(
    trigger: TriggerContext,
    resolution: TriggerResolution,
    decision: DecisionResult,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None = None,
) -> MessagePlan:
    """
    Assembles the MessagePlan from all deterministic components.
    This is the single most important function in the system.
    """

    # ── Voice (from category) ─────────────────────────────────
    voice = category.voice.tone  # e.g., "peer_clinical", "friendly_enthusiastic"

    # ── CTA selection ─────────────────────────────────────────
    # Pick the best CTA from allowed options based on trigger kind
    cta = _select_cta(resolution.allowed_ctas, trigger.kind, resolution.objective)

    # ── Required facts — validator will check these appear verbatim ──
    required_facts = _extract_required_facts(resolution.evidence, trigger)

    # ── Owner name & honorific for personalization ────────────────
    owner_first = merchant.identity.owner_first_name or merchant.identity.name
    if category.slug in ("dentists",) and not owner_first.lower().startswith("dr"):
        owner_first = f"Dr. {owner_first}"

    merchant_name = merchant.identity.name
    if merchant.identity.locality and merchant.identity.locality not in merchant_name:
        locality_context = f"{merchant_name} ({merchant.identity.locality})"
    else:
        locality_context = merchant_name

    # ── Audience-specific send_as ─────────────────────────────
    send_as = "merchant_on_behalf" if decision.audience == "customer" else "vera"

    plan = MessagePlan(
        audience=decision.audience,
        objective=resolution.objective,
        why_now=resolution.why_now,
        evidence=resolution.evidence,
        angle=resolution.angle,
        emotional_lever=resolution.emotional_lever,
        voice=voice,
        language=decision.language,
        cta=cta,
        required_facts=required_facts,
        merchant_name=merchant_name,
        owner_first_name=owner_first,
        send_as=send_as,
    )

    logger.info(
        "Built plan: audience=%s, objective=%s, angle=%s, cta=%s, lang=%s",
        plan.audience, plan.objective, plan.angle, plan.cta, plan.language,
    )

    return plan


def _select_cta(
    allowed: list[str],
    trigger_kind: str,
    objective: str,
) -> str:
    """
    Select the single best CTA from the allowed list.
    Prefer yes_no for action triggers, open_ended for insight triggers.
    """
    # Action-oriented triggers prefer binary CTA
    action_triggers = {
        "renewal_due", "recall_due", "winback_eligible",
        "gbp_unverified", "active_planning_intent",
        "chronic_refill_due", "supply_alert",
    }

    # Information-oriented triggers prefer open_ended or none
    info_triggers = {
        "research_digest", "cde_opportunity", "milestone_reached",
        "curious_ask_due",
    }

    if trigger_kind in action_triggers and "yes_no" in allowed:
        return "yes_no"
    elif trigger_kind in info_triggers:
        if "open_ended" in allowed:
            return "open_ended"
        elif "none" in allowed:
            return "none"

    # Default: prefer open_ended (more engagement-friendly)
    if "open_ended" in allowed:
        return "open_ended"
    elif "yes_no" in allowed:
        return "yes_no"
    elif "none" in allowed:
        return "none"

    return "open_ended"


def _extract_required_facts(
    evidence: list[str],
    trigger: TriggerContext,
) -> list[str]:
    """
    From the full evidence list, pick the facts that MUST appear verbatim
    in the composed message. The validator checks for these.

    We prioritize:
    - Numbers (verifiable)
    - Source citations
    - Dates/deadlines
    """
    required = []

    for fact in evidence:
        # Always require facts with numbers
        if any(char.isdigit() for char in fact):
            # Extract the key numeric part
            required.append(fact)
        # Always require source citations
        elif fact.startswith("Source:"):
            required.append(fact)

    # Limit to top 3 to avoid over-constraining the LLM
    return required[:3]
