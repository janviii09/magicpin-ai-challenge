"""
Vera 2.0 — Trigger Engine
Maps trigger_kind → {objective, angle_options, allowed_ctas, evidence extraction rules}.
This is the deterministic "what to say" layer — NOT how to say it.

Each trigger kind has a resolution function that:
  1. Reads the trigger payload + merchant/category context
  2. Returns a TriggerResolution with pre-determined objective, angle, evidence, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.models import (
    CategoryContext,
    CustomerContext,
    MerchantContext,
    TriggerContext,
)

logger = logging.getLogger("vera.trigger_engine")


@dataclass
class TriggerResolution:
    """The deterministic output of trigger interpretation."""
    objective: str                          # what we want to achieve
    why_now: str                            # literal justification for messaging now
    evidence: list[str]                     # verifiable facts the LLM must include
    angle: str                              # framing approach
    emotional_lever: str                    # primary compulsion lever
    allowed_ctas: list[str]                 # valid CTA types for this trigger
    audience: str = "merchant"              # merchant | customer
    send_as: str = "vera"                   # vera | merchant_on_behalf
    priority: int = 5                       # 1-10, higher = more important


def resolve(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None = None,
) -> TriggerResolution:
    """
    Main dispatch: trigger.kind → resolution function.
    Each resolver is a pure function that extracts evidence from the contexts.
    """
    resolver = _RESOLVERS.get(trigger.kind, _resolve_generic)
    return resolver(trigger, merchant, category, customer)


# ═══════════════════════════════════════════════════════════════
#  RESOLVER FUNCTIONS — one per trigger kind
# ═══════════════════════════════════════════════════════════════

def _resolve_research_digest(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    """External research digest — share relevant finding with merchant."""
    payload = trigger.payload
    top_item_id = payload.get("top_item_id", "")

    # Find the digest item in category
    digest_item = None
    for d in category.digest:
        if d.id == top_item_id:
            digest_item = d
            break

    evidence = []
    why_now = "New research digest released for your category"

    if digest_item:
        why_now = f"{digest_item.source} published: \"{digest_item.title}\""
        evidence.append(digest_item.title)
        if digest_item.trial_n:
            evidence.append(f"{digest_item.trial_n:,}-patient trial")
        if digest_item.summary:
            # Extract key numbers from summary
            evidence.append(digest_item.summary)
        evidence.append(f"Source: {digest_item.source}")

    # Merchant-specific relevance
    if "high_risk_adult_cohort" in merchant.signals and digest_item and digest_item.patient_segment == "high_risk_adults":
        evidence.append(f"Relevant to your {merchant.customer_aggregate.high_risk_adult_count or 'high-risk adult'} patients")

    return TriggerResolution(
        objective="share_research_insight",
        why_now=why_now,
        evidence=evidence,
        angle="peer_authority",
        emotional_lever="curiosity",
        allowed_ctas=["open_ended", "none"],
        priority=3,
    )


def _resolve_regulation_change(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    top_item_id = payload.get("top_item_id", "")
    deadline = payload.get("deadline_iso", "")

    digest_item = None
    for d in category.digest:
        if d.id == top_item_id:
            digest_item = d
            break

    evidence = []
    why_now = "Regulatory change affecting your practice"

    if digest_item:
        why_now = f"Compliance deadline: {digest_item.title}"
        evidence.append(digest_item.title)
        evidence.append(f"Source: {digest_item.source}")
        if digest_item.actionable:
            evidence.append(f"Action needed: {digest_item.actionable}")
    if deadline:
        evidence.append(f"Deadline: {deadline}")

    return TriggerResolution(
        objective="ensure_compliance_awareness",
        why_now=why_now,
        evidence=evidence,
        angle="compliance_alert",
        emotional_lever="loss_aversion",
        allowed_ctas=["open_ended", "yes_no"],
        priority=8,
    )


def _resolve_recall_due(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    """Customer-facing: recall reminder sent on behalf of merchant."""
    payload = trigger.payload
    service_due = payload.get("service_due", "appointment")
    last_service = payload.get("last_service_date", "")
    slots = payload.get("available_slots", [])

    evidence = []
    if last_service:
        evidence.append(f"Last visit: {last_service}")
    evidence.append(f"Service due: {service_due.replace('_', ' ')}")
    for slot in slots:
        evidence.append(f"Available: {slot.get('label', slot.get('iso', ''))}")

    # Add merchant offer if relevant
    active_offers = [o for o in merchant.offers if o.status == "active"]
    for offer in active_offers:
        if any(kw in offer.title.lower() for kw in ["cleaning", "checkup", "consultation"]):
            evidence.append(f"Current offer: {offer.title}")
            break

    customer_name = customer.identity.name if customer else "Customer"

    return TriggerResolution(
        objective="drive_recall_booking",
        why_now=f"{customer_name}'s {service_due.replace('_', ' ')} window is due",
        evidence=evidence,
        angle="care_continuity",
        emotional_lever="reciprocity",
        allowed_ctas=["open_ended"],
        audience="customer",
        send_as="merchant_on_behalf",
        priority=6,
    )


def _resolve_perf_dip(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    metric = payload.get("metric", "views")
    delta_pct = payload.get("delta_pct", 0)
    window = payload.get("window", "7d")
    vs_baseline = payload.get("vs_baseline", 0)

    evidence = [
        f"{metric.capitalize()} dropped {abs(delta_pct)*100:.0f}% over {window}",
    ]
    if vs_baseline:
        evidence.append(f"Previous baseline: {vs_baseline} {metric}/week")

    # Add peer comparison
    peer_val = getattr(category.peer_stats, f"avg_{metric}_30d", None)
    merchant_val = getattr(merchant.performance, metric, None)
    if peer_val and merchant_val:
        evidence.append(f"Your {metric}: {merchant_val} vs peer avg: {peer_val}")

    return TriggerResolution(
        objective="diagnose_and_recover_performance",
        why_now=f"Your {metric} dropped {abs(delta_pct)*100:.0f}% this {window}",
        evidence=evidence,
        angle="diagnostic_insight",
        emotional_lever="loss_aversion",
        allowed_ctas=["open_ended", "yes_no"],
        priority=7,
    )


def _resolve_perf_spike(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    metric = payload.get("metric", "views")
    delta_pct = payload.get("delta_pct", 0)
    likely_driver = payload.get("likely_driver", "")

    evidence = [
        f"{metric.capitalize()} up {delta_pct*100:.0f}% this week",
    ]
    if likely_driver:
        evidence.append(f"Likely driver: {likely_driver.replace('_', ' ')}")

    return TriggerResolution(
        objective="capitalize_on_momentum",
        why_now=f"Your {metric} are up {delta_pct*100:.0f}% — good time to double down",
        evidence=evidence,
        angle="momentum_amplifier",
        emotional_lever="reciprocity",
        allowed_ctas=["open_ended", "yes_no"],
        priority=4,
    )


def _resolve_renewal_due(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    days_remaining = payload.get("days_remaining", 0)
    plan = payload.get("plan", "")
    amount = payload.get("renewal_amount", 0)

    evidence = [
        f"Plan: {plan}, expires in {days_remaining} days",
    ]
    if amount:
        evidence.append(f"Renewal: ₹{amount:,}")

    # Add value-proof from performance
    perf = merchant.performance
    if perf.views > 0:
        evidence.append(f"Your profile got {perf.views:,} views this month")
    if perf.calls > 0:
        evidence.append(f"{perf.calls} calls generated")

    return TriggerResolution(
        objective="drive_renewal",
        why_now=f"Subscription expires in {days_remaining} days",
        evidence=evidence,
        angle="value_proof",
        emotional_lever="loss_aversion",
        allowed_ctas=["yes_no", "open_ended"],
        priority=8,
    )


def _resolve_festival_upcoming(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    festival = payload.get("festival", "")
    days_until = payload.get("days_until", 0)

    evidence = [f"{festival} in {days_until} days"]

    # Suggest relevant offer from catalog
    for offer in category.offer_catalog:
        if offer.audience == "new_user":
            evidence.append(f"Suggested seasonal offer: {offer.title}")
            break

    return TriggerResolution(
        objective="drive_seasonal_campaign",
        why_now=f"{festival} is {days_until} days away",
        evidence=evidence,
        angle="seasonal_opportunity",
        emotional_lever="social_proof",
        allowed_ctas=["open_ended", "yes_no"],
        priority=3 if days_until > 30 else 6,
    )


def _resolve_competitor_opened(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    comp_name = payload.get("competitor_name", "a new competitor")
    distance = payload.get("distance_km", 0)
    their_offer = payload.get("their_offer", "")

    evidence = [
        f"New competitor: {comp_name}, {distance}km away",
    ]
    if their_offer:
        evidence.append(f"Their offer: {their_offer}")

    # Add merchant's own positioning
    active_offers = [o.title for o in merchant.offers if o.status == "active"]
    if active_offers:
        evidence.append(f"Your active offer: {active_offers[0]}")

    # Add merchant's strengths
    for theme in merchant.review_themes:
        if theme.sentiment == "pos":
            evidence.append(f"Your strength: {theme.common_quote}")
            break

    return TriggerResolution(
        objective="strengthen_competitive_positioning",
        why_now=f"{comp_name} opened {distance}km from you",
        evidence=evidence,
        angle="competitive_awareness",
        emotional_lever="loss_aversion",
        allowed_ctas=["open_ended"],
        priority=5,
    )


def _resolve_review_theme_emerged(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    theme = payload.get("theme", "")
    occurrences = payload.get("occurrences_30d", 0)
    trend = payload.get("trend", "")
    quote = payload.get("common_quote", "")

    evidence = [
        f"Review theme: \"{theme.replace('_', ' ')}\" mentioned {occurrences}x this month",
    ]
    if trend:
        evidence.append(f"Trend: {trend}")
    if quote:
        evidence.append(f"Example: \"{quote}\"")

    return TriggerResolution(
        objective="address_review_pattern",
        why_now=f"\"{theme.replace('_', ' ')}\" appeared {occurrences}x in recent reviews",
        evidence=evidence,
        angle="diagnostic_insight",
        emotional_lever="loss_aversion",
        allowed_ctas=["open_ended"],
        priority=6,
    )


def _resolve_milestone_reached(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    metric = payload.get("metric", "")
    value_now = payload.get("value_now", 0)
    milestone = payload.get("milestone_value", 0)
    is_imminent = payload.get("is_imminent", False)

    if is_imminent:
        why_now = f"You're at {value_now} {metric.replace('_', ' ')} — just {milestone - value_now} away from {milestone}!"
    else:
        why_now = f"Crossed {milestone} {metric.replace('_', ' ')}!"

    evidence = [
        f"Current {metric.replace('_', ' ')}: {value_now}",
        f"Milestone: {milestone}",
    ]

    return TriggerResolution(
        objective="celebrate_and_capitalize",
        why_now=why_now,
        evidence=evidence,
        angle="momentum_amplifier",
        emotional_lever="social_proof",
        allowed_ctas=["open_ended", "none"],
        priority=3,
    )


def _resolve_curious_ask(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    ask_template = payload.get("ask_template", "general_check_in")

    return TriggerResolution(
        objective="gather_merchant_insight",
        why_now="Weekly check-in to understand your business better",
        evidence=[f"Ask type: {ask_template.replace('_', ' ')}"],
        angle="curious_ask",
        emotional_lever="reciprocity",
        allowed_ctas=["open_ended"],
        priority=2,
    )


def _resolve_active_planning_intent(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    topic = payload.get("intent_topic", "")
    last_msg = payload.get("merchant_last_message", "")

    evidence = [f"Topic: {topic.replace('_', ' ')}"]
    if last_msg:
        evidence.append(f"Merchant said: \"{last_msg}\"")

    return TriggerResolution(
        objective="execute_planning_action",
        why_now=f"Merchant expressed interest in {topic.replace('_', ' ')}",
        evidence=evidence,
        angle="action_execution",
        emotional_lever="effort_externalization",
        allowed_ctas=["open_ended"],
        priority=9,  # High — merchant is ready to act
    )


def _resolve_winback_eligible(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    days_since = payload.get("days_since_expiry", 0)
    perf_dip = payload.get("perf_dip_pct", 0)
    lapsed_added = payload.get("lapsed_customers_added_since_expiry", 0)

    evidence = [
        f"Subscription expired {days_since} days ago",
    ]
    if perf_dip:
        evidence.append(f"Performance down {abs(perf_dip)*100:.0f}% since expiry")
    if lapsed_added:
        evidence.append(f"{lapsed_added} customers lapsed since you left")

    return TriggerResolution(
        objective="win_back_churned_merchant",
        why_now=f"It's been {days_since} days since your subscription expired",
        evidence=evidence,
        angle="value_proof",
        emotional_lever="loss_aversion",
        allowed_ctas=["yes_no", "open_ended"],
        priority=5,
    )


def _resolve_dormant_with_vera(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    days_since = payload.get("days_since_last_merchant_message", 0)
    last_topic = payload.get("last_topic", "")

    evidence = [f"No interaction in {days_since} days"]
    if last_topic:
        evidence.append(f"Last conversation topic: {last_topic.replace('_', ' ')}")

    # Add something valuable to re-engage
    if merchant.performance.views > 0:
        evidence.append(f"Your profile still got {merchant.performance.views:,} views this month")

    return TriggerResolution(
        objective="re_engage_dormant_merchant",
        why_now=f"Haven't heard from you in {days_since} days",
        evidence=evidence,
        angle="diagnostic_insight",
        emotional_lever="curiosity",
        allowed_ctas=["open_ended"],
        priority=3,
    )


def _resolve_ipl_match(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    match = payload.get("match", "")
    venue = payload.get("venue", "")
    city = payload.get("city", "")

    evidence = [f"Match: {match}"]
    if venue:
        evidence.append(f"Venue: {venue}, {city}")

    return TriggerResolution(
        objective="drive_event_campaign",
        why_now=f"{match} tonight at {venue}",
        evidence=evidence,
        angle="event_opportunity",
        emotional_lever="social_proof",
        allowed_ctas=["yes_no", "open_ended"],
        priority=6,
    )


def _resolve_supply_alert(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    molecule = payload.get("molecule", "")
    batches = payload.get("affected_batches", [])
    manufacturer = payload.get("manufacturer", "")

    evidence = [
        f"Drug: {molecule}",
        f"Affected batches: {', '.join(batches)}",
    ]
    if manufacturer:
        evidence.append(f"Manufacturer: {manufacturer}")

    # Find the digest item for more detail
    alert_id = payload.get("alert_id", "")
    for d in category.digest:
        if d.id == alert_id:
            evidence.append(f"Source: {d.source}")
            if d.actionable:
                evidence.append(f"Action: {d.actionable}")
            break

    return TriggerResolution(
        objective="communicate_supply_alert",
        why_now=f"Supply alert: {molecule} recall affecting batches {', '.join(batches)}",
        evidence=evidence,
        angle="compliance_alert",
        emotional_lever="loss_aversion",
        allowed_ctas=["yes_no", "open_ended"],
        audience="merchant",
        priority=10,  # Highest — patient safety
    )


def _resolve_chronic_refill(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    molecules = payload.get("molecule_list", [])
    runs_out = payload.get("stock_runs_out_iso", "")
    delivery_saved = payload.get("delivery_address_saved", False)

    evidence = [f"Medications: {', '.join(molecules)}"]
    if runs_out:
        evidence.append(f"Current stock runs out: {runs_out[:10]}")
    if delivery_saved:
        evidence.append("Delivery address on file")

    customer_name = customer.identity.name if customer else "Patient"

    return TriggerResolution(
        objective="drive_refill_order",
        why_now=f"{customer_name}'s medication refill is due",
        evidence=evidence,
        angle="care_continuity",
        emotional_lever="effort_externalization",
        allowed_ctas=["open_ended", "yes_no"],
        audience="customer",
        send_as="merchant_on_behalf",
        priority=7,
    )


def _resolve_customer_lapsed_hard(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    days_since = payload.get("days_since_last_visit", 0)
    previous_focus = payload.get("previous_focus", "")

    evidence = [f"Last visit: {days_since} days ago"]
    if previous_focus:
        evidence.append(f"Previous focus: {previous_focus.replace('_', ' ')}")

    customer_name = customer.identity.name if customer else "Customer"

    return TriggerResolution(
        objective="win_back_lapsed_customer",
        why_now=f"{customer_name} hasn't visited in {days_since} days",
        evidence=evidence,
        angle="care_continuity",
        emotional_lever="reciprocity",
        allowed_ctas=["open_ended"],
        audience="customer",
        send_as="merchant_on_behalf",
        priority=5,
    )


def _resolve_trial_followup(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    trial_date = payload.get("trial_date", "")
    next_sessions = payload.get("next_session_options", [])

    evidence = [f"Trial date: {trial_date}"]
    for session in next_sessions:
        evidence.append(f"Next session available: {session.get('label', session.get('iso', ''))}")

    customer_name = customer.identity.name if customer else "Customer"

    return TriggerResolution(
        objective="convert_trial_to_regular",
        why_now=f"Follow-up after {customer_name}'s trial session",
        evidence=evidence,
        angle="care_continuity",
        emotional_lever="effort_externalization",
        allowed_ctas=["open_ended"],
        audience="customer",
        send_as="merchant_on_behalf",
        priority=5,
    )


def _resolve_wedding_package_followup(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    wedding_date = payload.get("wedding_date", "")
    days_to = payload.get("days_to_wedding", 0)
    next_step = payload.get("next_step_window_open", "")

    evidence = [f"Wedding date: {wedding_date}"]
    if days_to:
        evidence.append(f"{days_to} days to wedding")
    if next_step:
        evidence.append(f"Next step: {next_step.replace('_', ' ')}")

    customer_name = customer.identity.name if customer else "Customer"

    return TriggerResolution(
        objective="drive_bridal_package_conversion",
        why_now=f"{customer_name}'s wedding is {days_to} days away — time for {next_step.replace('_', ' ')}",
        evidence=evidence,
        angle="care_continuity",
        emotional_lever="loss_aversion",
        allowed_ctas=["open_ended"],
        audience="customer",
        send_as="merchant_on_behalf",
        priority=6,
    )


def _resolve_cde_opportunity(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    digest_item_id = payload.get("digest_item_id", "")
    credits = payload.get("credits", 0)
    fee = payload.get("fee", "")

    digest_item = None
    for d in category.digest:
        if d.id == digest_item_id:
            digest_item = d
            break

    evidence = []
    why_now = "CDE opportunity available"

    if digest_item:
        why_now = f"CDE event: {digest_item.title}"
        evidence.append(digest_item.title)
        if digest_item.date:
            evidence.append(f"Date: {digest_item.date[:10]}")
        evidence.append(f"Source: {digest_item.source}")
        if digest_item.summary:
            evidence.append(digest_item.summary)
    if credits:
        evidence.append(f"{credits} CDE credits")
    if fee:
        evidence.append(f"Fee: {fee}")

    return TriggerResolution(
        objective="drive_cde_registration",
        why_now=why_now,
        evidence=evidence,
        angle="peer_authority",
        emotional_lever="curiosity",
        allowed_ctas=["open_ended", "none"],
        priority=3,
    )


def _resolve_seasonal_perf_dip(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    metric = payload.get("metric", "views")
    delta_pct = payload.get("delta_pct", 0)
    is_expected = payload.get("is_expected_seasonal", False)
    season_note = payload.get("season_note", "")

    evidence = [
        f"{metric.capitalize()} down {abs(delta_pct)*100:.0f}% this week",
    ]
    if is_expected:
        evidence.append(f"This is expected: {season_note.replace('_', ' ')}")
    evidence.append("Good time to plan counter-seasonal campaigns")

    return TriggerResolution(
        objective="prepare_counter_seasonal_strategy",
        why_now=f"Seasonal dip in {metric} — expected for this time of year",
        evidence=evidence,
        angle="strategic_planning",
        emotional_lever="effort_externalization",
        allowed_ctas=["open_ended"],
        priority=3,
    )


def _resolve_category_seasonal(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    season = payload.get("season", "")
    trends = payload.get("trends", [])

    evidence = [f"Season: {season.replace('_', ' ')}"]
    for t in trends:
        evidence.append(t.replace("_", " "))

    return TriggerResolution(
        objective="optimize_for_seasonal_demand",
        why_now=f"Seasonal demand shifts for {season.replace('_', ' ')}",
        evidence=evidence,
        angle="strategic_planning",
        emotional_lever="social_proof",
        allowed_ctas=["open_ended"],
        priority=4,
    )


def _resolve_gbp_unverified(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    payload = trigger.payload
    uplift = payload.get("estimated_uplift_pct", 0)
    path = payload.get("verification_path", "")

    evidence = [
        "Your Google Business Profile is unverified",
        f"Estimated visibility uplift after verification: {uplift*100:.0f}%",
    ]
    if path:
        evidence.append(f"Verification method: {path.replace('_', ' ')}")

    return TriggerResolution(
        objective="drive_gbp_verification",
        why_now="Unverified profiles get significantly less visibility",
        evidence=evidence,
        angle="diagnostic_insight",
        emotional_lever="loss_aversion",
        allowed_ctas=["yes_no", "open_ended"],
        priority=7,
    )


def _resolve_generic(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    customer: CustomerContext | None,
) -> TriggerResolution:
    """Fallback for unknown trigger kinds."""
    logger.warning("No specific resolver for trigger kind: %s", trigger.kind)

    evidence = []
    for k, v in trigger.payload.items():
        if isinstance(v, (str, int, float)):
            evidence.append(f"{k}: {v}")

    return TriggerResolution(
        objective="general_engagement",
        why_now=f"Trigger: {trigger.kind.replace('_', ' ')}",
        evidence=evidence,
        angle="diagnostic_insight",
        emotional_lever="curiosity",
        allowed_ctas=["open_ended", "none"],
        priority=2,
    )


# ── Dispatch table ────────────────────────────────────────────

_RESOLVERS = {
    "research_digest":          _resolve_research_digest,
    "regulation_change":        _resolve_regulation_change,
    "recall_due":               _resolve_recall_due,
    "perf_dip":                 _resolve_perf_dip,
    "perf_spike":               _resolve_perf_spike,
    "renewal_due":              _resolve_renewal_due,
    "festival_upcoming":        _resolve_festival_upcoming,
    "competitor_opened":        _resolve_competitor_opened,
    "review_theme_emerged":     _resolve_review_theme_emerged,
    "milestone_reached":        _resolve_milestone_reached,
    "curious_ask_due":          _resolve_curious_ask,
    "active_planning_intent":   _resolve_active_planning_intent,
    "winback_eligible":         _resolve_winback_eligible,
    "dormant_with_vera":        _resolve_dormant_with_vera,
    "ipl_match_today":          _resolve_ipl_match,
    "supply_alert":             _resolve_supply_alert,
    "chronic_refill_due":       _resolve_chronic_refill,
    "customer_lapsed_hard":     _resolve_customer_lapsed_hard,
    "trial_followup":           _resolve_trial_followup,
    "wedding_package_followup": _resolve_wedding_package_followup,
    "cde_opportunity":          _resolve_cde_opportunity,
    "seasonal_perf_dip":        _resolve_seasonal_perf_dip,
    "category_seasonal":        _resolve_category_seasonal,
    "gbp_unverified":           _resolve_gbp_unverified,
}
