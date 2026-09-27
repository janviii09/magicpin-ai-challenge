"""
Vera 2.0 — Composition Engine
Audience-branched LLM prompts (merchant vs customer) + fallback templates.

Flow:
  1. Build prompt from MessagePlan
  2. Call LLM
  3. Parse JSON response
  4. Validate → retry with violations appended (up to MAX_COMPOSE_RETRIES)
  5. If still failing → use deterministic fallback template
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from config import (
    LLM_API_KEY,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    MAX_COMPOSE_RETRIES,
    get_model,
)
from core.models import CategoryContext, ComposedMessage, MessagePlan
from core.validator import ValidationResult, validate

logger = logging.getLogger("vera.composer")


# ═══════════════════════════════════════════════════════════════
#  PROMPT TEMPLATES — hard fork between merchant and customer
# ═══════════════════════════════════════════════════════════════

_MERCHANT_SYSTEM_PROMPT = """You are Vera, a merchant AI assistant for magicpin. You compose WhatsApp messages for Indian merchants.

RULES (non-negotiable):
1. Use the EXACT voice/tone specified. Never deviate.
2. Include ALL required facts verbatim — numbers, percentages, citations must appear exactly.
3. Keep the message concise — WhatsApp, not email. No long preambles.
4. ENGAGEMENT COMPULSION & CTA: The CTA goes in the LAST sentence. Make it a high-value, actionable, low-friction next step (e.g. "Want me to pull the abstract and draft a patient WhatsApp for you? Reply YES or tell me", "Want me to activate this boost for this weekend? Reply YES to confirm."). NEVER use generic or passive asks like "Want to know more?".
5. NEVER use taboo words. NEVER fabricate data not in the evidence.
6. Hindi-English code-mix is fine and preferred for "hinglish" language.
7. Address the merchant by name or owner first name — peer tone, not sales pitch.
8. No "I hope you're doing well", no "AMAZING DEAL!", no "!!!"

OUTPUT: Return ONLY a JSON object:
{"body": "...", "cta": "...", "rationale": "..."}
- body: the WhatsApp message text
- cta: must be exactly "{cta_type}"
- rationale: 1-2 sentences explaining why this message, which compulsion levers used"""

_MERCHANT_USER_PROMPT = """Compose a WhatsApp message for this merchant:

MERCHANT: {merchant_name} ({owner_first_name}), {locality}, {city}
VOICE: {voice} (category: {category_slug})
LANGUAGE: {language}
TABOO WORDS: {taboos}

OBJECTIVE: {objective}
WHY NOW: {why_now}
ANGLE: {angle}
EMOTIONAL LEVER: {emotional_lever}
CTA TYPE: {cta}

EVIDENCE (include these facts):
{evidence_block}

REQUIRED FACTS (must appear verbatim):
{required_facts_block}

{violations_block}"""

_CUSTOMER_SYSTEM_PROMPT = """You compose WhatsApp messages sent FROM a merchant TO their customer. You speak AS the merchant's business, not as Vera.

RULES (non-negotiable):
1. Identify the merchant by name: "X clinic here" or "Hi [customer], [merchant] here"
2. NEVER claim to be AI. You ARE the merchant's business communicating.
3. Include ALL required facts — dates, times, prices must be exact.
4. ENGAGEMENT COMPULSION & CTA: Provide a direct, low-friction call to action with concrete choices (e.g., "Reply 1 for [Slot A], 2 for [Slot B], or reply with a time that works best for you").
5. Clinical/service categories: NO overclaims ("guaranteed", "cure"). Keep it professional.
6. Respect the customer's language preference.
7. Keep it short — this is a WhatsApp reminder, not a brochure.
8. NEVER fabricate services, prices, or relationships not in the evidence.

OUTPUT: Return ONLY a JSON object:
{"body": "...", "cta": "...", "rationale": "..."}
- body: the WhatsApp message text
- cta: must be exactly "{cta_type}"
- rationale: 1-2 sentences explaining the approach"""

_CUSTOMER_USER_PROMPT = """Compose a WhatsApp message from this merchant to their customer:

MERCHANT: {merchant_name}, {locality}, {city}
CUSTOMER: {customer_name}
LANGUAGE: {language}
VOICE: {voice}
TABOO WORDS: {taboos}

OBJECTIVE: {objective}
WHY NOW: {why_now}
ANGLE: {angle}
EMOTIONAL LEVER: {emotional_lever}
CTA TYPE: {cta}

EVIDENCE (include these facts):
{evidence_block}

REQUIRED FACTS (must appear verbatim):
{required_facts_block}

{violations_block}"""


# ═══════════════════════════════════════════════════════════════
#  LLM CALL
# ═══════════════════════════════════════════════════════════════

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Provider-agnostic LLM call. Returns the raw text response.
    Supports: openai, anthropic, gemini, deepseek.
    """
    model = get_model()
    provider = LLM_PROVIDER.lower()

    if not LLM_API_KEY:
        logger.warning("No LLM_API_KEY set — falling back to template")
        raise RuntimeError("No LLM API key configured")

    try:
        if provider in ("openai", "deepseek"):
            return _call_openai_compatible(system_prompt, user_prompt, model, provider)
        elif provider == "anthropic":
            return _call_anthropic(system_prompt, user_prompt, model)
        elif provider == "gemini":
            return _call_gemini(system_prompt, user_prompt, model)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise


def _call_openai_compatible(system: str, user: str, model: str, provider: str) -> str:
    base_urls = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
    }
    base_url = base_urls.get(provider, "https://api.openai.com/v1")

    with httpx.Client(timeout=LLM_TIMEOUT) as client:
        resp = client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _call_anthropic(system: str, user: str, model: str) -> str:
    with httpx.Client(timeout=LLM_TIMEOUT) as client:
        resp = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": LLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


def _call_gemini(system: str, user: str, model: str) -> str:
    with httpx.Client(timeout=LLM_TIMEOUT) as client:
        resp = client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={LLM_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": LLM_TEMPERATURE,
                    "maxOutputTokens": LLM_MAX_TOKENS,
                    "responseMimeType": "application/json",
                },
            },
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


# ═══════════════════════════════════════════════════════════════
#  COMPOSE — the main entry point
# ═══════════════════════════════════════════════════════════════

def compose_with_llm(
    plan: MessagePlan,
    category: CategoryContext,
    customer_name: str = "",
) -> ComposedMessage:
    """
    Compose a message using the LLM, with retry on validation failure.
    Falls back to deterministic template if LLM fails entirely.
    """
    violations_block = ""

    for attempt in range(1 + MAX_COMPOSE_RETRIES):
        try:
            # Build prompt
            system_prompt, user_prompt = _build_prompts(
                plan, category, customer_name, violations_block,
            )

            # Call LLM
            raw = _call_llm(system_prompt, user_prompt)
            logger.debug("LLM response (attempt %d): %s", attempt + 1, raw[:200])

            # Parse JSON from response
            parsed = _parse_llm_response(raw)

            # Build ComposedMessage
            message = ComposedMessage(
                body=parsed.get("body", ""),
                cta=plan.cta,  # Always use plan's CTA, not LLM's
                send_as=plan.send_as,
                suppression_key="",  # Set by caller
                rationale=parsed.get("rationale", ""),
            )

            # Validate
            result = validate(message, plan, category)
            if result.valid:
                return message

            # Build violations block for retry
            violations_block = "PREVIOUS ATTEMPT FAILED. Fix these violations:\n"
            for v in result.violations:
                violations_block += f"  - {v}\n"

            logger.warning("Validation failed (attempt %d): %s", attempt + 1, result.violations)

        except Exception as e:
            logger.error("Compose attempt %d failed: %s", attempt + 1, e)
            violations_block = ""  # Reset on hard failure

    # All retries exhausted → fallback template
    logger.warning("All LLM attempts exhausted, using fallback template")
    return _fallback_template(plan, category, customer_name)


def _build_prompts(
    plan: MessagePlan,
    category: CategoryContext,
    customer_name: str,
    violations_block: str,
) -> tuple[str, str]:
    """Build the system + user prompt pair from the plan."""

    evidence_block = "\n".join(f"  • {e}" for e in plan.evidence)
    required_facts_block = "\n".join(f"  ✓ {f}" for f in plan.required_facts)
    taboos = ", ".join(category.voice.vocab_taboo) if category.voice.vocab_taboo else "(none)"

    if plan.audience == "customer":
        system = _CUSTOMER_SYSTEM_PROMPT.format(cta_type=plan.cta)
        user = _CUSTOMER_USER_PROMPT.format(
            merchant_name=plan.merchant_name,
            locality="",  # Not always needed
            city="",
            customer_name=customer_name or "Customer",
            language=plan.language,
            voice=plan.voice,
            taboos=taboos,
            objective=plan.objective,
            why_now=plan.why_now,
            angle=plan.angle,
            emotional_lever=plan.emotional_lever,
            cta=plan.cta,
            evidence_block=evidence_block,
            required_facts_block=required_facts_block,
            violations_block=violations_block,
        )
    else:
        system = _MERCHANT_SYSTEM_PROMPT.format(cta_type=plan.cta)
        user = _MERCHANT_USER_PROMPT.format(
            merchant_name=plan.merchant_name,
            owner_first_name=plan.owner_first_name,
            locality="",
            city="",
            voice=plan.voice,
            category_slug=category.slug,
            language=plan.language,
            taboos=taboos,
            objective=plan.objective,
            why_now=plan.why_now,
            angle=plan.angle,
            emotional_lever=plan.emotional_lever,
            cta=plan.cta,
            evidence_block=evidence_block,
            required_facts_block=required_facts_block,
            violations_block=violations_block,
        )

    return system, user


def _parse_llm_response(raw: str) -> dict[str, Any]:
    """Extract JSON from the LLM response (handles markdown code blocks and stringified JSON)."""
    # Try direct JSON parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            try:
                nested = json.loads(data)
                if isinstance(nested, dict):
                    return nested
            except Exception:
                return {"body": data, "rationale": ""}
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict):
                return data
            if isinstance(data, str):
                nested = json.loads(data)
                if isinstance(nested, dict):
                    return nested
        except Exception:
            pass

    # Try finding JSON object in the text
    brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    logger.error("Could not parse LLM response as JSON: %s", raw[:300])
    raise ValueError("LLM response is not valid JSON")


# ═══════════════════════════════════════════════════════════════
#  FALLBACK TEMPLATES — deterministic, always valid
# ═══════════════════════════════════════════════════════════════

def _fallback_template(
    plan: MessagePlan,
    category: CategoryContext,
    customer_name: str = "",
) -> ComposedMessage:
    """
    Deterministic fallback when LLM fails. Produces a valid but less
    natural message that still contains all required facts.
    """
    if plan.audience == "customer":
        body = _customer_fallback(plan, customer_name)
    else:
        body = _merchant_fallback(plan)

    return ComposedMessage(
        body=body,
        cta=plan.cta,
        send_as=plan.send_as,
        suppression_key="",
        rationale=f"[fallback] {plan.objective} — {plan.angle}",
    )


def _merchant_fallback(plan: MessagePlan) -> str:
    """Construct a high-converting, action-oriented merchant-facing message."""
    name = plan.owner_first_name or plan.merchant_name
    parts = [f"Hi {name},"]
    parts.append(plan.why_now.rstrip(".") + ".")

    # Add evidence items that aren't already included in why_now
    for evidence in plan.evidence:
        ev_clean = evidence.strip().rstrip(".")
        if ev_clean.lower() not in plan.why_now.lower():
            parts.append(ev_clean + ".")
            if len(parts) >= 3:
                break

    # High-converting action CTA
    if plan.cta == "yes_no":
        parts.append("Want me to prepare this update and draft it for you? Reply YES or tell me what to change.")
    elif plan.cta == "open_ended":
        if "research" in plan.objective or "digest" in plan.objective:
            parts.append("Want me to pull the abstract and draft a 2-minute patient WhatsApp you can share?")
        elif "compliance" in plan.objective:
            parts.append("Want me to pull the official checklist for your staff to review?")
        elif "event" in plan.objective or "campaign" in plan.objective:
            parts.append("Want me to draft a high-reach campaign post for you right away?")
        else:
            parts.append("Shall I prepare the draft for you right now?")

    return " ".join(parts)


def _customer_fallback(plan: MessagePlan, customer_name: str) -> str:
    """Construct an engaging, specific customer-facing message."""
    parts = [f"Hi {customer_name or 'there'}, {plan.merchant_name} here 🦷" if "dentist" in plan.merchant_name.lower() or "dental" in plan.merchant_name.lower() else f"Hi {customer_name or 'there'}, {plan.merchant_name} here."]
    parts.append(plan.why_now.rstrip(".") + ".")

    for evidence in plan.evidence:
        ev_clean = evidence.strip().rstrip(".")
        if ev_clean.lower() not in plan.why_now.lower():
            parts.append(ev_clean + ".")
            if len(parts) >= 3:
                break

    if plan.cta == "open_ended":
        parts.append("Reply with your preferred slot or time and we'll confirm it for you right away.")
    elif plan.cta == "yes_no":
        parts.append("Reply YES to block your spot or let us know a time that works best.")

    return " ".join(parts)
