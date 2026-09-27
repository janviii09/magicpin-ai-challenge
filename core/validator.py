"""
Vera 2.0 — Validator
Post-LLM quality gate. Checks the composed message against the MessagePlan.

Checks:
  1. Schema: body is non-empty string, CTA matches plan
  2. Taboo words: none from category.voice.vocab_taboo appear
  3. Fact grounding: required_facts from the plan appear (fuzzy match)
  4. CTA count: exactly one primary CTA (or none if plan.cta == "none")
  5. Length: within MIN/MAX bounds
  6. Anti-patterns: no "AMAZING DEAL!", no multi-CTA, no long preambles
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from core.models import CategoryContext, ComposedMessage, MessagePlan
from config import MAX_BODY_LENGTH, MIN_BODY_LENGTH

logger = logging.getLogger("vera.validator")


@dataclass
class ValidationResult:
    valid: bool
    violations: list[str]


# ── Anti-pattern regexes ──────────────────────────────────────
_PROMO_PATTERNS = [
    re.compile(r"AMAZING\s+DEAL", re.IGNORECASE),
    re.compile(r"LIMITED\s+TIME\s+OFFER", re.IGNORECASE),
    re.compile(r"ACT\s+NOW", re.IGNORECASE),
    re.compile(r"DON'T\s+MISS\s+OUT", re.IGNORECASE),
    re.compile(r"HURRY", re.IGNORECASE),
    re.compile(r"!!!", re.IGNORECASE),
]

_PREAMBLE_PATTERNS = [
    re.compile(r"^(Hi|Hello|Hey).*I\s+hope\s+you'?re\s+doing\s+well", re.IGNORECASE),
    re.compile(r"^I\s+am\s+reaching\s+out\s+to(day)?", re.IGNORECASE),
    re.compile(r"^I\s+wanted\s+to\s+take\s+a\s+moment", re.IGNORECASE),
    re.compile(r"^Thank\s+you\s+for\s+your\s+time", re.IGNORECASE),
]

_MULTI_CTA_PATTERNS = [
    re.compile(r"Reply\s+(YES|1)\s+for\s+.+,\s+(NO|2)\s+for\s+.+,\s+(MAYBE|3)\s+for", re.IGNORECASE),
]


def validate(
    message: ComposedMessage,
    plan: MessagePlan,
    category: CategoryContext,
) -> ValidationResult:
    """Run all validation checks and return the result."""
    violations: list[str] = []

    # 1. Schema checks
    if not message.body or not message.body.strip():
        violations.append("SCHEMA: body is empty")
    if message.cta != plan.cta:
        violations.append(f"SCHEMA: CTA mismatch (got '{message.cta}', expected '{plan.cta}')")
    if message.send_as != plan.send_as:
        violations.append(f"SCHEMA: send_as mismatch (got '{message.send_as}', expected '{plan.send_as}')")

    body = message.body

    # 2. Taboo words
    for taboo in category.voice.vocab_taboo:
        # Clean up taboo for matching (remove parenthetical notes)
        clean_taboo = re.sub(r'\s*\(.*?\)', '', taboo).strip().lower()
        if clean_taboo and clean_taboo in body.lower():
            violations.append(f"TABOO: '{clean_taboo}' found in body (category forbids this)")

    # 3. Fact grounding — check required_facts appear (fuzzy)
    for fact in plan.required_facts:
        if not _fact_present(body, fact):
            violations.append(f"FACT_MISSING: required fact not found: '{fact[:60]}...'")

    # 4. Length
    body_len = len(body)
    if body_len < MIN_BODY_LENGTH:
        violations.append(f"LENGTH: body too short ({body_len} chars < {MIN_BODY_LENGTH})")
    if body_len > MAX_BODY_LENGTH:
        violations.append(f"LENGTH: body too long ({body_len} chars > {MAX_BODY_LENGTH})")

    # 5. Anti-patterns
    for pattern in _PROMO_PATTERNS:
        if pattern.search(body):
            violations.append(f"ANTI_PATTERN: promotional language detected: '{pattern.pattern}'")

    for pattern in _PREAMBLE_PATTERNS:
        if pattern.search(body):
            violations.append(f"ANTI_PATTERN: long preamble detected")
            break

    for pattern in _MULTI_CTA_PATTERNS:
        if pattern.search(body):
            violations.append(f"ANTI_PATTERN: multiple CTAs detected")

    valid = len(violations) == 0
    if not valid:
        logger.warning("Validation failed: %s", violations)

    return ValidationResult(valid=valid, violations=violations)


def _fact_present(body: str, fact: str) -> bool:
    """
    Check if a required fact appears in the body (fuzzy match).
    We extract the key numeric/identifiable parts and check those.
    """
    body_lower = body.lower()
    fact_lower = fact.lower()

    # Direct substring match
    if fact_lower in body_lower:
        return True

    # Extract numbers from the fact and check if they appear
    numbers = re.findall(r'\d[\d,]*\.?\d*', fact)
    if numbers:
        # Check if the key number appears somewhere in the body
        for num in numbers:
            # Normalize: remove commas for matching
            clean_num = num.replace(",", "")
            # Check both with and without commas
            if clean_num in body.replace(",", "") or num in body:
                return True

    # Extract percentage patterns
    pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', fact)
    if pct_match:
        pct = pct_match.group(1)
        if pct in body:
            return True

    # Check for key quoted phrases
    quoted = re.findall(r'"([^"]+)"', fact)
    for q in quoted:
        if q.lower() in body_lower:
            return True

    return False
