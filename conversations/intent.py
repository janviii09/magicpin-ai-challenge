"""
Vera 2.0 — Intent Detection
Detects auto-replies and direct merchant intents from incoming messages.

Two critical capabilities:
1. Auto-reply detection: WhatsApp Business canned responses → don't burn turns
2. Direct-intent detection: "I want to join" / "go ahead" → switch to EXECUTE mode
"""

from __future__ import annotations

import re
import logging
from enum import Enum

logger = logging.getLogger("vera.intent")


class DetectedIntent(str, Enum):
    AUTO_REPLY = "auto_reply"           # WhatsApp Business canned response
    DIRECT_YES = "direct_yes"           # "yes", "go ahead", "do it"
    DIRECT_NO = "direct_no"             # "no", "not interested", "stop"
    JOIN_INTENT = "join_intent"         # "I want to join", "sign me up"
    QUESTION = "question"               # Merchant asking something
    ACTION_REQUEST = "action_request"   # "update my profile", "post this"
    GENERAL_REPLY = "general_reply"     # Meaningful but not a clear intent


# ── Auto-reply detection patterns ─────────────────────────────
# These are common WhatsApp Business canned auto-replies in Indian context
_AUTO_REPLY_PATTERNS = [
    # English auto-replies
    re.compile(r"thank\s+you\s+for\s+(contacting|reaching\s+out|your\s+message)", re.IGNORECASE),
    re.compile(r"we\s+will\s+get\s+back\s+to\s+you", re.IGNORECASE),
    re.compile(r"(our|my)\s+team\s+will\s+(respond|reply|get\s+back)", re.IGNORECASE),
    re.compile(r"this\s+is\s+an?\s+auto(mated|matic)?\s*(reply|response|message)", re.IGNORECASE),
    re.compile(r"i\s+am\s+(an?\s+)?automated\s+assistant", re.IGNORECASE),
    re.compile(r"currently\s+(unavailable|out\s+of\s+office|away)", re.IGNORECASE),
    re.compile(r"business\s+hours\s+are", re.IGNORECASE),
    re.compile(r"for\s+immediate\s+assistance.*call", re.IGNORECASE),

    # Hindi auto-replies
    re.compile(r"aapki\s+jaankari\s+ke\s+liye.*shukriya", re.IGNORECASE),
    re.compile(r"hum\s+jald\s+hi\s+aapse\s+sampark", re.IGNORECASE),
    re.compile(r"hamari\s+team\s+tak\s+pahuncha", re.IGNORECASE),
    re.compile(r"aapka\s+(call|message|sandesh)\s+(receive|mila|praapt)", re.IGNORECASE),
    re.compile(r"kripya\s+pratiksha\s+kar", re.IGNORECASE),
    re.compile(r"yeh\s+ek\s+automated?\s+(reply|jawab|sandesh)", re.IGNORECASE),

    # Generic short canned
    re.compile(r"^(thank\s*you|thanks|shukriya|dhanyavaad)\s*[.!]*$", re.IGNORECASE),
]

# ── Direct YES patterns ──────────────────────────────────────
_YES_PATTERNS = [
    re.compile(r"^(yes|yeah|yep|ya|yup|haan|ha|ok|okay|sure|go\s+ahead|proceed|do\s+it|let'?s\s+do\s+it|sounds\s+good|chalega|thik\s+hai|theek\s+hai|sahi\s+hai|kar\s+do|karo|chalo)", re.IGNORECASE),
    re.compile(r"^YES$"),  # Exact "YES" (the standard CTA reply)
    # Affirmative + action phrases
    re.compile(r"(let'?s\s+do\s+it|lets\s+do\s+it|go\s+ahead|proceed\s+with|count\s+me\s+in)", re.IGNORECASE),
    re.compile(r"^(yes|haan|sure|ok|okay|ha|ji)[,\s]+(send|share|show|pull|draft|do|update|go|bhejo|batao|dikhao|karo)", re.IGNORECASE),
]

# ── Direct NO / Hostile patterns ─────────────────────────────
_NO_PATTERNS = [
    re.compile(r"^(no|nah|nahi|nope|not\s+interested|stop|unsubscribe|band\s+karo|mat\s+karo|ruko|rukh?o|not\s+now)\s*[,.]?\s*(thanks|thank\s+you|shukriya)?[.!]*\s*$", re.IGNORECASE),
    re.compile(r"^STOP$"),  # The standard stop CTA
    re.compile(r"(stop\s+messaging|don'?t\s+message|useless\s+spam|spam|waste\s+of\s+time|leave\s+me\s+alone|don'?t\s+contact)", re.IGNORECASE),
    re.compile(r"please\s+(stop|don'?t|do\s+not)\s+(send|message|contact)", re.IGNORECASE),
    re.compile(r"remove\s+me\s+from", re.IGNORECASE),
    re.compile(r"^not\s+(interested|now|needed|required)", re.IGNORECASE),
]


# ── Join/action intent patterns ───────────────────────────────
_JOIN_PATTERNS = [
    re.compile(r"(I|mujhe|hume|main)\s+(want\s+to|chahta|chahti|chahte)\s+(join|judna|shuru\s+karna)", re.IGNORECASE),
    re.compile(r"sign\s+me\s+up", re.IGNORECASE),
    re.compile(r"how\s+(do\s+I|can\s+I|to)\s+join", re.IGNORECASE),
    re.compile(r"mujhe\s+magicpin\s+.*judna", re.IGNORECASE),
    re.compile(r"let'?s\s+(start|begin|get\s+started)", re.IGNORECASE),
]

_ACTION_PATTERNS = [
    re.compile(r"(update|fix|change|edit|modify)\s+(my|the|mera|hamara)\s+(profile|listing|hours|description|photos)", re.IGNORECASE),
    re.compile(r"(post|publish|share)\s+(this|it|ye|yeh)", re.IGNORECASE),
    re.compile(r"(send|draft|create|bana|bhejo)\s+(it|the|ek|ye)", re.IGNORECASE),
    re.compile(r"please\s+(go\s+ahead|do\s+it|proceed|update)", re.IGNORECASE),
]

_QUESTION_PATTERNS = [
    re.compile(r"\?$"),
    re.compile(r"^(what|how|when|where|why|which|who|kya|kab|kaise|kahan|kyun|kitna|kitne|konsa)", re.IGNORECASE),
    re.compile(r"(can\s+you|could\s+you|will\s+you|kya\s+aap|aap\s+kya)", re.IGNORECASE),
]


def detect_intent(
    message: str,
    conversation_turns: list[dict] | None = None,
) -> DetectedIntent:
    """
    Classify a merchant/customer message into an intent category.
    Priority: auto-reply > direct_yes/no > join > action > question > general
    """
    msg = message.strip()

    if not msg:
        return DetectedIntent.GENERAL_REPLY

    # 1. Auto-reply detection (highest priority — prevents wasted turns)
    if _is_auto_reply(msg, conversation_turns):
        return DetectedIntent.AUTO_REPLY

    # 2. Direct YES
    for pattern in _YES_PATTERNS:
        if pattern.search(msg):
            return DetectedIntent.DIRECT_YES

    # 3. Direct NO / STOP
    for pattern in _NO_PATTERNS:
        if pattern.search(msg):
            return DetectedIntent.DIRECT_NO

    # 4. Join intent
    for pattern in _JOIN_PATTERNS:
        if pattern.search(msg):
            return DetectedIntent.JOIN_INTENT

    # 5. Action request
    for pattern in _ACTION_PATTERNS:
        if pattern.search(msg):
            return DetectedIntent.ACTION_REQUEST

    # 6. Question
    for pattern in _QUESTION_PATTERNS:
        if pattern.search(msg):
            return DetectedIntent.QUESTION

    # 7. Default: general reply
    return DetectedIntent.GENERAL_REPLY


def _is_auto_reply(
    message: str,
    conversation_turns: list[dict] | None = None,
) -> bool:
    """
    Detect auto-replies using:
    1. Pattern matching against known canned responses
    2. Repetition detection (same message 2+ times = auto-reply)
    """
    # Pattern match
    for pattern in _AUTO_REPLY_PATTERNS:
        if pattern.search(message):
            logger.info("Auto-reply detected (pattern match): %s", message[:50])
            return True

    # Repetition detection: check if this exact message was sent before
    # Note: conversation_turns may already include the current message (appended
    # by the state machine before calling detect_intent), so we need >= 2
    # occurrences to confirm a repeat (current + at least one prior).
    if conversation_turns and len(conversation_turns) >= 2:
        msg_lower = message.strip().lower()
        prev_merchant_msgs = [
            t.get("body", "").strip().lower()
            for t in conversation_turns
            if t.get("from") == "merchant" or t.get("from_role") == "merchant"
        ]
        repeat_count = sum(1 for m in prev_merchant_msgs if m == msg_lower)
        if repeat_count >= 2:  # Current + at least one prior identical → auto-reply
            logger.info("Auto-reply detected (repetition): %s", message[:50])
            return True


    return False
