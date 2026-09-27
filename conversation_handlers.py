"""
Vera 2.0 — Conversation Handlers (optional submission artifact)
Multi-turn handling for merchant responses.

Implements:
  respond(state, merchant_message) → reply dict

This wraps the conversation state machine + intent detection +
contextual reply composition into the challenge's optional interface.
"""

from __future__ import annotations

from typing import Any

from core.context_store import get_store
from core.models import ConversationPhase, ConversationState
from conversations.intent import DetectedIntent, detect_intent
from conversations.state_machine import get_conversation_manager


def respond(state: dict | ConversationState, merchant_message: str) -> dict:
    """
    Given the conversation so far + the merchant's latest message, produce the reply.

    Args:
        state: ConversationState dict with conversation_id, merchant_id, turns
        merchant_message: the merchant's latest message text

    Returns:
        dict with keys: action, body, cta, rationale
        action: "send" | "wait" | "end"
    """
    conv_manager = get_conversation_manager()
    store = get_store()

    # Extract conversation ID
    if isinstance(state, dict):
        conv_id = state.get("conversation_id", "unknown")
        merchant_id = state.get("merchant_id", "unknown")
    else:
        conv_id = state.conversation_id
        merchant_id = state.merchant_id

    # Process through state machine
    updated_state, action, reason = conv_manager.process_reply(
        conversation_id=conv_id,
        from_role="merchant",
        message=merchant_message,
    )

    # Build response based on action
    if action == "end":
        body = _build_exit(updated_state, merchant_message, reason)
        return {
            "action": "end",
            "body": body,
            "cta": "none",
            "rationale": reason,
        }

    if action == "wait":
        return {
            "action": "wait",
            "body": "",
            "cta": "none",
            "rationale": reason,
            "wait_seconds": 1800,
        }

    # action == "send"
    merchant = store.get("merchant", merchant_id)
    body = _build_reply(updated_state, merchant_message, merchant, reason)

    conv_manager.record_bot_message(conv_id, body)

    return {
        "action": "send",
        "body": body,
        "cta": "open_ended",
        "rationale": reason,
    }


def _build_exit(state: ConversationState, msg: str, reason: str) -> str:
    """Build a graceful exit message."""
    if "auto" in reason.lower():
        return "No worries — I'll check back when the owner is available. Best wishes! 🙂"
    elif "stop" in reason.lower() or "no" in reason.lower():
        return "Understood, won't follow up. Message me anytime if you need help."
    return "Thanks for your time! Reach out whenever you need anything. 🙂"


def _build_reply(
    state: ConversationState,
    msg: str,
    merchant: Any,
    reason: str,
) -> str:
    """Build a contextual reply based on conversation phase."""
    phase = state.phase
    name = ""
    if merchant:
        name = getattr(merchant.identity, 'owner_first_name', '') or getattr(merchant.identity, 'name', '')

    if phase == ConversationPhase.AUTO_REPLY_DETECTED:
        return (
            f"Samajh gayi — automated reply lag raha hai. "
            f"Kya {name or 'aap'} ya koi team member directly reply kar sakte hain? "
            f"2 minute ka kaam hai. 🙂"
        )

    if phase == ConversationPhase.EXECUTE:
        return (
            f"Perfect{', ' + name if name else ''}! I'm on it. "
            f"I'll complete this and share the update shortly. "
            f"Anything else you'd like me to handle?"
        )

    if phase == ConversationPhase.CLARIFY:
        return (
            f"Good question{', ' + name if name else ''}! "
            f"Let me pull the details — I'll have an answer in a moment."
        )

    if phase == ConversationPhase.ENGAGED:
        return (
            f"Got it{', ' + name if name else ''}. "
            f"Shall I go ahead and set this up for you?"
        )

    return "Thanks for your message! How can I help you further?"
