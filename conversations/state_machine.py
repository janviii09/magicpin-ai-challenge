"""
Vera 2.0 — Conversation State Machine
FSM: IDLE → PROPOSED → ENGAGED/EXECUTE/BACKOFF/CLARIFY/END

Transition rules:
  - PROPOSED + auto_reply     → AUTO_REPLY_DETECTED → wait/try once → END
  - PROPOSED + direct_yes     → EXECUTE (priority handoff)
  - PROPOSED + direct_no      → END (graceful exit)
  - PROPOSED + join_intent    → EXECUTE
  - PROPOSED + question       → CLARIFY
  - PROPOSED + general_reply  → ENGAGED
  - ENGAGED  + direct_yes     → EXECUTE
  - ENGAGED  + direct_no      → END
  - ENGAGED  + 3 unanswered   → END
  - any      + auto_reply x2  → END
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from core.models import ConversationPhase, ConversationState
from conversations.intent import DetectedIntent, detect_intent

logger = logging.getLogger("vera.state_machine")


class ConversationManager:
    """Manages all active conversations and their state transitions."""

    def __init__(self) -> None:
        # conversation_id → ConversationState
        self._conversations: dict[str, ConversationState] = {}

    def create_conversation(
        self,
        conversation_id: str,
        merchant_id: str,
        trigger_id: str,
        customer_id: str | None = None,
    ) -> ConversationState:
        """Create a new conversation in PROPOSED state (first message sent)."""
        now = datetime.now(timezone.utc).isoformat()
        state = ConversationState(
            conversation_id=conversation_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            phase=ConversationPhase.PROPOSED,
            trigger_id=trigger_id,
            created_at=now,
            last_activity_at=now,
        )
        self._conversations[conversation_id] = state
        logger.info("Conversation created: %s (merchant=%s)", conversation_id, merchant_id)
        return state

    def get_conversation(self, conversation_id: str) -> ConversationState | None:
        return self._conversations.get(conversation_id)

    def process_reply(
        self,
        conversation_id: str,
        from_role: str,
        message: str,
        turn_number: int = 0,
    ) -> tuple[ConversationState, str, str]:
        """
        Process an incoming reply and transition the state machine.

        Returns: (updated_state, action, reason)
          action: "send" | "wait" | "end"
          reason: why this action was chosen
        """
        state = self._conversations.get(conversation_id)
        if state is None:
            # Create a new state for unknown conversations
            state = ConversationState(
                conversation_id=conversation_id,
                merchant_id="unknown",
                phase=ConversationPhase.PROPOSED,
                created_at=datetime.now(timezone.utc).isoformat(),
                last_activity_at=datetime.now(timezone.utc).isoformat(),
            )
            self._conversations[conversation_id] = state

        # Record the turn
        state.turns.append({
            "from": from_role,
            "body": message,
            "turn_number": turn_number,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        state.last_activity_at = datetime.now(timezone.utc).isoformat()

        # Detect intent
        intent = detect_intent(message, state.turns)
        logger.info(
            "Intent detected: %s for conversation %s (phase=%s)",
            intent.value, conversation_id, state.phase.value,
        )

        # Apply state transitions
        action, reason = self._transition(state, intent, message)

        return state, action, reason

    def _transition(
        self,
        state: ConversationState,
        intent: DetectedIntent,
        message: str,
    ) -> tuple[str, str]:
        """
        Apply FSM transitions. Returns (action, reason).
        """
        phase = state.phase

        # ── Auto-reply handling (any phase) ───────────────────
        if intent == DetectedIntent.AUTO_REPLY:
            state.auto_reply_count += 1
            latest_turn_num = state.turns[-1].get("turn_number", 0) if state.turns else 0
            if state.auto_reply_count >= 2 or latest_turn_num >= 3:
                # Two auto-replies or turn >= 3 → stop trying
                state.phase = ConversationPhase.END
                return "end", "Auto-reply detected — gracefully exiting to avoid wasting turns"
            else:
                # First auto-reply → try once more with a redirect
                state.phase = ConversationPhase.AUTO_REPLY_DETECTED
                return "send", "Auto-reply detected — trying once to reach the real person"

        # ── Direct NO / STOP (any phase) ──────────────────────
        if intent == DetectedIntent.DIRECT_NO:
            state.phase = ConversationPhase.END
            return "end", "Merchant said no / stop — gracefully exiting"

        # ── Phase-specific transitions ────────────────────────

        if phase in (ConversationPhase.PROPOSED, ConversationPhase.AUTO_REPLY_DETECTED):
            if intent == DetectedIntent.DIRECT_YES:
                state.phase = ConversationPhase.EXECUTE
                return "send", "Merchant said yes — switching to execution mode"

            elif intent == DetectedIntent.JOIN_INTENT:
                state.phase = ConversationPhase.EXECUTE
                return "send", "Join intent detected — routing to action immediately"

            elif intent == DetectedIntent.ACTION_REQUEST:
                state.phase = ConversationPhase.EXECUTE
                return "send", "Action request detected — executing"

            elif intent == DetectedIntent.QUESTION:
                state.phase = ConversationPhase.CLARIFY
                return "send", "Merchant asked a question — providing clarification"

            elif intent == DetectedIntent.GENERAL_REPLY:
                state.phase = ConversationPhase.ENGAGED
                state.unanswered_nudge_count = 0
                return "send", "Merchant engaged — continuing conversation"

        elif phase == ConversationPhase.ENGAGED:
            if intent == DetectedIntent.DIRECT_YES:
                state.phase = ConversationPhase.EXECUTE
                return "send", "Merchant confirmed — executing action"

            elif intent == DetectedIntent.JOIN_INTENT:
                state.phase = ConversationPhase.EXECUTE
                return "send", "Join intent in engaged state — routing to action"

            elif intent == DetectedIntent.ACTION_REQUEST:
                state.phase = ConversationPhase.EXECUTE
                return "send", "Action request while engaged — executing"

            elif intent == DetectedIntent.QUESTION:
                state.phase = ConversationPhase.CLARIFY
                return "send", "Question while engaged — clarifying"

            elif intent == DetectedIntent.GENERAL_REPLY:
                return "send", "Continuing engaged conversation"

        elif phase == ConversationPhase.EXECUTE:
            # In execution mode, keep going until done
            if intent in (DetectedIntent.DIRECT_YES, DetectedIntent.ACTION_REQUEST):
                return "send", "Continuing execution"
            elif intent == DetectedIntent.QUESTION:
                state.phase = ConversationPhase.CLARIFY
                return "send", "Question during execution — clarifying before proceeding"
            elif intent == DetectedIntent.GENERAL_REPLY:
                return "send", "Acknowledging and continuing"

        elif phase == ConversationPhase.CLARIFY:
            if intent == DetectedIntent.DIRECT_YES:
                state.phase = ConversationPhase.EXECUTE
                return "send", "Clarification resolved — proceeding to execute"
            elif intent == DetectedIntent.GENERAL_REPLY:
                state.phase = ConversationPhase.ENGAGED
                return "send", "Clarification response received — continuing"

        elif phase == ConversationPhase.END:
            return "end", "Conversation already ended"

        # Default: send a response
        return "send", f"Default transition from {phase.value} with intent {intent.value}"

    def record_bot_message(
        self,
        conversation_id: str,
        body: str,
    ) -> None:
        """Record that the bot sent a message (for turn tracking)."""
        state = self._conversations.get(conversation_id)
        if state:
            state.turns.append({
                "from": "vera",
                "body": body,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            state.last_activity_at = datetime.now(timezone.utc).isoformat()

    def increment_unanswered(self, conversation_id: str) -> bool:
        """
        Increment unanswered nudge count. Returns True if should end (3+).
        """
        state = self._conversations.get(conversation_id)
        if state:
            state.unanswered_nudge_count += 1
            if state.unanswered_nudge_count >= 3:
                state.phase = ConversationPhase.END
                return True
        return False

    def list_active(self) -> list[str]:
        """Return IDs of conversations not in END state."""
        return [
            cid for cid, state in self._conversations.items()
            if state.phase != ConversationPhase.END
        ]

    def count(self) -> int:
        return len(self._conversations)


# ── Singleton ─────────────────────────────────────────────────
_manager: ConversationManager | None = None


def get_conversation_manager() -> ConversationManager:
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager
