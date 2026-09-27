"""
Vera 2.0 — Suppression Engine
Prevents duplicate/redundant messages to the same merchant for the same trigger archetype.

Key = (merchant_id, suppression_key)
TTL = by trigger kind (configurable)

On /v1/tick:
  if key exists and now - last_sent_at < TTL → skip
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any

logger = logging.getLogger("vera.suppression")


# ── TTL by trigger kind (hours) ──────────────────────────────
# These represent "don't message about the same thing within this window"
SUPPRESSION_TTL_HOURS: dict[str, int] = {
    # Internal triggers
    "perf_dip":                 24,
    "perf_spike":               24,
    "seasonal_perf_dip":        168,    # 7 days
    "milestone_reached":        168,
    "renewal_due":              48,
    "dormant_with_vera":        168,
    "recall_due":               168,    # per-customer
    "customer_lapsed_hard":     168,
    "trial_followup":           72,
    "chronic_refill_due":       24,
    "review_theme_emerged":     72,
    "curious_ask_due":          168,
    "active_planning_intent":   4,      # fast follow-up is OK
    "winback_eligible":         168,
    "gbp_unverified":           168,
    "wedding_package_followup": 168,

    # External triggers
    "research_digest":          168,    # 7 days — once per digest cycle
    "regulation_change":        168,
    "festival_upcoming":        72,
    "ipl_match_today":          12,
    "competitor_opened":        48,
    "category_seasonal":        168,
    "cde_opportunity":          168,
    "supply_alert":             24,
}

DEFAULT_TTL_HOURS = 48


class _SuppressionRecord:
    __slots__ = ("last_sent_at", "conversation_state", "trigger_version")

    def __init__(
        self,
        last_sent_at: datetime,
        conversation_state: str = "",
        trigger_version: int = 1,
    ):
        self.last_sent_at = last_sent_at
        self.conversation_state = conversation_state
        self.trigger_version = trigger_version


class SuppressionEngine:
    """Thread-safe suppression checker."""

    def __init__(self) -> None:
        # (merchant_id, suppression_key) → _SuppressionRecord
        self._records: dict[tuple[str, str], _SuppressionRecord] = {}
        self._lock = Lock()

    def is_suppressed(
        self,
        merchant_id: str,
        suppression_key: str,
        trigger_kind: str,
        now: datetime | None = None,
    ) -> bool:
        """Check if this (merchant, suppression_key) pair is within its TTL window."""
        if not suppression_key:
            return False

        now = now or datetime.now(timezone.utc)
        key = (merchant_id, suppression_key)

        with self._lock:
            record = self._records.get(key)
            if record is None:
                return False

            ttl_hours = SUPPRESSION_TTL_HOURS.get(trigger_kind, DEFAULT_TTL_HOURS)
            ttl = timedelta(hours=ttl_hours)
            elapsed = now - record.last_sent_at

            if elapsed < ttl:
                logger.debug(
                    "Suppressed: %s (%.1fh < %dh TTL)",
                    key, elapsed.total_seconds() / 3600, ttl_hours,
                )
                return True

            return False

    def record_sent(
        self,
        merchant_id: str,
        suppression_key: str,
        now: datetime | None = None,
        conversation_state: str = "",
        trigger_version: int = 1,
    ) -> None:
        """Record that we sent a message for this suppression key."""
        if not suppression_key:
            return

        now = now or datetime.now(timezone.utc)
        key = (merchant_id, suppression_key)

        with self._lock:
            self._records[key] = _SuppressionRecord(
                last_sent_at=now,
                conversation_state=conversation_state,
                trigger_version=trigger_version,
            )
            logger.info("Recorded send: %s at %s", key, now.isoformat())

    def clear(self) -> None:
        """Clear all suppression records (for testing)."""
        with self._lock:
            self._records.clear()

    def count(self) -> int:
        return len(self._records)


# ── Singleton ─────────────────────────────────────────────────
_engine: SuppressionEngine | None = None


def get_suppression_engine() -> SuppressionEngine:
    global _engine
    if _engine is None:
        _engine = SuppressionEngine()
    return _engine
