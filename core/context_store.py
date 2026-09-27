"""
Vera 2.0 — Context Store
Versioned, idempotent, in-memory context storage.
Rules:
  - incoming.version <= stored.version → ignore (return stale_version)
  - incoming.version >  stored.version → replace atomically
  - Re-posting same version is a no-op (returns accepted)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

from core.models import (
    CategoryContext,
    CustomerContext,
    MerchantContext,
    TriggerContext,
)

logger = logging.getLogger("vera.context_store")


class _VersionedEntry:
    """Internal wrapper that pairs a context with its version."""
    __slots__ = ("version", "raw", "parsed")

    def __init__(self, version: int, raw: dict[str, Any], parsed: Any):
        self.version = version
        self.raw = raw
        self.parsed = parsed


_PARSERS = {
    "category":  CategoryContext,
    "merchant":  MerchantContext,
    "customer":  CustomerContext,
    "trigger":   TriggerContext,
}


class ContextStore:
    """Thread-safe, versioned context store."""

    def __init__(self) -> None:
        # scope → context_id → _VersionedEntry
        self._data: dict[str, dict[str, _VersionedEntry]] = {
            "category": {},
            "merchant": {},
            "customer": {},
            "trigger": {},
        }
        self._lock = Lock()

    # ── Public API ────────────────────────────────────────────

    def upsert(
        self,
        scope: str,
        context_id: str,
        version: int,
        payload: dict[str, Any],
    ) -> tuple[bool, int | None]:
        """
        Returns (accepted, current_version_or_None).
        - (True, None)   → stored successfully
        - (False, N)      → stale; current version is N
        """
        with self._lock:
            bucket = self._data.get(scope)
            if bucket is None:
                raise ValueError(f"Unknown scope: {scope}")

            existing = bucket.get(context_id)

            # Same version → idempotent no-op (still accepted)
            if existing and existing.version == version:
                return True, None

            # Stale version → reject
            if existing and existing.version > version:
                return False, existing.version

            # Parse into typed model
            parser = _PARSERS.get(scope)
            parsed = parser.model_validate(payload) if parser else payload

            bucket[context_id] = _VersionedEntry(version, payload, parsed)
            logger.info("Stored %s/%s v%d", scope, context_id, version)
            return True, None

    def get_raw(self, scope: str, context_id: str) -> dict[str, Any] | None:
        """Return the raw dict for a context, or None."""
        entry = self._data.get(scope, {}).get(context_id)
        return entry.raw if entry else None

    def get(self, scope: str, context_id: str) -> Any | None:
        """Return the parsed Pydantic model for a context, or None."""
        entry = self._data.get(scope, {}).get(context_id)
        return entry.parsed if entry else None

    def get_version(self, scope: str, context_id: str) -> int | None:
        entry = self._data.get(scope, {}).get(context_id)
        return entry.version if entry else None

    def list_ids(self, scope: str) -> list[str]:
        return list(self._data.get(scope, {}).keys())

    def count(self, scope: str) -> int:
        return len(self._data.get(scope, {}))

    def counts(self) -> dict[str, int]:
        return {scope: len(bucket) for scope, bucket in self._data.items()}

    def get_all(self, scope: str) -> dict[str, Any]:
        """Return all parsed objects in a scope, keyed by context_id."""
        return {
            cid: entry.parsed
            for cid, entry in self._data.get(scope, {}).items()
        }

    def get_merchant_for_trigger(self, trigger_id: str) -> MerchantContext | None:
        """Convenience: given a trigger_id, look up its merchant."""
        trigger = self.get("trigger", trigger_id)
        if trigger is None:
            return None
        return self.get("merchant", trigger.merchant_id)

    def get_category_for_merchant(self, merchant_id: str) -> CategoryContext | None:
        """Convenience: given a merchant_id, look up its category."""
        merchant = self.get("merchant", merchant_id)
        if merchant is None:
            return None
        return self.get("category", merchant.category_slug)

    def get_customer_for_trigger(self, trigger_id: str) -> CustomerContext | None:
        """Convenience: given a trigger_id with customer scope, look up customer."""
        trigger = self.get("trigger", trigger_id)
        if trigger is None or trigger.customer_id is None:
            return None
        return self.get("customer", trigger.customer_id)

    # ── Bootstrap from dataset files ─────────────────────────

    def load_from_dataset(self, dataset_dir: str | Path) -> None:
        """Load seed data from the dataset directory structure."""
        dataset_path = Path(dataset_dir)

        # Load categories
        categories_dir = dataset_path / "categories"
        if categories_dir.exists():
            for f in categories_dir.glob("*.json"):
                data = json.loads(f.read_text())
                slug = data.get("slug", f.stem)
                self.upsert("category", slug, 1, data)

        # Load merchants
        merchants_file = dataset_path / "merchants_seed.json"
        if merchants_file.exists():
            data = json.loads(merchants_file.read_text())
            for m in data.get("merchants", []):
                mid = m["merchant_id"]
                self.upsert("merchant", mid, 1, m)

        # Load customers
        customers_file = dataset_path / "customers_seed.json"
        if customers_file.exists():
            data = json.loads(customers_file.read_text())
            for c in data.get("customers", []):
                cid = c["customer_id"]
                self.upsert("customer", cid, 1, c)

        # Load triggers
        triggers_file = dataset_path / "triggers_seed.json"
        if triggers_file.exists():
            data = json.loads(triggers_file.read_text())
            for t in data.get("triggers", []):
                tid = t["id"]
                self.upsert("trigger", tid, 1, t)

        logger.info(
            "Dataset loaded: %s",
            {s: self.count(s) for s in self._data},
        )


# ── Singleton (one store per process) ─────────────────────────
_store: ContextStore | None = None


def get_store() -> ContextStore:
    global _store
    if _store is None:
        _store = ContextStore()
    return _store
