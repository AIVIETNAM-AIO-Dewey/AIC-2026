"""Stable identifiers make re-ingestion idempotent."""

from __future__ import annotations

from uuid import NAMESPACE_URL, UUID, uuid5


def point_id(*, collection: str, source_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"aic26/{collection}/{source_id}")
