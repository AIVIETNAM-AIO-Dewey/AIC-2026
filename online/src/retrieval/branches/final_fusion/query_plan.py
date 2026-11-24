"""Deterministic authoring plan for the KIS Fusion UI.

This module turns one overall query into the existing six-role bundle and an
optional ordered event list.  It performs no retrieval, embedding, model
inference, or translation.  A matching manually authored bilingual bundle is
preserved verbatim so preparing ordered events never degrades a richer query.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from ...infrastructure.query_parser import LocalQueryParser
from ..branch1.contracts import QUERY_ROLES

MAX_QUERY_CHARACTERS = 4096


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())[:MAX_QUERY_CHARACTERS].strip()


def _normalized(value: Any) -> str:
    return _clean(value).casefold()


def _query_by_role(query_bundle: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if (
        not isinstance(query_bundle, dict)
        or query_bundle.get("schema_version") != "branch1.query.v1"
    ):
        return {}
    queries = query_bundle.get("queries")
    if not isinstance(queries, list):
        return {}
    by_role = {str(query.get("role") or ""): query for query in queries if isinstance(query, dict)}
    if set(by_role) != set(QUERY_ROLES) or len(by_role) != len(QUERY_ROLES):
        return {}
    if any(
        not _clean(by_role[role].get(language)) for role in QUERY_ROLES for language in ("vi", "en")
    ):
        return {}
    return by_role


def _split_bilingual_source(source: str) -> tuple[str, str] | None:
    if "||" not in source:
        return None
    vi, en = source.split("||", 1)
    vi_text = _clean(vi)
    en_text = _clean(en)
    return (vi_text, en_text) if vi_text and en_text else None


def _source_languages(
    source: str,
    existing_by_role: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    explicit = _split_bilingual_source(source)
    if explicit is not None:
        return explicit
    original = existing_by_role.get("original", {})
    original_vi = _clean(original.get("vi"))
    original_en = _clean(original.get("en"))
    source_normalized = _normalized(source)
    if source_normalized and source_normalized == _normalized(original_vi):
        return original_vi, original_en
    if source_normalized and source_normalized == _normalized(original_en):
        return original_vi, original_en
    return source, source


def _bundle_matches_source(
    source: str,
    existing_by_role: dict[str, dict[str, Any]],
) -> bool:
    if not existing_by_role:
        return False
    original = existing_by_role["original"]
    original_vi = _clean(original.get("vi"))
    original_en = _clean(original.get("en"))
    candidates = {
        _normalized(original_vi),
        _normalized(original_en),
        _normalized(f"{original_vi} || {original_en}"),
    }
    explicit = _split_bilingual_source(source)
    if explicit is not None:
        return _normalized(explicit[0]) == _normalized(original_vi) and _normalized(
            explicit[1]
        ) == _normalized(original_en)
    return _normalized(source) in candidates


def _shared_context(text: str) -> str:
    """Extract a conservative shared prefix without inventing new concepts."""

    clean = _clean(text)
    marker_patterns = (
        r"\bE\s*1\s*[:.)-]",
        r"(?:^|\s)1\s*[.)-]\s+",
        r"\b(?:đầu tiên|first)\b",
    )
    cut_positions = [
        match.start()
        for pattern in marker_patterns
        if (match := re.search(pattern, clean, flags=re.IGNORECASE)) is not None
    ]
    colon = clean.find(":")
    if colon >= 0:
        cut_positions.append(colon)
    for position in sorted(cut_positions):
        prefix = clean[:position].strip(" ,.;:-")
        if len(prefix.split()) >= 3:
            return prefix
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean) if part.strip()]
    if len(sentences) > 1 and len(sentences[0].split()) >= 3:
        return sentences[0]
    return clean


def _event_texts(parsed: Any) -> list[str]:
    values: list[str] = []
    for event in list(getattr(parsed, "trake_events", []) or [])[:6]:
        text = _clean(getattr(event, "scene_en", "") or getattr(event, "description", ""))
        if text:
            values.append(text)
    return values


def _role_text(parsed: Any, source: str, role: str) -> str:
    objects = [_clean(value) for value in list(getattr(parsed, "objects_en", []) or [])]
    objects = [value for value in objects if value]
    events = _event_texts(parsed)
    keywords = [_clean(value) for value in list(getattr(parsed, "ocr_keywords", []) or [])]
    keywords = [value for value in keywords if value]
    if role == "original":
        return source
    if role == "entity":
        return ", ".join(objects) or source
    if role == "action":
        return " -> ".join(events) or source
    if role == "context":
        return _shared_context(source)
    if role == "synonym":
        # The deterministic parser cannot invent semantic synonyms.  Reusing
        # its extracted entities is honest and keeps the required stream valid.
        return ", ".join(objects) or source
    if role == "keyword":
        return ", ".join([*keywords, *objects]) or source
    raise ValueError(f"Unknown KIS query role: {role}")


def _ordered_events(vi_parsed: Any, en_parsed: Any) -> list[dict[str, Any]]:
    vi_events = list(getattr(vi_parsed, "trake_events", []) or [])[:6]
    en_events = list(getattr(en_parsed, "trake_events", []) or [])[:6]
    if len(vi_events) < 2 and len(en_events) < 2:
        return []

    primary = vi_events if len(vi_events) >= 2 else en_events
    paired = len(vi_events) == len(en_events) == len(primary)
    events: list[dict[str, Any]] = []
    for index, primary_event in enumerate(primary):
        vi_event = vi_events[index] if paired else primary_event
        en_event = en_events[index] if paired else primary_event
        vi_text = _clean(getattr(vi_event, "scene_en", "") or getattr(vi_event, "description", ""))
        en_text = _clean(getattr(en_event, "scene_en", "") or getattr(en_event, "description", ""))
        if not vi_text or not en_text:
            continue
        events.append(
            {
                "order": len(events) + 1,
                "description": en_text,
                "vi": vi_text,
                "en": en_text,
            }
        )
    return events if len(events) >= 2 else []


def build_kis_query_plan(
    *,
    query: str,
    task_type: str,
    parser: LocalQueryParser,
    query_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a no-inference KIS authoring plan from one overall query."""

    source = _clean(query)
    if not source:
        raise ValueError("Overall KIS query cannot be empty")
    existing_by_role = _query_by_role(query_bundle)
    vi_source, en_source = _source_languages(source, existing_by_role)
    vi_parsed = parser.parse(vi_source, task_type="KIS")
    en_parsed = (
        vi_parsed
        if _normalized(en_source) == _normalized(vi_source)
        else parser.parse(en_source, task_type="KIS")
    )

    preserve_bundle = _bundle_matches_source(source, existing_by_role)
    if preserve_bundle:
        planned_bundle = deepcopy(query_bundle)
    else:
        planned_bundle = {
            "schema_version": "branch1.query.v1",
            "queries": [
                {
                    "role": role,
                    "vi": _role_text(vi_parsed, vi_source, role)[:MAX_QUERY_CHARACTERS],
                    "en": _role_text(en_parsed, en_source, role)[:MAX_QUERY_CHARACTERS],
                }
                for role in QUERY_ROLES
            ],
        }

    events = _ordered_events(vi_parsed, en_parsed)
    return {
        "schema_version": "kis.query-plan.v1",
        "task_type": task_type,
        "source_query": source,
        "query_bundle": planned_bundle,
        "bundle_source": "preserved_matching_bundle" if preserve_bundle else "local_deterministic",
        "bundle_preserved": preserve_bundle,
        "events": events,
        "event_count": len(events),
        "is_temporal": len(events) >= 2,
        "shared_context": {
            "vi": str(
                next(item for item in planned_bundle["queries"] if item["role"] == "context")["vi"]
            ),
            "en": str(
                next(item for item in planned_bundle["queries"] if item["role"] == "context")["en"]
            ),
        },
        "translation_generated": False,
        "external_llm_used": False,
        "retrieval_invoked": False,
    }


__all__ = ["build_kis_query_plan"]
