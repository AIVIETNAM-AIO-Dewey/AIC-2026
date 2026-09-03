"""Branch-3 OCR service with bilingual, auditable FTS retrieval."""

from __future__ import annotations

import threading
import time
from typing import Any

from ...modalities.ocr import OcrFtsIndex
from .contracts import DEFAULT_FINAL_TOP_K, QUERY_ROLES


OCR_RESULT_SCHEMA_VERSION = "branch3.ocr.result.v1"


class Branch3OcrSearch:
    """Own the OCR heavy-search lock and six-role bilingual contract."""

    def __init__(self, index: OcrFtsIndex, search_lock: threading.Lock | None = None) -> None:
        self.index = index
        self._search_lock = search_lock or threading.Lock()

    def health(self, audit_sources: bool = False) -> dict[str, Any]:
        try:
            payload = dict(
                self.index.health(audit_sources=True)
                if audit_sources
                else self.index.health()
            )
        except Exception as error:  # health must never take down /api/health
            payload = {
                "ready": False,
                "production_ready": False,
                "error": str(error),
                "fail_closed": True,
            }
        payload.setdefault("status", "ready" if payload.get("ready") is True else "not_ready")
        payload.setdefault("branch", "branch3")
        payload.setdefault("modality", "ocr")
        payload.setdefault("required", False)
        payload.setdefault("production_ready", False)
        payload.setdefault(
            "artifact_summary",
            {
                "source_total": 0,
                "source_verified": 0,
                "source_failed": 0,
                "hash_recomputed": 0,
            },
        )
        payload.setdefault("fail_closed", payload.get("ready") is not True)
        # OCR source exports do not carry a cryptographically verified
        # immutable checkpoint revision in this phase.  Operational readiness
        # must never be presented as production qualification.
        if payload.get("revision_verified") is not True:
            payload["production_ready"] = False
        return payload

    def execute(
        self,
        query_bundle: dict[str, Any],
        per_stream_top_k: int,
        final_top_k: int,
        *,
        _lock_already_held: bool = False,
    ) -> dict[str, Any]:
        if not 1 <= int(per_stream_top_k) <= 2_000:
            raise ValueError("Branch-3 OCR per_stream_top_k must be between 1 and 2000")
        if not 1 <= int(final_top_k) <= DEFAULT_FINAL_TOP_K:
            raise ValueError(
                f"Branch-3 OCR final_top_k must be between 1 and {DEFAULT_FINAL_TOP_K}"
            )
        acquired = False
        if not _lock_already_held:
            if not self._search_lock.acquire(blocking=False):
                raise RuntimeError("BRANCH3_OCR_SEARCH_BUSY")
            acquired = True
        started = time.perf_counter()
        try:
            if not isinstance(query_bundle, dict):
                raise ValueError("query_bundle must be an object")
            if query_bundle.get("schema_version") != "branch1.query.v1":
                raise ValueError("query_bundle.schema_version must be branch1.query.v1")
            queries = query_bundle.get("queries")
            if not isinstance(queries, list) or len(queries) != len(QUERY_ROLES):
                raise ValueError("OCR requires exactly six query variants")
            by_role: dict[str, dict[str, str]] = {}
            for item in queries:
                if not isinstance(item, dict):
                    raise ValueError("each OCR query variant must be an object")
                role = str(item.get("role") or "")
                if role in by_role:
                    raise ValueError("OCR query roles must contain each role exactly once")
                by_role[role] = {
                    "vi": str(item.get("vi") or "").strip(),
                    "en": str(item.get("en") or "").strip(),
                }
            if set(by_role) != set(QUERY_ROLES):
                raise ValueError("OCR query roles must contain each role exactly once")
            if any(not values[language] for values in by_role.values() for language in ("vi", "en")):
                raise ValueError("OCR Vietnamese and English query variants must not be empty")
            streams = {
                f"{role}:{language}": values[language]
                for role in QUERY_ROLES
                for language, values in (("vi", by_role[role]), ("en", by_role[role]))
            }
            payload = self.index.search_many(
                streams,
                per_stream_top_k=per_stream_top_k,
                final_top_k=final_top_k,
            )
            # Enforce the 500-frame API pool at the service boundary even if
            # an index adapter accidentally returns an oversized result list.
            raw_results = list(payload.get("results") or [])
            results = raw_results[: int(final_top_k)]
            candidate_count_before_gate = max(
                int(payload.get("candidate_frame_count", 0) or 0),
                len(raw_results),
            )
            timing = dict(payload.get("timing") or {})
            timing["ocr_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            timing["total_ms"] = timing["ocr_ms"]
            return {
                "schema_version": OCR_RESULT_SCHEMA_VERSION,
                "fusion_applied": False,
                "reranking_applied": False,
                "query_roles": list(QUERY_ROLES),
                "query_languages": ["vi", "en"],
                "query_streams": [
                    {
                        "role": role,
                        "language": language,
                        "stream": f"{role}:{language}",
                    }
                    for role in QUERY_ROLES
                    for language in ("vi", "en")
                ],
                "stream_count": len(streams),
                "per_stream_top_k": per_stream_top_k,
                "final_top_k": final_top_k,
                "candidate_frame_count": candidate_count_before_gate,
                "stream_counts": payload.get("stream_counts", {}),
                "result_count": len(results),
                "candidate_count_before_gate": candidate_count_before_gate,
                "gate_top_k": DEFAULT_FINAL_TOP_K,
                "future_fusion_eligible": True,
                "timing": timing,
                "results": results,
            }
        finally:
            if acquired:
                self._search_lock.release()

    def _execute_locked(
        self,
        query_bundle: dict[str, Any],
        per_stream_top_k: int,
        final_top_k: int,
    ) -> dict[str, Any]:
        """Run OCR while the shared fusion lock is already held."""

        return self.execute(
            query_bundle,
            per_stream_top_k,
            final_top_k,
            _lock_already_held=True,
        )

    def execute_single(
        self,
        query: str,
        top_k: int,
        *,
        _lock_already_held: bool = False,
    ) -> list[dict[str, Any]]:
        """Run the legacy OCR endpoint through one canonical FTS stream."""

        if not 1 <= int(top_k) <= DEFAULT_FINAL_TOP_K:
            raise ValueError(
                f"Branch-3 OCR top_k must be between 1 and {DEFAULT_FINAL_TOP_K}"
            )
        query = str(query or "").strip()
        if not query:
            raise ValueError("OCR requires one non-empty query")
        acquired = False
        if not _lock_already_held:
            if not self._search_lock.acquire(blocking=False):
                raise RuntimeError("BRANCH3_OCR_SEARCH_BUSY")
            acquired = True
        try:
            payload = self.index.search_many(
                {"legacy": query},
                # Keep the compatibility stream within the same public
                # per-stream ceiling as the six-query endpoint.  Without the
                # upper bound a valid top_k=500 request would ask SQLite for
                # 10,000 rows and be rejected by OcrFtsIndex itself.
                per_stream_top_k=min(2_000, max(500, int(top_k) * 20)),
                final_top_k=int(top_k),
                _allow_single=True,
            )
            return list(payload.get("results") or [])[: int(top_k)]
        finally:
            if acquired:
                self._search_lock.release()


__all__ = ["Branch3OcrSearch", "OCR_RESULT_SCHEMA_VERSION"]
