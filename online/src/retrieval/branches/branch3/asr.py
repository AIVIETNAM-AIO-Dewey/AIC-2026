"""Branch-3 ASR service orchestration."""

from __future__ import annotations

import threading
import time
from typing import Any

from ...modalities.asr import AsrFtsIndex
from .contracts import ASR_RESULT_SCHEMA_VERSION, DEFAULT_FINAL_TOP_K, QUERY_ROLES


class Branch3AsrSearch:
    def __init__(self, index: AsrFtsIndex, search_lock: threading.Lock | None = None) -> None:
        self.index = index
        self._search_lock = search_lock or threading.Lock()

    def health(self) -> dict[str, Any]:
        # The ASR index performs several filesystem/SQLite validations.  A
        # malformed manifest, a transient atomic publication, or an adapter
        # exception must never bubble out of the health route and tear down
        # the API connection.  Keep the component fail-closed just like OCR.
        try:
            payload = dict(self.index.health())
        except Exception as error:  # health must never take down /api/health
            payload = {
                "ready": False,
                "production_ready": False,
                "error": str(error),
                "fail_closed": True,
            }
        payload.setdefault("status", "ready" if payload.get("ready") is True else "not_ready")
        payload.setdefault("branch", "branch3")
        payload.setdefault("modality", "asr")
        payload.setdefault("required", False)
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
            raise ValueError("Branch-3 ASR per_stream_top_k must be between 1 and 2000")
        if not 1 <= int(final_top_k) <= DEFAULT_FINAL_TOP_K:
            raise ValueError(
                f"Branch-3 ASR final_top_k must be between 1 and {DEFAULT_FINAL_TOP_K}"
            )
        acquired = False
        if not _lock_already_held:
            if not self._search_lock.acquire(blocking=False):
                raise RuntimeError("BRANCH3_ASR_SEARCH_BUSY")
            acquired = True
        started = time.perf_counter()
        try:
            if not isinstance(query_bundle, dict):
                raise ValueError("query_bundle must be an object")
            if query_bundle.get("schema_version") != "branch1.query.v1":
                raise ValueError("query_bundle.schema_version must be branch1.query.v1")
            queries = query_bundle.get("queries")
            if not isinstance(queries, list) or len(queries) != len(QUERY_ROLES):
                raise ValueError("ASR requires exactly six query variants")
            if any(not isinstance(item, dict) for item in queries):
                raise ValueError("each ASR query variant must be an object")
            query_by_role = {
                str(item.get("role") or ""): {
                    "vi": str(item.get("vi") or "").strip(),
                    "en": str(item.get("en") or "").strip(),
                }
                for item in queries
            }
            if set(query_by_role) != set(QUERY_ROLES) or len(query_by_role) != len(QUERY_ROLES):
                raise ValueError("ASR query roles must contain each role exactly once")
            if any(not value["vi"] or not value["en"] for value in query_by_role.values()):
                raise ValueError("ASR Vietnamese and English query variants must not be empty")
            stream_queries = {
                f"{role}:{language}": value[language]
                for role in QUERY_ROLES
                for value in (query_by_role[role],)
                for language in ("vi", "en")
            }
            payload = self.index.search_many(
                stream_queries,
                per_stream_top_k=per_stream_top_k,
                final_top_k=final_top_k,
            )
            # Keep the API gate defensive even when a custom/index adapter
            # returns more rows than requested.  The candidate count remains
            # the pre-gate diagnostic; only the public result pool is sliced.
            results = list(payload.get("results") or [])[: int(final_top_k)]
            timing = dict(payload.get("timing") or {})
            timing["asr_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            timing["total_ms"] = timing["asr_ms"]
            return {
                "schema_version": ASR_RESULT_SCHEMA_VERSION,
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
                "stream_count": len(stream_queries),
                "per_stream_top_k": per_stream_top_k,
                "final_top_k": final_top_k,
                "candidate_segment_count": payload.get("candidate_segment_count", 0),
                "candidate_frame_count": payload.get("candidate_frame_count", 0),
                "stream_counts": payload.get("stream_counts", {}),
                "result_count": len(results),
                "candidate_count_before_gate": payload.get("candidate_frame_count", 0),
                "gate_top_k": 500,
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
        """Run ASR while the shared fusion lock is already held."""

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
        """Run the compatibility one-query ASR endpoint through the shared lock."""

        acquired = False
        if not _lock_already_held:
            if not self._search_lock.acquire(blocking=False):
                raise RuntimeError("BRANCH3_ASR_SEARCH_BUSY")
            acquired = True
        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("ASR query must not be empty")
            if int(top_k) < 1 or int(top_k) > 500:
                raise ValueError("ASR top_k must be between 1 and 500")
            return self.index.search(query.strip(), int(top_k))
        finally:
            if acquired:
                self._search_lock.release()


__all__ = ["Branch3AsrSearch"]
