"""Health adapter for the canonical KIS fusion service.

Keeping this tiny adapter separate makes the health route easy to exercise in
isolation while leaving :class:`KisFusionSearch` as the single owner of the
readiness policy.  The adapter deliberately does not probe branches itself;
doing so would duplicate Qdrant/index checks and could produce a snapshot
different from the one used by a search request.
"""

from __future__ import annotations

from typing import Any


def fusion_health(service: Any) -> dict[str, Any]:
    """Return a fail-closed health payload for ``service``.

    Route code can use this helper without having to know whether the service
    has already been constructed during application lifespan startup.  Any
    malformed or exceptional response is represented as not-ready rather than
    escaping as a 500 from a health endpoint.
    """

    if service is None:
        return {
            "schema_version": "kis.fusion.health.v1",
            "branch": "final_fusion",
            "task_type": "KIS",
            "status": "starting",
            "ready": False,
            "required": False,
            "production_ready": False,
            "fail_closed": True,
            "error": "KIS fusion service is not initialized",
        }
    try:
        payload = service.health()
    except Exception as error:  # pragma: no cover - defensive route boundary
        return {
            "schema_version": "kis.fusion.health.v1",
            "branch": "final_fusion",
            "task_type": "KIS",
            "status": "not_ready",
            "ready": False,
            "required": False,
            "production_ready": False,
            "fail_closed": True,
            "error": str(error),
        }
    if not isinstance(payload, dict):
        return {
            "schema_version": "kis.fusion.health.v1",
            "branch": "final_fusion",
            "task_type": "KIS",
            "status": "not_ready",
            "ready": False,
            "required": False,
            "production_ready": False,
            "fail_closed": True,
            "error": "invalid KIS fusion health response",
        }
    result = dict(payload)
    result.setdefault("schema_version", "kis.fusion.health.v1")
    result.setdefault("branch", "final_fusion")
    result.setdefault("task_type", "KIS")
    result.setdefault("required", False)
    result.setdefault("ready", False)
    result.setdefault("production_ready", False)
    result.setdefault("status", "ready" if result["ready"] is True else "not_ready")
    result.setdefault("fail_closed", result["ready"] is not True)
    return result


__all__ = ["fusion_health"]
