"""Strict, retry-bounded GPT-4o adapter. No raw frames or keys are logged."""

from __future__ import annotations

import json
import time
from typing import Any

from aic2026.contracts.query import QuerySpec

from ..retrieval.models import SearchHit


class CapabilityUnavailable(RuntimeError):
    pass


class GPT4oAdapter:
    def __init__(
        self, *, api_key: str | None, model: str, timeout_s: float = 30.0, client: Any | None = None
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        if client is not None:
            self.client = client
        elif api_key:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)
        else:
            self.client = None

    def _create(self, **kwargs: Any) -> Any:
        if self.client is None:
            raise CapabilityUnavailable("OpenAI is not configured")
        transient: Exception | None = None
        for attempt in range(3):
            try:
                return self.client.responses.create(**kwargs)
            except (
                Exception
            ) as error:  # SDK categorizes transport/status errors; retry is intentionally bounded.
                transient = error
                if attempt == 2:
                    break
                time.sleep(0.25 * (2**attempt))
        raise CapabilityUnavailable("OpenAI request failed after retries") from transient

    @staticmethod
    def _json(response: Any) -> dict[str, Any]:
        text = getattr(response, "output_text", None)
        if not text:
            raise CapabilityUnavailable("OpenAI returned no structured output")
        return json.loads(text)

    def parse(self, *, task_type: str, raw_query_vi: str) -> QuerySpec:
        schema = QuerySpec.model_json_schema()
        response = self._create(
            model=self.model,
            store=False,
            input=[
                {
                    "role": "user",
                    "content": f"Parse this Vietnamese {task_type} retrieval query: {raw_query_vi}",
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "aic26_query",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        return QuerySpec.model_validate(self._json(response))

    def answer(
        self, *, query: QuerySpec, frames: list[SearchHit], use_images: bool
    ) -> tuple[str, float, list[str]]:
        del use_images
        context = [
            {
                "frame_uid": hit.frame_uid,
                "evidence": [item.text for item in hit.evidence if item.text],
            }
            for hit in frames[:8]
        ]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "confidence", "evidence_frame_uids"],
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_frame_uids": {"type": "array", "items": {"type": "string"}},
            },
        }
        response = self._create(
            model=self.model,
            store=False,
            input=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {query.question_en}\n"
                        f"Evidence: {json.dumps(context, ensure_ascii=False)}"
                    ),
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "aic26_answer",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        parsed = self._json(response)
        allowed = {hit.frame_uid for hit in frames[:8]}
        evidence = [uid for uid in parsed["evidence_frame_uids"] if uid in allowed]
        return str(parsed["answer"]), float(parsed["confidence"]), evidence
