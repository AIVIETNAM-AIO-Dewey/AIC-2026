"""ONNX adapter for multilingual-e5-base, loaded only by ingestion/query paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class E5OnnxEncoder:
    model_id = "intfloat/multilingual-e5-base"

    def __init__(self, tokenizer: object, model: object) -> None:
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def from_pretrained(cls, *, model_path: str | Path, device: str = "cpu") -> E5OnnxEncoder:
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - environment-specific
            raise RuntimeError("Install backend[encoders] to use the E5 ONNX encoder") from error
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            providers.insert(0, "CUDAExecutionProvider")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = ORTModelForFeatureExtraction.from_pretrained(
            str(model_path), local_files_only=True, provider=providers[0]
        )
        return cls(tokenizer, model)

    def encode(self, texts: list[str], *, query: bool) -> np.ndarray:
        prefix = "query: " if query else "passage: "
        inputs = self.tokenizer(
            [prefix + text for text in texts], padding=True, truncation=True, return_tensors="np"
        )
        values = np.asarray(self.model(**inputs).last_hidden_state)
        mask = np.asarray(inputs["attention_mask"])[..., None]
        pooled = (values * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
