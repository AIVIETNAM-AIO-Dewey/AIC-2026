"""Unified Model Embedding & Inference Wrappers for Online Retrieval."""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton-style registry managing model weights on device."""

    def __init__(
        self,
        bge_model_id: str = "BAAI/bge-m3",
        siglip_model_id: str = "google/siglip2-base-patch16-224",
        reranker_model_id: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.bge_model_id = bge_model_id
        self.siglip_model_id = siglip_model_id
        self.reranker_model_id = reranker_model_id

        self._bge_tokenizer = None
        self._bge_model = None

        self._siglip_tokenizer = None
        self._siglip_model = None

        self._reranker_tokenizer = None
        self._reranker_model = None

        logger.info(f"ModelRegistry initialized with device: {self.device}")

    # --- BGE-M3 (Multilingual Dense Text Embeddings) ---
    def get_bge_m3(self):
        if self._bge_model is None:
            logger.info(f"Loading BGE-M3 ({self.bge_model_id}) on {self.device}...")
            self._bge_tokenizer = AutoTokenizer.from_pretrained(self.bge_model_id)
            self._bge_model = AutoModel.from_pretrained(self.bge_model_id).to(self.device).eval()
        return self._bge_tokenizer, self._bge_model

    def encode_bge_m3(self, texts: list[str], max_length: int = 512, batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        tokenizer, model = self.get_bge_m3()
        all_embeds = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = model(**inputs)
                # Normalized CLS token
                embeds = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1).cpu().numpy()
                all_embeds.append(embeds)

        return np.vstack(all_embeds).astype(np.float32)

    # --- SigLIP-2 Text Encoder (768-d Visual Query Embeddings) ---
    def get_siglip(self):
        if self._siglip_model is None:
            logger.info(f"Loading SigLIP-2 ({self.siglip_model_id}) on {self.device}...")
            self._siglip_tokenizer = AutoTokenizer.from_pretrained(self.siglip_model_id)
            self._siglip_model = AutoModel.from_pretrained(self.siglip_model_id).to(self.device).eval()
        return self._siglip_tokenizer, self._siglip_model

    def encode_siglip_text(self, texts: list[str], max_length: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        tokenizer, model = self.get_siglip()
        with torch.no_grad():
            inputs = tokenizer(
                texts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            if hasattr(model, "get_text_features"):
                out = model.get_text_features(**inputs)
                if hasattr(out, "pooler_output"):
                    embeds = out.pooler_output
                elif isinstance(out, torch.Tensor):
                    embeds = out
                else:
                    embeds = out[0]
            elif hasattr(model, "text_model"):
                out = model.text_model(**inputs)
                embeds = out.pooler_output if hasattr(out, "pooler_output") else out[0][:, 0]
            else:
                out = model(**inputs)
                embeds = out.pooler_output if hasattr(out, "pooler_output") else out[0][:, 0]

            embeds = F.normalize(embeds, p=2, dim=1).cpu().numpy()
            return embeds.astype(np.float32)

    # --- BGE-Reranker-v2-m3 (Stage 2 Cross-Attention Re-ranking) ---
    def get_reranker(self):
        if self._reranker_model is None:
            logger.info(f"Loading BGE-Reranker ({self.reranker_model_id}) on {self.device}...")
            self._reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_id)
            self._reranker_model = (
                AutoModelForSequenceClassification.from_pretrained(self.reranker_model_id)
                .to(self.device)
                .eval()
            )
        return self._reranker_tokenizer, self._reranker_model

    def rerank_pairs(self, query_doc_pairs: list[tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        if not query_doc_pairs:
            return np.empty(0, dtype=np.float32)

        tokenizer, model = self.get_reranker()
        all_scores = []

        for i in range(0, len(query_doc_pairs), batch_size):
            batch = query_doc_pairs[i : i + batch_size]
            with torch.no_grad():
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                logits = model(**inputs, return_dict=True).logits.view(-1)
                scores = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(scores)

        return np.concatenate(all_scores).astype(np.float32)

    def warmup(self) -> None:
        """Pre-load all model weights into device memory for zero-latency first query."""
        logger.info(f"⚡ Pre-warming models on {self.device}...")
        self.encode_bge_m3(["warmup test"])
        self.encode_siglip_text(["warmup test"])
        self.rerank_pairs([("query", "document")])
        logger.info("✅ All models loaded and warmed up in GPU memory!")
