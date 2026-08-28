"""Model Registry for Query Embeddings (SigLIP-2 & BGE-M3).

Encodes text sub-queries on Mac MPS / CPU with PyTorch:
- SigLIP-2 text encoder -> 768-d float32 L2-normalized
- BGE-M3 text encoder -> 1024-d float32 L2-normalized
- BGE-Reranker-v2-m3 -> Cross-encoder logits/probabilities
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton model registry managing embedding and re-ranking models."""

    _instance: ModelRegistry | None = None

    def __init__(
        self,
        siglip_id: str = "google/siglip2-base-patch16-224",
        siglip_revision: str = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        bge_id: str = "BAAI/bge-m3",
        reranker_id: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.siglip_id = siglip_id
        self.siglip_revision = siglip_revision
        self.bge_id = bge_id
        self.reranker_id = reranker_id

        self._siglip_tokenizer = None
        self._siglip_model = None

        self._bge_tokenizer = None
        self._bge_model = None

        self._reranker_tokenizer = None
        self._reranker_model = None

        logger.info(f"⚡ ModelRegistry initialized (Device: {self.device})")

    @classmethod
    def get_instance(cls, **kwargs) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    # ──────────────────────────────────────────────────────────────────────────
    # 1. SigLIP-2 Text Embedding (768-d)
    # ──────────────────────────────────────────────────────────────────────────
    def _load_siglip(self):
        if self._siglip_model is None:
            logger.info(
                "Loading SigLIP-2 model (%s @ %s)...",
                self.siglip_id,
                self.siglip_revision,
            )
            self._siglip_tokenizer = AutoTokenizer.from_pretrained(
                self.siglip_id,
                revision=self.siglip_revision,
                trust_remote_code=False,
            )
            self._siglip_model = AutoModel.from_pretrained(
                self.siglip_id,
                revision=self.siglip_revision,
                trust_remote_code=False,
            ).to(self.device)
            self._siglip_model.eval()
            logger.info("✅ SigLIP-2 loaded successfully!")

    @torch.inference_mode()
    def embed_siglip_text(self, text: str) -> np.ndarray:
        """Embed text using SigLIP-2 text encoder into 768-d normalized float32 vector."""
        self._load_siglip()
        inputs = self._siglip_tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)

        # Extract text features
        if hasattr(self._siglip_model, "get_text_features"):
            features = self._siglip_model.get_text_features(**inputs)
        else:
            features = self._siglip_model(**inputs)

        if isinstance(features, torch.Tensor):
            text_embeds = features
        elif hasattr(features, "pooler_output") and features.pooler_output is not None:
            text_embeds = features.pooler_output
        elif hasattr(features, "last_hidden_state"):
            text_embeds = features.last_hidden_state[:, 0]
        else:
            text_embeds = features[0]

        normalized = F.normalize(text_embeds, p=2, dim=-1)
        return normalized.cpu().to(torch.float32).numpy()[0]

    # ──────────────────────────────────────────────────────────────────────────
    # 2. BGE-M3 Dense Embedding (1024-d)
    # ──────────────────────────────────────────────────────────────────────────
    def _load_bge(self):
        if self._bge_model is None:
            logger.info(f"Loading BGE-M3 model ({self.bge_id})...")
            self._bge_tokenizer = AutoTokenizer.from_pretrained(self.bge_id)
            self._bge_model = AutoModel.from_pretrained(self.bge_id).to(self.device)
            self._bge_model.eval()
            logger.info("✅ BGE-M3 loaded successfully!")

    @torch.inference_mode()
    def embed_bge_text(self, texts: list[str] | str) -> np.ndarray:
        """Embed list of strings (or single string) into 1024-d normalized float32 vectors."""
        self._load_bge()
        if isinstance(texts, str):
            text_list = [texts if texts.strip() else " "]
            single = True
        else:
            text_list = [t if (t and t.strip()) else " " for t in texts]
            single = False

        inputs = self._bge_tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        outputs = self._bge_model(**inputs)
        cls_repr = outputs.last_hidden_state[:, 0]
        normalized = F.normalize(cls_repr, p=2, dim=-1)
        res = normalized.cpu().to(torch.float32).numpy()
        return res[0] if single else res

    # ──────────────────────────────────────────────────────────────────────────
    # 3. BGE-Reranker-v2-m3 Cross-Encoder
    # ──────────────────────────────────────────────────────────────────────────
    def _load_reranker(self):
        if self._reranker_model is None:
            logger.info(f"Loading BGE-Reranker ({self.reranker_id})...")
            self._reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_id)
            self._reranker_model = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_id
            ).to(self.device)
            self._reranker_model.eval()
            logger.info("✅ BGE-Reranker loaded successfully!")

    @torch.inference_mode()
    def compute_rerank_scores(self, query: str, documents: list[str]) -> list[float]:
        """Compute cross-encoder relevance scores in [0.0, 1.0] for query vs candidate documents."""
        if not documents:
            return []
        self._load_reranker()
        pairs = [[query, doc] for doc in documents]
        inputs = self._reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        outputs = self._reranker_model(**inputs)
        logits = outputs.logits.squeeze(-1)
        scores = torch.sigmoid(logits).cpu().tolist()
        return [scores] if isinstance(scores, float) else scores
