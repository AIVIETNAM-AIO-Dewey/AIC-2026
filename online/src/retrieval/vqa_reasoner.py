"""VQA Extractive Reasoner for targeted answer extraction from evidence frames."""

from __future__ import annotations

import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv

from online.src.contracts.query import SearchResult

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class VQAReasoner:
    """Extracts concise answers to VQA questions using multimodal evidence from top keyframe."""

    def __init__(self, gemini_model_id: str = "gemini-2.0-flash") -> None:
        self.gemini_model_id = gemini_model_id
        self._gemini_client = None
        self._init_gemini()

    def _init_gemini(self) -> None:
        if GEMINI_API_KEY:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"VQAReasoner Gemini initialization failed: {e}")
                self._gemini_client = None

    def answer_question(self, question: str, top_result: SearchResult) -> str:
        """Read evidence dossier from top_result and extract a 1-5 word answer."""
        if not question:
            return ""

        # Build evidence context
        evidence_lines = []
        if top_result.dam_full_captions:
            evidence_lines.append(f"Visual Object Captions: {' '.join(top_result.dam_full_captions)}")
        if top_result.speech_evidence:
            evidence_lines.append(f"Spoken Speech Transcript: {top_result.speech_evidence.transcript_raw}")
        if top_result.ocr_text:
            evidence_lines.append(f"On-Screen Graphic Text: {top_result.ocr_text}")

        evidence_text = "\n".join(evidence_lines)
        if not evidence_text:
            evidence_text = f"Keyframe at timestamp {top_result.pts_time_s:.1f}s in video {top_result.video_id}"

        # 1. Try Gemini Flash if available
        if self._gemini_client is not None:
            try:
                from google.genai import types
                prompt = (
                    f"[EVIDENCE FROM RETRIEVED VIDEO KEYFRAME {top_result.video_id}:{top_result.frame_idx}]\n"
                    f"{evidence_text}\n\n"
                    f"[QUESTION]\n{question}\n\n"
                    "INSTRUCTION: Answer the question concisely using ONLY the facts in the evidence above.\n"
                    "FORMAT: Provide ONLY the direct answer in 1 to 5 words (e.g. 'màu xanh', '3 người', 'Công ty ABC'). No explanations."
                )

                response = self._gemini_client.models.generate_content(
                    model=self.gemini_model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=30),
                )
                answer = response.text.strip()
                return answer
            except Exception as e:
                logger.warning(f"Gemini VQA answering error: {e}")

        # 2. Fallback to local heuristic answer extraction
        return self._local_answer_extraction(question, evidence_text)

    def _local_answer_extraction(self, question: str, evidence_text: str) -> str:
        """Fast fallback rule-based extractor."""
        q_lower = question.lower()
        ev_lower = evidence_text.lower()

        # Color questions
        if any(w in q_lower for w in ["màu gì", "color", "màu sắc"]):
            colors_en = ["red", "blue", "green", "white", "black", "yellow", "orange", "pink", "purple", "gray", "checkered", "striped"]
            colors_vi = ["đỏ", "xanh dương", "xanh lá", "trắng", "đen", "vàng", "cam", "hồng", "tím", "xám", "kẻ caro", "sọc"]
            for en, vi in zip(colors_en, colors_vi):
                if en in ev_lower:
                    return f"Màu {vi}"

        # Number / count questions
        if any(w in q_lower for w in ["mấy", "bao nhiêu", "how many", "count"]):
            import re
            numbers = re.findall(r"\b\d+\b", evidence_text)
            if numbers:
                return numbers[0]

        # Return snippet from first evidence sentence
        sentences = evidence_text.split(".")
        return sentences[0].strip() if sentences else "Không có đủ thông tin"
