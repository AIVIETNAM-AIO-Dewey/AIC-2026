"""VQA Extractive Question Answering Reasoner.

Reads question and candidate keyframe multimodal evidence dossier,
extracting concise 1-to-5 word answers (<= 100 chars) for official AIC submission.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class VQAReasoner:
    """Extractive Visual Question Answering Reasoner with Multi-Tier Fallback."""

    def __init__(
        self,
        gemini_model_id: str = "gemini-3.6-flash",
        qwen_model_id: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434/api/chat",
    ):
        self.gemini_model_id = gemini_model_id
        self.qwen_model_id = qwen_model_id
        self.ollama_url = ollama_url
        self._gemini_client = None
        self._init_gemini()

    def _init_gemini(self):
        if GEMINI_API_KEY:
            try:
                from google import genai

                self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                logger.info(f"Initialized VQA Gemini Client with {self.gemini_model_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize VQA Gemini Client: {e}")

    def _answer_with_qwen(self, prompt: str) -> str | None:
        """Local extractive VQA using Ollama Qwen."""
        try:
            payload = {
                "model": self.qwen_model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an official AI Challenge extractive judge. Extract only a concise 1-5 word answer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            }
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                content = res.get("message", {}).get("content", "").strip()
                if content:
                    ans = content.strip().strip('"').strip("'").strip(".")
                    return ans[:100]
        except Exception as e:
            logger.warning(f"Ollama Qwen VQA extraction failed: {e}")
        return None

    def answer_question(
        self,
        question: str,
        evidence_dossier: dict[str, Any],
        raw_query: str | None = None,
    ) -> str:
        """Extract a concise answer (1-5 words, <= 100 chars) from the evidence frame dossier."""
        video_id = evidence_dossier.get("video_id", "")
        pts_time = evidence_dossier.get("pts_time_s", 0.0)
        dam_summary = evidence_dossier.get("dam_summary", "")
        asr_transcript = evidence_dossier.get("asr_transcript", "")
        ocr_text = evidence_dossier.get("ocr_text", "")
        matched_boxes = evidence_dossier.get("matched_boxes", [])

        box_descriptions = "; ".join(
            [f"{b.get('class_entity')}: {b.get('caption')}" for b in matched_boxes]
        )

        context_text = f"""
Video ID: {video_id} (Timestamp: {pts_time:.1f}s)
DAM Visual Entities & Descriptions: {dam_summary} | {box_descriptions}
Spoken Speech Transcript (ASR): {asr_transcript}
On-Screen Text (OCR): {ocr_text}
Original Query / Context: {raw_query or ""}
Question to Answer: {question}
"""

        prompt = f"""You are an official AI Challenge (AIC) VQA Extractive Judge.
Given the evidence extracted from the top retrieved keyframe, provide a concise, exact answer to the question.

RULES:
1. The answer must be extremely concise: 1 to 5 words maximum.
2. Must NOT exceed 100 characters.
3. Return ONLY the answer text, no explanations, no punctuation at the end.
4. If it is a food recipe name, return the Vietnamese dish name (e.g., "Thịt heo xào rau củ", "Măng tây chiên giòn").

{context_text}

ANSWER:"""

        # 1. Try Gemini
        if self._gemini_client:
            try:
                response = self._gemini_client.models.generate_content(
                    model=self.gemini_model_id,
                    contents=prompt,
                )
                if response and response.text:
                    ans = response.text.strip().strip('"').strip("'").strip(".")
                    return ans[:100]
            except Exception as e:
                logger.warning(f"Gemini VQA extraction failed ({e}). Falling back to local Qwen...")

        # 2. Try Local Ollama Qwen
        qwen_ans = self._answer_with_qwen(prompt)
        if qwen_ans:
            return qwen_ans

        # 3. Fallback to heuristic extraction
        if "công thức" in asr_transcript.lower():
            match = re.search(r"công thức (?:của |món )?([^,\.\n]+)", asr_transcript, flags=re.I)
            if match:
                return match.group(1).strip()[:100]

        return "Món ăn gia đình"
