"""Query Decomposer & Router for Multimodal Video Retrieval.

Supports:
1. Gemini 3.6 Flash (Primary High-Precision Cloud LLM)
2. Qwen 2.5 1.5B via Ollama (Local Metal GPU Engine for Offline Competition)
3. Regex Rule Fallback (Zero-dependency local backup)
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

from dotenv import load_dotenv

from online.src.contracts.query import ParsedQuery, TaskType, TrakeEvent

ROOT_DIR = Path(__file__).resolve().parents[3]
load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class QueryParser:
    """Intelligent Query Decomposer supporting Gemini 3.6 Flash, Local Qwen, and Rule fallback."""

    def __init__(
        self,
        gemini_model_id: str = "gemini-3.6-flash",
        qwen_model_id: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434/api/chat",
    ) -> None:
        self.gemini_model_id = gemini_model_id
        self.qwen_model_id = qwen_model_id
        self.ollama_url = ollama_url
        self._gemini_client = None
        self._init_gemini()

    def _init_gemini(self) -> None:
        if GEMINI_API_KEY:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                logger.info(f"Initialized Gemini Client with model: {self.gemini_model_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Client: {e}")
                self._gemini_client = None

    def parse(
        self,
        query_text: str,
        task_type: Optional[TaskType] = None,
        engine: str = "gemini",  # "gemini", "qwen", or "rule"
    ) -> ParsedQuery:
        """Parse raw query into structured 4-channel sub-queries."""
        query_text = query_text.strip()
        if not query_text:
            return ParsedQuery(task_type=task_type or "KIS", original_query="")

        if task_type is None:
            task_type = self._detect_task_type(query_text)

        # 1. Gemini Engine
        if engine == "gemini" and self._gemini_client is not None:
            try:
                return self._parse_with_gemini(query_text, task_type)
            except Exception as e:
                logger.warning(f"Gemini API parsing failed ({e}). Trying Qwen...")
                engine = "qwen"

        # 2. Local Qwen Engine (via Ollama)
        if engine == "qwen":
            try:
                return self._parse_with_qwen(query_text, task_type)
            except Exception as e:
                logger.warning(f"Local Qwen parsing failed ({e}). Falling back to rule parser.")

        # 3. Rule Fallback
        return self._parse_local(query_text, task_type)

    def _detect_task_type(self, query_text: str) -> TaskType:
        q_lower = query_text.lower()
        if bool(re.search(r"E\d+:|khoảnh khắc đầu tiên|khoảnh khắc tiếp theo|sau đó|tiếp theo|sau đấy", query_text, re.I)):
            return "TRAKE"
        if any(w in q_lower for w in ["hỏi", "là gì", "màu gì", "bao nhiêu", "ai", "ở đâu", "khi nào", "?", "what", "which", "how many"]):
            return "VQA"
        return "KIS"

    def _get_system_prompt(self, task_type: TaskType) -> str:
        return (
            "You are an expert query parser for a High-Speed Multimodal Video Retrieval Engine.\n"
            "Your job is to analyze the user's query and decompose it into distinct sub-queries for 4 specialized search channels:\n\n"
            "CHANNELS & LANGUAGE REQUIREMENTS:\n"
            "1. 'global_scene_en' (ENGLISH for SigLIP-2): High-level background, lighting, camera angle, setting, environmental scene. TRANSLATE TO DESCRIPTIVE ENGLISH.\n"
            "2. 'objects_en' (ENGLISH for DAM): List of specific visual objects, people, clothing, vehicles, ingredients, accessories, actions. TRANSLATE TO CONCISE ENGLISH OBJECT PHRASES.\n"
            "3. 'speech_vi' (VIETNAMESE for Audio ASR): Dialogue keywords, spoken discussion topics, voiceover speech. KEEP IN VIETNAMESE (strip question preambles).\n"
            "4. 'ocr_keywords' (VIETNAMESE/ORIGINAL for OCR): Exact text, recipe titles, numbers, brand names, on-screen subtitles, lower-thirds.\n\n"
            "TASK SPECIFIC HANDLING:\n"
            "- If task is VQA: extract 'vqa_question' (the exact question to be answered).\n"
            "- If task is TRAKE or query has sequential steps (E1, E2, ...): extract 'trake_events' array where each item has:\n"
            "  - order (int 1, 2, 3...)\n"
            "  - description (original text)\n"
            "  - scene_en (English visual setting)\n"
            "  - objects_en (English object list)\n"
            "  - speech_vi (Vietnamese speech if any)\n"
            "  - ocr_keywords (Vietnamese on-screen text if any)\n\n"
            "OUTPUT FORMAT: Return ONLY a valid JSON object matching the requested schema."
        )

    def _parse_with_gemini(self, query_text: str, task_type: TaskType) -> ParsedQuery:
        from google.genai import types

        system_instruction = self._get_system_prompt(task_type)
        prompt = (
            f"Assigned Task Type: {task_type}\n"
            f"Raw User Query: {query_text}\n\n"
            "Decompose this query into the exact structured JSON format."
        )

        response = self._gemini_client.models.generate_content(
            model=self.gemini_model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        data = json.loads(response.text)
        return self._build_parsed_query(data, query_text, task_type)

    def _parse_with_qwen(self, query_text: str, task_type: TaskType) -> ParsedQuery:
        """Local inference using Qwen 2.5 on Ollama."""
        system_instruction = self._get_system_prompt(task_type)
        prompt = (
            f"Assigned Task Type: {task_type}\n"
            f"Raw User Query: {query_text}\n\n"
            "Output JSON with keys: global_scene_en, objects_en, speech_vi, ocr_keywords, trake_events, vqa_question."
        )

        payload = {
            "model": self.qwen_model_id,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0},
        }

        req = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data.get("message", {}).get("content", "{}").strip()
            
            # Robust JSON extraction (strip ```json ... ``` if present)
            if "```" in content:
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                if m:
                    content = m.group(1)
            
            data = json.loads(content)

        return self._build_parsed_query(data, query_text, task_type)

    def _build_parsed_query(self, data: dict, query_text: str, task_type: TaskType) -> ParsedQuery:
        trake_events = []
        raw_events = data.get("trake_events") or []
        for idx, ev in enumerate(raw_events, 1):
            if isinstance(ev, dict):
                trake_events.append(
                    TrakeEvent(
                        order=ev.get("order") or idx,
                        description=ev.get("description") or "",
                        scene_en=ev.get("scene_en") or "",
                        objects_en=ev.get("objects_en") or [],
                        speech_vi=ev.get("speech_vi") or "",
                        ocr_keywords=ev.get("ocr_keywords") or [],
                    )
                )

        weights = {"vis": 0.35, "dam": 0.30, "asr": 0.35, "ocr": 0.00}
        speech_raw = data.get("speech_vi")
        if isinstance(speech_raw, list):
            speech_v = " ".join([str(s) for s in speech_raw if s])
        else:
            speech_v = str(speech_raw) if speech_raw is not None else ""

        ocr_raw = data.get("ocr_keywords")
        if isinstance(ocr_raw, list):
            ocr_kw = [str(k) for k in ocr_raw if k]
        elif isinstance(ocr_raw, str):
            ocr_kw = [ocr_raw]
        else:
            ocr_kw = []

        objs_raw = data.get("objects_en")
        if isinstance(objs_raw, list):
            objs_en = [str(o) for o in objs_raw if o]
        elif isinstance(objs_raw, str):
            objs_en = [objs_raw]
        else:
            objs_en = []

        if ocr_kw:
            weights["ocr"] = 0.20
            weights["vis"] = 0.35
            weights["dam"] = 0.35
            weights["asr"] = 0.10
        if speech_v and len(speech_v) > 15:
            weights["asr"] = 0.35
            weights["vis"] = 0.35
            weights["dam"] = 0.30
        if not objs_en:
            weights["dam"] = 0.10
            weights["vis"] = 0.60

        is_vi = bool(re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", query_text, re.I))

        return ParsedQuery(
            task_type=task_type,
            language="vi" if is_vi else "en",
            original_query=query_text,
            global_scene_en=data.get("global_scene_en") or query_text,
            objects_en=objs_en if objs_en else [query_text],
            ocr_keywords=ocr_kw,
            speech_vi=speech_v,
            is_temporal_trake=bool(trake_events or task_type == "TRAKE"),
            trake_events=trake_events,
            vqa_question=data.get("vqa_question") or (query_text if task_type == "VQA" else ""),
            weights=weights,
        )

    def _parse_local(self, query_text: str, task_type: TaskType) -> ParsedQuery:
        """Fast offline rule-based parser."""
        is_vi = bool(re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", query_text, re.I))
        ocr_matches = re.findall(r'["\']([^"\']+)["\']|\b\d+g\b|\b\d+\s*kg\b', query_text)
        flat_ocr = [m if isinstance(m, str) else m[0] for m in ocr_matches if m]

        trake_events = []
        e_matches = re.findall(r"(?:E(\d+):|\b(\d+)\.\s*)(.*?)(?=(?:E\d+:|\b\d+\.\s*|$))", query_text, re.DOTALL | re.I)
        if e_matches:
            for idx_str1, idx_str2, content in e_matches:
                order = int(idx_str1 or idx_str2 or len(trake_events) + 1)
                clean = content.strip().strip(".")
                if clean:
                    trake_events.append(
                        TrakeEvent(
                            order=order,
                            description=clean,
                            scene_en=clean,
                            objects_en=[clean],
                        )
                    )

        objects = [q.strip() for q in re.split(r",| và | with | and | đang ", query_text) if len(q.strip()) > 3]
        if not objects:
            objects = [query_text]

        vqa_q = ""
        if task_type == "VQA":
            if "hỏi" in query_text.lower():
                vqa_q = query_text.lower().split("hỏi", 1)[1].strip(" ?:.")
            else:
                vqa_q = query_text

        return ParsedQuery(
            task_type=task_type,
            language="vi" if is_vi else "en",
            original_query=query_text,
            global_scene_en=query_text,
            objects_en=objects,
            ocr_keywords=flat_ocr,
            speech_vi=query_text if is_vi else "",
            is_temporal_trake=bool(trake_events or task_type == "TRAKE"),
            trake_events=trake_events,
            vqa_question=vqa_q,
            weights={"vis": 0.35, "dam": 0.30, "asr": 0.35, "ocr": 0.00},
        )
