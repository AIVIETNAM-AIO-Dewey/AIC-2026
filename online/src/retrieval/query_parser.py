"""Query Decomposition and Parsing Component using Gemini Flash & Local Fallback."""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Optional
from dotenv import load_dotenv

from online.src.contracts.query import ParsedQuery, TRAKEEvent, TaskType

load_dotenv()
logger = logging.getLogger(__name__)

# Check for Gemini API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class QueryParser:
    """Intelligent Query Decomposer supporting Gemini Flash and Local Rule-based fallback."""

    def __init__(self, gemini_model_id: str = "gemini-3.5-flash-lite") -> None:
        self.gemini_model_id = gemini_model_id
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
        task_type: TaskType = "KIS",
        force_local: bool = False,
    ) -> ParsedQuery:
        """Parse raw user query into structured 4-channel sub-queries."""
        query_text = query_text.strip()
        if not query_text:
            return ParsedQuery(task_type=task_type, original_query="")

        # Use Gemini Flash if available and not forced local
        if self._gemini_client is not None and not force_local:
            try:
                return self._parse_with_gemini(query_text, task_type)
            except Exception as e:
                logger.warning(f"Gemini parsing failed ({e}). Falling back to local parser.")

        # Fallback to local rule-based parser
        return self._parse_local(query_text, task_type)

    def _parse_with_gemini(self, query_text: str, task_type: TaskType) -> ParsedQuery:
        from google.genai import types

        system_instruction = (
            "You are an expert query parser for a High-Speed Video Retrieval Search Engine.\n"
            "Given a user query (in Vietnamese, English, or Vietlish), decompose it into a clean structured JSON object:\n"
            "1. 'global_scene_en': High-level background, lighting, camera angle, setting (in English for SigLIP).\n"
            "2. 'objects_en': List of specific visual objects, people, clothing, vehicles, accessories (in English for DAM).\n"
            "3. 'ocr_keywords': Exact on-screen text, brand logos, numbers, license plates (original text).\n"
            "4. 'speech_vi': Spoken dialogue, voiceover topic, interview quotes (in Vietnamese for ASR).\n"
            "5. 'is_temporal_trake': Boolean, true if the query describes a chronological sequence of events (First A, then B, after that C).\n"
            "6. 'trake_events': List of sequential sub-events if is_temporal_trake is true.\n"
            "7. 'vqa_question': If task_type is VQA, summarize the exact question being asked."
        )

        prompt = (
            f"Target Task Type: {task_type}\n"
            f"User Query: {query_text}\n\n"
            "Decompose this query into the required structured JSON format."
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
        
        # Build TRAKE events if present
        trake_events = []
        for idx, ev in enumerate(data.get("trake_events", []), 1):
            if isinstance(ev, dict):
                trake_events.append(
                    TRAKEEvent(
                        event_index=idx,
                        description=ev.get("description", ""),
                        scene_en=ev.get("scene_en", ""),
                        objects_en=ev.get("objects_en", []),
                        speech_vi=ev.get("speech_vi", ""),
                        ocr_keywords=ev.get("ocr_keywords", []),
                    )
                )

        # Dynamic channel weights
        weights = {"vis": 0.40, "dam": 0.40, "asr": 0.20, "ocr": 0.0}
        if data.get("speech_vi") and len(data["speech_vi"]) > 10:
            weights["asr"] = 0.35
            weights["vis"] = 0.35
            weights["dam"] = 0.30
        if not data.get("objects_en"):
            weights["dam"] = 0.10
            weights["vis"] = 0.60

        return ParsedQuery(
            task_type=task_type,
            language="vi" if re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", query_text, re.I) else "en",
            original_query=query_text,
            global_scene_en=data.get("global_scene_en", query_text),
            objects_en=data.get("objects_en", [query_text]),
            ocr_keywords=data.get("ocr_keywords", []),
            speech_vi=data.get("speech_vi", ""),
            is_temporal_trake=bool(data.get("is_temporal_trake", task_type == "TRAKE")),
            trake_events=trake_events,
            vqa_question=data.get("vqa_question", query_text if task_type == "VQA" else ""),
            weights=weights,
        )

    def _parse_local(self, query_text: str, task_type: TaskType) -> ParsedQuery:
        """Fast offline rule-based parser."""
        is_vi = bool(re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", query_text, re.I))
        
        # 1. Extract text in quotes for OCR
        ocr_matches = re.findall(r'["\']([^"\']+)["\']', query_text)

        # 2. Check for speech markers
        speech_match = ""
        speech_triggers = ["nói về", "kể về", "phát biểu", "bản tin", "chương trình", "nói rằng"]
        for trig in speech_triggers:
            if trig in query_text.lower():
                parts = query_text.lower().split(trig, 1)
                if len(parts) > 1 and len(parts[1].strip()) > 3:
                    speech_match = parts[1].strip()
                    break

        # 3. Simple object token extraction
        objects = [q.strip() for q in re.split(r",| và | with | and | standing next to ", query_text) if len(q.strip()) > 3]
        if not objects:
            objects = [query_text]

        # 4. Check for TRAKE sequence prepositions
        is_trake = task_type == "TRAKE" or bool(re.search(r"đầu tiên|sau đó|tiếp theo|sau đấy|first|then|after that", query_text, re.I))
        trake_events = []
        if is_trake:
            steps = re.split(r"sau đó|tiếp theo|sau đấy|then|after that", query_text, flags=re.I)
            for idx, step in enumerate(steps, 1):
                clean_step = step.strip()
                if clean_step:
                    trake_events.append(
                        TRAKEEvent(
                            event_index=idx,
                            description=clean_step,
                            scene_en=clean_step,
                            objects_en=[clean_step],
                        )
                    )

        weights = {"vis": 0.45, "dam": 0.40, "asr": 0.15, "ocr": 0.0}

        return ParsedQuery(
            task_type=task_type,
            language="vi" if is_vi else "en",
            original_query=query_text,
            global_scene_en=query_text,
            objects_en=objects,
            ocr_keywords=ocr_matches,
            speech_vi=speech_match if speech_match else (query_text if is_vi else ""),
            is_temporal_trake=is_trake,
            trake_events=trake_events,
            vqa_question=query_text if task_type == "VQA" else "",
            weights=weights,
        )
