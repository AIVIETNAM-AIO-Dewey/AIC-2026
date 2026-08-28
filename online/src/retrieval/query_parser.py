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
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from online.src.contracts.query import ParsedQuery, TaskType, TrakeEvent

ROOT_DIR = Path(__file__).resolve().parents[3]
load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# SigLIP-2 has a hard 64-token text window. Keeping generated scene prompts at
# 40 words leaves room for sub-word tokenization and special tokens instead of
# silently losing the most discriminative details at the end of the query.
MAX_VISUAL_QUERY_WORDS = 40
MAX_DAM_OBJECT_QUERIES = 3
MAX_DAM_OBJECT_WORDS = 14


def _clean_text(value: object) -> str:
    """Return one whitespace-normalized scalar string."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _limit_words(text: str, limit: int) -> str:
    """Deterministically cap a model-generated query without adding semantics."""
    words = _clean_text(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:-")


def _clean_unique_phrases(
    values: object,
    *,
    max_items: int,
    max_words: int,
) -> list[str]:
    """Normalize, cap, and case-insensitively deduplicate query phrases."""
    if isinstance(values, str):
        candidates: list[object] = [values]
    elif isinstance(values, list):
        candidates = values
    else:
        candidates = []

    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str | int | float):
            continue
        phrase = _limit_words(str(candidate), max_words)
        folded = phrase.casefold()
        if not phrase or folded in seen:
            continue
        seen.add(folded)
        cleaned.append(phrase)
        if len(cleaned) == max_items:
            break
    return cleaned


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
        task_type: TaskType | None = None,
        engine: str = "gemini",  # "gemini", "qwen", "rule", or "direct"
        *,
        allow_qwen_fallback: bool = True,
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
                if allow_qwen_fallback:
                    logger.warning(f"Gemini API parsing failed ({e}). Trying Qwen...")
                    engine = "qwen"
                else:
                    logger.warning(
                        "Gemini API parsing failed (%s). Qwen fallback is disabled; "
                        "using the deterministic rule parser.",
                        e,
                    )
                    engine = "rule"

        # 2. Local Qwen Engine (via Ollama)
        if engine == "qwen":
            try:
                return self._parse_with_qwen(query_text, task_type)
            except Exception as e:
                logger.warning(f"Local Qwen parsing failed ({e}). Falling back to rule parser.")

        # 3. Rule/direct fallback. "direct" is an explicit API-safe alias for
        # deterministic local parsing; the scoped-search UI may also bypass
        # this endpoint and construct a visual-only ParsedQuery directly.
        return self._parse_local(query_text, task_type)

    def _detect_task_type(self, query_text: str) -> TaskType:
        q_lower = query_text.lower()
        if bool(
            re.search(
                r"E\d+:|khoảnh khắc đầu tiên|khoảnh khắc tiếp theo|sau đó|tiếp theo|sau đấy|lần lượt|theo thứ tự|thứ tự.{0,40}nhất.{0,40}nhì|first.{0,80}second",
                query_text,
                re.I | re.S,
            )
        ):
            return "TRAKE"
        if any(
            w in q_lower
            for w in [
                "hỏi",
                "là gì",
                "màu gì",
                "bao nhiêu",
                "ai",
                "ở đâu",
                "khi nào",
                "?",
                "what",
                "which",
                "how many",
            ]
        ):
            return "VQA"
        return "KIS"

    def _get_system_prompt(self, task_type: TaskType) -> str:
        return (
            "You are an expert query parser for a High-Speed Multimodal Video Retrieval Engine.\n"
            "Your job is to analyze the user's query and decompose it into distinct sub-queries for 4 specialized search channels:\n\n"
            "CHANNELS & LANGUAGE REQUIREMENTS:\n"
            "1. 'global_scene_en' (ENGLISH for SigLIP-2): Write ONE compact visual caption of 18-40 words. Start with the rarest visible evidence, never a generic person/background when a distinctive layout or action is supplied. If a detailed diagram, chart, slide, screen, or document is described, begin with that content and omit the presenter unless the presenter's action is discriminative. Translate relationships into visible geometry (for example, 'an orange outlined group enclosing two boxes' or 'a wide blue middle bar'), while preserving exact counts, containment, top/middle/bottom order, colors, and arrow direction. Drop generic context before dropping topology. Omit filler and do not exceed 40 words.\n"
            "2. 'objects_en' (ENGLISH for DAM): Return 1-3 caption-like phrases for concrete regions likely to have their own mask (person, clothing, vehicle, tool, screen/slide, food). Keep each phrase under 14 words. Keep attributes with their object; do not turn colors, directions, or abstract relationships into standalone objects. Treat a screen/slide and the diagram shown on it as one region phrase rather than duplicate queries.\n"
            "3. 'speech_vi' (VIETNAMESE for Audio ASR): Include only dialogue, narration, or a spoken topic explicitly stated by the user. Otherwise return an empty string. Never copy a purely visual scene description here.\n"
            "4. 'ocr_keywords' (VIETNAMESE/ORIGINAL for OCR): Include only literal text the user says should be visible on screen, such as a quoted title, number, brand, subtitle, or lower-third. Do not infer text from the visual topic; otherwise return an empty list.\n\n"
            "CHANNEL PURITY:\n"
            "- Never invent speech or on-screen text.\n"
            "- Do not copy the whole raw query into every channel.\n"
            "- Prefer one precise query over a long inventory of generic details.\n\n"
            "VISUAL PRIORITY EXAMPLE:\n"
            "- Weak: 'A teacher beside a presentation slide.'\n"
            "- Strong: 'A three-level diagram with paired top and bottom boxes around one middle box, connected by downward arrows, displayed on a slide.'\n\n"
            "TASK SPECIFIC HANDLING:\n"
            "- If task is VQA: extract 'vqa_question' (the exact question to be answered).\n"
            "- If task is TRAKE or query has sequential steps (E1, E2, ...): extract 'trake_events' array where each item has:\n"
            "  - order (int 1, 2, 3...)\n"
            "  - description (original text)\n"
            "  - scene_en (English visual setting)\n"
            "  - objects_en (English object list)\n"
            "  - speech_vi (Vietnamese speech if any)\n"
            "  - ocr_keywords (Vietnamese on-screen text if any)\n\n"
            "For ordered arrivals or participants described with words such as 'lần lượt', 'nhất, nhì, ba', first/second/third, create one event per ordered participant. Make every scene_en independently searchable: repeat the shared camera angle and setting, describe only that event's visible subject, and do not rely on ordinal words to carry visual meaning.\n\n"
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

        with urllib.request.urlopen(req, timeout=60) as response:
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
                        description=_clean_text(ev.get("description")) or f"Event {idx}",
                        scene_en=_limit_words(
                            _clean_text(ev.get("scene_en")),
                            MAX_VISUAL_QUERY_WORDS,
                        ),
                        objects_en=_clean_unique_phrases(
                            ev.get("objects_en"),
                            max_items=MAX_DAM_OBJECT_QUERIES,
                            max_words=MAX_DAM_OBJECT_WORDS,
                        ),
                        speech_vi=_clean_text(ev.get("speech_vi")),
                        ocr_keywords=_clean_unique_phrases(
                            ev.get("ocr_keywords"),
                            max_items=8,
                            max_words=64,
                        ),
                    )
                )

        weights = {"vis": 0.35, "dam": 0.30, "asr": 0.35, "ocr": 0.00}
        speech_raw = data.get("speech_vi")
        if isinstance(speech_raw, list):
            speech_v = _clean_text(" ".join([str(s) for s in speech_raw if s]))
        else:
            speech_v = _clean_text(speech_raw)

        ocr_kw = _clean_unique_phrases(
            data.get("ocr_keywords"),
            max_items=8,
            max_words=64,
        )
        objs_en = _clean_unique_phrases(
            data.get("objects_en"),
            max_items=MAX_DAM_OBJECT_QUERIES,
            max_words=MAX_DAM_OBJECT_WORDS,
        )
        global_scene = _limit_words(
            _clean_text(data.get("global_scene_en")) or query_text,
            MAX_VISUAL_QUERY_WORDS,
        )

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

        is_vi = bool(
            re.search(
                r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
                query_text,
                re.I,
            )
        )

        return ParsedQuery(
            task_type=task_type,
            language="vi" if is_vi else "en",
            original_query=query_text,
            global_scene_en=global_scene,
            objects_en=objs_en if objs_en else [_limit_words(query_text, MAX_DAM_OBJECT_WORDS)],
            ocr_keywords=ocr_kw,
            speech_vi=speech_v,
            is_temporal_trake=bool(trake_events or task_type == "TRAKE"),
            trake_events=trake_events,
            vqa_question=data.get("vqa_question") or (query_text if task_type == "VQA" else ""),
            weights=weights,
        )

    def _parse_local(self, query_text: str, task_type: TaskType) -> ParsedQuery:
        """Fast offline rule-based parser."""
        is_vi = bool(
            re.search(
                r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
                query_text,
                re.I,
            )
        )
        ocr_matches = re.findall(r'["\']([^"\']+)["\']|\b\d+g\b|\b\d+\s*kg\b', query_text)
        flat_ocr = [m if isinstance(m, str) else m[0] for m in ocr_matches if m]

        trake_events = []
        e_matches = re.findall(
            r"(?:E(\d+):|\b(\d+)\.\s*)(.*?)(?=(?:E\d+:|\b\d+\.\s*|$))", query_text, re.DOTALL | re.I
        )
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

        # Deterministic fallback for ordered lists such as Vietnamese
        # "lần lượt là A, B và C" when both Gemini and local Qwen are unavailable.
        # The original language is retained rather than inventing a translation;
        # the UI exposes these lines for explicit editing before SigLIP search.
        if not trake_events and task_type == "TRAKE":
            ordered_match = re.search(
                r"\b(?:lần\s+lượt(?:\s+là)?|respectively(?:\s+are)?)\s*:?[\s]+(.+)$",
                query_text,
                re.I | re.S,
            )
            if ordered_match:
                shared_context = query_text[: ordered_match.start()].strip(" .,:;-\n")
                shared_context = re.sub(
                    r"(?:theo\s+)?thứ\s+tự.*$",
                    "",
                    shared_context,
                    flags=re.I | re.S,
                ).strip(" .,:;-\n")
                raw_items = re.split(
                    r"\s*,\s*|\s+và\s+|\s+and\s+",
                    ordered_match.group(1).strip().strip('"\' .'),
                    flags=re.I,
                )
                ordered_items = [
                    item.strip().strip('"\' .')
                    for item in raw_items
                    if item.strip().strip('"\' .')
                ][:6]
                if len(ordered_items) >= 2:
                    for order, item in enumerate(ordered_items, 1):
                        scene = f"{shared_context}. {item}" if shared_context else item
                        trake_events.append(
                            TrakeEvent(
                                order=order,
                                description=item,
                                scene_en=_limit_words(scene, MAX_VISUAL_QUERY_WORDS),
                                objects_en=[_limit_words(item, MAX_DAM_OBJECT_WORDS)],
                            )
                        )

        objects = [
            q.strip()
            for q in re.split(r",| và | with | and | đang ", query_text)
            if len(q.strip()) > 3
        ]
        if not objects:
            objects = [query_text]
        objects = _clean_unique_phrases(
            objects,
            max_items=MAX_DAM_OBJECT_QUERIES,
            max_words=MAX_DAM_OBJECT_WORDS,
        )

        # A generic Vietnamese scene description is not evidence that the same
        # words were spoken. The rule fallback opts into ASR only for explicit
        # speech/audio language instead of contaminating the audio pool.
        speech_markers = (
            "nói",
            "phát biểu",
            "lời thoại",
            "đối thoại",
            "thuyết minh",
            "giọng đọc",
            "voiceover",
            "says",
            "speaks",
            "speech",
            "dialogue",
        )
        speech_query = (
            query_text if any(marker in query_text.casefold() for marker in speech_markers) else ""
        )

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
            global_scene_en=_limit_words(query_text, MAX_VISUAL_QUERY_WORDS),
            objects_en=objects,
            ocr_keywords=flat_ocr,
            speech_vi=speech_query,
            is_temporal_trake=bool(trake_events or task_type == "TRAKE"),
            trake_events=trake_events,
            vqa_question=vqa_q,
            weights={"vis": 0.35, "dam": 0.30, "asr": 0.35, "ocr": 0.00},
        )
