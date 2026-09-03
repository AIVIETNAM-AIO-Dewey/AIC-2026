"""Deterministic, dependency-free query parsing for the CPU-only server."""

from __future__ import annotations

import re

from online.src.contracts.query import ParsedQuery, TaskType, TrakeEvent


_VIETNAMESE_MARKS = set(
    "ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "các",
    "có",
    "của",
    "đang",
    "được",
    "for",
    "from",
    "in",
    "là",
    "một",
    "những",
    "of",
    "on",
    "the",
    "to",
    "trong",
    "và",
    "với",
}


def _language(text: str) -> str:
    if any(character in _VIETNAMESE_MARKS for character in text):
        return "vi"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "mixed"


def _ocr_keywords(text: str, limit: int = 10) -> list[str]:
    quoted = [
        match.strip()
        for match in re.findall(r'["“”‘’\']([^"“”‘’\']{2,80})["“”‘’\']', text)
        if match.strip()
    ]
    tokens = re.findall(r"[0-9A-Za-zÀ-ỹĐđ][0-9A-Za-zÀ-ỹĐđ:_\-\.]{1,31}", text)
    candidates = quoted + [
        token
        for token in tokens
        if token.casefold() not in _STOPWORDS and (len(token) >= 3 or any(c.isdigit() for c in token))
    ]
    deduplicated: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        folded = candidate.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        deduplicated.append(candidate)
        if len(deduplicated) >= limit:
            break
    return deduplicated


def _temporal_events(text: str) -> list[TrakeEvent]:
    marked = re.sub(r"(?:^|\s)\d+[.)]\s+", " | ", text)
    parts = re.split(
        r"\s*(?:\||\b(?:sau đó|tiếp theo|kế tiếp|rồi|cuối cùng|then|next|after that|finally)\b)\s*",
        marked,
        flags=re.IGNORECASE,
    )
    clean_parts = [" ".join(part.strip(" ,.;:-").split()) for part in parts]
    clean_parts = [part for part in clean_parts if len(part.split()) >= 2][:6]
    if len(clean_parts) < 2:
        return []
    speech_markers = (
        "nói", "phát biểu", "lời thoại", "đối thoại", "thuyết minh",
        "giọng đọc", "voiceover", "says", "speaks", "speech", "dialogue",
    )
    return [
        TrakeEvent(
            order=index,
            description=part,
            scene_en=part,
            objects_en=[part],
            speech_vi=part if any(marker in part.casefold() for marker in speech_markers) else "",
            ocr_keywords=_ocr_keywords(part),
        )
        for index, part in enumerate(clean_parts, 1)
    ]


class LocalQueryParser:
    """Map one raw query to existing UI fields without any generative model."""

    def parse(self, query: str, task_type: TaskType = "KIS") -> ParsedQuery:
        clean = " ".join(query.split())
        if not clean:
            raise ValueError("Query cannot be empty")
        if task_type != "KIS":
            raise ValueError("The CPU-only server currently supports KIS queries only")
        events = _temporal_events(clean)
        speech_markers = (
            "nói", "phát biểu", "lời thoại", "đối thoại", "thuyết minh",
            "giọng đọc", "voiceover", "says", "speaks", "speech", "dialogue",
        )
        speech_query = clean if any(marker in clean.casefold() for marker in speech_markers) else ""
        return ParsedQuery(
            task_type="KIS",
            language=_language(clean),
            original_query=clean,
            global_scene_en=clean,
            objects_en=[clean],
            speech_vi=speech_query,
            ocr_keywords=_ocr_keywords(clean),
            is_temporal_trake=bool(events),
            trake_events=events,
            vqa_question="",
        )
