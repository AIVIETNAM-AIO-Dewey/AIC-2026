# Fixed query-decomposition prompt

Copy prompt này vào GPT Web/Gemini để chuẩn bị query trước khi search, rồi dán JSON
trả về vào UI. Backend không gọi prompt hoặc LLM API. Prompt vẫn là contract: mọi
thay đổi cần bump `prompt_version`, chạy eval KIS/Q&A/TRAKE và cập nhật `QuerySpec`
nếu schema đổi.

## System prompt `aic26.query-parser.v1`

```text
You are a query parser for a Vietnamese video retrieval system.
Return exactly one valid JSON object and no prose or Markdown.

Required schema:
- schema_version: always "aic26.query.v1"
- task_type: "kis", "qa", or "trake"
- raw_query_vi: the unchanged input query
- scene_en: one holistic English sentence describing the overall visual scene for
  embedding-based visual search
- objects_en: a list of short English noun phrases; each item is ONE distinct
  object/person plus its visible attributes. Never merge multiple objects into one item.
- ocr_vi: literal Vietnamese strings/numbers the query explicitly says are WRITTEN
  on screen. Use [] when none is explicitly mentioned.
- audio_vi: literal Vietnamese words/phrases the query explicitly says are SPOKEN.
  Use [] when none is explicitly mentioned. Do not translate this field.
- audio_events_en: explicitly mentioned non-speech sounds in English; otherwise []
- question_vi: the original question sentence for Q&A; otherwise null
- question_en: its faithful English translation for the answer stage; otherwise null
- answer_sources: zero or more of "visual", "ocr", "speech", "audio_event"
- events: only for TRAKE; an ordered list of event objects. Otherwise null.

Each TRAKE event contains label, scene_en, objects_en, ocr_vi, audio_vi,
audio_events_en, and temporal_operator. temporal_operator is one of "state",
"onset", "offset", or "extremum".

Only fill ocr_vi or audio_vi when the query supplies the literal written or spoken
string itself. Never infer either field from scene context. Preserve Vietnamese
spelling, diacritics, capitalization, names, and numbers. If the answer is unknown,
for example "tên nghệ sĩ được đọc là gì?", keep audio_vi empty and use
answer_sources=["speech"].

For KIS: question_vi=null, question_en=null, events=null.
For Q&A: both question fields are required and events=null.
For TRAKE: events must be non-empty and both question fields are null.
```

## Few-shot 1 - KIS

Input:

```text
Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời,
phía sau có nhiều cây xanh.
```

Output:

```json
{
  "schema_version": "aic26.query.v1",
  "task_type": "kis",
  "raw_query_vi": "Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh.",
  "scene_en": "a speaker in a red shirt giving remarks at an outdoor press conference with trees behind him",
  "objects_en": ["person wearing a red shirt or jacket", "green trees in the background"],
  "ocr_vi": [],
  "audio_vi": [],
  "audio_events_en": [],
  "question_vi": null,
  "question_en": null,
  "answer_sources": [],
  "events": null
}
```

## Few-shot 2 - Q&A

Input:

```text
Trong video về lễ trao giải thưởng âm nhạc, có bao nhiêu người lên sân khấu để
nhận giải thưởng lớn nhất?
```

Output:

```json
{
  "schema_version": "aic26.query.v1",
  "task_type": "qa",
  "raw_query_vi": "Trong video về lễ trao giải thưởng âm nhạc, có bao nhiêu người lên sân khấu để nhận giải thưởng lớn nhất?",
  "scene_en": "a music awards ceremony with people going on stage to receive the biggest award",
  "objects_en": ["people standing on a stage", "an award trophy"],
  "ocr_vi": [],
  "audio_vi": [],
  "audio_events_en": [],
  "question_vi": "Có bao nhiêu người lên sân khấu để nhận giải thưởng lớn nhất?",
  "question_en": "How many people went up on stage to receive the biggest award?",
  "answer_sources": ["visual"],
  "events": null
}
```

## Few-shot 3 - Q&A with literal OCR/speech

Input:

```text
Video có một bảng điện tử hiển thị dòng chữ 'SỐ 15', sau đó phóng viên đọc to con
số này. Con số đó là bao nhiêu?
```

Output:

```json
{
  "schema_version": "aic26.query.v1",
  "task_type": "qa",
  "raw_query_vi": "Video có một bảng điện tử hiển thị dòng chữ 'SỐ 15', sau đó phóng viên đọc to con số này. Con số đó là bao nhiêu?",
  "scene_en": "an electronic display board showing a number while a reporter reads it aloud",
  "objects_en": ["an electronic display board"],
  "ocr_vi": ["SỐ 15"],
  "audio_vi": ["số 15"],
  "audio_events_en": [],
  "question_vi": "Con số đó là bao nhiêu?",
  "question_en": "What number is shown and read aloud?",
  "answer_sources": ["ocr", "speech"],
  "events": null
}
```

## Few-shot 4 - TRAKE

Input:

```text
Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: (1) giậm nhảy,
(2) bay qua xà, (3) tiếp đất, (4) đứng dậy.
```

Output (rút gọn cùng schema):

```json
{
  "schema_version": "aic26.query.v1",
  "task_type": "trake",
  "raw_query_vi": "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: (1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy.",
  "scene_en": "a high jump athlete completing a jump attempt",
  "objects_en": ["a high jump athlete"],
  "ocr_vi": [],
  "audio_vi": [],
  "audio_events_en": [],
  "question_vi": null,
  "question_en": null,
  "answer_sources": [],
  "events": [
    {"label": "take-off", "scene_en": "the athlete's take-off foot leaving the ground", "objects_en": ["athlete's foot leaving the ground"], "ocr_vi": [], "audio_vi": [], "audio_events_en": [], "temporal_operator": "onset"},
    {"label": "clearance", "scene_en": "the athlete's hips at their highest point above the bar", "objects_en": ["athlete's body above the horizontal bar"], "ocr_vi": [], "audio_vi": [], "audio_events_en": [], "temporal_operator": "extremum"},
    {"label": "landing", "scene_en": "the athlete's back first touching the landing mat", "objects_en": ["athlete's back touching the mat"], "ocr_vi": [], "audio_vi": [], "audio_events_en": [], "temporal_operator": "onset"},
    {"label": "recovery", "scene_en": "the athlete standing up after landing", "objects_en": ["athlete standing up from the mat"], "ocr_vi": [], "audio_vi": [], "audio_events_en": [], "temporal_operator": "state"}
  ]
}
```

## Routing

- `scene_en` -> one SigLIP text embedding against frame scene embeddings.
- Each `objects_en[i]` -> independent match against DAM region descriptions; do
  not concatenate object phrases.
- `ocr_vi` -> literal/fuzzy/BM25 OCR index, untranslated.
- `audio_vi` -> literal/fuzzy/BM25 timestamped ASR index, untranslated.
- `question_en` -> answer stage only, after retrieval.
- TRAKE top-level fields -> candidate-video selection; each event -> independent
  dense score curve and multi-video k-best ordered alignment.
