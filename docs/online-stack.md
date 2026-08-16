# Online retrieval stack

The online system runs from processed artifacts only. Original videos and GPU
models remain in the offline Colab/Kaggle stages. Every result uses the canonical
pair `video_id` + `frame_idx`; it never derives a submission identifier from a
thumbnail filename or `keyframe_n`.

## Local deployment

```bash
copy .env.example .env
docker compose up --build -d
docker compose --profile ingest run --rm ingest
```

Qdrant is pinned to `qdrant/qdrant:v1.18.2`. Ingest creates versioned collections
for `frames_sparse`, `frames_dense`, `regions`, `ocr`, and `asr`; only a successful
ingest publishes their `*_current` aliases. The ingest command rejects incomplete
manifests, bad output checksums, missing canonical fields, and manifest/record-count
mismatches before it writes points.

The Vite web application is served at `http://localhost:8080`; FastAPI is at
`http://localhost:8000`. The public endpoint is `POST /api/v1/search`. KIS falls
back to raw-query retrieval only when GPT-4o is unavailable. Q&A and TRAKE return
`503 capability_unavailable` rather than fabricating an answer or event sequence.

`OPENAI_API_KEY` belongs only in the API environment/secret. The adapter uses the
Responses API with `store=false`, default model `gpt-4o-2024-11-20`, a 30-second
timeout, and two bounded retries. It sends at most eight selected keyframes with
DAM/OCR/ASR evidence when image answers are enabled.

Windows runs all Compose services CPU-only. On macOS, Compose is also CPU-only;
developers may run the API natively with `AIC_DEVICE=mps` for optional local text
encoder acceleration. No auth is included because v1 is single-user localhost/LAN.
