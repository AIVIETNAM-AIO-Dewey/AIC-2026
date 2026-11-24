# System Architecture — AIC 2026

## Design Principles

Source videos and organizer keyframe maps form the canonical frame-identity layer across the entire pipeline:

- Every indexed record and submission row resolves to a canonical zero-based `(video_id, frame_idx)` identity (`<video_id>:<frame_idx>`), paired with exact `pts_time_s` and `fps` metadata.
- The corpus spans **873 videos**, **247,956 canonical keyframes**, **681,355 localized DAM object regions**, and **55,168 timestamped ASR segments** (`872` videos with speech segments plus `1` valid zero-segment video, `L30_V029`).
- Offline feature extraction, index preparation, and online retrieval are separated by explicit data contracts and fail-closed readiness gates (`RunManifest`, `branch1_data_gate.json`, `branch2_dam_manifest.json`, `branch3_ocr_manifest.json`, `branch3_asr_manifest.json`, and `qdrant_ingestion_manifest.json`).

---

## Offline Pipeline (`src/aic2026/`, `scripts/`)

1. **Frame Manifest & Identity Alignment** (`scripts/build_frame_manifest.py`):
   - Validates organizer `map-keyframes` CSVs and builds canonical `FrameRef` rows (`video_id`, `keyframe_n`, `frame_idx`, `pts_time_s`, `fps`, `frame_relpath`, image dimensions).
2. **Object Masks & Localized Descriptions** (`scripts/prepare_object_masks.py`, `scripts/run_dam_descriptions.py`, `scripts/run_dam_batch.py`):
   - Filters/deduplicates organizer bounding boxes, refines masks via `facebook/sam-vit-base` (with bbox fallback), and generates localized English descriptions (max 20 words) using `nvidia/DAM-3B`, producing `ObjectFrameRecord` JSONL (`aic26.object_regions.v1`) and `RunManifest` sidecars.
3. **Full-Frame Visual Embeddings**:
   - Offline keyframe feature exports for **SigLIP2** (`768d`), **MetaCLIP 2** (`1024d`), and **BEiT-3 COCO** (`768d`), plus **BGE-M3** (`1024d`) embeddings for DAM region descriptions.
4. **Vietnamese Speech Recognition (ASR)** (`scripts/run_phowhisper_asr.py`, `src/aic2026/asr/`):
   - Transcribes audio in 30-second sliding windows with 15-second stride using `vinai/PhoWhisper-large` (`kiendt/PhoWhisper-large-ct2`) and exports timestamped `[start_ms, end_ms]` segments rather than duplicating text per frame.
5. **Vietnamese OCR**:
   - Frame-level OCR JSONL exports aligned to the 247,956 canonical keyframes.

---

## Index & State Preparation (`scripts/qdrant/`)

Before serving queries, preparation commands validate source parity and atomically publish persistent vector collections and SQLite indexes:

- **Qdrant Collections** (`scripts/qdrant/ingest.py`):
  - `aic_frames`: 247,956 points with float16 named vectors `siglip2` (`768d`) and `metaclip2` (`1024d`), scalar INT8 quantization pinned in RAM, disk-backed HNSW.
  - `aic_beit3_frames`: 247,956 points with `beit3` (`768d`) COCO retrieval vectors.
  - `aic_dam_regions`: 681,355 points with `dam` (`1024d`) BGE-M3 vectors and `parent_point_id` links back to `aic_frames`.
- **Branch-1 & Branch-2 Gates** (`prepare_branch1.py`, `prepare_branch2.py`):
  - Validate matrix shapes, finiteness, L2 normalization, and canonical point ID ordering; build `branch2_dam_bm25.sqlite3`.
- **Branch-3 SQLite FTS5 Indexes** (`prepare_text_indexes.py`, `prepare_asr_index.py`):
  - Publish atomic `ocr.sqlite3` (`branch3.ocr-index.v3`, schema v3) and `asr.sqlite3` (`branch3.asr-index.v2`, schema v4) with Vietnamese diacritic folding and mojibake repair.

---

## Online Retrieval Engine (`online/`)

`online.cpu_server:app` serves both the compiled Retrieval Studio frontend and the 4-branch search API (`http://127.0.0.1:8890`).

### 1. Deterministic 6-Role Bilingual Query Planning

`LocalQueryParser` and `build_kis_query_plan` decompose each query deterministically without calling external LLM APIs:

- Six query roles: `original`, `entity`, `action`, `context`, `synonym`, `keyword` in Vietnamese and English (`branch1.query.v1`).
- Manual bilingual edits in the UI are preserved and checked with non-blocking lexical diagnostics.

### 2. Isolated Encoder Workers & Persistent Cache

- `EncoderWorkerManager` runs heavy text/image encoders in isolated per-model worker processes (supporting CPU and Apple Silicon MPS with per-worker CPU fallback).
- Workers stay warm for up to 30 seconds before exiting so the OS reclaims RAM; `PersistentQueryEmbeddingCache` (`query_embeddings.sqlite3`) avoids recomputing embeddings on cache hits.

### 3. Four Retrieval Branches + KIS Fusion

| Branch | Modality & Sources | Streams & Pool Cap | Reranking |
|---|---|---|---|
| **Branch 1** | Visual full-frame (`SigLIP2`, `MetaCLIP2`, `BEiT-3`) | 30 cosine streams (12 SigLIP2 VI/EN, 12 MetaCLIP2 VI/EN, 6 BEiT-3 EN) → **1,500** pool | Model-level weighted fusion (`45 / 30 / 25`) |
| **Branch 2** | DAM localized objects (`BGE-M3` + `BM25`) | 6 EN dense + 6 EN BM25 → **500** hybrid pool | Top **100** validated with `BEiT-3` COCO cosine (`40% BEiT-3 + 60% hybrid`) |
| **Branch 3A** | Speech transcripts (`PhoWhisper` SQLite FTS5) | 12 VI/EN FTS5 streams → **500** pool | Okapi BM25 + n-gram coverage mapped to nearest canonical keyframe |
| **Branch 3B** | On-screen text (OCR SQLite FTS5) | 12 VI/EN FTS5 streams → **500** pool | Exact + folded Vietnamese lexical matching |
| **KIS Fusion** | Cross-branch weighted RRF (`k=60`, `40/30/15/15`) | Combines the 4 branch pools → **150** frames | Top **100** reranked via `BEiT-3` COCO (`25% BEiT-3 + 75% RRF`); ranks 101–150 keep RRF order |

### 4. Task Modes & Submission Workbench

- **KIS / Q&A**: Full 4-branch fusion, frame inspector with bounding-box/OCR overlays, YouTube timestamp sync, video-scoped visual search (`SigLIP2 + MetaCLIP2 + BEiT-3`), and related-frame expansion to populate 100-row CSV submissions.
- **Ordered KIS Events (TRAKE)**: Splits 2–6 chronological events (each inheriting only the parent's shared `context` role), executes the full KIS pipeline per event, and aligns strictly monotonic `(video_id, frame_idx, pts_time_s)` paths via temporal search.

---

## Failure Policy & Integrity Gates

- Missing or corrupted manifests, mismatched `build_id` values, or unreachable Qdrant keep capabilities fail-closed in `/api/health`.
- Index builders write to staging files and atomically swap artifacts into `STATE_ROOT`; read-only API connections never rebuild indexes during startup.
- Offline shard runners write `.partial` files and publish atomically alongside `RunManifest` sidecars; `--resume` only skips completed, checksum-verified work.
