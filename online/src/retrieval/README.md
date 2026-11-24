# Canonical CPU retrieval runtime

`online.cpu_server:app` is the only supported online entry point. Retrieval code
is owned by `infrastructure/`, `encoders/`, `modalities/`, and `branches/`; there
are no compatibility imports back to the removed legacy pipeline.

Heavy text/image encoders run in isolated worker processes. Checkpoints stay in
the persistent model volume; a worker serves one model identity, can be reused
for at most 30 seconds, and then exits so the OS reclaims model RAM. Persistent
query embeddings avoid repeated model loads altogether on cache hits.

Branch 1 performs 30 cosine streams: six roles in both Vietnamese and English
for SigLIP2 and MetaCLIP2, plus six English streams for BEiT-3. Model-level
fusion returns an API pool of up to 1,500 frames; the workbench displays 150.
Branch 2 performs six English BGE-M3 DAM streams plus six English BM25 streams,
returns an API pool of up to 500 frames, and uses six English BEiT-3 COCO cosine
streams to reorder only the first 100 with a 40/60 BEiT-3/hybrid blend.

Branch-2 runtime never builds indexes. Prepare and validate its state explicitly:

```bash
python scripts/qdrant/prepare_branch2.py --data-root /data --state-root /state
```

Branch 1 publishes its matrix/metadata and encoder compatibility gates with
`scripts/qdrant/prepare_branch1.py` after the pinned model setup has completed.

The OCR FTS database is prepared explicitly with
`scripts/qdrant/prepare_text_indexes.py`; API startup only opens the existing
database and reports a not-ready capability when it is absent. Branch 3 ASR and
OCR each run an independent 12-stream bilingual (VI/EN) FTS search, capped at
500 frames and displayed as 150. ASR has an independent preparation command,
`scripts/qdrant/prepare_asr_index.py`, which atomically builds
`branch3.asr-index.v2` and maps every segment to its nearest canonical
keyframe. The OCR command atomically builds `branch3.ocr-index.v3` and publishes
`branch3_ocr_manifest.json`; OCR preparation never rebuilds the ASR database.

The `final_fusion` branch owns KIS cross-branch orchestration. It invokes the
four branch services in-process under one shared heavy-search lock, applies
weighted rank-only RRF (`k=60`, `40/30/15/15`) to the 1,500/500/500/500 pools,
keeps the top 150, and uses the reusable BEiT-3 COCO dual-encoder cosine scorer
on only the first 100 (`25% BEiT-3 + 75% RRF`). It never calls a branch HTTP
endpoint or uses cross-attention.
