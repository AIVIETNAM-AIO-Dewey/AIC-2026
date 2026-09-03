# Online data contract

The online CPU runtime reads the source data tree read-only. Qdrant stores the
searchable vectors; derived SQLite files and readiness manifests live in the
state volume. Do not point the server at the old `unified_index` or embedded
Qdrant layouts.

```text
data/
├── keyframes/<VIDEO_ID>/<FRAME_IDX:08d>.jpg
├── visual_embeddings/
│   ├── metaclip2/
│   │   ├── keyframes_visual_vectors.f16.npy       # (247956, 1024), float16
│   │   ├── keyframes_metadata.jsonl                # canonical frame rows
│   │   ├── keyframe_index.csv
│   │   └── run_manifest.json
│   └── beit3/
│       ├── keyframes_visual_vectors.f16.npy       # (247956, 768), float16
│       ├── keyframes_metadata.jsonl
│       ├── keyframe_index.csv
│       └── run_manifest.json
├── dense_text_embeddings/
│   ├── dam_vectors.f16.npy                         # (681355, 1024), float16
│   └── dam_metadata.jsonl
├── asr_segments/*.jsonl                            # 55168 segments
├── ocr_transcripts/*.jsonl                         # frame OCR rows
└── unified_metadata/<VIDEO_ID>.jsonl               # optional detail fallback

state/
├── qdrant_ingestion_manifest.json
├── branch1_data_gate.json
├── branch1_encoder_compatibility.json
├── branch2_dam_manifest.json
├── branch2_dam_bm25.sqlite3
├── branch3_asr_manifest.json
├── branch3_ocr_manifest.json                      # branch3.ocr-index.v3
├── ocr.sqlite3                                     # SQLite user_version 3
├── asr.sqlite3
├── query_embeddings.sqlite3
├── runtime_fingerprint.json
└── resource_qualification.json
```

`asr.sqlite3` is built by `scripts/qdrant/prepare_asr_index.py` using SQLite
schema/user version 4. Each ASR segment stores its transcript, time interval,
and one canonical frame identity. The v2 index stores a deterministic `build_id`
and source/canonical fingerprints in `asr_meta`; `branch3_asr_manifest.json`
binds the index to the source segment files and canonical metadata by fingerprint/stat.
The manifest distinguishes all 873 source videos from the 872 videos that have
searchable ASR segments; `L30_V029` is retained as a valid zero-segment source.

## Canonical frame identity

There are exactly 247,956 frames. The canonical row order is the global point
order used by both visual matrices and both Qdrant frame collections:

```json
{
  "point_id": 1,
  "video_id": "L21_V001",
  "keyframe_n": 1,
  "frame_idx": 4,
  "pts_time_s": 0.1333,
  "fps": 30.0,
  "frame_uid": "L21_V001:4",
  "image_relpath": "keyframes/L21_V001/00000004.jpg"
}
```

`frame_uid` must equal `<video_id>:<frame_idx>`, and `point_id` is sequential
from 1 through 247,956. MetaCLIP2 metadata is the canonical frame mapping;
BEiT-3 metadata must match its identity fields row-for-row.

## DAM identity and join

Each of the 681,355 DAM rows has one vector and a unique `region_id`:

```json
{
  "video_id": "L21_V001",
  "frame_idx": 4,
  "region_id": "L21_V001:4:d000",
  "class_entity": "sky",
  "bbox": [0.002, 0.0002, 0.614, 1.0],
  "description_en": "..."
}
```

The current DAM export may omit `keyframe_n`. Preparation and ingestion join
`video_id/frame_idx` against canonical frame metadata and persist the derived
`keyframe_n`, `frame_uid`, `parent_point_id`, `pts_time_s`, `fps`, and
`image_relpath` in SQLite/Qdrant. A sparse-only BM25 hit therefore retains the
same canonical detail payload as a dense hit. A mismatch or unknown parent is a
hard data-gate failure.

## Vector and model contract

| Space | Collection/vector | Rows × dimension | Query text | Normalization |
|---|---|---:|---|---|
| SigLIP2 | `aic_frames/siglip2` | 247,956 × 768 | VI + EN (Branch 1) | L2 |
| MetaCLIP2 | `aic_frames/metaclip2` | 247,956 × 1,024 | VI + EN (Branch 1) | L2 |
| BEiT-3 COCO | `aic_beit3_frames/beit3` | 247,956 × 768 | EN | L2 |
| DAM BGE-M3 | `aic_dam_regions/dam` | 681,355 × 1,024 | EN caption | L2 |

The online text encoders must use the same model family, pooling, dimension,
tokenizer limits, and immutable revision that produced each space. Branch 1
uses 12 VI/EN SigLIP2 streams, 12 VI/EN MetaCLIP2 streams, and six EN BEiT-3
streams. Branch 2 uses six EN BGE-M3 streams for DAM text retrieval and six EN
BM25 streams; BEiT-3 is only full-frame cosine validation on the hybrid
candidate set. It does not re-embed DAM descriptions or decode images online.

`scripts/qdrant/prepare_branch1.py` and `prepare_branch2.py` publish the
manifests only after shape, dtype, finite values, ordering, identity, and
normalization checks pass. Runtime health reads those manifests and never
rebuilds an index during server startup.
