# No-Fusion UI Index Contract

## Source layout

```text
AIC_HCM_BATCH_1/
├── artifacts/
│   ├── keyframes/<VIDEO_ID>/<FRAME_IDX:08d>.jpg
│   ├── map-keyframes/<VIDEO_ID>.csv
│   ├── unified_metadata/<VIDEO_ID>.jsonl
│   ├── scene_embeddings/<VIDEO_ID>.safetensors
│   ├── asr_aligned/
│   │   ├── keyframes_speech_vectors.f16.npy
│   │   └── keyframes_asr_metadata.jsonl
│   ├── dense_text_embeddings/
│   │   ├── dam_vectors.f16.npy
│   │   └── dam_metadata.jsonl
│   └── media-info/<VIDEO_ID>.json
└── unified_index/
    ├── keyframes_visual_vectors.f16.npy   # (247956, 768), float16
    ├── keyframes_speech_vectors.f16.npy   # (247956, 1024), float16
    ├── keyframes_metadata.jsonl           # 247,956 rows
    ├── dam_vectors.f16.npy                # (681355, 1024), float16
    ├── dam_metadata.jsonl                 # 681,355 rows
    ├── unified_dataset_summary.json
    └── manifest.json
```

## Canonical ordering

Frame order is sorted `map-keyframes/*.csv` filename order followed by row
order inside each CSV. The same global row must address:

- `keyframes_metadata.jsonl[row]`
- `keyframes_visual_vectors.f16.npy[row]`
- `keyframes_speech_vectors.f16.npy[row]`

`point_id` is globally sequential from 1 to 247,956. `visual_vector_row` and
`speech_vector_row` are globally sequential from 0 to 247,955.

## Keyframe metadata

Each JSONL row contains the source DAM and OCR fields plus aligned ASR fields:

```json
{
  "point_id": 1,
  "video_id": "L21_V001",
  "keyframe_n": 1,
  "frame_idx": 4,
  "pts_time_s": 0.1333,
  "fps": 30.0,
  "frame_uid": "L21_V001:4",
  "image_relpath": "keyframes/L21_V001/00000004.jpg",
  "visual_vector_row": 0,
  "speech_vector_row": 0,
  "dam_summary_en": "...",
  "dam_regions": [],
  "ocr_text": "71 06:30:11 giây",
  "asr_transcript_vi": "",
  "has_speech": false,
  "asr_alignment_method": "nearest_legacy_keyframe_same_video"
}
```

Empty OCR text and silent ASR rows are valid. Silent ASR rows contain zero
vectors and are excluded from ASR search.

## DAM metadata

Each DAM vector row has one metadata row with a parent frame identity:

```json
{
  "video_id": "L21_V001",
  "frame_idx": 4,
  "keyframe_n": 1,
  "region_id": "L21_V001:4:d000",
  "class_entity": "sky",
  "bbox": [0.002, 0.0002, 0.614, 1.0],
  "description_en": "..."
}
```

All 247,956 frames have between one and three DAM regions.

## Model provenance

The visual vectors and online text queries use:

```text
model: google/siglip2-base-patch16-224
revision: 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
dimension: 768
normalization: L2 unit vectors
```

The revision is pinned in `online/configs/server_config.yaml` and passed to
both the tokenizer and model loader.
