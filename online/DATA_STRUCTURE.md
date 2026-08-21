# AIC-2026 Online Retrieval System — Data Structure & Assumptions

This document outlines the filesystem hierarchy and metadata schemas used by the retrieval engine and web UI.

---

## 1. Directory Hierarchy

```
AIC_Challenger/
├── data/
│   └── keyframes/                          <-- paths.keyframes_root
│       ├── L21_V001/
│       │   ├── 001.jpg                     (zero-padded 3 digits, e.g. 001.jpg, 002.jpg)
│       │   ├── 002.jpg
│       │   └── ...
│       ├── L21_V002/
│       └── ... (L21 to L30)
│
├── map-keyframes/                          <-- paths.map_keyframes
│   ├── L21_V001.csv                        (Columns: n, pts_time, fps, frame_idx)
│   ├── L21_V002.csv
│   └── ...
│
└── AIC_HCM/
    ├── unified_index/                      <-- paths.unified_index
    │   ├── keyframes_visual_vectors.f16.npy    (177321 x 768 float16: SigLIP-2)
    │   ├── keyframes_speech_vectors.f16.npy    (177321 x 1024 float16: BGE-M3 Speech)
    │   ├── dam_vectors.f16.npy                 (435713 x 1024 float16: DAM Objects)
    │   ├── keyframes_metadata.jsonl            (177321 JSON lines)
    │   ├── dam_metadata.jsonl                  (435713 JSON lines)
    │   └── unified_dataset_summary.json        (Summary statistics)
    │
    └── artifacts/
        ├── asr_segments/                   <-- paths.asr_segments
        │   ├── L21_V001.jsonl              (PhoWhisper segment transcripts)
        │   └── ...
        └── dam_descriptions/              <-- paths.dam_descriptions
            ├── L21_V001.jsonl              (DAM object bounding boxes & captions)
            └── ...
```

---

## 2. Metadata JSONL Schemas

### `keyframes_metadata.jsonl` (1 line per keyframe)
```json
{
  "point_id": 1,
  "video_id": "L21_V001",
  "keyframe_n": 1,
  "frame_idx": 0,
  "pts_time_s": 0.0,
  "fps": 30.0,
  "frame_uid": "L21_V001:0",
  "image_relpath": "keyframes/L21_V001/001.jpg",
  "asr_transcript_vi": "chào mừng quý vị đến với chương trình...",
  "has_speech": true,
  "dam_summary_en": "a cyclist wearing yellow jersey, red helmet...",
  "num_objects": 4,
  "ocr_text": "HTV THỂ THAO"
}
```

### `dam_metadata.jsonl` (1 line per detected object)
```json
{
  "video_id": "L21_V001",
  "frame_idx": 0,
  "keyframe_n": 1,
  "region_id": "L21_V001:0:d000",
  "class_entity": "Bicycle Helmet",
  "bbox": [0.4686, 0.3664, 0.6361, 0.4671],
  "description_en": "A red aerodynamic racing bicycle helmet worn by the lead cyclist."
}
```
*Note: `bbox` format is `[ymin, xmin, ymax, xmax]` or `[x1, y1, x2, y2]` normalized between 0.0 and 1.0.*

---

## 3. Image Access & Missing Frames Fallback

1. **Local Access via `file://`**:
   The frontend can resolve images directly using the local path:
   `file:///Users/khoale/Downloads/AIC_Challenger/data/keyframes/<video_id>/<keyframe_n:03d>.jpg`
2. **Server Static Route**:
   The server also exposes `http://localhost:8890/keyframes/<video_id>/<keyframe_n:03d>.jpg`.
3. **Missing Keyframe Fallback**:
   If an image file is not downloaded on the local machine, the frontend automatically falls back to an aesthetic SVG/CSS placeholder displaying the exact Keyframe label (e.g. `L23_V007 #029 | 129.8s`).
