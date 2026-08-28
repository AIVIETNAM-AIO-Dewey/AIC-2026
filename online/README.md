# AIC-2026 No-Fusion Retrieval Lab

This branch is a KIS-only modality evaluation UI. It searches SigLIP, DAM,
OCR, and ASR independently and never combines their result pools.

## Retrieval behavior

| Pool | Query field | Ranking score |
|---|---|---|
| SigLIP | `global_scene_en` | Raw image/text cosine |
| DAM | `objects_en` | Mean of each object query's best region cosine |
| OCR | `ocr_keywords` | Exact case-insensitive keyword coverage |
| ASR | `speech_vi`, then explicit `original_query` fallback | Raw transcript cosine |

There is no weighted RRF, cross-modal synergy, score normalization, Stage-2
cross-encoder, VQA reasoning, or TRAKE sequence solver in the server path. The
UI provides `All Pools`, `SigLIP`, `DAM`, `OCR`, and `ASR` views and shows the
exact subquery and score type used by every pool.

OCR is lexical because the dataset contains OCR text but no OCR embedding
matrix. DAM's region-to-frame aggregation is displayed explicitly and applies
no coverage bonus.

## Dataset

The configured dataset is:

```text
/Users/macbookpro/Downloads/AIC-HCM-BATCH-1/AIC_HCM_BATCH_1
```

Validated totals:

- 873 videos
- 247,956 keyframes and SigLIP vectors
- 247,956 aligned ASR rows, including 146,244 speech-bearing frames
- 681,355 DAM region vectors
- 216,506 frames with non-empty OCR text
- 873 media-info records

See [DATA_STRUCTURE.md](DATA_STRUCTURE.md) for the exact index contract.

## Build the derived UI index

The builder reads the source artifacts without changing them. It validates all
frame identities, vector rows, images, media-info records, and DAM parents. It
refuses to overwrite an existing output directory.

```bash
cd "/Users/macbookpro/Documents/Final AIC/AIC-2026"
source aic/bin/activate
python -m online.src.index.build_nofusion_index
```

The default output is:

```text
/Users/macbookpro/Downloads/AIC-HCM-BATCH-1/AIC_HCM_BATCH_1/unified_index
```

Use `--asset-mode hardlink` to avoid duplicating the already-built ASR and DAM
files on the same filesystem. The default is a portable copy.

## Run

```bash
cd "/Users/macbookpro/Documents/Final AIC/AIC-2026"
npm ci
npm run build
source aic/bin/activate
pip install -r online/requirements.txt
python -m online.server
```

Open [http://127.0.0.1:8890](http://127.0.0.1:8890).

Re-run `npm run build` after changing files under `online/frontend`. FastAPI
serves the compiled Vite output rather than the TypeScript source tree.

Startup loads the metadata and warms the pinned SigLIP and BGE-M3 query
encoders. The BGE reranker is not loaded. Matrix scans are block-based to avoid
multi-gigabyte temporary float32 copies on Mac.

## Verify

```bash
python -m unittest discover -s online/tests -p 'test_nofusion.py' -v
npm ci
npm run typecheck
npm run build
```

The focused tests verify isolated rankings, silent-ASR filtering, transparent
DAM aggregation, lexical OCR labeling, zero-weight independence, and the
non-destructive dataset builder.
