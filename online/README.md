# AIC-2026 No-Fusion Retrieval Lab

This branch is a KIS-only modality evaluation UI. It searches SigLIP, DAM,
OCR, and ASR independently and never combines their result pools.

## Retrieval behavior

| Pool | Query field | Ranking score |
|---|---|---|
| SigLIP | `global_scene_en` | Raw image/text cosine |
| DAM | `objects_en` | Mean of each object query's best region cosine |
| OCR | `ocr_keywords` | Exact case-insensitive keyword coverage |
| ASR | `speech_vi` only | Raw transcript cosine |

There is no weighted RRF, cross-modal synergy, score normalization, Stage-2
cross-encoder, VQA reasoning, or TRAKE sequence solver in the server path. The
UI provides `All Pools`, `SigLIP`, `DAM`, `OCR`, and `ASR` views and shows the
exact subquery and score type used by every pool.

OCR is lexical because the dataset contains OCR text but no OCR embedding
matrix. DAM's region-to-frame aggregation is displayed explicitly and applies
no coverage bonus. A configurable `0.50` cosine threshold labels weak DAM
region evidence as unmatched without changing the raw rank. ASR is skipped
when the parser finds no explicit speech or narration, so a visual description
can never leak into the audio pool.

The query parser keeps SigLIP captions within a 40-word budget for the model's
64-token text window, prioritizes distinctive visible geometry, and limits DAM
to three concrete region phrases. The UI displays SigLIP token diagnostics and
the effective query whenever manual JSON exceeds the model window.

Every result card also offers `Search inside this video`. This is a manual,
auditable drill-down: it reruns the unchanged `global_scene_en` SigLIP cosine
against every indexed frame in the selected video. The source card only chooses
the video scope; its score is not reused. The UI reports the number of evaluated
frames and keeps `no fusion` / `no reranking` visible, with a button to return to
the untouched four original pools.

The opt-in `Discover videos → frames` view addresses video discovery without
silently changing those raw pools. It runs each `objects_en` phrase as its own
DAM search, deduplicates the Top 20 raw DAM frames into candidate video scopes,
then keeps the Top 10 raw SigLIP frames inside each candidate video. Each object
has a separate cascade section. Final frame order uses only SigLIP cosine; DAM
scores are shown as provenance and are never added to the final score. Because
DAM determines the video scope, the UI labels this operation explicitly as
cross-modal gating rather than presenting it as a modality-only result.

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

The focused tests verify isolated rankings, speech-query purity, transparent
DAM aggregation and unmatched labeling, parser budgets, direct-JSON handling,
lexical OCR labeling, zero-weight independence, and the non-destructive dataset
builder. They also verify that video drill-down results contain only frames from
the explicitly selected video and remain raw SigLIP cosine results. Discovery
tests verify explicit DAM gating, per-object provenance, and that DAM scores do
not enter the final SigLIP rank.
