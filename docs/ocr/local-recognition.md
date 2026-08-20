# Local Vietnamese/English recognition pilot

This pilot keeps OCR Phase 1 unchanged. It evaluates a local VietOCR recognizer on the
165 immutable, verified L23 crops before any full-dataset run. Exact-match accuracy is
reported, but it is not a hard launch requirement unless an explicit threshold is passed.

## Runtime isolation

Use a separate environment for Phase 2. Do not install VietOCR or PyTorch into the Phase 1
environment. The recognizer must receive a local config, a local weight file, and the exact
SHA-256 of that weight file. The runner never asks VietOCR to download a model implicitly.
It also forces `cnn.pretrained: false`: VietOCR 0.3.13 otherwise asks torchvision for an
unused ImageNet VGG19 initialization (about 548 MB) before strictly loading the complete
local OCR checkpoint.

The official `vgg_seq2seq.pth` acquired for this pilot is 89,575,371 bytes with SHA-256
`0921503a41375a0584268e23ef3d414ea478a8fe8777865c7745d38f2d0bc5db`.

## Prepare the read-only evaluation set

```bash
.venv/bin/python offline/scripts/run_ocr_phase2_local.py prepare-l23 \
  --annotation-root artifacts/ocr_annotation/L23_v1 \
  --output artifacts/ocr_recognition_eval/L23_verified_165_v1/manifest.jsonl \
  --expected-samples 165
```

This command reads the annotation database in SQLite read-only mode, validates every crop
path and SHA-256, and writes a new evaluation manifest. It does not create training data or
change annotation state.

## Run VietOCR

```bash
PHASE2_PY=/path/to/phase2-vietocr/bin/python
WEIGHTS=/path/to/vgg_seq2seq.pth
WEIGHTS_SHA256=0921503a41375a0584268e23ef3d414ea478a8fe8777865c7745d38f2d0bc5db

test "$(wc -c < "$WEIGHTS" | tr -d ' ')" = 89575371
test "$(shasum -a 256 "$WEIGHTS" | awk '{print $1}')" = "$WEIGHTS_SHA256"

"$PHASE2_PY" offline/scripts/run_ocr_phase2_local.py run-vietocr \
  --manifest artifacts/ocr_recognition_eval/L23_verified_165_v1/manifest.jsonl \
  --crop-root artifacts/ocr_annotation/L23_v1 \
  --output artifacts/ocr_recognition_eval/L23_verified_165_v1/vietocr-results.jsonl \
  --weights "$WEIGHTS" \
  --weights-sha256 "$WEIGHTS_SHA256" \
  --device cuda:0
```

One canonical JSONL record is fsynced after each crop. If the process stops, rerun the same
command with `--resume`; committed records are validated and are not inferred again.

## Score without a hard accuracy gate

```bash
.venv/bin/python offline/scripts/run_ocr_phase2_local.py score \
  --manifest artifacts/ocr_recognition_eval/L23_verified_165_v1/manifest.jsonl \
  --results artifacts/ocr_recognition_eval/L23_verified_165_v1/vietocr-results.jsonl \
  --output artifacts/ocr_recognition_eval/L23_verified_165_v1/vietocr-score.json
```

The default threshold is zero: the command passes only when at least one crop is recognized
and no inference record has a runtime error. Exact match, case-insensitive exact match, CER,
character accuracy, and Vietnamese-diacritic recall are still reported for comparison.

## Run one production Phase 1 shard

After Phase 1 detect/track/select has completed, recognize its representatives without
materializing crop files:

```bash
"$PHASE2_PY" offline/scripts/run_ocr_phase2_local.py run-representatives \
  --phase1-config offline/configs/offline/ocr_phase1_kaggle_gpu.yaml \
  --frame-manifest /kaggle/working/shards/shard-000001.frames.jsonl \
  --data-root /kaggle/input/aic-test-dataset/data \
  --detections /kaggle/working/phase1/shard-000001/detections.jsonl \
  --trajectories /kaggle/working/phase1/shard-000001/trajectories.jsonl \
  --representatives /kaggle/working/phase1/shard-000001/representatives.jsonl \
  --output /kaggle/working/phase2/shard-000001/recognition.jsonl \
  --weights "$WEIGHTS" \
  --weights-sha256 "$WEIGHTS_SHA256" \
  --source-commit-sha "$(git rev-parse HEAD)" \
  --batch-size 256 \
  --commit-interval-records 256 \
  --frame-cache-capacity 8 \
  --frame-cache-max-bytes 268435456 \
  --device cuda:0
```

This path verifies the linked Phase 1 artifacts before inference, reconstructs every crop
from its source frame and immutable provenance, and rechecks source/canonical/crop hashes.
Its running receipt commits progress every 32 records. The v2 receipt binds the exact batch
size, bounded canonical-frame LRU settings, execution-policy hash, GPU/CUDA/cuDNN/TF32
runtime evidence, and cumulative cache/batch statistics. A failed batch writes no records.
After interruption, rerun the exact command with `--resume`; any torn or uncommitted tail is
truncated. A model error leaves the stage resumable and prevents final publication. A v1
receipt must remain with its original runner/output root; do not reinterpret it as v2.

The production runner stops at raw representative recognition. Temporal consensus and the
final backend adapter are separate deterministic commands:

```bash
"$PHASE2_PY" offline/scripts/run_ocr_phase2_local.py consensus \
  --trajectories /kaggle/working/phase1/shard-000001/trajectories.jsonl \
  --representatives /kaggle/working/phase1/shard-000001/representatives.jsonl \
  --recognition-output /kaggle/working/phase2/shard-000001/recognition.jsonl \
  --output /kaggle/working/phase2/shard-000001/consensus.jsonl \
  --run-id vietocr-local-consensus-v1

"$PHASE2_PY" offline/scripts/run_ocr_phase2_local.py build-final \
  --trajectories /kaggle/working/phase1/shard-000001/trajectories.jsonl \
  --consensus /kaggle/working/phase2/shard-000001/consensus.jsonl \
  --output /kaggle/working/ocr/vietocr-local-v1/shard-000001.jsonl \
  --run-id vietocr-local-v1
```

Consensus first uses exact agreement; disagreement is resolved by support count, confidence
weighted by Phase 1 crop quality, then representative rank. Empty trajectories remain
explicit but are marked unaccepted, so backend ingestion does not index them. This stage has
no VLM fallback and does not rewrite model text.
