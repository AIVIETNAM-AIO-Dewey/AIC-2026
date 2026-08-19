# OCR Phase 1 real-model quality gate

Phase 1 cannot be approved from synthetic fixtures or detection counts. Mount the exact
detector files listed in `offline/configs/offline/ocr_phase1.yaml`; do not download or
substitute a model. The CPU profile is the only approved execution identity. The Kaggle GPU
profile remains `blocked_unverified_runtime`.

## Separate performance and evaluation inputs

The performance manifest contains 500–1.000 real, supported source images and is used for
throughput, peak RSS, detection distributions, and tracking metrics. Unsupported image modes
must not appear in this successful manifest.

The evaluation JSONL is an independently reviewed, fully annotated subset of the performance
manifest. It has at least 100 labeled frames, 200 non-ignored text instances, and at least 15
distinct frames for each of `positive_text`, `multi_box`, `horizontal`, `perspective`,
`clipped_edge`, and `near_vertical`. Each record has exactly `frame_uid`, `strata`, and
`instances`. Frame `strata` is reserved for the frame-level `multi_box` group and must be
present exactly when the frame has at least two non-ignored instances. Every instance has
exactly `instance_id`, `polygon_xy`, `ignore`, and `strata`; non-ignored instance strata are
drawn from `positive_text`, `horizontal`, `perspective`, `clipped_edge`, and
`near_vertical`, while ignored instances have no strata. Per-stratum recall for these five
groups counts only instances carrying that label. `multi_box` recall separately counts all
non-ignored instances in frames carrying the frame-level label.

The locked `aic26.ocr_detection_quality.v1` policy uses one-to-one matching at polygon IoU
0,50 and requires:

- overall recall ≥ 0,95;
- recall for every required stratum ≥ 0,90;
- overall precision ≥ 0,50.

The gate reports TP, FP, FN, precision, recall, F1, per-stratum recall, matched-IoU
distribution, and the hashes of both manifests and the threshold policy. Detection bytes and
their receipt are snapshotted into one hash baseline before semantic verification; quality
metrics are parsed from those exact bytes and both files are re-hashed before report
publication.

## Negative source fixtures

Unsupported/corrupt source fixtures use a separate strict manifest. Run this preflight without
constructing Paddle or the detector:

```bash
.venv/bin/python offline/scripts/verify_ocr_phase1_negative_fixtures.py \
  --config offline/configs/offline/ocr_phase1.yaml \
  --negative-manifest /mounted/pilot/negative-sources.jsonl \
  --negative-data-root /mounted/pilot/negative-data \
  --receipt /writable/pilot/negative-suite.receipt.json
```

Every fixture must fail with its exact expected stable error code and reason. An accepted
fixture, missing fixture, manifest drift, or mismatched receipt fails the final gate.

## Kaggle execution attestation

The final gate requires strict external operator/platform evidence containing the provider,
kernel identifier, source commit, config/model/runtime identity, exact boolean
`internet_enabled: false`, CPU device, timestamp, approver, and payload checksum. Its SHA-256
is bound into the report. This is operational evidence supplied by the operator/platform; it
is not cryptographic proof that every possible network route was blocked. The Python socket
guard remains best-effort only and is not network-isolation evidence.

Phase 1 detector pilot/runs execute in a Kaggle Internet-off session. Phase 2 OpenAI API and
rclone operations execute separately in an Internet-on session/notebook and may read only
verified Phase 1 artifacts.

## Run the real gate

```bash
mkdir -p /writable/pilot/paddlex-cache
.venv/bin/python offline/scripts/gate_ocr_phase1_real_model.py \
  --config offline/configs/offline/ocr_phase1.yaml \
  --model-root /mounted/read-only-model-root \
  --runtime-cache-root /writable/pilot/paddlex-cache \
  --performance-manifest /mounted/pilot/performance-frames.jsonl \
  --evaluation-manifest /mounted/pilot/evaluation-ground-truth.jsonl \
  --data-root /mounted/pilot/data \
  --execution-attestation /mounted/pilot/kaggle-execution-attestation.json \
  --expected-source-commit-sha "$AIC_PHASE1_SOURCE_COMMIT_SHA" \
  --negative-manifest /mounted/pilot/negative-sources.jsonl \
  --negative-data-root /mounted/pilot/negative-data \
  --negative-suite-receipt /writable/pilot/negative-suite.receipt.json
```

Missing model/data, insufficient labels, low quality, online/mismatched attestation, or a
missing/tampered negative suite returns nonzero. Unit-test fixtures may exercise matching but
never constitute a `real_model_gate_pass` report.

Do not start the 177.000-frame run until the real CPU pilot passes. If a video exceeds a locked
frame/detection/resource limit, stateful cross-shard tracking must be designed instead of
renaming trajectory IDs.
