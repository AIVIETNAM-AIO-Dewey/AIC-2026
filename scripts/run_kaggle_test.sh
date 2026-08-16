#!/usr/bin/env bash
# Quick Kaggle Smoke Test & Visual Verification Launcher
set -euo pipefail

VIDEO_ID="${AIC_VIDEO_ID:-L21_V001}"
DATA_ROOT="${AIC_DATA_ROOT:-/kaggle/input/aic2026-data}"
ARTIFACT_ROOT="${AIC_ARTIFACT_ROOT:-/kaggle/working/aic2026-artifacts}"
LIMIT="${AIC_LIMIT:-5}"

SMOKE_ROOT="$ARTIFACT_ROOT/smoke"
FRAME_MANIFEST="$SMOKE_ROOT/frame_manifests/$VIDEO_ID.jsonl"
MASK_ARTIFACT="$SMOKE_ROOT/object_description/masks/$VIDEO_ID.jsonl"
DESCRIPTION_ARTIFACT="$SMOKE_ROOT/object_description/descriptions/$VIDEO_ID.jsonl"
MAP_CSV="$DATA_ROOT/map-keyframes/$VIDEO_ID.csv"
FRAMES_DIR="$DATA_ROOT/keyframes/$VIDEO_ID"
OBJECTS_DIR="$DATA_ROOT/objects/$VIDEO_ID"
VIS_DIR="$SMOKE_ROOT/visualizations/$VIDEO_ID"

echo "================================================================="
echo " 🚀 Running DAM Smoke Test for Video: $VIDEO_ID (Limit: $LIMIT)"
echo "================================================================="

echo "1/4. Building Frame Manifest..."
python scripts/build_frame_manifest.py \
  --config configs/offline/object_description.yaml \
  --video-id "$VIDEO_ID" \
  --map-csv "$MAP_CSV" \
  --frames-dir "$FRAMES_DIR" \
  --output "$FRAME_MANIFEST" \
  --resume \
  --limit "$LIMIT"

echo "2/4. Generating SAM Region Masks..."
python scripts/prepare_object_masks.py \
  --config configs/offline/object_description.yaml \
  --video-id "$VIDEO_ID" \
  --frame-manifest "$FRAME_MANIFEST" \
  --objects-dir "$OBJECTS_DIR" \
  --output "$MASK_ARTIFACT" \
  --device cuda \
  --resume \
  --limit "$LIMIT"

echo "3/4. Generating DAM Descriptions with Combined Prompt..."
python scripts/run_dam_descriptions.py \
  --config configs/offline/object_description.yaml \
  --video-id "$VIDEO_ID" \
  --mask-artifact "$MASK_ARTIFACT" \
  --output "$DESCRIPTION_ARTIFACT" \
  --device cuda \
  --resume \
  --limit "$LIMIT"

echo "4/4. Running Visual Inspection & Generating Image Cards..."
python scripts/inspect_dam_results.py \
  --artifact "$DESCRIPTION_ARTIFACT" \
  --data-root "$DATA_ROOT" \
  --output-dir "$VIS_DIR" \
  --limit "$LIMIT"

echo "================================================================="
echo " ✅ DAM Test Completed Successfully!"
echo " Visual Inspection Images Saved To: $VIS_DIR"
echo "================================================================="
