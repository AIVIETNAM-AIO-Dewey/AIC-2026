# Model Registry & License Gate

`configs/models.yaml` is the machine-readable source of truth for model identities, revisions, and license metadata. Runtime setup scripts validate checksums and snapshot inventories against these specifications.

| Role | Model / Checkpoint | Pinned Revision / SHA-256 | License | Status |
|---|---|---|---|---|
| Bbox-to-mask refinement | `facebook/sam-vit-base` | `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f` | Apache-2.0 | Locked |
| Region caption weights | `nvidia/DAM-3B` | `0797bedd98d645cd021379a4661ee233da279bba` | NVIDIA Noncommercial | Locked (competition rule check required) |
| DAM source code | `NVlabs/describe-anything` | `153ad3d33c29324e9197f565547c6bc8500da02d` | Apache-2.0 | Locked |
| Branch-1 scene embedding (768d) | `google/siglip2-base-patch16-224` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | Apache-2.0 | Locked |
| Branch-1 visual-text embedding (1024d) | `facebook/metaclip-2-worldwide-huge-quickgelu` | `2431b607fc8e05dd43b73797ba1a7a042514bcf4` | CC-BY-NC-4.0 | Locked |
| Branch-1 & KIS reranker (768d) | `beit3_base_patch16_384_coco_retrieval.pth` (`microsoft/unilm`) | UniLM `ca43e4cd19445a536f133bf2bc25b573b2f0c7c5`, checkpoint SHA-256 `df39666a88508ccd356567616582bc62cd56fa86ad6a8f8e50471b35217c8629` | MIT | Locked |
| Branch-2 DAM dense encoder (1024d) | `BAAI/bge-m3` | Snapshot hashed in `query_models.json` | MIT | Active (commit revision pin pending) |
| Branch-3A Vietnamese ASR | `vinai/PhoWhisper-large` (`kiendt/PhoWhisper-large-ct2`) | Model registry locked | MIT | Locked |
| Branch-3B Vietnamese OCR | Offline OCR JSONL (`branch3.ocr-index.v3` SQLite FTS5) | Verified against 247,956 keyframes in `branch3_ocr_manifest.json` | — | Active (offline provenance unverified → `production_ready=false`) |
| Query decomposition | `LocalQueryParser` + `build_kis_query_plan` | Deterministic in-repo (`branch1.query.v1`) | Repo license | Active (no external LLM calls) |

## Model Update Procedure

1. Verify competition rules and redistribution rights for weights and derived outputs.
2. Review the model card and upstream source if `trust_remote_code` would be needed (default remains `false`).
3. Pin the immutable commit SHA in `configs/models.yaml` and the setup scripts (`scripts/qdrant/setup_query_models.py`, `scripts/qdrant/setup_branch1_models.py`).
4. Verify SHA-256 checksums for external archive/checkpoint downloads (`beit3`, Qdrant binary, offline Kaggle wheels).
5. Run the test suite and data gates (`prepare_branch1.py`, `prepare_branch2.py`, `qdrant/ingest.py --verify-only`) before enabling runtime search.

## Provenance Notes

- Online DAM installs require PEP 610 VCS metadata matching `code_revision`; Kaggle Internet-off runs enforce `SHA256SUMS` verification on private runtime wheels.
- Because legacy offline visual/DAM/OCR exports did not carry immutable checkpoint-hash provenance in their original metadata files, `/api/health` keeps `production_ready=false` intentionally while remaining operationally `ready=true` when all data gates pass.
