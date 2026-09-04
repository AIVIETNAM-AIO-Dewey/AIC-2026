# AIC-2026 Native Retrieval Workbench

## Native Apple Silicon run (no Docker)

On Apple Silicon, the local launcher uses MPS for SigLIP2, MetaCLIP2,
BGE-M3, BEIT3, and DAM exact tensor scoring. Qdrant and SQLite FTS5 run as
native ARM64 CPU services. An unsupported Metal operation falls back only its
own encoder worker to CPU and is exposed in `/api/health`.

```bash
./scripts/native_macos/aic_local.sh setup
./scripts/native_macos/aic_local.sh start
./scripts/native_macos/aic_local.sh status
./scripts/native_macos/aic_local.sh stop
```

`setup` validates and copies the downloaded MetaCLIP2/BEIT3 artifacts into the
main artifact tree, creates a Python 3.12 environment, installs native Qdrant,
builds all indexes and collections, runs the frontend build gate, and starts
the compiled UI at <http://127.0.0.1:8890>. It deliberately performs no model
inference or retrieval query; the first inference is initiated from the UI.

Paths can be overridden with `AIC_DATA_ROOT`, `AIC_METACLIP2_SOURCE`,
`AIC_BEIT3_SOURCE`, `AIC_LOCAL_RUNTIME_ROOT`, and `AIC_LOCAL_VENV_ROOT`.
Because the main dataset is under `Downloads`, macOS may ask once for file
access. If the launcher reports that access is blocked, enable Downloads for
your terminal app under **System Settings → Privacy & Security → Files and
Folders**, then run `start` again.

The workbench runs locally through `online.cpu_server:app`, Qdrant, persistent
SQLite indexes, and isolated encoder workers. Gemini, Qwen, the former BGE
cross-encoder, and the legacy server/CLI are not part of this runtime.

### Source-frame indexing contract

Video submission uses the original zero-based `frame_idx`, not the sparse
keyframe number. `GET /api/video/{video_id}/timeline` publishes the verified
frame range and exact organizer keyframe anchors. `GET
/api/video/{video_id}/source-frame/{frame_idx}` validates any in-range source
frame without replacing its identity. Exact anchors keep their stored
timestamps; frames between anchors use deterministic piecewise-linear timing.
The nearest indexed frame is labelled separately and is used only for image
preview and stored-vector related-frame suggestions.

## Prepare data

Start Qdrant, prepare the pinned query/branch gates and local text indexes,
then verify the existing collections before any in-place repair:

```bash
docker compose -f docker-compose.qdrant.yml up -d qdrant
docker compose -f docker-compose.qdrant.yml --profile tools run --rm query-model-setup
docker compose -f docker-compose.qdrant.yml --profile tools run --rm branch1-model-setup
docker compose -f docker-compose.qdrant.yml --profile tools run --rm branch1-prepare
docker compose -f docker-compose.qdrant.yml --profile tools run --rm branch2-prepare
docker compose -f docker-compose.qdrant.yml --profile tools run --rm text-index-prepare
docker compose -f docker-compose.qdrant.yml --profile tools run --rm asr-index-prepare
docker compose -f docker-compose.qdrant.yml --profile tools run --rm qdrant-ingest --verify-only
```

The Branch-1 preparer publishes a `branch1.data-gate.v4` report with artifact
fingerprints before ingestion. The read-only `qdrant-ingest --verify-only` pass
must succeed before a repair run. The ingester verifies exact IDs, named-vector
coverage, canonical payloads, and vector direction. A normal run repairs incomplete existing points in place and writes
`qdrant_ingestion_manifest.json` to the shared state volume; it never prunes
unexpected points automatically. Use `--verify-only` for a read-only check and
pass `--recreate` only when the collection vector schema itself is incompatible.

The legacy visual and DAM embedding exports do not contain sufficient immutable
offline revision evidence. Search remains operational when data gates pass, but
health keeps `production_ready=false`; runtime checkpoint hashes do not prove
which checkpoint produced older offline vectors. This is intentional and does
not re-embed or alter source data.

Before a RAM qualification, generate a host-side runtime fingerprint from the
actual image IDs, compose file, and model manifests, set its digest in
`AIC_COMPOSE_FINGERPRINT`, then restart `search-api`. The bootstrap value
`unqualified` can run the UI but can never make production health ready.

## Run

```bash
docker compose -f docker-compose.qdrant.yml up -d qdrant search-api
```

Open `http://127.0.0.1:8890`. Branch 1 runs 30 bilingual/English cosine
streams (SigLIP2 12, MetaCLIP2 12, BEiT-3 6), fuses them, and returns a fixed
API pool of up to 1,500 frames; the UI displays the first 150. Branch 2 uses
six English BGE-M3 DAM streams plus six English BM25 streams, returns up to 500
hybrid frames, and validates only its top 100 with BEiT-3 COCO cosine using a
40% BEiT-3 / 60% hybrid blend. Branch 3 ASR and OCR are independent 12-stream
VI/EN FTS searches, each returning up to 500 frames and displaying 150. The
`KIS Fusion` workspace combines these fixed pools with weighted RRF (`k=60`,
default `40/30/15/15`) and keeps up to 150 canonical frames, then applies
BEiT-3 COCO cosine to ranks 1-100 with a `25% BEiT-3 + 75% RRF` blend. Ranks
101-150 retain their RRF order.
The BEiT-3 stage is dual-encoder cosine scoring through its `language_head`
against precomputed full-frame vectors; it is not cross-attention and is never
trained online.

`Ordered KIS Events` is an opt-in KIS/VQA/TRAKE action. It focuses the same
six-role bundle on each of two to six events, runs the complete unchanged KIS
pipeline once per event, and then keeps only same-video paths with strictly
increasing source-frame indexes and timestamps. Its retrieval cost therefore
scales with the number of events; ordinary KIS search is unaffected. The
sticky overall-query editor is the authoring source for both the six-role
bundle and ordered events. `Prepare bundle & events` uses only the local
deterministic parser and does not run retrieval or model inference. Matching
manually authored bilingual bundles are preserved; manual changes are never
silently overwritten, and lexical mismatch checks warn without blocking. Each
ordered event receives its own text plus only the parent's shared `context`
role, so actions or entities belonging to other events do not leak into its
search. Ordered results always contain every event, but indexed-frame search
does not by itself prove a first appearance or camera motion; verify those in
the video player.

`Search inside this video` is deliberately narrower: it filters Qdrant
to the selected video and fuses only SigLIP2, MetaCLIP2, and BEIT3 at their
Branch-1 weights. DAM, OCR, ASR, and final cross-branch KIS reranking are not
invoked for that scoped action.

After code updates, rebuild the ASR index explicitly and then recreate only the
API service so its read-only SQLite connection observes the new build:

```bash
docker compose -f docker-compose.qdrant.yml --profile tools run --rm asr-index-prepare
docker compose -f docker-compose.qdrant.yml up -d --build --force-recreate search-api
```

OCR is prepared independently and publishes an atomic v3 database/manifest
pair. Run the OCR preparation command once to migrate an older v2 index, and
again when its source or canonical metadata has changed, then recreate only
`search-api`:

```bash
docker compose -f docker-compose.qdrant.yml --profile tools run --rm text-index-prepare
docker compose -f docker-compose.qdrant.yml up -d --build --force-recreate search-api
```

Check `GET /api/branch3/ocr/health` before searching. A healthy v3 OCR index
shows `frames`, `fts_frames`, and `mapped_frames` equal to 247,956, `videos`
equal to 873, `fts_content_verified: true`, and matching
`manifest_build_id`, `internal_build_id`, and `opened_build_id`.
`production_ready` remains false while the offline OCR checkpoint revision is
unverified. Runtime search uses a fast manifest/database gate; the dedicated
OCR health route may perform the cached raw-source audit and report
`source_stale` as a non-blocking rebuild warning.

Check `GET /api/branch3/asr/health` before searching. A healthy v2 index shows
matching `build_id`, `internal_metadata_matches: true`, and a new
`connection_generation`; `production_ready` remains false while the offline ASR
revision is unverified.

The ASR source contains 873 completed video manifests and 55,168 searchable
segments. `L30_V029` is a valid zero-segment video, so health reports
`videos: 873`, `indexed_videos: 872`, `empty_videos: 1`, and
`empty_video_ids: ["L30_V029"]`.
