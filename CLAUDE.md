# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Team repo for the HCMC AI Challenge 2026 (AIC 2026) preliminary round: a video
retrieval pipeline over the organizer's video corpus, handling KIS, Q&A and TRAKE
queries. Designed to run on **Google Colab / Kaggle GPUs**, reproducible from the
CLI, so each member can own one modality without depending on anyone's notebook.

All prose docs (`README.md`, `CONTRIBUTING.md`, `docs/`) are in Vietnamese; code and
data contracts are in English.

Only **one** branch is actually implemented: organizer Objects → SAM masks → DAM
English region captions. SigLIP2, Vietnamese OCR, PhoWhisper ASR, query parser,
fusion, Q&A and TRAKE alignment are agreed architecture but **not implemented** until
the subsystem owner ships code + model pin + validator + tests.

## Commands

```bash
# Dev setup (CPU only — no models, no competition data)
python -m pip install -r requirements/dev.txt   # == pip install -e ".[dev]"

# The full pre-push check (identical to CI)
pre-commit run --all-files
ruff check .
ruff format --check .
pytest -m "not gpu and not slow"

# Single test / single file
pytest tests/unit/test_rle.py::test_roundtrip_random_masks
pytest tests/unit/test_geometry.py -v
```

Pytest markers: `gpu` (needs CUDA + weights), `integration`, `slow`. Default CI runs
everything except `gpu` and `slow`. `tests/conftest.py` puts both `src/` and
`scripts/` on `sys.path`, which is why tests can `import _common` directly.

CI (`.github/workflows/ci.yml`) runs on Python 3.10, and also JSON-validates both
notebooks and YAML-validates `configs/models.yaml` + `configs/offline/*.yaml`.
`pyproject.toml` allows `>=3.10,<3.13`.

**Torch is deliberately absent from every requirements file** — the Colab/Kaggle GPU
image supplies the CUDA-matched wheel, and cloud installs use `--no-deps`. Never add
torch to `requirements/*.txt`.

### Running a stage on a GPU runtime

```bash
python scripts/verify_environment.py --config configs/offline/object_description.yaml \
  --device cuda --write-report        # ALWAYS run before loading any weights

python scripts/build_frame_manifest.py --config ... --video-id "$AIC_VIDEO_ID" \
  --map-csv ... --frames-dir ... --output ... --resume
python scripts/prepare_object_masks.py  --config ... --frame-manifest ... --objects-dir ... --device cuda --resume
python scripts/run_dam_descriptions.py  --config ... --mask-artifact ... --device cuda --resume
python scripts/validate_object_artifacts.py --artifact ... --manifest ... --require-captions
```

`verify_environment.py` defaults to `--minimum-vram-gb 14.0`, sized for DAM-3B. A
lighter stage should pass a lower threshold rather than skip the preflight.

## Architecture

### Three runtime roots — the portable path contract

Every runner reads paths from env vars (or `--data-root/--output-root/--cache-root`
overrides), resolved in `scripts/_common.py::runtime_roots`:

```
AIC_DATA_ROOT      competition data, read-only
AIC_ARTIFACT_ROOT  generated JSONL + manifests
AIC_CACHE_ROOT     HF/torch model snapshots
```

Runner code must never contain `/content`, `/kaggle`, Drive or Windows paths. Notebooks
are thin launchers that only set env vars and shell out to `scripts/`.

### The stage pattern

Every offline stage is the same four pieces, and a new modality should mirror them:

```
configs/offline/<stage>.yaml   pinned model ids/revisions + thresholds
scripts/run_<stage>.py         CLI only: parse args, build manifest, call the package
src/aic2026/<subsystem>/       all logic
src/aic2026/contracts/         strict Pydantic schemas shared across the team
```

`scripts/*.py` each do `sys.path.insert(0, REPO_ROOT / "src")` at import time — there is
no editable install on cloud runtimes.

Stage N reads stage N-1's published artifact, and every stage writes
`<output>.jsonl` plus a sidecar `<output>.manifest.json`:

```
$AIC_ARTIFACT_ROOT/frame_manifests/<VIDEO_ID>.jsonl
$AIC_ARTIFACT_ROOT/object_description/masks/<VIDEO_ID>.jsonl
$AIC_ARTIFACT_ROOT/object_description/descriptions/<VIDEO_ID>.jsonl
```

The shard unit is one `video_id`, which is what makes Kaggle's 12h session limit
survivable.

### `FrameRef` is the join key for the whole system

`src/aic2026/common/frame_manifest.py` turns the organizer `map-keyframes` CSV
(`n,pts_time,fps,frame_idx`) plus a keyframe image dir into `FrameRef` rows.
`frame_uid` is exactly `<video_id>:<frame_idx>`, and `frame_relpath` is relative to
`AIC_DATA_ROOT`. Every downstream modality (objects, embeddings, OCR, ASR) joins on
`frame_uid` / `video_id` + timestamp. **Never re-derive `frame_idx` or `pts_time_s`
from a filename by assuming an FPS**, and never scan an image directory instead of
reading the frame manifest.

### Contracts are strict on purpose

`src/aic2026/contracts/models.py` builds everything on `StrictModel`
(`extra="forbid"`, `strict=True`, `allow_inf_nan=False`). Schemas are versioned
strings (`aic26.object_regions.v1`, `aic26.run_manifest.v1`, `aic26.query.v1`) —
a breaking change means a new version, never a redefined field.

Two chokepoints when adding a stage:

- `RunManifest.stage` is a closed `Literal[...]`. A new stage name must be added
  there or the manifest will not validate.
- `contracts/__init__.py` re-exports everything; append there too.

`docs/data-contracts.md` is the human-readable mirror and must be updated with any
schema change.

### Resume / atomic publish machinery

`src/aic2026/common/manifest.py` and `object_description/pipeline.py::_ResumableJsonl`
implement rules that are easy to break by accident:

- Records stream to `<output>.jsonl.partial`, fsynced per line, then atomically
  renamed to the final path only at the end.
- `prepare_resume` refuses to start when a manifest exists without `--resume`, and
  refuses an artifact that exists with **no** manifest. It also handles the crash
  window where the rename succeeded but the completed sidecar was never written —
  it revalidates the final artifact instead of rerunning the GPU model.
- Resume is only allowed when run_id, stage, config SHA-256, input checksums and
  model revisions all match. A resumed partial must be an exact **ordered prefix** of
  the expected records.
- A description shard is published only when every caption is `ok`; any error/OOM
  leaves `.partial` for a retry and marks the sidecar failed.
- `--limit` smoke runs must go to a **separate artifact tree** (e.g. `.../smoke`).
  The config hash intentionally will not match a full run, so a full run can never
  resume from a limited one.

### Model registry and license gating

`configs/models.yaml` is the machine-readable source of truth; `docs/model-registry.md`
mirrors it. Policy is `immutable_revision_required: true` — pin a commit SHA, never a
branch or tag, and never fall back to `main`. `trust_remote_code` defaults to false.
Backends load lazily (`SamMaskGenerator.from_pretrained`, `DamCaptioner.from_pretrained`)
and respect `HF_HUB_OFFLINE=1` for Kaggle Internet-off runs.

DAM-3B **weights** are NVIDIA Noncommercial (the source code is Apache-2.0);
competition eligibility is unconfirmed. The repo has no `LICENSE` — do not assume
redistribution rights for code, organizer data or third-party weights.

### Online side (design only, not implemented)

A fixed few-shot prompt (`docs/query-decomposition.md`) turns a Vietnamese query into
`aic26.query.v1`. Invariants: `scene_en` is exactly **one** string for SigLIP; each
`objects_en[i]` is a **separate** query matched against DAM regions via
maximum-weight one-to-one assignment; `ocr_vi` / `audio_vi` keep literal Vietnamese
and are never translated or inferred. Fusion is
`S = w_scene*S_scene + w_object*S_object + w_ocr*S_ocr + w_audio*S_audio`, with unused
modalities at weight 0 and the rest renormalized.

### Known divergence between branches

`origin/feature/phowhisper-asr` (ASR owner, ~1.8k lines) solved the same problems a
different way: its schema lives in a separate `contracts/asr.py` with a **copied**
`StrictModel`, and it reimplements its own `AsrVideoManifest` instead of using
`aic2026.common`'s `RunManifest` + resume helpers. `object_description` is the
reference implementation. Prefer reusing `aic2026.common`, but put new schemas in
their own `contracts/<subsystem>.py` file to avoid merge conflicts in `models.py`.

## Conventions that will trip you up

- Logic goes in `src/aic2026`; `scripts/` only parses CLI and calls the package;
  notebooks only install, set env vars, call scripts, and print a few lines.
- No network or downloads at module import time.
- Every runner needs `--help`, must validate inputs **before** loading weights, exit
  non-zero on missing/duplicate/invalid artifacts, and be safe to rerun.
- Branch names: `feat|fix|exp/<area>-<slug>`, `docs/<slug>`, `chore/<slug>`. Commit
  prefixes: `feat: fix: perf: refactor: test: docs: chore:`. One reviewable concern
  per PR, squash merge, at least one non-author approval; contract/config/model/
  dependency changes need the owning subsystem's review.
- Never commit data, media, weights, embeddings, submissions, notebook outputs, local
  paths or secrets. `data/`, `artifacts/`, `runs/` are gitignored except their READMEs;
  only tiny licensed fixtures under `tests/fixtures/` are allowed.
- pre-commit runs nbstripout, detect-secrets (with an allowlist for
  `configs/models.yaml`, `configs/offline/object_description.yaml`,
  `docs/model-registry.md`) and blocks files over 1024 KB.
- Benchmark numbers are meaningless here without resolved config, dataset/split,
  model id + revision, seed, Git SHA, GPU, runtime and peak VRAM; small experiment
  writeups go in `reports/experiments/<id>.md`.

## Current session task: SigLIP2 scene embeddings

Branch `feat/siglip2` (currently 0 commits ahead of `main`). Goal: embed all keyframe
images with SigLIP2 so `S_scene` in the fusion formula becomes computable — the
`F --> S["SigLIP scene embeddings"]` edge of the README architecture diagram.

Scope is the keyframes listed in `frame_manifests/<VIDEO_ID>.jsonl`. Dense full-FPS
embedding is a later stage. Source data lives on Kaggle, but **this stage runs locally
on Apple Silicon (M4 Pro, 24 GB unified memory, MPS)** — not on a Kaggle T4.

### Verified facts

- `google/siglip2-base-patch16-224` revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`
  exists (latest commit on `main`, 2025-02-21) — safe to lock.
- **Blocker:** `requirements/runtime-base.txt` pins `transformers==4.48.3`, but the
  `Siglip2` architecture only landed in the `v4.49.0-SigLIP-2` tag. The pin must be
  bumped to a stable release that contains it. That file is shared with the SAM/DAM
  stage, so it needs its own PR and the object-description owner's smoke rerun.
- Device is **MPS**, so pass `--device mps` explicitly. No code change is needed for
  this: `_common.py::resolve_device` only special-cases `"auto"` (which has no MPS
  branch and would silently fall back to `cpu`) and returns any other value verbatim.
  The backend must therefore never hardcode `.to("cuda")`.
- `verify_environment.py` skips its CUDA/VRAM checks entirely for a non-`cuda` device,
  so the preflight only validates paths. It still must be run.
- MPS has no `torch.cuda.OutOfMemoryError`, so `pipeline.py::_is_cuda_oom` never fires
  here. Do not build MPS retry logic on that helper.
- Prefer fp32 on MPS unless fp16 is measured to be both faster and numerically equal;
  SigLIP2-base in fp32 is small enough for 24 GB unified memory.

### Local environment (set up; use `.venv`, never the system Python)

The system Python is 3.14.6, which `pyproject.toml` (`>=3.10,<3.13`) rejects. A
`uv`-managed **Python 3.12.14 venv lives at `.venv/`** — run everything through
`.venv/bin/python` and `.venv/bin/ruff`. Installed: torch 2.13.0 (MPS available),
transformers 4.57.6, plus `-e ".[dev]"`.

Torch is in the venv only and must never enter `requirements/*.txt` — cloud images
supply it and installs there use `--no-deps`.

Not yet done: `pip install kaggle` and `~/.kaggle/kaggle.json` (mode 600) for pulling
the data down. The IDE is still pointed at the system interpreter, so its
"could not be resolved" import warnings are noise until it is repointed at `.venv`.

### Source data layout

The organizer ships `Videos/`, `Keyframes/<VIDEO_ID>/*.jpg`,
`Objects/<VIDEO_ID>/*.json` (Faster R-CNN on OpenImages V4), a single `.npy` of
`clip-ViT-B-32` features, and per-video YouTube `Metadata/<VIDEO_ID>.json` (some
videos have none). This stage needs only **keyframes + map-keyframes** — do not
download videos or CLIP features for it. `AIC_DATA_ROOT` must end up as:

```
$AIC_DATA_ROOT/keyframes/<VIDEO_ID>/<n>.jpg
$AIC_DATA_ROOT/map-keyframes/<VIDEO_ID>.csv     # columns exactly: n,pts_time,fps,frame_idx
```

The organizer's `clip-ViT-B-32` features are a supporting baseline only; SigLIP2 is
the scene-embedding signal the retrieval design actually uses.

**Keyframe numbering must be verified against the real data before the first run.**
`FrameRef.keyframe_n` is `PositiveInt` and `read_frame_map` rejects `n < 1`, i.e. the
repo assumes **1-based** numbering (the fixture pairs `n=1` with `000001.jpg`). The
organizer docs show `L01_V001/0000.jpg`, which is 0-based. `_index_frames` keys files
by `int(path.stem)`, so if files are 0-based while the CSV is 1-based, every `n` pairs
with its neighbour's image. A **full** run is saved by the surplus check in
`build_frame_refs` ("Frame directory contains n values absent from mapping: [0]"), but
that check is skipped when `--limit` is set — so a smoke run mispairs every frame
silently and would poison the embedding index. Check one video by hand first.

### Planned placement

```
src/aic2026/scene_embedding/siglip_backend.py   lazy HF loader, mirrors sam_backend.py
src/aic2026/scene_embedding/pipeline.py         resumable batched embedding
src/aic2026/scene_embedding/store.py            .npy + index JSONL atomic writer
src/aic2026/scene_embedding/validation.py       input checks before weights load
src/aic2026/contracts/scene_embedding.py        SceneEmbeddingRecord, aic26.scene_embeddings.v1
scripts/run_scene_embeddings.py                 CLI, mirrors prepare_object_masks.py
configs/offline/scene_embedding.yaml
```

A `notebooks/kaggle_scene_embedding.ipynb` launcher is deferred: the first run is
local, and the runner must stay platform-agnostic anyway. Add it only when someone
needs to rerun this stage on a cloud GPU.

Output layout — vectors go to `.npy`, **not** inline in JSONL (768 floats as JSON text
is ~10x larger and not mmap-able):

```
$AIC_ARTIFACT_ROOT/scene_embeddings/<VIDEO_ID>.f16.npy        [N, D] float16
$AIC_ARTIFACT_ROOT/scene_embeddings/<VIDEO_ID>.jsonl          index, one row per frame
$AIC_ARTIFACT_ROOT/scene_embeddings/<VIDEO_ID>.manifest.json  RunManifest sidecar
```

Contract: `.npy` row order matches the frame manifest line order exactly, and vectors
are L2-normalized (normalize in fp32, then cast to fp16) so online scoring is a plain
dot product. Declare both files in `complete_manifest(output_paths=[...])` so each gets
a checksum. Sizing: ~1.5 KB/frame, so ~150k keyframes ≈ 230 MB of output; the local
disk has ~387 GB free, so the keyframe download dominates, not the embeddings.

### Status

Implemented and smoke-tested on synthetic data — see
`reports/experiments/2026-08-15-siglip2-mps-smoke.md`. `transformers` was bumped
4.48.3 → 4.57.6 (with `tokenizers` and `huggingface-hub`) in the shared
`requirements/runtime-base.txt`; **the SAM/DAM owner still has to re-smoke that stage
on the new pin.** Measured ~105 img/s at `batch_size: 32` on the M4 Pro, so
`siglip2-base` over ~150k keyframes is roughly 24 minutes.

### Remaining

1. Install the kaggle CLI + credentials and pull `keyframes/` + `map-keyframes/`.
2. **Verify keyframe numbering on the real data before the first run** (see above).
3. Run per video with `--device mps --resume`; benchmark `so400m-patch14-384` against
   the base model once there is a real-data baseline.
4. Share `scene_embeddings/` with the team through the private artifact storage.

Benchmark `siglip2-so400m-patch14-384` against the base model only after step 4 has a
measured baseline — it is several times slower, so the choice should be driven by
numbers on this machine, not assumed up front.
