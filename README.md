# AIC 2026 MultiRetrieval

Hệ thống video retrieval cho HCMC AI Challenge 2026, gồm pipeline offline chạy
trên Colab/Kaggle và ứng dụng online chạy FastAPI, Qdrant, React.

## Bắt đầu từ đâu?

| Bạn phụ trách | Thư mục | Việc chính |
|---|---|---|
| Offline/ML | `offline/` | Chuẩn bị dữ liệu, DAM, OCR, ASR, SigLIP2, dense frames |
| Backend | `backend/` | Ingest Qdrant, retrieval KIS/Q&A/TRAKE, API |
| Frontend | `frontend/` | Search UI, keyframe grid, answer/timeline, submission basket |
| Hạ tầng | `docker/`, `.github/` | Docker images, Compose và CI |

Chi tiết kiến trúc nằm trong [`docs/architecture.md`](docs/architecture.md). Hướng
dẫn vận hành cloud/online nằm trong [`docs/runbook.md`](docs/runbook.md).

## Luồng hệ thống

```text
ZIP/video/keyframe
      │
      ▼
offline/scripts/prepare_data.py
      │
      ├── Object JSON → SAM → DAM captions
      ├── Keyframes → SigLIP2 embeddings
      ├── Keyframes → EasyOCR
      ├── Videos → PhoWhisper segments
      └── Videos → dense 5 FPS → SigLIP2
      │
      ▼
artifact JSONL + NPY + manifest/checksum
      │
      ▼
backend ingest → Qdrant aliases
      │
      ▼
FastAPI → React search UI → video_id/frame_idx
```

Không dùng CLIP vector do ban tổ chức cung cấp vì không có model manifest. ID nộp
bài luôn là `video_id` và canonical `frame_idx`; không fallback sang filename hoặc
`keyframe_n`.

## Cấu trúc repository

```text
backend/                   online API, retrieval, ingest, LLM
frontend/                  React/Vite client
offline/
  configs/                 immutable model revisions và stage defaults
  notebooks/               thin Colab/Kaggle launchers
  requirements/            dependency profiles theo model
  scripts/                 CLI entrypoints
  src/aic2026/             contracts và pipeline implementation
  tests/                   offline unit/integration tests
docker/                    API/web Dockerfiles và Nginx config
docs/                      architecture, runbook, model/query notes
compose.yaml               Qdrant + API + web + ingest profile
```

`data/`, `artifacts/`, model weights, embeddings và submissions đều bị Git ignore.

## 1. Cài môi trường phát triển

Python hỗ trợ `>=3.10,<3.13`.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r offline/requirements/dev.txt
```

Frontend:

```bash
cd frontend
npm ci
npm test
npm run build
```

## 2. Chuẩn bị dữ liệu

Đặt sáu ZIP gốc trong `data/raw/`, không sửa nội dung archive. Sau đó chạy từ
repository root:

```bash
python offline/scripts/prepare_data.py \
  --raw-root data/raw \
  --prepared-root data/prepared \
  --subset L21 \
  --resume
```

L21 đã được kiểm chứng với 29 video, 7.800 keyframe raw và 7.790 frame canonical
sau khi giữ keyframe thứ hai của 10 cặp `frame_idx` trùng. Kết quả prepare nằm tại
`data/prepared/L21/inventory.json`.

## 3. Chạy pipeline offline

Các stage dùng chung `AIC_DATA_ROOT`, `AIC_ARTIFACT_ROOT`, `AIC_CACHE_ROOT` và hỗ
trợ `--config`, `--device`, `--seed`, `--resume`, `--limit`, `--video-id`.

```bash
# Frame manifest canonical
python offline/scripts/build_frame_manifest.py --help

# Organizer objects → SAM masks → DAM descriptions
python offline/scripts/prepare_object_masks.py --help
python offline/scripts/run_dam_descriptions.py --help

# Các modality do từng member chạy độc lập
python offline/scripts/run_scene_embeddings.py --help
python offline/scripts/run_easyocr.py --help
python offline/scripts/run_phowhisper_asr.py --help
python offline/scripts/sample_dense_frames.py --help

# Validate trước ingest
python offline/scripts/validate_artifacts.py --help
```

Model IDs/revisions nằm trong
[`offline/configs/models.yaml`](offline/configs/models.yaml). Object-description
quick start trên cloud dùng:

- [`offline/notebooks/colab_object_description.ipynb`](offline/notebooks/colab_object_description.ipynb)
- [`offline/notebooks/kaggle_object_description.ipynb`](offline/notebooks/kaggle_object_description.ipynb)

Không chạy DAM, PhoWhisper hoặc dense decode trong API container.

## 4. Ingest và chạy ứng dụng online

Tạo `.env` từ `.env.example`, sau đó:

```bash
docker compose up -d qdrant api web
docker compose --profile ingest run --rm ingest
```

Mở UI tại <http://localhost:8080>. API docs nằm tại
<http://localhost:8000/docs> và readiness tại
<http://localhost:8000/api/v1/capabilities>.

Nếu chạy frontend native:

```bash
cd frontend
npm run dev
```

Vite proxy `/api` sang `http://localhost:8000`. Search form luôn hiển thị nhưng bị
khóa rõ ràng cho tới khi collection/model tương ứng sẵn sàng.

## 5. Kiểm thử

```bash
pytest offline/tests backend/tests -m "not gpu and not slow"
ruff check offline/src offline/tests offline/scripts backend/src backend/tests
ruff format --check offline/src offline/tests offline/scripts backend/src backend/tests

cd frontend
npm test
npm run build
```

CI không tải GPU model và không gọi OpenAI thật. GPU smoke test phải chạy riêng trên
Colab T4 và Kaggle T4 trước full corpus.

## Contracts quan trọng

- Frame: `video_id`, `frame_idx`, `keyframe_n`, `pts_time_s`, `frame_relpath`.
- Object: một record/frame, mỗi region giữ bbox, COCO RLE mask và DAM caption.
- Query: `aic26.query.v1`, scene/object/OCR/audio được tách thành signal độc lập.
- Ingest: chỉ nhận manifest `completed`, checksum đúng và model revision đã pin.
- API: mọi result bắt buộc có canonical `video_id` và `frame_idx`.

## Quy tắc repository

- Không commit data, video, model weight, artifact, secret hoặc submission.
- Không push trực tiếp `main`; dùng branch nhỏ và squash merge.
- Xem [`CONTRIBUTING.md`](CONTRIBUTING.md) và [`SECURITY.md`](SECURITY.md).
- Model/license notes nằm trong [`docs/model-registry.md`](docs/model-registry.md).
- Query parser prompt nằm trong
  [`docs/query-decomposition.md`](docs/query-decomposition.md).
