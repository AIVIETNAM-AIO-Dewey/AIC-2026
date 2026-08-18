# Cloud runbook: Colab và Kaggle

## Contract chung

Tất cả lệnh chạy từ repo root và nhận path qua ba biến:

```text
AIC_DATA_ROOT      dữ liệu gốc, read-only
AIC_ARTIFACT_ROOT  manifests, masks, captions và run metadata
AIC_CACHE_ROOT     Hugging Face/Torch model cache
```

Dùng `%env` trong notebook vì `export` trong một `%%bash` cell không phải giao kèo
bền vững cho cell sau. Không đặt token trong các biến này.

## Colab

1. Chọn GPU runtime và mount Drive trước khi đặt bất kỳ path `/content/drive` nào.
2. Clone private repo bằng token read-only trong Colab Secrets, `%cd` repo root,
   rồi đặt ba `%env` path theo tài khoản. Không đưa token vào URL hoặc output.
3. Chạy cell install rồi cell `verify_environment.py` trước khi download model.
4. Dùng `offline/notebooks/colab_object_description.ipynb` làm thin launcher.

Drive phù hợp để giữ artifact qua nhiều session nhưng I/O file nhỏ chậm. Nên ghi
shard tạm ở `/content` rồi copy artifact đã đóng/manifest atomically sang Drive.
Không cache model trong Git working tree.

## Kaggle

1. Bật GPU. Attach competition data dưới `/kaggle/input` (read-only).
2. Với Internet-off, attach ba private Dataset snapshots: competition data, source
   tree của đúng Git SHA, và runtime mirror gồm wheelhouse + model cache + DAM wheel.
   Không clone Git hoặc tải lại từ `main`.
3. Đặt artifact/cache dưới `/kaggle/working`, rồi tạo Dataset version riêng tư khi
   cần lưu qua session.
4. Dùng `offline/notebooks/kaggle_object_description.ipynb` làm thin launcher.

`/kaggle/input` không ghi được. `AIC_ARTIFACT_ROOT` và `AIC_CACHE_ROOT` không được
trỏ vào đó trừ cache snapshot đã tồn tại và chỉ đọc.

Internet-off setup dùng private Dataset có layout cache đã chuẩn hóa và manifest
checksum của team:

```python
%env AIC_CACHE_ROOT=/kaggle/input/aic2026-runtime/cache
%env AIC_WHEELHOUSE=/kaggle/input/aic2026-runtime/wheelhouse
%env AIC_DAM_WHEEL=/kaggle/input/aic2026-runtime/wheelhouse/dam-1.0.0-py3-none-any.whl
%env AIC_DAM_CODE_REVISION=153ad3d33c29324e9197f565547c6bc8500da02d
%env HF_HUB_OFFLINE=1
```

```bash
%%bash
set -euo pipefail
cd /kaggle/input/aic2026-runtime
sha256sum -c SHA256SUMS
cd /kaggle/input/aic2026-source
sha256sum -c SHA256SUMS
cd /kaggle/working/AIC-2026
python -m pip install --no-index --no-deps \
  --find-links "$AIC_WHEELHOUSE" \
  -r offline/requirements/runtime-base.txt
python -m pip install --no-index --no-deps "$AIC_DAM_WHEEL"
```

`wheelhouse/` phải chứa wheel tương thích Python của **mọi** pin trong
`offline/requirements/runtime-base.txt`; không dùng sdist. Source snapshot được copy từ
`/kaggle/input/aic2026-source/AIC-2026` sang `/kaggle/working/AIC-2026` trước cell
trên và `SHA256SUMS` của source Dataset phải được tạo từ đúng Git SHA đã duyệt. Vì
runner tự thêm `offline/src/` vào import path, không cần editable install hoặc PEP 517 build
trong phiên Internet-off.

Không cài `offline/requirements/object-description.txt` trực tiếp trong chế độ này vì file đó cài DAM từ
Git commit qua Internet. Snapshot/wheel sai hash hoặc sai revision phải fail trước
khi model load.

## Preflight và resume

`offline/scripts/verify_environment.py` kiểm tra CUDA, VRAM và path trước model load. DAM
3B có thể không chạy ổn trên GPU VRAM thấp; preflight mặc định yêu cầu 14 GiB và
config có OOM retry ngắn hơn. Mỗi shard phải publish atomically và chỉ được resume
khi config hash/model revision/input checksum khớp.

Smoke artifacts có `--limit` phải nằm trong output tree riêng. Full run không được
resume từ chúng; dùng output tree mới để config hash và ETA không bị lẫn.

## PP-OCRv6-small offline

PP-OCRv6 không dùng Hugging Face cache và không tự tải weights. Trước khi chạy, copy
đúng sáu file detector/recognizer vào layout sau dưới `AIC_CACHE_ROOT`:

```text
ocr/ppocrv6-small/
  detector/{inference.json,inference.pdiparams,inference.yml}
  recognizer/{inference.json,inference.pdiparams,inference.yml}
```

Checksum và kích thước chính xác được khóa trong
`offline/configs/offline/ocr_ppocrv6.yaml`. Luôn chạy preflight trước:

```bash
python -m pip install -r offline/requirements/ppocrv6.txt
python offline/scripts/run_ppocrv6.py --preflight-only \
  --cache-root "$AIC_CACHE_ROOT"
```

Preflight phải PASS trước khi PaddleOCR được construct. Cấu hình đã duyệt hiện chỉ
cho phép `ppocrv6-small` trên CPU; medium/GPU cần identity và acceptance riêng, không
được tự fallback. Khi chạy full, manifest đầu vào vẫn là canonical keyframe manifest
như các offline stage khác. Một frame lỗi chỉ tạo terminal error record và không làm
mất identity/coverage của các frame còn lại.

Description shard chỉ được đổi từ `.partial` sang final khi mọi region có caption
`ok`. Nếu một region lỗi/OOM, sidecar ở trạng thái failed và `--resume` cắt partial
về prefix thành công trước đó rồi retry từ frame lỗi; không xóa final artifact để
che lỗi. OOM primary nhưng retry thành công được đếm trong `manifest.counters`.

Không dùng DVC trong bootstrap v1. Dữ liệu cuộc thi được mount thủ công; artifacts
được chia sẻ qua private Drive/Kaggle Dataset/object storage cùng manifest và
checksum. Chỉ thêm DVC sau khi team chốt remote và quyền lưu dữ liệu.

## Online stack

```bash
copy .env.example .env
docker compose up --build -d
docker compose --profile ingest run --rm ingest
```

Qdrant dùng image `qdrant/qdrant:v1.18.2`. Ingest đọc JSONL + NPY thật,
kiểm manifest/checksum/model revision, tạo collection versioned rồi mới atomically
đổi alias `*_current`. API chạy tại `http://localhost:8000`, web tại
`http://localhost:8080`. `OPENAI_API_KEY` chỉ tồn tại trong API environment.

```bash
python -m aic_backend.ingest.cli --artifact-root /artifacts --all --activate \
  --e5-model-path /artifacts/models/multilingual-e5-base
```
