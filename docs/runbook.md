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

Python socket guard của OCR chỉ là lớp bảo vệ best-effort cho socket được tạo trong
constructor/inference. Nó không chặn socket đã mở trước đó, native code hoặc subprocess;
vì vậy job Phase 1 bắt buộc chạy với Internet bị tắt ở execution environment. Thiếu
local model/cache hoặc package/runtime lệch pin phải fail trước khi construct model.

Real-model gate yêu cầu execution-attestation strict từ operator/platform với
`internet_enabled` là boolean `false`, identity/config/source commit khớp và payload checksum
hợp lệ. Hash attestation được bind vào report. Attestation này là bằng chứng vận hành, không
phải chứng minh mật mã rằng mọi network route đều bị chặn.

Tách session vận hành bắt buộc: detector pilot/run của Phase 1 chạy Kaggle Internet off.
Phase 2 gọi OpenAI API và các thao tác rclone chạy ở session/notebook Internet on riêng và chỉ
đọc artifact Phase 1 đã verify. Không bật OCR GPU profile; profile đó tiếp tục
`blocked_unverified_runtime` cho tới khi runtime GPU được xác minh riêng.

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

Phase 1 OCR xử lý track/select theo artifact shard, với giới hạn mặc định 25.000 frame,
250.000 detection mỗi shard, 2.000 detection mỗi frame, 20.000 trajectory active và
250.000 candidate edge mỗi component. Các limit này nằm trong locked tracking config và
receipt provenance. Dataset khoảng 177.000 frame phải được chia thành các
frame manifest bằng `offline/scripts/plan_ocr_phase1_shards.py`; planner bin-pack nguyên
video và không bao giờ cắt một `video_id` qua nhiều shard. Video đơn lẻ vượt giới hạn phải
fail cho tới khi có stateful cross-shard tracking. Không ghép nhiều shard vào một invocation vì
track/select giữ metadata của đúng một shard trong RAM. Mỗi stage dùng path độc lập:
`detect` cần detections, `track` cần detections + trajectories, và `select` cần
trajectories + representatives.

Receipt và checksum ở đây là metadata integrity trong trust boundary của filesystem/job;
chúng không phải chữ ký mật mã và không chống được người có quyền sửa đồng thời artifact
và receipt. Global shard verifier dùng chúng để phát hiện drift/crash, không tuyên bố trusted
signing.
Mỗi stage re-verify input/output sau khi publish completed receipt và trước khi return;
mutation đồng thời xảy ra sau khi hàm đã return nằm ngoài filesystem trust boundary này.

### OCR Phase 1: resume an toàn qua nhiều Kaggle session

Resume có hai tầng tách biệt. Trong cùng session, `--resume` dùng receipt và committed byte
offset dưới `/kaggle/working`; detection tiếp tục từ frame đã commit kế tiếp, còn track/select
được replay canonical. Browser disconnect không làm mất tiến trình kernel, và kernel crash chỉ
làm mất tail chưa có receipt. Không trỏ output đang chạy vào `/kaggle/input`.

Checkpoint qua session được export vào đúng cây sau và phải được giữ nguyên cả cây sequence:

```text
/kaggle/working/checkpoints/<run_id>/<shard_id>/
  checkpoint-000001-<content-hash>/
    manifests/...
    state/...
    checkpoint.json
```

`checkpoint.json` là commit marker được ghi cuối cùng. Thư mục tạm
`.checkpoint-publishing-*` hoặc bundle thiếu marker không được restore. Bundle khóa Git SHA,
config, model/runtime/resource identity, source/global/shard manifest, video ownership, mọi
artifact/receipt hash và committed record/byte offset. Publisher không overwrite sequence cũ;
exact replay của cùng trạng thái trả lại bundle hiện có. Chạy export sau ít nhất mỗi shard hoàn
thành; có thể chạy lệnh `checkpoint` sau một detection bị dừng để lưu prefix đã có receipt:

```bash
python offline/scripts/run_ocr_phase1.py checkpoint \
  --config offline/configs/offline/ocr_phase1.yaml \
  --output-root "/kaggle/working/ocr/$RUN_ID/$SHARD_ID" \
  --checkpoint-root "/kaggle/working/checkpoints/$RUN_ID/$SHARD_ID" \
  --source-manifest /kaggle/input/aic-manifests/source.frames.jsonl \
  --global-manifest /kaggle/input/aic-manifests/global-shards.json \
  --frame-manifest "/kaggle/input/aic-manifests/$SHARD_ID.frames.jsonl" \
  --data-root /kaggle/input/aic-data \
  --shard-id "$SHARD_ID" \
  --git-commit-sha "$AIC_GIT_COMMIT_SHA"
```

Workflow bắt buộc:

```text
Session 1:
run/resume shard N
→ verify shard
→ publish checkpoint bundle
→ Save Kaggle Output/Version

Session 2:
attach previous output read-only dưới /kaggle/input
→ verify rồi restore/copy checkpoint vào /kaggle/working
→ resume shard/stage tiếp theo
```

Session mới dùng `--resume-from`; CLI verify toàn bộ bundle và semantic crop/artifact trước khi
copy, giữ nguyên toàn bộ checkpoint hash-chain dưới `--checkpoint-root`, rồi copy artifact
payload trước receipt. Vì vậy checkpoint mới tiếp tục đúng sequence/previous hash của session
cũ. Nó không sửa checkpoint read-only:

```bash
python offline/scripts/run_ocr_phase1.py run \
  --resume-from "/kaggle/input/previous-output/checkpoints/$RUN_ID/$SHARD_ID" \
  --output-root "/kaggle/working/ocr/$RUN_ID/$SHARD_ID" \
  --checkpoint-root "/kaggle/working/checkpoints/$RUN_ID/$SHARD_ID" \
  --source-manifest /kaggle/input/aic-manifests/source.frames.jsonl \
  --global-manifest /kaggle/input/aic-manifests/global-shards.json \
  --frame-manifest "/kaggle/input/aic-manifests/$SHARD_ID.frames.jsonl" \
  --data-root /kaggle/input/aic-data \
  --cache-root /kaggle/input/aic2026-runtime/cache \
  --runtime-cache-root /kaggle/working/runtime-cache \
  --shard-id "$SHARD_ID" \
  --git-commit-sha "$AIC_GIT_COMMIT_SHA"
```

Không đưa OpenAI key, Google OAuth token hoặc rclone credential vào output/log/checkpoint.
Google Drive/rclone chỉ chạy cuối pipeline ở session Internet-on; Kaggle Output/Version là lớp
durability trung gian, không phải bản phát hành cuối. Resume sau session loss chỉ khả dụng nếu
checkpoint gần nhất đã được Save thành Kaggle Output/Version. Nếu session chết trước lần save
đó, mọi thay đổi sau durable checkpoint gần nhất có thể mất; cơ chế này không tuyên bố chống
mất dữ liệu trong khoảng đó.

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
