# Quy tắc đóng góp

Repo này ưu tiên hai mục tiêu: mọi kết quả phải tái lập được, và thay đổi của một
thành viên không được phá vỡ data contract của các thành viên khác.

## Nhánh và commit

- Không push trực tiếp lên `main`. Tạo nhánh ngắn hạn từ `main` mới nhất.
- Đặt tên nhánh theo một trong các mẫu: `feat/<area>-<slug>`,
  `fix/<area>-<slug>`, `exp/<area>-<slug>`, `docs/<slug>`, `chore/<slug>`.
- Dùng commit prefix: `feat:`, `fix:`, `perf:`, `refactor:`, `test:`, `docs:` hoặc
  `chore:`. Commit phải mô tả hành vi thay đổi, không dùng tên chung như `update`.
- Mỗi pull request chỉ giải quyết một vấn đề có thể review và chạy độc lập. Khi
  merge, dùng squash merge và xóa nhánh.

## Pull request

Một PR chỉ sẵn sàng review khi:

1. Có issue hoặc mô tả rõ mục tiêu, phạm vi và phần không làm.
2. Có lệnh `%%bash`/`python scripts/<runner>.py ...` để reviewer tái lập.
3. Ghi rõ dataset/split, config, model ID + revision, seed, Git SHA, GPU, runtime
   và peak VRAM cho mọi kết quả benchmark.
4. CI CPU xanh; smoke test dùng fixture nhỏ, không tải model hay dữ liệu cuộc thi.
5. Không có token, đường dẫn máy cá nhân, video, model weight, embedding, output
   notebook hoặc submission.
6. Thay đổi schema/config dùng chung phải cập nhật tài liệu contract và nêu rõ
   backward compatibility hoặc migration.

Cần ít nhất một approval không phải tác giả. Thay đổi trong data contracts,
model registry, dependency hoặc scoring phải được người phụ trách subsystem đó
review. Khi team cung cấp GitHub handles, maintainer sẽ thêm `CODEOWNERS`; không
commit file `CODEOWNERS` với username giả.

## Quy tắc code và notebook

- Logic nằm trong `src/aic2026`; `scripts/` chỉ parse CLI và gọi package. Notebook
  chỉ cài môi trường, đặt biến, gọi script và xem vài dòng kết quả.
- Không hard-code `/content`, `/kaggle`, Google Drive hoặc đường dẫn Windows trong
  code. Dùng `AIC_DATA_ROOT`, `AIC_ARTIFACT_ROOT`, `AIC_CACHE_ROOT` hoặc CLI flag.
- Mọi runner phải có `--help`, validate input trước khi tải model, trả exit code
  khác 0 khi artifact thiếu/trùng/sai schema, và hỗ trợ chạy lại an toàn.
- Notebook phải được clear output và strip execution count trước commit.
- Code Python theo Ruff, type hint tại public interface, docstring cho logic khó,
  và không thực hiện network/download khi import module.

## Data, artifact và thí nghiệm

- `data/raw` là read-only. Không ghi đè dữ liệu do ban tổ chức cung cấp.
- Dữ liệu sinh ra đặt dưới `AIC_ARTIFACT_ROOT`; mỗi stage ghi manifest sidecar.
  `runs/<run_id>/` chỉ dành cho aggregate metadata/log. Các payload đều bị Git ignore.
- Chỉ commit fixture nhỏ, có quyền phân phối, trong `tests/fixtures/`.
- Kết quả thí nghiệm nhỏ được ghi trong `reports/experiments/<id>.md`; liên kết
  artifact riêng tư bằng ID/checksum, không bằng URL chứa credential.
- Không công bố metric nếu thiếu resolved config, input version, model revision,
  seed và Git SHA. Ghi rõ kernel CUDA nào không deterministic.

## Checklist trước khi push

```bash
python -m pip install -r requirements/dev.txt
pre-commit run --all-files
ruff check .
ruff format --check .
pytest -m "not gpu and not slow"
```

DVC chưa là dependency bắt buộc. Chỉ đưa DVC vào sau khi team có private remote
được duyệt và thực sự cần version derived datasets; không commit credential hoặc
`.dvc/config.local`.
