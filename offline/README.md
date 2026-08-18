# Offline pipelines

Thư mục này chứa toàn bộ công việc chạy trước trận thi trên Colab/Kaggle:

- `configs/`: model revisions và cấu hình từng stage;
- `notebooks/`: thin launchers cho Colab/Kaggle;
- `requirements/`: dependency profile theo từng model;
- `scripts/`: CLI entrypoints, không chứa business logic;
- `src/aic2026/`: contracts và pipeline importable;
- `tests/`: unit/integration tests cho artifact offline.

Chạy lệnh từ repository root. Ví dụ:

```bash
python offline/scripts/prepare_data.py \
  --raw-root data/raw \
  --prepared-root data/prepared \
  --subset L21 \
  --resume
```

Xem `README.md` ở repository root cho luồng end-to-end và `docs/runbook.md` cho
hướng dẫn cloud chi tiết.

## PP-OCRv6-small

```bash
python -m pip install -r offline/requirements/ppocrv6.txt
python offline/scripts/run_ppocrv6.py --preflight-only \
  --cache-root "$AIC_CACHE_ROOT"
python offline/scripts/run_ppocrv6.py --help
```

Model files phải được provision sẵn theo layout/checksum trong
`configs/offline/ocr_ppocrv6.yaml`; runner không tự tải model. Output là
`aic26.ocr.v2` JSONL với đúng một trạng thái `success`, `empty` hoặc `error` cho mỗi
frame. EasyOCR runner cũ chỉ được giữ để tương thích với artifact trước đây.
