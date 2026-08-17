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
