## Mục tiêu

<!-- Vấn đề cần giải quyết, phạm vi và phần cố ý không thay đổi. -->

## Cách tái lập

```bash
# Exact command; dùng path qua env/CLI, không dùng path máy cá nhân.
python offline/scripts/<runner>.py --config offline/configs/<config>.yaml
```

- Dataset/split + checksum/version:
- Model ID + immutable revision:
- Seed, Git SHA:
- Platform, GPU, runtime, peak VRAM:
- Artifact ID/path + checksum (không signed URL):

## Kết quả và compatibility

- Metric trước/sau hoặc lý do không áp dụng:
- Schema/config/API thay đổi:
- Migration/backward compatibility:

## Checklist

- [ ] CI CPU và smoke test xanh.
- [ ] Notebook đã clear output; không có binary/data/model/submission.
- [ ] Không có secret, local absolute path hoặc network call khi import.
- [ ] Config, seed, model revision và run manifest đủ để tái lập.
- [ ] Data contract/runbook/model registry đã cập nhật nếu cần.
- [ ] Đã kiểm tra license, điều lệ cuộc thi và tác động bảo mật.
