# Security policy

## Dữ liệu và secret

- Source repository có thể public, nhưng tuyệt đối không upload hoặc chia sẻ lại
  video, metadata hạn chế, ground truth hay tài nguyên do ban tổ chức cấp. Nếu điều
  lệ yêu cầu private repository, maintainer phải đổi visibility trước khi chạy thật.
- Lưu token trong Colab Secrets, Kaggle Secrets hoặc GitHub Actions Secrets. Không
  đặt token trong notebook, YAML, command history, log hay ảnh chụp màn hình.
- Không commit `.env`, service-account JSON, signed URL, cookie hoặc model weight.
  Nếu secret từng xuất hiện trong Git, phải revoke/rotate ngay; xóa file khỏi HEAD
  không làm secret biến mất khỏi history.
- Output có thể chứa OCR/ASR hoặc hình người. Chỉ lưu trong backend riêng tư đã
  được team duyệt và áp dụng chính sách giữ/xóa của cuộc thi.

## Model và dependency

- Chỉ tải model đúng ID và immutable revision trong `offline/configs/models.yaml`.
- Ưu tiên `safetensors`; không load pickle/checkpoint không rõ nguồn gốc.
- `trust_remote_code` mặc định là `false`. Ngoại lệ phải pin revision, review source
  và được maintainer phê duyệt trước khi chạy với secret hoặc dữ liệu cuộc thi.
- Download ngoài registry phải dùng HTTPS và verify SHA-256. Không thực thi script
  vừa tải xuống bằng `curl | bash` hoặc tương đương.
- Kiểm tra lại license/terms của model và dataset trước khi đổi revision hoặc công
  khai repo. DAM weights dùng NVIDIA Noncommercial License; việc dùng trong cuộc
  thi phải được đối chiếu với điều lệ trước production run.

## Báo cáo sự cố

Không mở public issue có chứa secret, dữ liệu hoặc đường dẫn private. Báo trực tiếp
cho repository maintainers qua kênh nội bộ của team, kèm loại sự cố, commit/run ID
và phạm vi ảnh hưởng nhưng không đính kèm dữ liệu nhạy cảm.
