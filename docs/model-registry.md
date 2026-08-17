# Model registry và license gate

Nguồn máy đọc là `configs/models.yaml`. Mọi production run phải dùng immutable
revision trong file đó; `main` hoặc model alias trôi nổi không được chấp nhận.

| Role | Model/code | Revision | License | Trạng thái |
|---|---|---|---|---|
| Bbox-to-mask refinement | `facebook/sam-vit-base` | `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f` | Apache-2.0 | locked |
| Region caption weights | `nvidia/DAM-3B` | `0797bedd98d645cd021379a4661ee233da279bba` | NVIDIA Noncommercial | locked, cần rule check |
| DAM code | `NVlabs/describe-anything` | `153ad3d33c29324e9197f565547c6bc8500da02d` | Apache-2.0 | locked |
| Scene embedding | `google/siglip2-base-patch16-224` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | Apache-2.0 | locked |

Vietnamese OCR, PhoWhisper và query/answer model do các subsystem owners phụ trách.
Chúng không được production run cho đến khi có model ID/API version, immutable
revision, license URL, hardware profile và kết quả smoke test.

SigLIP2 yêu cầu `transformers>=4.49.0`; kiến trúc `Siglip2` không tồn tại trong các
bản cũ hơn. `requirements/runtime-base.txt` đã nâng pin từ `4.48.3` lên `4.57.6`
(cùng `tokenizers` và `huggingface-hub`), nên **owner của stage SAM/DAM phải chạy lại
smoke test** trước khi pin mới được coi là an toàn cho nhánh object.

## Quy trình đổi model

1. Đối chiếu điều lệ cuộc thi và quyền tải/redistribute/derived output.
2. Review model card, license và source nếu cần `trust_remote_code`.
3. Pin commit SHA, không pin branch/tag có thể di chuyển.
4. Chạy smoke + benchmark cùng split/seed/config với baseline.
5. Cập nhật registry, runbook và PR với disk/VRAM/runtime.
6. Với Kaggle Internet-off, tạo snapshot riêng tư được phép, verify checksum và
   giữ nguyên upstream attribution/license.

Online DAM install phải có PEP 610 VCS metadata khớp `code_revision`. Kaggle
offline chỉ được set `AIC_DAM_CODE_REVISION` sau khi `SHA256SUMS` của private wheel
mirror đã pass; runner fail closed nếu không chứng minh được code revision.

DAM weights là non-commercial, khác với Apache-2.0 của source code. Repo không tự
khẳng định rằng cuộc thi đáp ứng license; maintainer phải xác nhận trước khi chạy
trên toàn bộ corpus hoặc công khai artifact.
