# Data contracts v1

Nguồn chuẩn là Pydantic models trong `src/aic2026/contracts`. Tất cả JSONL dùng
UTF-8, mỗi dòng một JSON object, field thừa bị từ chối. Schema breaking change cần
version mới thay vì sửa nghĩa field cũ.

## `FrameRef`

| Field | Nghĩa |
|---|---|
| `video_id` | ID ổn định, chỉ chữ/số/`_`/`.`/`-` |
| `frame_uid` | Chính xác `<video_id>:<frame_idx>` |
| `keyframe_n` | Số thứ tự keyframe dương trong organizer map |
| `frame_idx` | Frame index 0-based của video gốc |
| `pts_time_s` | Timestamp giây từ organizer map |
| `fps` | FPS dương của video |
| `frame_relpath` | Path tương đối bên dưới `AIC_DATA_ROOT` |
| `width`, `height` | Kích thước image dương |

`frame_uid`, `frame_idx` và `pts_time_s` không được suy ra từ filename bằng giả
định FPS. Builder phải fail khi map thiếu/trùng/không tăng hoặc ảnh không tồn tại.

## Organizer object input

Mỗi keyframe có một JSON gồm năm arrays cùng độ dài:
`detection_scores`, `detection_class_names`, `detection_class_entities`,
`detection_boxes`, `detection_class_labels`. `detection_boxes` theo thứ tự
`[ymin, xmin, ymax, xmax]`, normalized trong `[0,1]`. Parser convert string sang số,
clip sai số biên tối đa `0.01`, và reject record không đồng bộ, NaN, bbox rỗng hoặc
tọa độ nằm ngoài tolerance.

## `ObjectFrameRecord` - `aic26.object_regions.v1`

Mỗi dòng kế thừa toàn bộ `FrameRef`, thêm `run_id` và `regions`.

Mỗi `ObjectRegion` có:

- `region_id` duy nhất trong frame và `source_detection_index` truy ngược input.
- Cả `bbox_yxyx_norm` và pixel `bbox_xyxy_px`; pixel convention luôn
  `xyxy_half_open`.
- `detector`: organizer class name/entity/label/score, source cố định
  `organizer_frcnn`.
- `segmentation`: `mask_source` (`sam` hoặc `bbox_fallback`), raw SAM predicted
  quality score (bắt buộc cho `sam`, `null` cho fallback; không giả định nằm trong
  `[0,1]`) và COCO-style RLE `{size: [height, width], counts: string}`.
- `caption`: `pending|ok|error|oom`. Khi `ok`, `description_en` không rỗng và tối
  đa 20 words; khi lỗi không được để caption cũ.

Mask size phải bằng frame size; bbox pixel phải khớp phép đổi deterministic từ bbox
normalized và không vượt frame. Duplicate `region_id` hoặc record frame trùng là
hard failure trước publish.

## `RunManifest` - `aic26.run_manifest.v1`

Manifest lưu `run_id`, stage, status, Git SHA/dirty flag, platform, resolved config
và SHA-256, seed, input source/checksum, model ID/revision/license, timestamps,
counters, completed shards và structured errors. Status terminal phải có
`ended_at`; `--resume` chỉ dùng completed shard khi config/input/model identity
khớp.

## `QuerySpec` - `aic26.query.v1`

`task_type` là `kis|qa|trake`. `scene_en` luôn có; `objects_en`, `ocr_vi`,
`audio_vi`, `audio_events_en` là lists. Q&A bắt buộc cả `question_vi` và
`question_en`; TRAKE bắt buộc ordered `events`; KIS không được có question/events.

Mỗi event có `label`, scene/object/modal fields và `temporal_operator` trong
`state|onset|offset|extremum`. `ocr_vi`/`audio_vi` chỉ chứa literal strings được
query nêu rõ và không dịch.

## Planned modality contracts

OCR, ASR, scene embeddings và online scores chưa được freeze trong v1. Owner phải
thêm versioned schema, timestamp/coordinate convention, validator và migration
note trước khi component khác tiêu thụ; không trao đổi ad-hoc dict giữa notebooks.
