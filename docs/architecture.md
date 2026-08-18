# Kiến trúc retrieval AIC 2026

## Mục tiêu thiết kế

Video gốc là nguồn dữ liệu chuẩn. Keyframes, object metadata và CLIP features do
ban tổ chức cung cấp chỉ là tín hiệu hỗ trợ; hệ thống phải quay lại full video khi
cần định vị frame chính xác. Thiết kế chia thành offline indexing và online query,
với stable IDs để mọi thành viên có thể phát triển độc lập.

## Offline pipeline

1. **Ingest video**: kiểm kê video, checksum, FPS, duration và tạo `FrameRef`.
   Không sửa file gốc.
2. **Sampling hai tầng**: keyframes/sparse frames phục vụ coarse retrieval trên
   toàn corpus; full-FPS frames chỉ giải mã/index dày cho candidate videos hoặc
   những đoạn có độ biến thiên cao.
3. **Object regions**: đọc bbox/class/score từ organizer Objects, lọc theo
   confidence/area, NMS trong nhãn và deduplicate xuyên nhãn; giữ tối đa số region
   trong config. SAM tinh chỉnh mỗi bbox thành mask (bbox fallback nếu SAM lỗi).
   DAM nhận image + retained mask và sinh caption tiếng Anh ngắn, cụ thể. Output
   là `ObjectRegion` JSONL, không phải ảnh crop rời. V1 không chạy lại detector;
   detector riêng chỉ được thêm sau benchmark chứng minh organizer boxes thiếu.
4. **Scene embedding**: SigLIP2 mã hóa toàn frame. Index thưa cho video selection;
   dense embeddings có thể materialize theo shard/candidate để kiểm soát chi phí.
5. **OCR**: lưu nguyên văn tiếng Việt, confidence và polygon/bbox theo frame.
   Không dịch trước khi index; literal/BM25/fuzzy matching cần giữ dấu và số.
6. **ASR**: PhoWhisper giải mã cửa sổ 30 giây với stride 15 giây nhưng lưu transcript
   theo segment `[start_ms, end_ms]`. Không nhân bản transcript vào từng frame;
   online join segment với timestamp của frame.
7. **Build indexes**: vector index cho scene/object; lexical index cho OCR/ASR;
   metadata index liên kết mọi record bằng `video_id`, `frame_idx`, `timestamp_ms`.
8. **Run manifest**: mỗi stage ghi resolved config, model revision, input hash,
   Git SHA, environment, seed, counters và output checksum trước khi publish.

## Online pipeline

### Query decomposition

Một prompt few-shot cố định chuyển câu truy vấn tiếng Việt thành schema versioned:

- `scene_en`: đúng một mô tả cảnh tổng thể bằng tiếng Anh.
- `objects_en`: danh sách phrase độc lập, mỗi phrase là một object/person cùng
  thuộc tính trực quan; mỗi item được score riêng.
- `ocr_vi`: chỉ literal text mà query nói là chữ viết trên màn hình.
- `audio_vi`: chỉ literal phrase mà query nói là lời nói/âm thanh.
- `question_en`: câu hỏi cho answer stage hoặc `null`.
- `events`: sequence event có thứ tự cho TRAKE hoặc `null`.

Không suy diễn `ocr_vi`/`audio_vi`, không dịch hai field này, và validate JSON trước
khi retrieval. Prompt/model/API version phải được ghi vào query log.

### Retrieval và fusion

1. `scene_en` vào SigLIP2 text tower, cosine với frame embedding.
2. Mỗi `objects_en[i]` được embed riêng và so với tất cả DAM captions trong một
   frame; dùng maximum-weight one-to-one assignment giữa query slots và regions.
   Một region không được thỏa nhiều object slots trong cùng frame.
3. `ocr_vi` và `audio_vi` dùng literal/fuzzy/BM25 tiếng Việt, không round-trip qua
   tiếng Anh.
4. Modality không được query đề cập có weight bằng 0; các weight còn lại được
   renormalize. Calibrate score theo modality trên validation set trước fusion.
5. Coarse search chọn nhiều candidate videos; dense full-FPS refinement, temporal
   NMS và reranking tạo tối đa 100 đáp án.

Điểm khởi đầu để tune, không phải hằng số của contract:

```text
S = w_scene*S_scene + w_object*S_object + w_ocr*S_ocr + w_audio*S_audio
```

### Theo loại truy vấn

- **KIS**: trả danh sách `(video_id, frame_idx)` theo score; temporal NMS giữ đa
  dạng ở cutoff 1/5/20/50/100.
- **Q&A**: retrieval giống KIS. Answer stage nhận `question_en`, visual crop/frame,
  DAM captions, OCR và ASR timestamp-nearby; output phải đúng cả video, frame và
  answer theo điều lệ.
- **TRAKE**: top-level scene/object tìm nhiều candidate videos. Với từng video,
  mỗi event tạo score curve dense; k-best dynamic programming chọn
  `t1 < t2 < ... < tn` cùng gap constraints. Merge sequences giữa nhiều video,
  vì chọn sai video làm score bằng 0.

## Hệ quả từ scoring

Final score là trung bình `R@1`, `R@5`, `R@20`, `R@50`, `R@100`, và mỗi query chỉ
được tối đa 100 câu trả lời. Vì vậy hệ thống phải tối ưu cả precision đầu bảng lẫn
candidate diversity; không thể chỉ đẩy nhiều frame lân cận của một peak. KIS/Q&A
là binary theo tổ hợp video-frame (và answer cho Q&A). TRAKE chỉ nhận credit khi
đúng video, rồi tính tỷ lệ event frames khớp; các cửa sổ ground truth thường dưới
10 frames nên coarse keyframes một mình không đủ.

## Failure policy

- Thiếu/trùng `frame_id` hoặc `region_id`, bbox ngoài ảnh, schema/model revision
  sai: fail shard trước khi publish.
- CUDA OOM trong DAM: retry một lần với `oom_retry_max_new_tokens`, ghi degradation
  vào record/run manifest; hết retry thì record lỗi, không âm thầm bỏ qua.
- Download/cache thiếu trong Kaggle offline: fail trước inference và chỉ ra model
  snapshot/revision cần attach.
- Stage publish output atomically; `--resume` chỉ skip record đã complete và đúng
  config hash.

## Data contracts

Nguồn chuẩn là Pydantic models trong `offline/src/aic2026/contracts`; JSONL dùng UTF-8,
field thừa bị từ chối. `frame_uid` luôn là `<video_id>:<frame_idx>`. Keyframe
artifact có `keyframe_n`; dense TRAKE frame để field này `null` và giữ decoded
`frame_idx` 0-based cùng PTS thật.

- `aic26.object_regions.v1`: một record/frame, region có bbox, SAM RLE và DAM caption.
- `aic26.scene_embeddings.v1`: JSONL metadata có `row` + NPY companion L2-normalized.
- `aic26.ocr.v2`: terminal success/empty/error, accepted/rejected text, confidence nullable
  và polygon native-coordinate theo keyframe. `aic26.ocr.v1` chỉ còn read compatibility.
- `aic26.asr_segments.v1`: segment timestamped tham chiếu danh sách keyframe.
- `aic26.query.v1`: decomposition versioned cho KIS/Q&A/TRAKE.
- `aic26.run_manifest.v1`: config, input/output hash, model revision, seed và counters.

Ingest explode DAM theo region, OCR theo span và ASR theo segment-keyframe. Mọi Qdrant
payload bắt buộc có canonical `video_id`, `frame_idx`, `pts_time_s`, `run_id`.
