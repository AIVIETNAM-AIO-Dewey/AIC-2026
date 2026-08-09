# Run metadata

Mỗi stage hiện ghi sidecar `<artifact>.manifest.json` dưới `AIC_ARTIFACT_ROOT`, chứa
Git SHA/dirty flag, resolved config, model revisions, input/output checksums, seed,
platform/Python/CUDA/GPU, thời gian và trạng thái. `runs/<run_id>/` được dành cho
aggregate manifest/log khi orchestration đa stage được thêm; payload lớn không vào repo.
