#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${AIC_LOCAL_RUNTIME_ROOT:-${REPO_ROOT}/.local/macos}"
VENV_ROOT="${AIC_LOCAL_VENV_ROOT:-${REPO_ROOT}/.venv}"
MODEL_ROOT="${AIC_MODEL_ROOT:-${REPO_ROOT}/models}"
BRANCH1_MODEL_ROOT="${AIC_BRANCH1_MODEL_ROOT:-${MODEL_ROOT}/branch1}"
HF_CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"
DATA_ROOT="${AIC_DATA_ROOT:-/Users/macbookpro/Downloads/AIC-HCM-BATCH-1/AIC_HCM_BATCH_1/artifacts}"
METACLIP2_SOURCE="${AIC_METACLIP2_SOURCE:-/Users/macbookpro/Downloads/metaclip2}"
BEIT3_SOURCE="${AIC_BEIT3_SOURCE:-/Users/macbookpro/Downloads/beit3}"
STATE_ROOT="${AIC_STATE_ROOT:-${RUNTIME_ROOT}/state}"
QDRANT_ROOT="${RUNTIME_ROOT}/qdrant"
QDRANT_BIN="${QDRANT_ROOT}/bin/qdrant"
QDRANT_STORAGE="${QDRANT_ROOT}/storage"
QDRANT_SNAPSHOTS="${QDRANT_ROOT}/snapshots"
LOG_ROOT="${RUNTIME_ROOT}/logs"
PID_ROOT="${RUNTIME_ROOT}/pids"
QDRANT_PID_FILE="${PID_ROOT}/qdrant.pid"
API_PID_FILE="${PID_ROOT}/api.pid"
QDRANT_SCREEN_SESSION="aic2026-native-qdrant"
API_SCREEN_SESSION="aic2026-native-api"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
API_URL="${AIC_API_URL:-http://127.0.0.1:8890}"
PYTHON_BIN="${VENV_ROOT}/bin/python"
PIP_BIN="${VENV_ROOT}/bin/pip"

QDRANT_VERSION="1.19.0"
QDRANT_ARCHIVE="qdrant-aarch64-apple-darwin.tar.gz"
QDRANT_DOWNLOAD_URL="https://github.com/qdrant/qdrant/releases/download/v${QDRANT_VERSION}/${QDRANT_ARCHIVE}"
QDRANT_SHA256="4e279a80cc1ebe73e859318ff86375af54c123887dd7ae46605c0eb6cb7c44e8"
METACLIP2_VECTOR_SHA256="9c68186574ba61e10ebb17886e4cbd3ae4b88fea9b2ab3f9139420ff5aaa78f9"
BEIT3_VECTOR_SHA256="84b7f250d9ef05338cd708e5db4105b80f150b1fefb1fbf1083a1a3de2636c15"

export AIC_DATA_ROOT="${DATA_ROOT}"
export AIC_STATE_ROOT="${STATE_ROOT}"
export AIC_MODEL_ROOT="${MODEL_ROOT}"
export AIC_BRANCH1_MODEL_ROOT="${BRANCH1_MODEL_ROOT}"
export AIC_QUERY_MODEL_MANIFEST="${MODEL_ROOT}/query_models.json"
export HF_HOME="${HF_CACHE_ROOT}"
export QDRANT_URL="${QDRANT_URL}"
export AIC_DEVICE="${AIC_DEVICE:-auto}"
export AIC_ALLOW_CPU_FALLBACK="${AIC_ALLOW_CPU_FALLBACK:-1}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM="false"
export HF_HUB_DISABLE_TELEMETRY="1"
export AIC_CPU_THREADS="${AIC_CPU_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-8}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

log() {
  printf '[AIC local] %s\n' "$*"
}

fail() {
  printf '[AIC local] ERROR: %s\n' "$*" >&2
  exit 1
}

qdrant_reachable() {
  curl --silent --fail --connect-timeout 2 --max-time 5 "${QDRANT_URL}/" \
    >/dev/null 2>&1
}

api_health() {
  curl --silent --show-error --fail --connect-timeout 2 --max-time 15 \
    "${API_URL}/api/health"
}

api_reachable() {
  api_health >/dev/null 2>&1
}

require_data_access() {
  local probe
  local probes=(
    "${DATA_ROOT}/visual_embeddings/metaclip2/keyframes_metadata.jsonl"
    "${DATA_ROOT}/dense_text_embeddings/dam_vectors.f16.npy"
  )
  for probe in "${probes[@]}"; do
    [[ -f "${probe}" ]] || fail "Required data file does not exist: ${probe}"
    head -c 1 "${probe}" >/dev/null 2>&1 || fail \
      "macOS blocked access to ${DATA_ROOT}. Allow your terminal app to access Downloads in System Settings > Privacy & Security > Files and Folders, then retry."
  done
}

ensure_directories() {
  mkdir -p \
    "${RUNTIME_ROOT}/downloads" \
    "${STATE_ROOT}" \
    "${QDRANT_ROOT}/bin" \
    "${QDRANT_STORAGE}" \
    "${QDRANT_SNAPSHOTS}" \
    "${LOG_ROOT}" \
    "${PID_ROOT}" \
    "${MODEL_ROOT}" \
    "${BRANCH1_MODEL_ROOT}"
}

require_native_mac() {
  [[ "$(uname -s)" == "Darwin" ]] || fail "This launcher supports macOS only."
  [[ "$(uname -m)" == "arm64" ]] || fail "Apple Silicon (arm64) is required."
  command -v /opt/homebrew/bin/python3.12 >/dev/null 2>&1 \
    || fail "Python 3.12 is required at /opt/homebrew/bin/python3.12."
  command -v curl >/dev/null 2>&1 || fail "curl is required."
  command -v npm >/dev/null 2>&1 || fail "Node.js/npm is required."
  command -v screen >/dev/null 2>&1 || fail "The macOS screen utility is required."
}

sha256_of() {
  shasum -a 256 "$1" | awk '{print $1}'
}

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256_of "${path}")"
  [[ "${actual}" == "${expected}" ]] \
    || fail "SHA-256 mismatch for ${path}: ${actual}"
}

sync_embedding_data() {
  local visual_root="${DATA_ROOT}/visual_embeddings"
  [[ -d "${DATA_ROOT}" ]] || fail "Data root does not exist: ${DATA_ROOT}"
  [[ -d "${METACLIP2_SOURCE}" ]] || fail "MetaCLIP2 source does not exist: ${METACLIP2_SOURCE}"
  [[ -d "${BEIT3_SOURCE}" ]] || fail "BEIT3 source does not exist: ${BEIT3_SOURCE}"
  mkdir -p "${visual_root}"
  log "Synchronizing MetaCLIP2 artifacts into the main data root"
  ditto "${METACLIP2_SOURCE}" "${visual_root}/metaclip2"
  log "Synchronizing BEIT3 artifacts into the main data root"
  ditto "${BEIT3_SOURCE}" "${visual_root}/beit3"
  verify_sha256 \
    "${visual_root}/metaclip2/keyframes_visual_vectors.f16.npy" \
    "${METACLIP2_VECTOR_SHA256}"
  verify_sha256 \
    "${visual_root}/beit3/keyframes_visual_vectors.f16.npy" \
    "${BEIT3_VECTOR_SHA256}"
  cmp -s \
    "${visual_root}/metaclip2/keyframes_metadata.jsonl" \
    "${visual_root}/beit3/keyframes_metadata.jsonl" \
    || fail "MetaCLIP2 and BEIT3 metadata ordering differs."
}

setup_python() {
  local recreate=0
  if [[ -x "${PYTHON_BIN}" ]]; then
    "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' \
      || recreate=1
  else
    recreate=1
  fi
  if [[ "${recreate}" == "1" ]]; then
    if [[ -e "${VENV_ROOT}" ]]; then
      mv "${VENV_ROOT}" "${VENV_ROOT}.incompatible.$(date +%Y%m%d%H%M%S)"
    fi
    log "Creating the Python 3.12 environment"
    /opt/homebrew/bin/python3.12 -m venv "${VENV_ROOT}"
  fi
  log "Installing pinned native macOS dependencies"
  "${PYTHON_BIN}" -m pip install --upgrade 'pip==25.2' 'setuptools>=75,<76' wheel
  "${PIP_BIN}" install -e "${REPO_ROOT}[dev]" -r "${SCRIPT_DIR}/requirements.txt"
  "${PYTHON_BIN}" -c \
    'import torch; assert torch.backends.mps.is_built(); assert torch.backends.mps.is_available(); print(f"PyTorch {torch.__version__}: MPS available")'
}

install_qdrant() {
  if [[ -x "${QDRANT_BIN}" ]] && "${QDRANT_BIN}" --version 2>&1 | grep -q "${QDRANT_VERSION}"; then
    return
  fi
  local archive_path="${RUNTIME_ROOT}/downloads/${QDRANT_ARCHIVE}"
  log "Downloading Qdrant ${QDRANT_VERSION} for Apple Silicon"
  curl --fail --location --retry 3 --continue-at - \
    --output "${archive_path}" "${QDRANT_DOWNLOAD_URL}"
  verify_sha256 "${archive_path}" "${QDRANT_SHA256}"
  tar -xzf "${archive_path}" -C "${QDRANT_ROOT}/bin"
  chmod 755 "${QDRANT_BIN}"
  "${QDRANT_BIN}" --version
}

screen_session_running() {
  local session="$1"
  local listing
  listing="$(screen -ls 2>/dev/null || true)"
  grep -Eq "[[:space:]][0-9]+\\.${session}[[:space:]]" <<<"${listing}"
}

record_screen_pid() {
  local session="$1"
  local pid_file="$2"
  local listing pid
  listing="$(screen -ls 2>/dev/null || true)"
  pid="$(
    awk -v suffix=".${session}" \
      '$1 ~ suffix "$" { split($1, parts, "."); print parts[1]; exit }' \
      <<<"${listing}"
  )"
  if [[ "${pid}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${pid}" >"${pid_file}"
  fi
}

stop_screen_job() {
  local session="$1"
  if ! screen_session_running "${session}"; then
    return
  fi
  # Let Qdrant/Uvicorn flush and close their stores before removing the
  # detached terminal.  Both services handle SIGINT cleanly.
  screen -S "${session}" -p 0 -X stuff $'\003' >/dev/null 2>&1 || true
  for _ in $(seq 1 40); do
    if ! screen_session_running "${session}"; then
      return 0
    fi
    sleep 0.25
  done
  screen -S "${session}" -X quit >/dev/null 2>&1 || true
}

run_qdrant_foreground() {
  ensure_directories
  cd "${QDRANT_ROOT}"
  exec env \
    QDRANT__TELEMETRY_DISABLED=true \
    QDRANT__SERVICE__HOST=127.0.0.1 \
    QDRANT__SERVICE__HTTP_PORT=6333 \
    QDRANT__SERVICE__GRPC_PORT=6334 \
    QDRANT__STORAGE__STORAGE_PATH="${QDRANT_STORAGE}" \
    QDRANT__STORAGE__SNAPSHOTS_PATH="${QDRANT_SNAPSHOTS}" \
    "${QDRANT_BIN}" >>"${LOG_ROOT}/qdrant.log" 2>>"${LOG_ROOT}/qdrant.error.log"
}

run_api_foreground() {
  ensure_directories
  cd "${REPO_ROOT}"
  exec env \
    AIC_DATA_ROOT="${DATA_ROOT}" \
    AIC_STATE_ROOT="${STATE_ROOT}" \
    AIC_MODEL_ROOT="${MODEL_ROOT}" \
    AIC_BRANCH1_MODEL_ROOT="${BRANCH1_MODEL_ROOT}" \
    AIC_QUERY_MODEL_MANIFEST="${MODEL_ROOT}/query_models.json" \
    AIC_DEVICE="${AIC_DEVICE}" \
    AIC_ALLOW_CPU_FALLBACK="${AIC_ALLOW_CPU_FALLBACK}" \
    PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK}" \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME="${HF_HOME}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    QDRANT_URL="${QDRANT_URL}" \
    AIC_CPU_THREADS="${AIC_CPU_THREADS}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS}" \
    PYTHONPATH="${PYTHONPATH}" \
    "${PYTHON_BIN}" -m uvicorn online.cpu_server:app \
      --host 127.0.0.1 --port 8890 --workers 1 \
      >>"${LOG_ROOT}/api.log" 2>>"${LOG_ROOT}/api.error.log"
}

start_qdrant() {
  ensure_directories
  if screen_session_running "${QDRANT_SCREEN_SESSION}" \
    && qdrant_reachable; then
    record_screen_pid "${QDRANT_SCREEN_SESSION}" "${QDRANT_PID_FILE}"
    return
  fi
  [[ -x "${QDRANT_BIN}" ]] || fail "Qdrant is not installed; run setup first."
  stop_screen_job "${QDRANT_SCREEN_SESSION}"
  if qdrant_reachable; then
    fail "Port 6333 is already served by a Qdrant process not managed by this launcher."
  fi
  log "Starting native Qdrant"
  screen -dmS "${QDRANT_SCREEN_SESSION}" \
    "${SCRIPT_DIR}/aic_local.sh" _run-qdrant
  local aic_qdrant_deadline=$((SECONDS + 90))
  while ((SECONDS < aic_qdrant_deadline)); do
    if qdrant_reachable; then
      record_screen_pid "${QDRANT_SCREEN_SESSION}" "${QDRANT_PID_FILE}"
      return
    fi
    sleep 1
  done
  tail -n 80 "${LOG_ROOT}/qdrant.error.log" >&2 || true
  fail "Qdrant did not become ready."
}

start_api() {
  ensure_directories
  start_qdrant
  if screen_session_running "${API_SCREEN_SESSION}" \
    && api_reachable; then
    record_screen_pid "${API_SCREEN_SESSION}" "${API_PID_FILE}"
    return
  fi
  require_data_access
  [[ -x "${PYTHON_BIN}" ]] || fail "Python environment is missing; run setup first."
  [[ -f "${REPO_ROOT}/online/frontend/dist/index.html" ]] \
    || fail "Frontend build is missing; run setup first."
  stop_screen_job "${API_SCREEN_SESSION}"
  if api_reachable; then
    fail "Port 8890 is already served by a process not managed by this launcher."
  fi
  log "Starting the native API and compiled frontend"
  screen -dmS "${API_SCREEN_SESSION}" \
    "${SCRIPT_DIR}/aic_local.sh" _run-api
  local aic_api_deadline=$((SECONDS + 180))
  while ((SECONDS < aic_api_deadline)); do
    if api_reachable; then
      record_screen_pid "${API_SCREEN_SESSION}" "${API_PID_FILE}"
      return
    fi
    sleep 1
  done
  tail -n 120 "${LOG_ROOT}/api.error.log" >&2 || true
  fail "The API did not become ready."
}

stop_all() {
  stop_screen_job "${API_SCREEN_SESSION}"
  stop_screen_job "${QDRANT_SCREEN_SESSION}"
  rm -f "${API_PID_FILE}" "${QDRANT_PID_FILE}"
  log "Native services stopped"
}

download_models() {
  log "Resolving and validating local query-model snapshots"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/qdrant/setup_query_models.py"
  log "Resolving and validating BEIT3 runtime assets"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/qdrant/setup_branch1_models.py"
}

prepare_indexes() {
  log "Validating Branch 1 data and static encoder contracts"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/qdrant/prepare_branch1.py" \
    --data-root "${DATA_ROOT}" \
    --state-root "${STATE_ROOT}" \
    --model-root "${BRANCH1_MODEL_ROOT}" \
    --query-manifest "${MODEL_ROOT}/query_models.json" \
    --skip-runtime-probe
  log "Preparing Branch 2 DAM and BM25 state"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/qdrant/prepare_branch2.py" \
    --data-root "${DATA_ROOT}" --state-root "${STATE_ROOT}"
  log "Preparing OCR SQLite FTS5 state"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/qdrant/prepare_text_indexes.py" \
    --data-root "${DATA_ROOT}" --state-root "${STATE_ROOT}"
  log "Preparing ASR SQLite FTS5 state"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/qdrant/prepare_asr_index.py" \
    --data-root "${DATA_ROOT}" --state-root "${STATE_ROOT}"
}

ingest_qdrant() {
  start_qdrant
  log "Ingesting or repairing all Qdrant collections"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/qdrant/ingest.py" \
    --url "${QDRANT_URL}" \
    --grpc-host 127.0.0.1 \
    --grpc-port 6334 \
    --data-root "${DATA_ROOT}" \
    --state-root "${STATE_ROOT}"
  log "Verifying exact Qdrant contents"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/qdrant/ingest.py" \
    --url "${QDRANT_URL}" \
    --grpc-host 127.0.0.1 \
    --grpc-port 6334 \
    --data-root "${DATA_ROOT}" \
    --state-root "${STATE_ROOT}" \
    --verify-only
}

build_frontend() {
  log "Installing frontend dependencies and running the production build gate"
  (cd "${REPO_ROOT}" && npm ci --no-audit --no-fund && npm run check)
}

verify_health() {
  local health_json
  health_json="$(api_health)"
  "${PYTHON_BIN}" -c '
import json, sys
health = json.load(sys.stdin)
required = ("branch1", "branch2", "branch3_asr", "branch3_ocr", "kis_fusion")
components = health.get("components") or {}
missing = [name for name in required if (components.get(name) or {}).get("ready") is not True]
devices = health.get("execution_devices") or {}
if health.get("ready") is not True or missing:
    status = health.get("status")
    raise SystemExit(f"Health is not operational; missing={missing}; status={status}")
if devices.get("preferred") != "mps":
    preferred = devices.get("preferred")
    raise SystemExit(f"Expected MPS preference, got {preferred!r}")
print(json.dumps({
    "status": health.get("status"),
    "ready": health.get("ready"),
    "production_ready": health.get("production_ready"),
    "device": health.get("device"),
    "branches": {name: components[name].get("ready") for name in required},
}, indent=2))
' <<<"${health_json}"
}

setup_all() {
  require_native_mac
  ensure_directories
  sync_embedding_data
  setup_python
  install_qdrant
  download_models
  prepare_indexes
  build_frontend
  ingest_qdrant
  start_api
  verify_health
  log "Setup complete: ${API_URL}"
}

resume_after_prepare() {
  require_native_mac
  ensure_directories
  setup_python
  install_qdrant
  build_frontend
  ingest_qdrant
  start_api
  verify_health
  log "Setup complete: ${API_URL}"
}

show_status() {
  printf 'Qdrant: '
  if qdrant_reachable; then
    printf 'running (%s)\n' "${QDRANT_URL}"
  else
    printf 'stopped\n'
  fi
  printf 'UI/API: '
  if api_reachable; then
    printf 'running (%s)\n' "${API_URL}"
    verify_health
  else
    printf 'stopped\n'
  fi
}

usage() {
  printf '%s\n' \
    'Usage: scripts/native_macos/aic_local.sh <setup|resume|start|stop|status>' \
    '' \
    '  setup   Install/prepare/build everything, then start the UI' \
    '  resume  Continue after models and local text indexes are prepared' \
    '  start   Start native Qdrant and the compiled UI/API' \
    '  stop    Stop only services managed by this launcher' \
    '  status  Show service and branch readiness'
}

case "${1:-}" in
  setup) setup_all ;;
  resume) resume_after_prepare ;;
  start) require_native_mac; start_api; verify_health; log "UI: ${API_URL}" ;;
  stop) stop_all ;;
  status) require_native_mac; show_status ;;
  _run-qdrant) run_qdrant_foreground ;;
  _run-api) run_api_foreground ;;
  *) usage; exit 2 ;;
esac
