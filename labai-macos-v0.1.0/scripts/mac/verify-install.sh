#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/.labai/config.toml"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
VENV_LABAI="${REPO_ROOT}/.venv/bin/labai"
LAUNCHER_DIR="${HOME}/Library/Application Support/LabAI/bin"
LAUNCHER_PATH="${LAUNCHER_DIR}/labai"
LABAI_LAUNCHER_PATH_LINE='export PATH="$HOME/Library/Application Support/LabAI/bin:$PATH"'
SMOKE_TIMEOUT_SECONDS="${LABAI_MAC_SMOKE_TIMEOUT_SECONDS:-180}"
LAUNCHER_PATH_STATUS="not_checked"
SHELL_PATH_STATUS="not_checked"
ZPROFILE_PATH_STATUS="not_checked"
LABAI_CMD=""

usage() {
  cat <<'USAGE'
Usage: scripts/mac/verify-install.sh [--launcher-dir PATH]
USAGE
}

step() {
  printf '\n[labai-verify] %s\n' "$1"
}

check_launcher_path_status() {
  if [ -x "$LAUNCHER_PATH" ]; then
    LAUNCHER_PATH_STATUS="ok"
  else
    LAUNCHER_PATH_STATUS="missing"
  fi
}

check_shell_path_status() {
  if command -v labai >/dev/null 2>&1; then
    SHELL_PATH_STATUS="ok"
  else
    SHELL_PATH_STATUS="missing"
  fi
}

check_zprofile_path_status() {
  zprofile="${HOME:-}/.zprofile"
  if [ -f "$zprofile" ] && grep -Fqx "$LABAI_LAUNCHER_PATH_LINE" "$zprofile" >/dev/null 2>&1; then
    ZPROFILE_PATH_STATUS="already_present"
  else
    ZPROFILE_PATH_STATUS="missing"
  fi
}

resolve_labai_command() {
  shell_labai="$(command -v labai 2>/dev/null || true)"
  if [ -n "$shell_labai" ]; then
    LABAI_CMD="$shell_labai"
  elif [ -x "$LAUNCHER_PATH" ]; then
    LABAI_CMD="$LAUNCHER_PATH"
  else
    LABAI_CMD="$VENV_LABAI"
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --launcher-dir)
      LAUNCHER_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done
LAUNCHER_PATH="${LAUNCHER_DIR}/labai"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Virtual environment Python is missing: $VENV_PYTHON" >&2
  exit 1
fi
if [ ! -x "$VENV_LABAI" ]; then
  echo "LabAI console entrypoint is missing: $VENV_LABAI" >&2
  exit 1
fi
if [ ! -f "$CONFIG_PATH" ]; then
  echo "LabAI config is missing: $CONFIG_PATH" >&2
  exit 1
fi

check_launcher_path_status
check_shell_path_status
check_zprofile_path_status
resolve_labai_command
echo "launcher_path_status: $LAUNCHER_PATH_STATUS"
echo "launcher_path: $LAUNCHER_PATH"
echo "shell_path_status: $SHELL_PATH_STATUS"
echo "zprofile_path_status: $ZPROFILE_PATH_STATUS"
echo "labai_command_used: $LABAI_CMD"
if [ "$LAUNCHER_PATH_STATUS" = "missing" ]; then
  echo "launcher_next_step: rerun ./Launch-LabAI-Setup.command or create the launcher at $LAUNCHER_PATH" >&2
  exit 1
fi
if [ "$SHELL_PATH_STATUS" = "missing" ]; then
  echo "shell_path_next_step_1: source ~/.zprofile"
  echo "shell_path_next_step_2: rehash"
  echo "shell_path_next_step_3: labai doctor"
  echo "shell_path_direct_command: \"${LAUNCHER_PATH}\" doctor"
fi
if [ "$ZPROFILE_PATH_STATUS" = "missing" ]; then
  echo "zprofile_path_next_step: add this line to ~/.zprofile: $LABAI_LAUNCHER_PATH_LINE"
fi

export LABAI_CONFIG_PATH="$CONFIG_PATH"
export PATH="${LAUNCHER_DIR}:${REPO_ROOT}/.venv/bin:${PATH}"

expected_bundled_claw_path() {
  arch="$(uname -m 2>/dev/null || echo unknown)"
  case "$arch" in
    arm64) printf '%s\n' "${REPO_ROOT}/runtime-assets/claw/macos-arm64/claw" ;;
    x86_64) printf '%s\n' "${REPO_ROOT}/runtime-assets/claw/macos-x64/claw" ;;
    *) printf '%s\n' "${REPO_ROOT}/runtime-assets/claw/macos-arm64/claw or ${REPO_ROOT}/runtime-assets/claw/macos-x64/claw" ;;
  esac
}

resolve_configured_claw_path() {
  "$VENV_PYTHON" - "$CONFIG_PATH" <<'PY'
import os
from pathlib import Path
import sys
import tomllib

config_path = Path(sys.argv[1]).resolve()
project_root = config_path.parent.parent
with config_path.open("rb") as handle:
    data = tomllib.load(handle)
raw = os.environ.get("LABAI_CLAW_BINARY") or str(data.get("claw", {}).get("binary", "")).strip()
expanded = os.path.expanduser(os.path.expandvars(raw))
if not expanded:
    print("")
elif any(separator in expanded for separator in ("/", "\\")):
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    print(candidate.resolve())
else:
    print(expanded)
PY
}

run_claw_version_smoke() {
  candidate="$1"
  smoke_output="$("$candidate" --version 2>&1 || true)"
  if [ -z "$smoke_output" ]; then
    smoke_output="$("$candidate" version 2>&1 || true)"
  fi
  printf '%s\n' "$smoke_output"
  printf '%s\n' "$smoke_output" | grep -E "Claw|claw|Version|version|0\\." >/dev/null
}

verify_configured_claw() {
  configured_claw="$(resolve_configured_claw_path)"
  expected_claw="$(expected_bundled_claw_path)"
  if [ -z "$configured_claw" ]; then
    echo "claw_runtime_status: missing"
    echo "claw_runtime_expected_bundled_path: $expected_claw"
    echo "claw_runtime_env_override_set: ${LABAI_CLAW_BINARY:+yes}"
    echo "claw_next_step: Provide the expected bundled macOS Claw binary or export LABAI_CLAW_BINARY=/path/to/claw."
    return 1
  fi

  if printf '%s' "$configured_claw" | grep -q '/'; then
    resolved_claw="$configured_claw"
  else
    resolved_claw="$(command -v "$configured_claw" 2>/dev/null || true)"
  fi

  if [ -z "$resolved_claw" ] || [ ! -f "$resolved_claw" ]; then
    echo "claw_runtime_status: missing"
    echo "claw_runtime_path: ${resolved_claw:-$configured_claw}"
    echo "claw_runtime_expected_bundled_path: $expected_claw"
    echo "claw_runtime_env_override_set: ${LABAI_CLAW_BINARY:+yes}"
    echo "claw_next_step: Provide the expected bundled macOS Claw binary or export LABAI_CLAW_BINARY=/path/to/claw."
    return 1
  fi
  if [ ! -s "$resolved_claw" ]; then
    echo "claw_runtime_status: missing"
    echo "claw_runtime_path: $resolved_claw"
    echo "claw_next_step: Claw binary exists but is empty; rebuild it with scripts/mac/build-claw-macos.sh on a Mac."
    return 1
  fi
  if [ ! -x "$resolved_claw" ]; then
    echo "claw_runtime_status: missing"
    echo "claw_runtime_path: $resolved_claw"
    echo "claw_next_step: Claw binary is not executable; run chmod +x \"$resolved_claw\" or rebuild the macOS asset."
    return 1
  fi

  if CLAW_VERSION_OUTPUT="$(run_claw_version_smoke "$resolved_claw")"; then
    echo "claw_runtime_status: ok"
    echo "claw_runtime_path: $resolved_claw"
    echo "claw_version_output: $(printf '%s\n' "$CLAW_VERSION_OUTPUT" | head -n 1)"
    return 0
  fi

  echo "claw_runtime_status: missing"
  echo "claw_runtime_path: $resolved_claw"
  echo "claw_next_step: Claw binary did not pass version smoke; rebuild it with scripts/mac/build-claw-macos.sh on a Mac."
  return 1
}

run_timed() {
  "$VENV_PYTHON" - "$SMOKE_TIMEOUT_SECONDS" "$@" <<'PY'
import subprocess
import sys
import time

timeout = int(sys.argv[1])
cmd = sys.argv[2:]
start = time.monotonic()
try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    timed_out = False
except subprocess.TimeoutExpired as exc:
    result = exc
    timed_out = True
duration_ms = int((time.monotonic() - start) * 1000)
print(f"duration_ms={duration_ms}")
print(f"timed_out={str(timed_out).lower()}")
def to_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

stdout = to_text(getattr(result, "stdout", ""))
stderr = to_text(getattr(result, "stderr", ""))
output = (stdout + "\n" + stderr).strip()
if timed_out:
    print("exit_code=-1")
else:
    print(f"exit_code={result.returncode}")
print("output_begin")
print(output)
print("output_end")
sys.exit(0)
PY
}

extract_field() {
  printf '%s\n' "$1" | awk -F= -v key="$2" '$1 == key { print substr($0, length(key) + 2); exit }'
}

output_contains_two() {
  printf '%s\n' "$1" | awk '
    $0 == "output_begin" { in_output = 1; next }
    $0 == "output_end" { in_output = 0; next }
    in_output { print }
  ' | grep -E '(^|[^0-9])2([^0-9]|$)' >/dev/null
}

command_output_body() {
  printf '%s\n' "$1" | awk '
    $0 == "output_begin" { in_output = 1; next }
    $0 == "output_end" { in_output = 0; next }
    in_output { print }
  '
}

command_output_field() {
  command_output_body "$1" | awk -F': ' -v key="$2" '$1 == key { print substr($0, length(key) + 3); exit }'
}

output_indicates_claw_model_syntax_failed() {
  command_output_body "$1" | grep -E "invalid_model_syntax|invalid model syntax|Expected provider/model|Did you mean.*qwen/" >/dev/null
}

labai_local_runtime_ok() {
  body="$(command_output_body "$1")"
  runtime_used="$(printf '%s\n' "$body" | awk -F': ' '$1 == "runtime_used" { print substr($0, length($1) + 3); exit }')"
  runtime_fallback="$(printf '%s\n' "$body" | awk -F': ' '$1 == "runtime_fallback" { print substr($0, length($1) + 3); exit }')"
  selected_model="$(printf '%s\n' "$body" | awk -F': ' '$1 == "selected_model" { print substr($0, length($1) + 3); exit }')"
  provider_used="$(printf '%s\n' "$body" | awk -F': ' '$1 == "provider_used" { print substr($0, length($1) + 3); exit }')"

  if output_indicates_claw_model_syntax_failed "$1"; then
    return 2
  fi
  if [ "$runtime_used" != "claw" ]; then
    return 1
  fi
  if [ "$runtime_fallback" != "none" ]; then
    return 1
  fi
  if [ "$selected_model" = "mock-static" ] || [ -z "$selected_model" ]; then
    return 1
  fi
  if [ "$provider_used" = "mock" ] || [ -z "$provider_used" ]; then
    return 1
  fi
  if printf '%s\n' "$body" | grep -F "Mock response" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

output_indicates_missing_claw() {
  printf '%s\n' "$1" | grep -E "runtime_check_claw_binary: (not_installed|misconfigured|not_built)|Claw binary .*not found|No bundled or configured macOS Claw binary|LABAI_CLAW_BINARY|claw_health: unavailable" >/dev/null
}

doctor_reports_missing_claw() {
  printf '%s\n' "$DOCTOR_OUTPUT" | grep -E "runtime_check_claw_binary: (not_installed|misconfigured|not_built)|Claw binary .*not found|No bundled or configured macOS Claw binary|LABAI_CLAW_BINARY|claw_health: unavailable" >/dev/null
}

classify_latency() {
  elapsed_ms="$1"
  if [ "$elapsed_ms" -le 15000 ]; then
    echo "local_ready"
  elif [ "$elapsed_ms" -le 45000 ]; then
    echo "local_works_but_slow"
  else
    echo "local_not_recommended"
  fi
}

step "Checking Python and shipped dependencies"
"$VENV_PYTHON" --version
"$VENV_PYTHON" -c "import click, nbclient, nbformat, numpy, pandas, typer, unidiff; import fitz; import pypdf; print('deps_ok')"
"$VENV_PYTHON" -c "import labai; import labai.aci; import labai.data_contracts; import labai.notebook_io; import labai.owner_detection; import labai.repo_map; import labai.runtime_exec; import labai.structured_edits; import labai.task_manifest; import labai.typed_validation; import labai.validator_routing; import labai.evidence_ledger; import labai.external.grep_ast_adapter; print('phase18_modules_ok')"

step "Checking configured macOS Claw runtime"
CLAW_BLOCKED=0
if ! verify_configured_claw; then
  CLAW_BLOCKED=1
fi

step "Checking LabAI doctor and tool surface"
DOCTOR_OUTPUT="$("$LABAI_CMD" doctor)"
printf '%s\n' "$DOCTOR_OUTPUT"
printf '%s\n' "$DOCTOR_OUTPUT" | grep -F "active_profile:" >/dev/null
printf '%s\n' "$DOCTOR_OUTPUT" | grep -F "active_generation_provider:" >/dev/null
printf '%s\n' "$DOCTOR_OUTPUT" | grep -F "selected_runtime:" >/dev/null
if doctor_reports_missing_claw; then
  CLAW_BLOCKED=1
  echo "claw_runtime_status: missing"
  echo "claw_next_step: Provide runtime-assets/claw/macos-arm64/claw, runtime-assets/claw/macos-x64/claw, or export LABAI_CLAW_BINARY=/path/to/claw."
elif [ "$CLAW_BLOCKED" -eq 0 ]; then
  echo "claw_runtime_status: available_or_not_blocking"
fi

"$LABAI_CMD" tools | grep -F "registered_tools:" >/dev/null

ACTIVE_PROFILE="$("$VENV_PYTHON" - "$CONFIG_PATH" <<'PY'
import sys
import tomllib
with open(sys.argv[1], "rb") as handle:
    data = tomllib.load(handle)
print(data.get("app", {}).get("active_profile", ""))
PY
)"

if [ "$ACTIVE_PROFILE" = "local" ]; then
  step "Running local performance smoke"
  MODEL="$("$VENV_PYTHON" - "$CONFIG_PATH" <<'PY'
import sys
import tomllib
with open(sys.argv[1], "rb") as handle:
    data = tomllib.load(handle)
print(data.get("models", {}).get("general_model", "qwen2.5:7b"))
PY
)"
  DIRECT_OUTPUT=""
  DIRECT_STATUS="not_run"
  if command -v ollama >/dev/null 2>&1; then
    DIRECT_OUTPUT="$(run_timed ollama run "$MODEL" "Say exactly 2 and nothing else.")"
    printf '%s\n' "$DIRECT_OUTPUT"
    DIRECT_TIMED_OUT="$(extract_field "$DIRECT_OUTPUT" "timed_out")"
    DIRECT_EXIT_CODE="$(extract_field "$DIRECT_OUTPUT" "exit_code")"
    if [ "$DIRECT_TIMED_OUT" = "true" ] || [ "$DIRECT_EXIT_CODE" != "0" ] || ! output_contains_two "$DIRECT_OUTPUT"; then
      echo "local_performance_classification: local_failed"
      echo "local_performance_direct_ollama_status: failed"
      echo "local_performance_next_step: Check Ollama API, model availability, network, and disk space."
      exit 1
    fi
    DIRECT_STATUS="ok"
  else
    echo "local_performance_classification: local_failed"
    echo "local_performance_direct_ollama_status: missing"
    echo "local_performance_next_step: Rerun scripts/mac/setup-local-ollama.sh --apply --yes."
    exit 1
  fi

  if [ "$CLAW_BLOCKED" -eq 1 ]; then
    echo "local_performance_classification: blocked_by_claw"
    echo "local_performance_direct_ollama_status: $DIRECT_STATUS"
    echo "local_performance_labai_ask_status: blocked_by_claw"
    echo "local_performance_direct_ollama_ms: $(extract_field "$DIRECT_OUTPUT" "duration_ms")"
    echo "local_performance_next_step: Direct Ollama passed, but LabAI ask needs a macOS Claw runtime. Provide a bundled macOS Claw binary or set LABAI_CLAW_BINARY."
    echo "verification_status: blocked_by_claw"
    exit 2
  fi

  export LABAI_PROGRESS=on
  LABAI_OUTPUT="$(run_timed "$LABAI_CMD" ask -- "Say exactly 2 and nothing else.")"
  printf '%s\n' "$LABAI_OUTPUT"
  LABAI_TIMED_OUT="$(extract_field "$LABAI_OUTPUT" "timed_out")"
  LABAI_EXIT_CODE="$(extract_field "$LABAI_OUTPUT" "exit_code")"
  LABAI_DURATION="$(extract_field "$LABAI_OUTPUT" "duration_ms")"
  if output_indicates_claw_model_syntax_failed "$LABAI_OUTPUT"; then
    echo "local_performance_classification: claw_model_syntax_failed"
    echo "local_performance_direct_ollama_status: $DIRECT_STATUS"
    echo "local_performance_labai_ask_status: claw_model_syntax_failed"
    echo "local_runtime_status: failed"
    echo "local_runtime_failure_reason: claw_model_syntax_failed"
    echo "local_performance_next_step: Direct Ollama passed, but Claw rejected the local model syntax before completing the LabAI local runtime path."
    echo "verification_status: claw_model_syntax_failed"
    exit 1
  fi
  if [ "$LABAI_TIMED_OUT" = "true" ] || [ "$LABAI_EXIT_CODE" != "0" ] || ! output_contains_two "$LABAI_OUTPUT"; then
    if output_indicates_missing_claw "$LABAI_OUTPUT"; then
      echo "local_performance_classification: blocked_by_claw"
      echo "local_performance_direct_ollama_status: $DIRECT_STATUS"
      echo "local_performance_labai_ask_status: blocked_by_claw"
      echo "local_performance_next_step: Direct Ollama passed, but LabAI ask needs a macOS Claw runtime. Provide a bundled macOS Claw binary or set LABAI_CLAW_BINARY."
      echo "verification_status: blocked_by_claw"
      exit 2
    fi
    echo "local_performance_classification: local_failed"
    echo "local_performance_direct_ollama_status: $DIRECT_STATUS"
    echo "local_performance_labai_ask_status: failed"
    echo "local_performance_next_step: If direct Ollama passed but LabAI failed, check the Claw macOS runtime or LABAI_CLAW_BINARY."
    exit 1
  fi
  if labai_local_runtime_ok "$LABAI_OUTPUT"; then
    :
  else
    local_runtime_check="$?"
    if [ "$local_runtime_check" -eq 2 ]; then
      echo "local_performance_classification: claw_model_syntax_failed"
      echo "local_performance_direct_ollama_status: $DIRECT_STATUS"
      echo "local_performance_labai_ask_status: claw_model_syntax_failed"
      echo "local_runtime_status: failed"
      echo "local_runtime_failure_reason: claw_model_syntax_failed"
      echo "verification_status: claw_model_syntax_failed"
      exit 1
    fi
    echo "local_performance_classification: local_failed"
    echo "local_performance_direct_ollama_status: $DIRECT_STATUS"
    echo "local_performance_labai_ask_status: fallback_or_mock"
    echo "local_runtime_status: failed"
    echo "local_runtime_failure_reason: fallback_or_mock"
    echo "local_runtime_used: $(command_output_field "$LABAI_OUTPUT" "runtime_used")"
    echo "local_runtime_fallback: $(command_output_field "$LABAI_OUTPUT" "runtime_fallback")"
    echo "local_selected_model: $(command_output_field "$LABAI_OUTPUT" "selected_model")"
    echo "local_provider_used: $(command_output_field "$LABAI_OUTPUT" "provider_used")"
    echo "local_performance_next_step: LabAI ask returned an answer, but it did not use runtime_used=claw with runtime_fallback=none and a non-mock provider/model."
    echo "verification_status: local_failed"
    exit 1
  fi
  CLASSIFICATION="$(classify_latency "$LABAI_DURATION")"
  echo "local_performance_classification: $CLASSIFICATION"
  echo "local_performance_direct_ollama_status: $DIRECT_STATUS"
  echo "local_performance_labai_ask_status: ok"
  echo "local_runtime_status: ok"
  echo "local_runtime_used: $(command_output_field "$LABAI_OUTPUT" "runtime_used")"
  echo "local_runtime_fallback: $(command_output_field "$LABAI_OUTPUT" "runtime_fallback")"
  echo "local_selected_model: $(command_output_field "$LABAI_OUTPUT" "selected_model")"
  echo "local_provider_used: $(command_output_field "$LABAI_OUTPUT" "provider_used")"
  echo "local_performance_labai_ask_ms: $LABAI_DURATION"
  echo "local_performance_direct_ollama_ms: $(extract_field "$DIRECT_OUTPUT" "duration_ms")"
  if [ "$CLASSIFICATION" = "local_not_recommended" ]; then
    echo "local_performance_next_step: Consider API mode or a smaller local model on this Mac."
  fi
else
  echo "local_performance_classification: not_measured"
  echo "local_performance_reason: active profile is ${ACTIVE_PROFILE:-unknown}, not local."
fi

step "Running lightweight ask and workflow smoke checks"
if [ "${CLAW_BLOCKED:-0}" -eq 1 ]; then
  echo "verification_status: blocked_by_claw"
  echo "verification_next_step: Python, LabAI, Ollama, and Qwen can be ready while the macOS Claw runtime is missing. Provide a macOS Claw binary or set LABAI_CLAW_BINARY."
  exit 2
fi
"$LABAI_CMD" ask "what is 1+1" | grep -E "2|answer:" >/dev/null
"$LABAI_CMD" ask "只输出 3+5 的答案，不要解释" | grep -F "8" >/dev/null
"$LABAI_CMD" workflow verify-workspace --preview | grep -F "target_workspace_root:" >/dev/null

echo ""
echo "Install verification passed."
