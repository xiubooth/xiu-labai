#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/.labai/config.toml"
LAUNCHER_DIR="${HOME}/Library/Application Support/LabAI/bin"
DEFAULT_LAUNCHER_DIR="${HOME}/Library/Application Support/LabAI/bin"
LABAI_LAUNCHER_PATH_LINE='export PATH="$HOME/Library/Application Support/LabAI/bin:$PATH"'
REPLACE_CONFIG=0
SKIP_MODEL_PULLS=0

PYTHON_STATUS="not_checked"
LABAI_INSTALL_STATUS="not_checked"
OLLAMA_INSTALL_STATUS="not_checked"
OLLAMA_API_STATUS="not_checked"
QWEN_MODELS_STATUS="not_checked"
CLAW_STATUS="not_checked"
CLAW_CONFIG_BINARY=""
VERIFY_STATUS="not_checked"
LOCAL_PERFORMANCE_STATUS="not_measured"
PYTHON_VERSION_STATUS="not_available"
PYTHON_PATH_STATUS="not_available"
VENV_PATH_STATUS="not_available"
PYTHON_INFO_FILE="${REPO_ROOT}/.labai/python-bootstrap.env"
PATH_CURRENT_STATUS="not_checked"
ZPROFILE_PATH_STATUS="not_checked"

usage() {
  cat <<'USAGE'
Usage: scripts/mac/bootstrap-mac.sh [options]

Options:
  --launcher-dir PATH
  --replace-config
  --skip-model-pulls
  -h, --help
USAGE
}

step() {
  printf '\n[labai-setup] %s\n' "$1"
}

claw_step() {
  printf '\n[labai-claw] %s\n' "$1"
}

launcher_path_line() {
  if [ "$LAUNCHER_DIR" = "$DEFAULT_LAUNCHER_DIR" ]; then
    printf '%s\n' "$LABAI_LAUNCHER_PATH_LINE"
  else
    printf 'export PATH="%s:$PATH"\n' "$LAUNCHER_DIR"
  fi
}

ensure_launcher_path_current() {
  case ":${PATH:-}:" in
    *":${LAUNCHER_DIR}:"*)
      PATH_CURRENT_STATUS="already_present"
      ;;
    *)
      export PATH="${LAUNCHER_DIR}:${PATH:-}"
      PATH_CURRENT_STATUS="updated"
      ;;
  esac
}

ensure_launcher_path_zprofile() {
  path_line="$(launcher_path_line)"
  if [ -z "${HOME:-}" ]; then
    ZPROFILE_PATH_STATUS="failed"
    echo "[labai-setup] warning: HOME is not set; could not update ~/.zprofile." >&2
    echo "[labai-setup] manual command: $path_line" >&2
    return 0
  fi
  zprofile="${HOME}/.zprofile"
  if [ -f "$zprofile" ] && grep -Fqx "$path_line" "$zprofile" >/dev/null 2>&1; then
    ZPROFILE_PATH_STATUS="already_present"
    echo "[labai-setup] launcher PATH already present in $zprofile."
    return 0
  fi
  if ! touch "$zprofile" >/dev/null 2>&1; then
    ZPROFILE_PATH_STATUS="failed"
    echo "[labai-setup] warning: could not update $zprofile." >&2
    echo "[labai-setup] manual command: $path_line" >&2
    return 0
  fi
  if ! printf '%s\n' "$path_line" >> "$zprofile"; then
    ZPROFILE_PATH_STATUS="failed"
    echo "[labai-setup] warning: could not write launcher PATH to $zprofile." >&2
    echo "[labai-setup] manual command: $path_line" >&2
    return 0
  fi
  ZPROFILE_PATH_STATUS="added"
  echo "[labai-setup] added launcher PATH to $zprofile: $path_line"
}

print_summary() {
  echo ""
  echo "[labai-setup] Final summary"
  echo "Python: ${PYTHON_STATUS}"
  echo "Python version: ${PYTHON_VERSION_STATUS}"
  echo "Python path: ${PYTHON_PATH_STATUS}"
  echo "Venv path: ${VENV_PATH_STATUS}"
  echo "LabAI install: ${LABAI_INSTALL_STATUS}"
  echo "Ollama install: ${OLLAMA_INSTALL_STATUS}"
  echo "Ollama API: ${OLLAMA_API_STATUS}"
  echo "Qwen models: ${QWEN_MODELS_STATUS}"
  echo "Claw: ${CLAW_STATUS}"
  echo "Verification: ${VERIFY_STATUS}"
  echo "Local performance: ${LOCAL_PERFORMANCE_STATUS}"
  echo "Launcher path: ${LAUNCHER_DIR}/labai"
  echo "Current process PATH: ${PATH_CURRENT_STATUS}"
  echo "~/.zprofile PATH: ${ZPROFILE_PATH_STATUS}"
  echo "Parent Terminal PATH: reload_may_be_required"
  echo "Run source ~/.zprofile: yes"
  echo "Next command for this Terminal:"
  echo "  source ~/.zprofile"
  echo "  rehash"
  echo "  labai doctor"
  echo "Alternative direct command:"
  echo "  \"${LAUNCHER_DIR}/labai\" doctor"
}

fail_component() {
  component="$1"
  message="$2"
  echo "[labai-setup] ${component} failed: ${message}" >&2
  print_summary
  exit 1
}

load_python_info() {
  if [ ! -f "$PYTHON_INFO_FILE" ]; then
    return 1
  fi
  PYTHON_STATUS="$(awk -F= '$1 == "python_status" { print substr($0, length($1) + 2); found = 1 } END { if (!found) print "ok" }' "$PYTHON_INFO_FILE")"
  PYTHON_VERSION_STATUS="$(awk -F= '$1 == "python_version" { print substr($0, length($1) + 2); found = 1 } END { if (!found) print "not_available" }' "$PYTHON_INFO_FILE")"
  PYTHON_PATH_STATUS="$(awk -F= '$1 == "python_path" { print substr($0, length($1) + 2); found = 1 } END { if (!found) print "not_available" }' "$PYTHON_INFO_FILE")"
  VENV_PATH_STATUS="$(awk -F= '$1 == "venv_path" { print substr($0, length($1) + 2); found = 1 } END { if (!found) print "not_available" }' "$PYTHON_INFO_FILE")"
  return 0
}

run_claw_smoke() {
  candidate="$1"
  if [ ! -s "$candidate" ]; then
    echo "Claw binary exists but is empty: $candidate" >&2
    return 1
  fi
  if [ ! -x "$candidate" ]; then
    echo "Claw binary is not executable: $candidate" >&2
    return 1
  fi

  smoke_output="$("$candidate" --version 2>&1 || true)"
  if [ -z "$smoke_output" ]; then
    smoke_output="$("$candidate" version 2>&1 || true)"
  fi
  if ! printf '%s\n' "$smoke_output" | grep -E "Claw|claw|Version|version|0\\." >/dev/null 2>&1; then
    echo "$smoke_output" >&2
    return 1
  fi
  echo "$smoke_output"
  return 0
}

configure_claw_binary_path() {
  binary_path="$1"
  if [ -z "$binary_path" ] || [ ! -f "$CONFIG_PATH" ]; then
    return 0
  fi
  if printf '%s' "$binary_path" | grep -q '"'; then
    fail_component "Claw" "Claw binary path must not contain double quotes: $binary_path"
  fi
  claw_step "configuring LabAI Claw binary path"
  awk -v replacement="$binary_path" '
    /^\[claw\]$/ { in_claw = 1; print; next }
    /^\[/ && $0 != "[claw]" { in_claw = 0 }
    in_claw && /^binary[[:space:]]*=/ { print "binary = \"" replacement "\""; replaced = 1; next }
    { print }
    END {
      if (!replaced) {
        print ""
        print "[claw]"
        print "binary = \"" replacement "\""
      }
    }
  ' "$CONFIG_PATH" > "${CONFIG_PATH}.tmp"
  mv "${CONFIG_PATH}.tmp" "$CONFIG_PATH"
}

detect_claw() {
  managed_claw="${HOME}/Library/Application Support/LabAI/runtime/claw/claw"
  bundled_arm64="${REPO_ROOT}/runtime-assets/claw/macos-arm64/claw"
  bundled_x64="${REPO_ROOT}/runtime-assets/claw/macos-x64/claw"
  mac_arch="$(uname -m 2>/dev/null || echo unknown)"
  claw_step "detecting mac architecture: $mac_arch"

  if [ -n "${LABAI_CLAW_BINARY:-}" ]; then
    claw_step "checking LABAI_CLAW_BINARY: $LABAI_CLAW_BINARY"
    if [ ! -x "$LABAI_CLAW_BINARY" ]; then
      CLAW_STATUS="failed"
      fail_component "Claw" "LABAI_CLAW_BINARY is set but is not executable: ${LABAI_CLAW_BINARY}"
    fi
    claw_step "verifying Claw binary"
    run_claw_smoke "$LABAI_CLAW_BINARY"
    echo "Using user-provided Claw binary from LABAI_CLAW_BINARY: $LABAI_CLAW_BINARY"
    CLAW_CONFIG_BINARY="$LABAI_CLAW_BINARY"
    CLAW_STATUS="configured"
    return 0
  fi

  case "$mac_arch" in
    arm64)
      expected_bundled="$bundled_arm64"
      expected_label="runtime-assets/claw/macos-arm64/claw"
      ;;
    x86_64)
      expected_bundled="$bundled_x64"
      expected_label="runtime-assets/claw/macos-x64/claw"
      ;;
    *)
      expected_bundled=""
      expected_label="runtime-assets/claw/macos-arm64/claw or runtime-assets/claw/macos-x64/claw"
      ;;
  esac

  claw_step "checking bundled Claw: $expected_label"
  if [ -n "$expected_bundled" ] && [ -f "$expected_bundled" ]; then
    claw_step "bundled Claw found"
    claw_step "ensuring executable bit"
    chmod +x "$expected_bundled"
    claw_step "verifying Claw binary"
    if ! run_claw_smoke "$expected_bundled"; then
      CLAW_STATUS="failed"
      fail_component "Claw" "bundled Claw failed version smoke: $expected_label"
    fi
    mkdir -p "$(dirname "$managed_claw")"
    cp "$expected_bundled" "$managed_claw"
    chmod +x "$managed_claw"
    if ! run_claw_smoke "$managed_claw" >/dev/null 2>&1; then
      CLAW_STATUS="failed"
      fail_component "Claw" "managed Claw copy failed version smoke: $managed_claw"
    fi
    echo "Managed macOS Claw runtime provisioned: $managed_claw"
    CLAW_CONFIG_BINARY="$managed_claw"
    CLAW_STATUS="ok"
  elif command -v claw >/dev/null 2>&1; then
    path_claw="$(command -v claw)"
    claw_step "verifying Claw binary"
    run_claw_smoke "$path_claw"
    echo "Using Claw from PATH: $path_claw"
    CLAW_CONFIG_BINARY="$path_claw"
    CLAW_STATUS="configured"
  else
    echo "No bundled or configured macOS Claw binary was found."
    echo "Expected bundled path for this Mac: $expected_label"
    echo "This does not block Ollama/Qwen setup, but LabAI Claw runtime verification will remain incomplete."
    echo "Provide one of:"
    echo "  A. runtime-assets/claw/macos-arm64/claw"
    echo "  B. runtime-assets/claw/macos-x64/claw"
    echo "  C. LABAI_CLAW_BINARY=/path/to/claw"
    echo "  D. explicit developer source-build fallback"
    CLAW_STATUS="missing"
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --launcher-dir)
      LAUNCHER_DIR="${2:-}"
      shift 2
      ;;
    --replace-config)
      REPLACE_CONFIG=1
      shift
      ;;
    --skip-model-pulls)
      SKIP_MODEL_PULLS=1
      shift
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

step "Starting macOS bootstrap"
echo "Repo root: $REPO_ROOT"
echo "Launcher dir: $LAUNCHER_DIR"

step "Selecting compliant Python"
rm -f "$PYTHON_INFO_FILE"
if "${SCRIPT_DIR}/install-labai.sh" --python-only --python-info-file "$PYTHON_INFO_FILE" --launcher-dir "$LAUNCHER_DIR"; then
  load_python_info || true
  PYTHON_STATUS="ok"
else
  load_python_info || true
  PYTHON_STATUS="failed"
  fail_component "Python" "could not find, install, or prepare Python >= 3.11"
fi

step "Checking macOS Claw provisioning state"
detect_claw

INSTALL_ARGS=(--profile local --launcher-dir "$LAUNCHER_DIR" --python-info-file "$PYTHON_INFO_FILE")
if [ "$REPLACE_CONFIG" -eq 1 ]; then
  INSTALL_ARGS+=(--replace-config)
fi

step "Installing LabAI"
if "${SCRIPT_DIR}/install-labai.sh" "${INSTALL_ARGS[@]}"; then
  load_python_info || true
  PYTHON_STATUS="ok"
  LABAI_INSTALL_STATUS="ok"
  configure_claw_binary_path "$CLAW_CONFIG_BINARY"
else
  install_code=$?
  if load_python_info; then
    PYTHON_STATUS="ok"
    LABAI_INSTALL_STATUS="failed"
  else
    PYTHON_STATUS="failed"
    LABAI_INSTALL_STATUS="not_completed"
  fi
  fail_component "LabAI install" "install-labai.sh failed"
fi
ensure_launcher_path_current
ensure_launcher_path_zprofile

if [ "$SKIP_MODEL_PULLS" -eq 1 ]; then
  step "Checking Ollama/Qwen readiness without pulling models"
  if "${SCRIPT_DIR}/setup-local-ollama.sh" --plan; then
    OLLAMA_INSTALL_STATUS="ok"
    OLLAMA_API_STATUS="ok"
    QWEN_MODELS_STATUS="not_pulled"
  else
    OLLAMA_INSTALL_STATUS="failed"
    OLLAMA_API_STATUS="failed"
    QWEN_MODELS_STATUS="not_checked"
    fail_component "Ollama/Qwen" "readiness check failed"
  fi
else
  step "Installing Ollama if needed and pulling local Qwen models"
  if "${SCRIPT_DIR}/setup-local-ollama.sh" --apply --yes; then
    OLLAMA_INSTALL_STATUS="ok"
    OLLAMA_API_STATUS="ok"
    QWEN_MODELS_STATUS="ok"
  else
    OLLAMA_INSTALL_STATUS="failed_or_unavailable"
    OLLAMA_API_STATUS="failed_or_unavailable"
    QWEN_MODELS_STATUS="failed_or_partial"
    fail_component "Ollama/Qwen" "setup-local-ollama.sh failed"
  fi
fi

step "Running install verification"
VERIFY_OUTPUT=""
if VERIFY_OUTPUT="$("${SCRIPT_DIR}/verify-install.sh" --launcher-dir "$LAUNCHER_DIR" 2>&1)"; then
  printf '%s\n' "$VERIFY_OUTPUT"
  VERIFY_STATUS="ok"
else
  verify_code=$?
  printf '%s\n' "$VERIFY_OUTPUT"
  LOCAL_PERFORMANCE_STATUS="$(printf '%s\n' "$VERIFY_OUTPUT" | awk -F': ' '$1 == "local_performance_classification" { print $2; found = 1 } END { if (!found) print "not_measured" }')"
  if [ "$LOCAL_PERFORMANCE_STATUS" = "blocked_by_claw" ] || printf '%s\n' "$VERIFY_OUTPUT" | grep -F "verification_status: blocked_by_claw" >/dev/null 2>&1; then
    VERIFY_STATUS="blocked_by_claw"
  elif [ "$LOCAL_PERFORMANCE_STATUS" = "claw_model_syntax_failed" ] || printf '%s\n' "$VERIFY_OUTPUT" | grep -F "verification_status: claw_model_syntax_failed" >/dev/null 2>&1; then
    VERIFY_STATUS="claw_model_syntax_failed"
  else
    VERIFY_STATUS="failed"
  fi
  print_summary
  exit "$verify_code"
fi
LOCAL_PERFORMANCE_STATUS="$(printf '%s\n' "$VERIFY_OUTPUT" | awk -F': ' '$1 == "local_performance_classification" { print $2; found = 1 } END { if (!found) print "not_measured" }')"

print_summary
echo ""
echo "macOS bootstrap completed."
echo "LabAI macOS setup finished."
echo "If local Qwen is slow on this Mac, use API mode or a smaller local model."
