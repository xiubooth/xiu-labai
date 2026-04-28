#!/usr/bin/env bash
set -euo pipefail

APPLY=0
YES=0
MODELS=("qwen2.5:7b" "qwen2.5-coder:7b" "qwen3-embedding:0.6b")
OLLAMA_API_URL="${OLLAMA_API_URL:-http://127.0.0.1:11434}"
OLLAMA_APP_PATH="${OLLAMA_APP_PATH:-/Applications/Ollama.app}"
OLLAMA_APP_CLI="${OLLAMA_APP_PATH}/Contents/Resources/ollama"
OLLAMA_CLI_LINK="${OLLAMA_CLI_LINK:-/usr/local/bin/ollama}"
OLLAMA_DOWNLOAD_URL="${OLLAMA_MAC_DOWNLOAD_URL:-https://ollama.com/download/Ollama-darwin.zip}"
OLLAMA_WAIT_SECONDS="${OLLAMA_WAIT_SECONDS:-120}"
TEMP_DIR=""
OLLAMA_BIN=""

usage() {
  cat <<'USAGE'
Usage: scripts/mac/setup-local-ollama.sh [--plan|--apply] [--yes]

Checks macOS Ollama/Qwen readiness. Use --apply to install Ollama when
missing, start the local API, and pull missing models.
USAGE
}

step() {
  printf '\n[labai-ollama] %s\n' "$1"
}

cleanup() {
  if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
  fi
}
trap cleanup EXIT

fail_ollama_install() {
  echo "Ollama install blocker: $1" >&2
  echo "Fallback:" >&2
  echo "  1. Download Ollama from https://ollama.com/download/mac" >&2
  echo "  2. Install and open Ollama.app" >&2
  echo "  3. Rerun ./Launch-LabAI-Setup.command from this package" >&2
  exit 1
}

find_ollama() {
  if [ -x "$OLLAMA_APP_CLI" ]; then
    printf '%s\n' "$OLLAMA_APP_CLI"
    return 0
  fi
  if command -v ollama >/dev/null 2>&1; then
    command -v ollama
    return 0
  fi
  if [ -x "$OLLAMA_CLI_LINK" ]; then
    printf '%s\n' "$OLLAMA_CLI_LINK"
    return 0
  fi
  return 1
}

require_macos_install_tools() {
  missing=()
  for tool in curl unzip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      missing+=("$tool")
    fi
  done
  if [ "${#missing[@]}" -ne 0 ]; then
    fail_ollama_install "required macOS install tool(s) missing: ${missing[*]}"
  fi
}

ensure_cli_link() {
  if [ ! -x "$OLLAMA_APP_CLI" ]; then
    return 0
  fi
  if command -v ollama >/dev/null 2>&1; then
    return 0
  fi

  step "ensuring ollama CLI link"
  echo "Admin permission may be requested to create ${OLLAMA_CLI_LINK}."
  mkdir -p "$(dirname "$OLLAMA_CLI_LINK")" 2>/dev/null || sudo mkdir -p "$(dirname "$OLLAMA_CLI_LINK")"
  ln -sf "$OLLAMA_APP_CLI" "$OLLAMA_CLI_LINK" 2>/dev/null || sudo ln -sf "$OLLAMA_APP_CLI" "$OLLAMA_CLI_LINK"
}

install_ollama_app() {
  if [ "$(uname -s)" != "Darwin" ]; then
    fail_ollama_install "automatic Ollama.app installation is only supported on macOS"
  fi
  if [ "$APPLY" -ne 1 ]; then
    echo "Ollama was not found."
    echo "Plan only. Rerun with --apply --yes to download and install Ollama.app from:"
    echo "  $OLLAMA_DOWNLOAD_URL"
    exit 0
  fi
  if [ "$YES" -ne 1 ]; then
    echo "Ollama was not found."
    echo "This will download the official macOS Ollama archive and install Ollama.app."
    echo "Pass --yes to confirm."
    exit 2
  fi

  step "Ollama not found; attempting macOS install"
  require_macos_install_tools
  TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/labai-ollama.XXXXXX")"
  echo "Downloading official Ollama macOS archive."
  if ! curl --fail --show-error --location --progress-bar -o "$TEMP_DIR/Ollama-darwin.zip" "$OLLAMA_DOWNLOAD_URL"; then
    fail_ollama_install "download failed from $OLLAMA_DOWNLOAD_URL"
  fi
  if ! unzip -q "$TEMP_DIR/Ollama-darwin.zip" -d "$TEMP_DIR"; then
    fail_ollama_install "could not unpack Ollama-darwin.zip"
  fi
  if [ ! -d "$TEMP_DIR/Ollama.app" ]; then
    fail_ollama_install "downloaded archive did not contain Ollama.app"
  fi

  step "installing Ollama.app"
  if [ -d "$OLLAMA_APP_PATH" ]; then
    echo "Ollama.app already exists at ${OLLAMA_APP_PATH}; keeping existing app."
  else
    echo "Admin permission may be requested to install Ollama.app into /Applications."
    mv "$TEMP_DIR/Ollama.app" "$OLLAMA_APP_PATH" 2>/dev/null || sudo mv "$TEMP_DIR/Ollama.app" "$OLLAMA_APP_PATH"
  fi
}

api_ready() {
  curl -fsS "${OLLAMA_API_URL}/api/version" >/dev/null 2>&1
}

start_ollama() {
  if api_ready; then
    echo "Ollama API already running."
    return 0
  fi

  step "starting Ollama"
  if [ "$(uname -s)" = "Darwin" ] && command -v open >/dev/null 2>&1 && [ -d "$OLLAMA_APP_PATH" ]; then
    open -a Ollama --args hidden >/dev/null 2>&1 || open -a Ollama >/dev/null 2>&1 || true
  elif [ -n "$OLLAMA_BIN" ]; then
    log_dir="${HOME}/Library/Logs/LabAI"
    mkdir -p "$log_dir" 2>/dev/null || true
    nohup "$OLLAMA_BIN" serve >"${log_dir}/ollama-serve.log" 2>&1 &
  fi
}

wait_for_ollama_api() {
  step "waiting for Ollama API"
  elapsed=0
  while [ "$elapsed" -le "$OLLAMA_WAIT_SECONDS" ]; do
    if api_ready; then
      echo "Ollama API ready."
      return 0
    fi
    echo "Waiting for ${OLLAMA_API_URL} (${elapsed}s/${OLLAMA_WAIT_SECONDS}s)"
    sleep 3
    elapsed=$((elapsed + 3))
  done
  fail_ollama_install "Ollama API did not become reachable at ${OLLAMA_API_URL}"
}

model_present() {
  model="$1"
  printf '%s\n' "$AVAILABLE_MODELS" | awk -v name="$model" '$1 == name { found = 1 } END { exit found ? 0 : 1 }'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --plan)
      APPLY=0
      shift
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --yes|-y)
      YES=1
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

step "checking Ollama command"
OLLAMA_BIN="$(find_ollama || true)"
if [ -z "$OLLAMA_BIN" ]; then
  install_ollama_app
  ensure_cli_link
  OLLAMA_BIN="$(find_ollama || true)"
fi
if [ -z "$OLLAMA_BIN" ]; then
  fail_ollama_install "Ollama was installed or detected, but the ollama CLI could not be located"
fi
echo "Ollama binary: $OLLAMA_BIN"

ensure_cli_link
OLLAMA_BIN="$(find_ollama || true)"

start_ollama
wait_for_ollama_api

step "checking required models"
AVAILABLE_MODELS="$("$OLLAMA_BIN" list 2>/dev/null || true)"
if [ -z "$AVAILABLE_MODELS" ]; then
  echo "Could not read Ollama model list. The API is reachable, but the CLI list call failed." >&2
  exit 1
fi

MISSING=()
for model in "${MODELS[@]}"; do
  if model_present "$model"; then
    echo "present: $model"
  else
    echo "missing: $model"
    MISSING+=("$model")
  fi
done

if [ "${#MISSING[@]}" -eq 0 ]; then
  echo "All required local Qwen models are present."
  echo "qwen_models_status: ok"
  exit 0
fi

if [ "$APPLY" -ne 1 ]; then
  echo ""
  echo "Plan only. To pull missing models:"
  echo "  scripts/mac/setup-local-ollama.sh --apply --yes"
  echo ""
  echo "Note: Qwen 7B models can be slow on weaker Macs. Verification will classify local performance."
  echo "qwen_models_status: missing"
  exit 0
fi

if [ "$YES" -ne 1 ]; then
  echo "This will pull ${#MISSING[@]} model(s). Pass --yes to confirm."
  exit 2
fi

for model in "${MISSING[@]}"; do
  step "pulling missing model: $model"
  if ! "$OLLAMA_BIN" pull "$model"; then
    echo "Qwen model setup failed while pulling: $model" >&2
    echo "Check network connectivity and available disk space, then rerun:" >&2
    echo "  scripts/mac/setup-local-ollama.sh --apply --yes" >&2
    exit 1
  fi
done

echo "Ollama/Qwen setup complete."
echo "qwen_models_status: ok"
