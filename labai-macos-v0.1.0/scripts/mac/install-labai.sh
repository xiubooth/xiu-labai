#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LABAI_DIR="${REPO_ROOT}/.labai"
CONFIG_PATH="${LABAI_DIR}/config.toml"
VENV_DIR="${REPO_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_LABAI="${VENV_DIR}/bin/labai"
BOOTSTRAP_TEMP_ROOT="${LABAI_DIR}/temp/macos-bootstrap"
BOOTSTRAP_TEMP_DIR="${BOOTSTRAP_TEMP_ROOT}/$(date +%Y%m%d%H%M%S)-$$"
DEFAULT_LAUNCHER_DIR="${HOME}/Library/Application Support/LabAI/bin"
LABAI_LAUNCHER_PATH_LINE='export PATH="$HOME/Library/Application Support/LabAI/bin:$PATH"'
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11
HOMEBREW_INSTALL_URL="${HOMEBREW_INSTALL_URL:-https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh}"
PROFILE="local"
REPLACE_CONFIG=0
DEV_EXTRAS=0
PYTHON_ONLY=0
LAUNCHER_DIR="${DEFAULT_LAUNCHER_DIR}"
PYTHON_INFO_FILE=""
SELECTED_PYTHON=""
SELECTED_PYTHON_VERSION=""
PATH_CURRENT_STATUS="not_checked"
ZPROFILE_PATH_STATUS="not_checked"

usage() {
  cat <<'USAGE'
Usage: scripts/mac/install-labai.sh [options]

Options:
  --profile local|api-deepseek|fallback
  --replace-config
  --dev-extras
  --python-only
  --launcher-dir PATH
  --python-info-file PATH
  -h, --help
USAGE
}

step() {
  printf '\n[labai-install] %s\n' "$1"
}

python_step() {
  printf '\n[labai-python] %s\n' "$1"
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
    echo "[labai-install] warning: HOME is not set; could not update ~/.zprofile." >&2
    echo "[labai-install] manual command: $path_line" >&2
    return 0
  fi
  zprofile="${HOME}/.zprofile"
  if [ -f "$zprofile" ] && grep -Fqx "$path_line" "$zprofile" >/dev/null 2>&1; then
    ZPROFILE_PATH_STATUS="already_present"
    echo "[labai-install] launcher PATH already present in $zprofile."
    return 0
  fi
  if ! touch "$zprofile" >/dev/null 2>&1; then
    ZPROFILE_PATH_STATUS="failed"
    echo "[labai-install] warning: could not update $zprofile." >&2
    echo "[labai-install] manual command: $path_line" >&2
    return 0
  fi
  if ! printf '%s\n' "$path_line" >> "$zprofile"; then
    ZPROFILE_PATH_STATUS="failed"
    echo "[labai-install] warning: could not write launcher PATH to $zprofile." >&2
    echo "[labai-install] manual command: $path_line" >&2
    return 0
  fi
  ZPROFILE_PATH_STATUS="added"
  echo "[labai-install] added launcher PATH to $zprofile: $path_line"
}

python_requirement() {
  printf '%s.%s\n' "$MIN_PYTHON_MAJOR" "$MIN_PYTHON_MINOR"
}

python_probe() {
  candidate="$1"
  "$candidate" - "$MIN_PYTHON_MAJOR" "$MIN_PYTHON_MINOR" <<'PY'
import importlib.util
import sys

min_major = int(sys.argv[1])
min_minor = int(sys.argv[2])
version = sys.version_info
print(f"version={version.major}.{version.minor}.{version.micro}")
print(f"executable={getattr(sys, 'executable')}")
print(f"compliant={str(version >= (min_major, min_minor)).lower()}")
print(f"venv={str(importlib.util.find_spec('venv') is not None).lower()}")
PY
}

probe_field() {
  printf '%s\n' "$1" | awk -F= -v key="$2" '$1 == key { print substr($0, length(key) + 2); exit }'
}

find_brew() {
  if command -v brew >/dev/null 2>&1; then
    command -v brew
    return 0
  fi
  if [ -x "/opt/homebrew/bin/brew" ]; then
    printf '%s\n' "/opt/homebrew/bin/brew"
    return 0
  fi
  if [ -x "/usr/local/bin/brew" ]; then
    printf '%s\n' "/usr/local/bin/brew"
    return 0
  fi
  return 1
}

configure_homebrew_shellenv() {
  brew_bin="$(find_brew || true)"
  if [ -z "$brew_bin" ]; then
    return 1
  fi
  python_step "configuring Homebrew shell environment"
  eval "$("$brew_bin" shellenv)"
  if ! command -v brew >/dev/null 2>&1; then
    echo "[labai-python] Homebrew shellenv did not expose brew on PATH."
    return 1
  fi
  return 0
}

ensure_zprofile_homebrew_shellenv() {
  brew_bin="$(find_brew || true)"
  if [ -z "$brew_bin" ] || [ -z "${HOME:-}" ]; then
    return 0
  fi
  if [ ! -f "${HOME}/.zprofile" ]; then
    case "${SHELL:-}" in
      */zsh) ;;
      *)
        echo "[labai-python] Homebrew shellenv is configured for this setup run; skipping .zprofile because the login shell is not zsh."
        return 0
        ;;
    esac
  fi
  case "$brew_bin" in
    /opt/homebrew/bin/brew) shellenv_line='eval "$(/opt/homebrew/bin/brew shellenv)"' ;;
    /usr/local/bin/brew) shellenv_line='eval "$(/usr/local/bin/brew shellenv)"' ;;
    *) return 0 ;;
  esac

  zprofile="${HOME}/.zprofile"
  if [ -f "$zprofile" ] && grep -F "$shellenv_line" "$zprofile" >/dev/null 2>&1; then
    echo "[labai-python] Homebrew shellenv already present in $zprofile."
    return 0
  fi

  touch "$zprofile"
  {
    printf '\n# Homebrew shell environment for LabAI setup\n'
    printf '%s\n' "$shellenv_line"
  } >> "$zprofile"
  echo "[labai-python] added Homebrew shellenv to $zprofile: $shellenv_line"
}

install_homebrew() {
  if configure_homebrew_shellenv; then
    return 0
  fi

  python_step "Homebrew not found; attempting automatic Homebrew install"
  if ! command -v curl >/dev/null 2>&1; then
    fail_python_install "Python/Homebrew install blocker: curl is required to install Homebrew automatically"
  fi
  if [ ! -x "/bin/bash" ]; then
    fail_python_install "Python/Homebrew install blocker: /bin/bash is required to install Homebrew automatically"
  fi

  echo "[labai-python] running official Homebrew installer: /bin/bash -c \"\$(curl -fsSL ${HOMEBREW_INSTALL_URL})\""
  if ! install_script="$(curl -fsSL "$HOMEBREW_INSTALL_URL")"; then
    fail_python_install "Python/Homebrew install blocker: could not download Homebrew installer from $HOMEBREW_INSTALL_URL"
  fi
  if ! /bin/bash -c "$install_script"; then
    fail_python_install "Python/Homebrew install blocker: automatic Homebrew install failed"
  fi

  if ! configure_homebrew_shellenv; then
    fail_python_install "Python/Homebrew install blocker: Homebrew install finished but brew was not found at /opt/homebrew/bin/brew or /usr/local/bin/brew"
  fi
  ensure_zprofile_homebrew_shellenv
}

check_python_candidate() {
  label="$1"
  candidate="$2"
  if [ -z "$candidate" ] || [ ! -x "$candidate" ]; then
    return 1
  fi

  probe="$(python_probe "$candidate" 2>/dev/null || true)"
  version="$(probe_field "$probe" "version")"
  executable="$(probe_field "$probe" "executable")"
  compliant="$(probe_field "$probe" "compliant")"
  has_venv="$(probe_field "$probe" "venv")"
  if [ -z "$version" ]; then
    echo "[labai-python] $label at $candidate is not usable."
    return 1
  fi
  if [ "$compliant" = "true" ] && [ "$has_venv" = "true" ]; then
    SELECTED_PYTHON="${executable:-$candidate}"
    SELECTED_PYTHON_VERSION="$version"
    echo "[labai-python] found compliant Python at $SELECTED_PYTHON ($SELECTED_PYTHON_VERSION)"
    return 0
  fi
  if [ "$compliant" != "true" ]; then
    if [ "$label" = "python3" ]; then
      echo "[labai-python] found python3 but version $version is below required >= $(python_requirement)"
    else
      echo "[labai-python] found $label but version $version is below required >= $(python_requirement)"
    fi
  elif [ "$has_venv" != "true" ]; then
    echo "[labai-python] found $label at $candidate but the venv module is unavailable"
  fi
  return 1
}

check_path_python3() {
  path_python="$(command -v python3 2>/dev/null || true)"
  if [ -n "$path_python" ]; then
    check_python_candidate "python3" "$path_python"
    return $?
  fi
  echo "[labai-python] python3 was not found on PATH."
  return 1
}

check_common_homebrew_python() {
  for candidate in \
    "/opt/homebrew/bin/python3" \
    "/usr/local/bin/python3" \
    "/opt/homebrew/bin/python3.13" \
    "/opt/homebrew/bin/python3.12" \
    "/opt/homebrew/bin/python3.11" \
    "/usr/local/bin/python3.13" \
    "/usr/local/bin/python3.12" \
    "/usr/local/bin/python3.11"
  do
    if check_python_candidate "Homebrew Python" "$candidate"; then
      return 0
    fi
  done
  return 1
}

check_brew_formula_python() {
  configure_homebrew_shellenv >/dev/null 2>&1 || true
  brew_bin="$(find_brew || true)"
  if [ -z "$brew_bin" ]; then
    echo "[labai-python] Homebrew was not found."
    return 1
  fi

  for formula in python@3.12 python@3.13 python@3.11 python; do
    prefix="$("$brew_bin" --prefix "$formula" 2>/dev/null || true)"
    if [ -z "$prefix" ]; then
      continue
    fi
    for candidate in \
      "$prefix/bin/python3" \
      "$prefix/bin/python3.13" \
      "$prefix/bin/python3.12" \
      "$prefix/bin/python3.11" \
      "$prefix/libexec/bin/python3"
    do
      if check_python_candidate "brew formula $formula" "$candidate"; then
        return 0
      fi
    done
  done
  return 1
}

find_compliant_python() {
  python_step "checking python3"
  if check_path_python3; then
    return 0
  fi

  python_step "checking Homebrew Python"
  if check_common_homebrew_python; then
    return 0
  fi
  if check_brew_formula_python; then
    return 0
  fi
  return 1
}

fail_python_install() {
  echo "Python install blocker: $1" >&2
  echo "LabAI requires Python >= $(python_requirement)." >&2
  echo "The setup attempted automatic remediation before reaching this blocker." >&2
  echo "Fallback commands:" >&2
  echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"" >&2
  echo "  eval \"\$(/opt/homebrew/bin/brew shellenv)\" || eval \"\$(/usr/local/bin/brew shellenv)\"" >&2
  echo "  brew install python@3.12" >&2
  echo "  ./Launch-LabAI-Setup.command" >&2
  exit 1
}

install_python_with_brew() {
  python_step "checking Homebrew"
  configure_homebrew_shellenv >/dev/null 2>&1 || true
  brew_bin="$(find_brew || true)"
  if [ -z "$brew_bin" ]; then
    install_homebrew
    brew_bin="$(find_brew || true)"
  fi
  if [ -z "$brew_bin" ]; then
    fail_python_install "Python/Homebrew install blocker: Homebrew install did not provide a brew command"
  fi
  configure_homebrew_shellenv
  brew_bin="$(find_brew || true)"

  python_step "Homebrew detected; attempting Python install"
  formulas=("${LABAI_MAC_PYTHON_FORMULA:-python@3.12}" "python@3.13" "python@3.11" "python")
  tried=""
  for formula in "${formulas[@]}"; do
    case " $tried " in
      *" $formula "*) continue ;;
    esac
    tried="$tried $formula"
    python_step "installing $formula"
    if "$brew_bin" list --versions "$formula" >/dev/null 2>&1; then
      echo "[labai-python] $formula is already installed."
    elif ! "$brew_bin" install "$formula"; then
      echo "[labai-python] brew install $formula failed; trying next candidate."
      continue
    fi
    if find_compliant_python; then
      return 0
    fi
  done
  fail_python_install "Homebrew was present, but installing a compliant Python did not produce a usable interpreter"
}

select_compliant_python() {
  if find_compliant_python; then
    python_step "selected Python: $SELECTED_PYTHON"
    python_step "Python version ok: $SELECTED_PYTHON_VERSION"
    return 0
  fi
  python_step "no compliant Python found"
  install_python_with_brew
  python_step "selected Python: $SELECTED_PYTHON"
  python_step "Python version ok: $SELECTED_PYTHON_VERSION"
}

venv_is_compliant() {
  if [ ! -x "$VENV_PYTHON" ]; then
    return 1
  fi
  probe="$(python_probe "$VENV_PYTHON" 2>/dev/null || true)"
  version="$(probe_field "$probe" "version")"
  compliant="$(probe_field "$probe" "compliant")"
  has_venv="$(probe_field "$probe" "venv")"
  if [ "$compliant" = "true" ] && [ "$has_venv" = "true" ]; then
    echo "[labai-python] existing venv Python is compliant: $version"
    return 0
  fi
  echo "[labai-python] existing venv Python is not compliant (${version:-unknown}); recreating .venv."
  return 1
}

write_python_info() {
  if [ -z "$PYTHON_INFO_FILE" ]; then
    return 0
  fi
  mkdir -p "$(dirname "$PYTHON_INFO_FILE")"
  {
    printf 'python_status=ok\n'
    printf 'python_path=%s\n' "$SELECTED_PYTHON"
    printf 'python_version=%s\n' "$SELECTED_PYTHON_VERSION"
    printf 'venv_path=%s\n' "$VENV_DIR"
    printf 'venv_python=%s\n' "$VENV_PYTHON"
  } > "$PYTHON_INFO_FILE"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --replace-config)
      REPLACE_CONFIG=1
      shift
      ;;
    --dev-extras)
      DEV_EXTRAS=1
      shift
      ;;
    --python-only)
      PYTHON_ONLY=1
      shift
      ;;
    --launcher-dir)
      LAUNCHER_DIR="${2:-}"
      shift 2
      ;;
    --python-info-file)
      PYTHON_INFO_FILE="${2:-}"
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

case "$PROFILE" in
  local) TEMPLATE_PATH="${REPO_ROOT}/templates/profiles/local-mac.toml" ;;
  api-deepseek) TEMPLATE_PATH="${REPO_ROOT}/templates/profiles/api-deepseek-mac.toml" ;;
  fallback) TEMPLATE_PATH="${REPO_ROOT}/templates/profiles/fallback-mac.toml" ;;
  *)
    echo "Unsupported profile: $PROFILE" >&2
    exit 2
    ;;
esac

if [ ! -f "$TEMPLATE_PATH" ]; then
  echo "Profile template not found: $TEMPLATE_PATH" >&2
  exit 1
fi

if [ "$PYTHON_ONLY" -eq 1 ]; then
  step "Preparing compliant Python for macOS"
else
  step "Installing LabAI for macOS"
fi
echo "Repo root: $REPO_ROOT"
echo "Selected profile: $PROFILE"
echo "Launcher dir: $LAUNCHER_DIR"
echo "Bootstrap temp dir: $BOOTSTRAP_TEMP_DIR"

mkdir -p "$BOOTSTRAP_TEMP_DIR"
export TMPDIR="$BOOTSTRAP_TEMP_DIR"

select_compliant_python

if [ -x "$VENV_PYTHON" ] && ! venv_is_compliant; then
  rm -rf "$VENV_DIR"
fi

if [ ! -x "$VENV_PYTHON" ]; then
  step "Creating virtual environment"
  "$SELECTED_PYTHON" -m venv "$VENV_DIR"
fi
"$VENV_PYTHON" -c "import sys; assert sys.version_info >= (3, 11)"
write_python_info
echo "Python interpreter: $SELECTED_PYTHON"
echo "Python version: $SELECTED_PYTHON_VERSION"
echo "Venv path: $VENV_DIR"

if [ "$PYTHON_ONLY" -eq 1 ]; then
  echo "Python setup complete."
  exit 0
fi

step "Upgrading pip"
"$VENV_PYTHON" -m pip install --upgrade pip

step "Installing LabAI package"
if [ "$DEV_EXTRAS" -eq 1 ]; then
  "$VENV_PYTHON" -m pip install -e "${REPO_ROOT}[dev]"
else
  "$VENV_PYTHON" -m pip install -e "$REPO_ROOT"
fi

if [ ! -x "$VENV_LABAI" ]; then
  echo "The LabAI console script was not created: $VENV_LABAI" >&2
  exit 1
fi

mkdir -p "$LABAI_DIR"
CONFIG_STATUS="preserved"
if [ ! -f "$CONFIG_PATH" ]; then
  cp "$TEMPLATE_PATH" "$CONFIG_PATH"
  CONFIG_STATUS="created"
elif [ "$REPLACE_CONFIG" -eq 1 ]; then
  BACKUP_DIR="${LABAI_DIR}/config.backups"
  mkdir -p "$BACKUP_DIR"
  BACKUP_PATH="${BACKUP_DIR}/config-$(date +%Y%m%d-%H%M%S).toml"
  cp "$CONFIG_PATH" "$BACKUP_PATH"
  cp "$TEMPLATE_PATH" "$CONFIG_PATH"
  CONFIG_STATUS="replaced"
fi

if [ -n "${LABAI_CLAW_BINARY:-}" ] && [ -f "$CONFIG_PATH" ]; then
  step "Applying LABAI_CLAW_BINARY override"
  if [ ! -x "$LABAI_CLAW_BINARY" ]; then
    echo "LABAI_CLAW_BINARY must point to an existing executable file: $LABAI_CLAW_BINARY" >&2
    exit 1
  fi
  if printf '%s' "$LABAI_CLAW_BINARY" | grep -q '"'; then
    echo "LABAI_CLAW_BINARY must not contain double quotes." >&2
    exit 1
  fi
  awk -v replacement="$LABAI_CLAW_BINARY" '
    /^\[claw\]$/ { in_claw = 1; print; next }
    /^\[/ && $0 != "[claw]" { in_claw = 0 }
    in_claw && /^binary[[:space:]]*=/ { print "binary = \"" replacement "\""; next }
    { print }
  ' "$CONFIG_PATH" > "${CONFIG_PATH}.tmp"
  mv "${CONFIG_PATH}.tmp" "$CONFIG_PATH"
fi

step "Creating launcher"
mkdir -p "$LAUNCHER_DIR"
cat > "${LAUNCHER_DIR}/labai" <<EOF
#!/bin/sh
export LABAI_CONFIG_PATH="${CONFIG_PATH}"
exec "${VENV_LABAI}" "\$@"
EOF
chmod +x "${LAUNCHER_DIR}/labai"
ensure_launcher_path_current
ensure_launcher_path_zprofile

echo "Install complete."
echo "Config status: $CONFIG_STATUS"
echo "Config path: $CONFIG_PATH"
echo "Launcher path: ${LAUNCHER_DIR}/labai"
echo "Current process PATH: $PATH_CURRENT_STATUS"
echo "~/.zprofile PATH: $ZPROFILE_PATH_STATUS"
echo "Parent Terminal PATH: reload_may_be_required"
echo "Run source ~/.zprofile: yes"
echo ""
echo "Next verification command:"
echo "  scripts/mac/verify-install.sh --launcher-dir \"${LAUNCHER_DIR}\""
echo "Next command for this Terminal:"
echo "  source ~/.zprofile"
echo "  rehash"
echo "  labai doctor"
echo "Alternative direct command:"
echo "  \"${LAUNCHER_DIR}/labai\" doctor"
