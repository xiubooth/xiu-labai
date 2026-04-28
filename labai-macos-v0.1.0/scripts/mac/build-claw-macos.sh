#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_ROOT="${CLAW_SOURCE_ROOT:-}"
BUILD_PROFILE="${CLAW_BUILD_PROFILE:-release}"

usage() {
  cat <<'USAGE'
Usage: scripts/mac/build-claw-macos.sh [--source PATH] [--profile release|debug]

Maintainer-only helper. Builds the real Claw macOS binary from a local
claw-code source checkout and writes it into runtime-assets/claw/macos-*/claw.
Normal RA one-click setup must not require Rust or this script.
USAGE
}

step() {
  printf '\n[labai-claw-build] %s\n' "$1"
}

fail_build() {
  echo "Claw source not found or build failed: $1" >&2
  echo "Expected a claw-code checkout containing rust/Cargo.toml." >&2
  echo "Set CLAW_SOURCE_ROOT or pass --source PATH, for example:" >&2
  echo "  scripts/mac/build-claw-macos.sh --source \"${HOME}/src/claw-code\"" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source)
      SOURCE_ROOT="${2:-}"
      shift 2
      ;;
    --profile)
      BUILD_PROFILE="${2:-}"
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

if [ "$(uname -s)" != "Darwin" ]; then
  fail_build "this helper must run on macOS so the output is a real Darwin executable"
fi

case "$BUILD_PROFILE" in
  release|debug) ;;
  *)
    fail_build "unsupported build profile: $BUILD_PROFILE"
    ;;
esac

if [ -z "$SOURCE_ROOT" ]; then
  for candidate in \
    "${REPO_ROOT}/../claw-code" \
    "${HOME}/src/claw-code" \
    "${HOME}/Developer/claw-code"
  do
    if [ -f "$candidate/rust/Cargo.toml" ]; then
      SOURCE_ROOT="$candidate"
      break
    fi
  done
fi

if [ -z "$SOURCE_ROOT" ] || [ ! -f "$SOURCE_ROOT/rust/Cargo.toml" ]; then
  fail_build "Claw source not found"
fi

if ! command -v cargo >/dev/null 2>&1; then
  fail_build "cargo is required for maintainer Claw builds"
fi

ARCH="$(uname -m)"
case "$ARCH" in
  arm64)
    ASSET_DIR="${REPO_ROOT}/runtime-assets/claw/macos-arm64"
    TARGET_LABEL="macos-arm64"
    ;;
  x86_64)
    ASSET_DIR="${REPO_ROOT}/runtime-assets/claw/macos-x64"
    TARGET_LABEL="macos-x64"
    ;;
  *)
    fail_build "unsupported macOS architecture: $ARCH"
    ;;
esac

WORKSPACE_DIR="${SOURCE_ROOT}/rust"
OUTPUT_PATH="${ASSET_DIR}/claw"

step "building Claw from source"
echo "Source root: $SOURCE_ROOT"
echo "Rust workspace: $WORKSPACE_DIR"
echo "Build profile: $BUILD_PROFILE"
echo "Target asset: $OUTPUT_PATH"

if [ "$BUILD_PROFILE" = "release" ]; then
  (cd "$WORKSPACE_DIR" && cargo build --release --bin claw)
  BUILT_BINARY="${WORKSPACE_DIR}/target/release/claw"
else
  (cd "$WORKSPACE_DIR" && cargo build --bin claw)
  BUILT_BINARY="${WORKSPACE_DIR}/target/debug/claw"
fi

if [ ! -s "$BUILT_BINARY" ]; then
  fail_build "cargo did not produce a non-empty Claw binary at $BUILT_BINARY"
fi

mkdir -p "$ASSET_DIR"
cp "$BUILT_BINARY" "$OUTPUT_PATH"
chmod +x "$OUTPUT_PATH"

if [ ! -x "$OUTPUT_PATH" ]; then
  fail_build "output binary is not executable: $OUTPUT_PATH"
fi

step "running Claw smoke"
SMOKE_OUTPUT="$("$OUTPUT_PATH" --version 2>&1 || true)"
if [ -z "$SMOKE_OUTPUT" ]; then
  SMOKE_OUTPUT="$("$OUTPUT_PATH" version 2>&1 || true)"
fi
if ! printf '%s\n' "$SMOKE_OUTPUT" | grep -E "Claw|claw|Version|version|0\\." >/dev/null 2>&1; then
  echo "$SMOKE_OUTPUT" >&2
  fail_build "Claw version smoke did not produce recognizable output"
fi
printf '%s\n' "$SMOKE_OUTPUT"

SOURCE_COMMIT="unknown"
if command -v git >/dev/null 2>&1; then
  SOURCE_COMMIT="$(git -C "$SOURCE_ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')"
fi
BUILD_DATE="$(date -u +%Y-%m-%d)"

cat > "${ASSET_DIR}/README.md" <<EOF
# LabAI Claw Runtime for ${TARGET_LABEL}

- Target OS/arch: macOS ${ARCH}
- Binary path: \`runtime-assets/claw/${TARGET_LABEL}/claw\`
- Built from: \`${SOURCE_ROOT}\`
- Source commit: \`${SOURCE_COMMIT}\`
- Build profile: \`${BUILD_PROFILE}\`
- Build date: \`${BUILD_DATE}\`
- Smoke command: \`${OUTPUT_PATH} --version\`
- Smoke output:

\`\`\`text
${SMOKE_OUTPUT}
\`\`\`

This binary is intended for RA one-click macOS setup once included in the
release archive. Normal RA setup must not require Rust or Cargo.
EOF

echo ""
echo "Claw macOS asset built: $OUTPUT_PATH"
