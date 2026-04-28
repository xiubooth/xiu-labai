#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage: scripts/mac/setup-api-provider.sh [--plan|--apply]

Configures the macOS API profile. Secrets stay in environment variables.
USAGE
}

MODE="plan"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --plan)
      MODE="plan"
      shift
      ;;
    --apply)
      MODE="apply"
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

echo "[labai-api] macOS API provider setup"
echo "Secrets are read from the environment. Do not write real API keys into config files."
echo ""
echo "Current-shell example:"
echo "  export DEEPSEEK_API_KEY=\"your_key_here\""
echo ""
echo "Persistent zsh example:"
echo "  echo 'export DEEPSEEK_API_KEY=\"your_key_here\"' >> ~/.zshrc"
echo "  source ~/.zshrc"
echo ""

if [ "$MODE" = "apply" ]; then
  exec "$SCRIPT_DIR/install-labai.sh" --profile api-deepseek --replace-config
fi

echo "Plan only. To switch the local install to the API profile:"
echo "  scripts/mac/setup-api-provider.sh --apply"
