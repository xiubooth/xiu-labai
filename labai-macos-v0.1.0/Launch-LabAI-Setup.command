#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

echo "LabAI macOS setup launcher"
echo "Repository: $ROOT_DIR"
echo ""
echo "This will run scripts/mac/bootstrap-mac.sh."
echo "If macOS blocks this file, run:"
echo "  chmod +x Launch-LabAI-Setup.command scripts/mac/*.sh"
echo "  xattr -dr com.apple.quarantine ."
echo ""

if /bin/bash "$ROOT_DIR/scripts/mac/bootstrap-mac.sh" "$@"; then
  echo ""
  echo "LabAI macOS setup finished."
else
  status=$?
  echo ""
  echo "LabAI macOS setup failed with exit code $status."
  echo "Check the component summary above: Python, LabAI install, Ollama, Qwen, Claw, or verification."
  exit "$status"
fi
