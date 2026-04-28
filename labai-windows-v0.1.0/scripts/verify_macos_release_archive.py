from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from macos_release_rules import (
    MACOS_CLAW_BINARIES,
    MACOS_CLAW_SHA256S,
    build_macos_file_set,
    build_macos_release_plan,
    validate_macos_archive,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a macOS LabAI release archive.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    archive_path = args.archive.resolve()
    names = validate_macos_archive(archive_path, repo_root=repo_root)
    plan = build_macos_release_plan(repo_root)
    expected = build_macos_file_set(repo_root)
    shipped_checks = [name for name in names if name.startswith(".continue/checks/")]
    shipped_macos_claw = [name for name in names if name in MACOS_CLAW_BINARIES]
    shipped_macos_claw_sha256s = [name for name in names if name in MACOS_CLAW_SHA256S]

    print("macos_release_archive_status: ok")
    print(f"archive_path: {archive_path}")
    print(f"package_kind: {plan.package_kind}")
    print(f"archive_entries: {len(names)}")
    print(f"required_entries: {len(expected)}")
    print(f"shipped_continue_checks: {len(shipped_checks)}")
    print(f"real_macos_claw_binary_present: {bool(shipped_macos_claw)}")
    if shipped_macos_claw:
        print("shipped_macos_claw_binaries: " + ", ".join(shipped_macos_claw))
    else:
        print("shipped_macos_claw_binaries: none")
    if shipped_macos_claw_sha256s:
        print("shipped_macos_claw_sha256s: " + ", ".join(shipped_macos_claw_sha256s))
    else:
        print("shipped_macos_claw_sha256s: none")
    print("forbidden_windows_dev_paths: absent")
    print("phase18_19_modules: present")
    print("python_source_syntax: ok")
    print("runtime_dependencies: present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
