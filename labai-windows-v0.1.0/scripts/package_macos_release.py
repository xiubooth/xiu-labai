from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from macos_release_rules import build_macos_release_plan, create_macos_release_archive


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a clean macOS-focused LabAI release zip.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dist-dir", type=Path, default=REPO_ROOT / "dist")
    parser.add_argument("--staging-dir", type=Path, default=REPO_ROOT / ".release-staging" / "macos")
    parser.add_argument("--archive-name", default="", help="Override the generated macOS archive name.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    plan = build_macos_release_plan(repo_root)
    archive_name = args.archive_name or plan.archive_name
    archive_path = args.dist_dir.resolve() / archive_name
    result = create_macos_release_archive(
        repo_root,
        archive_path,
        staging_root=args.staging_dir.resolve(),
    )
    extracted_folder = repo_root / archive_path.stem
    if extracted_folder.exists():
        shutil.rmtree(extracted_folder)
    shutil.copytree(result.staging_root, extracted_folder)

    print("macos_release_package_created")
    print(f"version: {result.plan.version}")
    print(f"package_kind: {result.plan.package_kind}")
    print(f"real_macos_claw_binary_present: {bool(result.plan.real_claw_binaries)}")
    if result.plan.real_claw_binaries:
        print("real_macos_claw_binaries: " + ", ".join(result.plan.real_claw_binaries))
    else:
        print("real_macos_claw_binaries: none")
    print(f"archive_path: {result.archive_path}")
    print(f"extracted_folder: {extracted_folder}")
    print(f"staging_root: {result.staging_root}")
    print(f"included_files: {len(result.included_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
