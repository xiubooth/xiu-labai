# LabAI Claw Runtime for macOS ARM64

- Target OS/arch: macOS arm64
- Binary path: `runtime-assets/claw/macos-arm64/claw`
- SHA256 sidecar: `runtime-assets/claw/macos-arm64/claw.sha256`
- SHA256: `1b56b244975c01062a4932a76b7834ee140a19db8c24731ef6204f66b0528612`
- Built from: GitHub Actions macOS Claw artifact
- Source commit: `11e2353585fac22568e2cd53d0cbffcd9d1b7e1b`
- Build date: GitHub Actions artifact supplied after the Windows packaging pass
- Smoke command: `runtime-assets/claw/macos-arm64/claw --version`
- Smoke output: must be re-run on the next real Mac retest; Windows packaging verified Mach-O magic and SHA256 only
- RA one-click setup: bundled for macOS arm64 one-click setup

This is a real macOS Mach-O executable, not a placeholder and not the Windows runtime.
Normal RA setup must not require Rust or Cargo.
