# LabAI Claw Runtime for macOS x64

- Target OS/arch: macOS x86_64
- Binary path: `runtime-assets/claw/macos-x64/claw`
- SHA256 sidecar: `runtime-assets/claw/macos-x64/claw.sha256`
- SHA256: `fc987a7ec1d8bef94061aef3e207f3b726cbd7e66f47997804ce34e1d7d67839`
- Built from: GitHub Actions macOS Claw artifact
- Source commit: `11e2353585fac22568e2cd53d0cbffcd9d1b7e1b`
- Build date: GitHub Actions artifact supplied after the Windows packaging pass
- Smoke command: `runtime-assets/claw/macos-x64/claw --version`
- Smoke output: must be re-run on the next real Mac retest; Windows packaging verified Mach-O magic and SHA256 only
- RA one-click setup: bundled for macOS x64 one-click setup

This is a real macOS Mach-O executable, not a placeholder and not the Windows runtime.
Normal RA setup must not require Rust or Cargo.
