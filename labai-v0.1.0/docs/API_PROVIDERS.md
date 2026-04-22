# API Providers

## Current First-Release API Profile

The first release only documents one API profile for end users:

- DeepSeek

This is optional. Local mode remains supported.

## DeepSeek Setup

Base URL:

- `https://api.deepseek.com`

OpenAI-compatible URL used by the Claw bridge:

- `https://api.deepseek.com/v1`

Environment variable:

- `DEEPSEEK_API_KEY`

Set it for the current PowerShell session:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

Switch to the API profile:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-api-provider.ps1 -Apply
labai doctor
```

## Security Rules

- do not write a real API key into `.labai/config.toml`
- do not commit a real `.env`
- do not write secrets into session or audit files
- use `.env.example` only as a placeholder example

## Verify The API Profile

```powershell
labai doctor
labai ask "reply with the word ready"
labai ask "Write a tiny Python function that adds two numbers and explain it briefly."
```

## If The Key Is Missing

`labai doctor` should say:

- `missing DEEPSEEK_API_KEY`

and show the PowerShell command shape needed to set it.
