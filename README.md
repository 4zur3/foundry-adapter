# foundry-adapter

Adapter to proxy OpenClaw requests to Azure OpenAI deployments. This repo is intended to be published publicly only after removing secrets and auditing git history.

## Quick start

1. Copy `.env.example` to `.env` and set secrets.
2. Run with `uvicorn app:app --host 0.0.0.0 --port 8000` or use the provided Dockerfile.

## Logging / Production

- Control logging with environment variables:
  - `LOG_LEVEL` — default `INFO`. Use `WARNING` or `ERROR` in production.
  - `LOG_TO_FILE` — `0` (disabled) or `1` (enabled). Disabled by default to avoid writing logs with secrets to disk.
  - `LOG_FILE_PATH` — file path when `LOG_TO_FILE=1`.
  - `ADAPTER_DEBUG` — `0` or `1` for verbose debug logs (do not enable in prod).

Example production env:

```bash
LOG_LEVEL=WARNING
LOG_TO_FILE=0
ADAPTER_DEBUG=0
```

## Security

- Never commit `.env` or any files containing secrets. Use `.gitignore` (already provided).
- Audit git history for leaked secrets before making the repo public. Use `git filter-repo` or BFG to remove secrets from history.
- Use CI/CD secrets for deploy keys (GitHub Actions secrets, etc.).

## Before publishing

- Ensure no secrets exist in commits or files.
- Replace `AZURE_OPENAI_API_KEY` with a secret stored in your deployment platform.
- Consider adding a license (MIT included by default in this repo).

## License

MIT
