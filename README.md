# foundry-adapter

Adapter for Microsoft Foundry — proxies SSE chat completions to Azure OpenAI legacy deployments.

**Not affiliated with or endorsed by Microsoft or OpenAI.**

## Quick start

1. Copy `.env.example` to `.env` and set your secrets (do not commit `.env`).
2. Run locally:

```bash
cp .env.example .env
# fill .env with values, then
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or build and run the provided `Dockerfile`.

## Logging / Production

Control logging with environment variables:

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

## Security and publishing checklist

- Never commit `.env` or any files containing secrets. `.gitignore` is provided.
- Audit git history for leaked secrets before making the repository public. Use `git filter-repo` or the BFG Repo-Cleaner to remove secrets from history.
- Use CI/CD secret stores (GitHub Actions secrets, etc.) for deployment keys.
- Avoid logging full request payloads in production; mask or remove sensitive fields.

## Recommended repository metadata

- Description (short): `Microsoft Foundry — Azure OpenAI legacy model adapter for SSE chat completions`
- Topics/tags: `fastapi`, `azure-openai`, `openclaw`, `foundry-vtt`, `sse`, `docker`, `python`

## Before publishing

- Ensure no secrets exist in commits or files. If you previously committed secrets, clean the history before pushing public.

## License

MIT

---

If you want I can add a simple GitHub Actions workflow to run a lint/smoke check on PRs and a pre-merge secret check.
