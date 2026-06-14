# NeuroEmbed API v0.1.0 — Release Notes

**Date:** 2026-06-14
**Status:** M0–M6 complete · 37 tests pass · 93% coverage · Low-risk security audit
**Repo path:** `C:\Users\AGC\neurosaas-loop\neuroembed-api\`
**Tarball:** `C:\Users\AGC\neurosaas-loop\neuroembed-api-v0.1.0.tar.gz` (583 KB, 43 tracked files)

## Highlights

- Hosted REVE EEG foundation-model inference — FastAPI + Pydantic v2
- `POST /v1/embeddings` — 256-dim embedding per 4s window + mean embedding
- `POST /v1/cognitive` — 5 zero-shot cognitive-state scores (sleep stage, PVT lapse, valence, arousal, seizure risk)
- API key auth (bcrypt, `nmb_` prefix) with `Bearer` header
- LRU embedding cache (sha256-keyed, deterministic)
- `/healthz`, `/readyz`, `/metrics`, `/docs` (Swagger UI), `/dashboard` (10KB static)
- Multi-stage Dockerfile (CPU + CUDA 12.4 GPU)
- GitHub Actions CI: lint, types, tests + coverage, pip-audit, docker smoke
- Security audit: 0 critical, 0 high, 1 medium (CVE-2025-3000 mitigated via torch upper-bound pin)

## Quick start

```bash
# Extract
tar xzf neuroembed-api-v0.1.0.tar.gz
cd neuroembed-api

# Run tests
pip install -e ".[dev]"
pytest tests/ -q

# Boot
neuroembed-api --reload
# → http://localhost:8000/docs
```

## Push to GitHub — step-by-step

```bash
# Once `gh` is installed and you are authenticated:
cd neuroembed-api

# Create the public repo (substitute your org/name)
gh repo create <ORG>/neuroembed-api --public --description "Hosted REVE EEG foundation-model inference API" --source .

# Push
git push -u origin main

# Optionally create a release
gh release create v0.1.0 --generate-notes
```

If you prefer SSH, set the remote first:
```bash
git remote add origin git@github.com:<ORG>/neuroembed-api.git
git push -u origin main
```

## What's NOT in v0.1.0 (logged for v1.1)

1. `POST /v1/auth/keys` (dashboard references it; provisioning currently via the test fixture)
2. Postgres-backed key store (currently in-memory; restarts wipe keys)
3. Rate limiting (60 rpm per key) at the FastAPI layer
4. Real REVE weights load on the GPU build (CPU path uses `FakeReve` for end-to-end testability)
5. Real-model contract tests against TUAB/TUEV accuracies (only the dev path is verified)
6. TLS termination (deployment responsibility; documented in README)

## License

MIT — see [LICENSE](LICENSE).
