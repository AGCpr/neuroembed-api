# NeuroEmbed API

> Hosted REVE EEG foundation-model inference.
> Upload EEG, get 256-dim brain-state embeddings + cognitive scores.
> **Research use only — not a medical device.**

[![Tests](https://img.shields.io/badge/tests-35%20passed-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)]()

## What this is

NeuroEmbed is a hosted REST API that wraps [REVE](https://brain-bzh.github.io/reve/),
a self-supervised transformer pretrained on **60,000+ hours of EEG from
25,000 subjects** (NeurIPS 2025). Send EEG samples, get back:

- A **256-dim embedding** per 4-second window (REVE's latent space)
- A **mean embedding** over the whole recording
- Optional **zero-shot cognitive-state scores** — sleep stage, PVT-lapse
  probability, valence, arousal, seizure risk

The whole thing runs in a single `docker run` and is small enough to
develop on a laptop (no GPU required for the dev path — the heavy model
weights are loaded on demand in the worker process).

## Quick start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Run the tests
pytest tests/ -q

# 3. Boot the API
neuroembed-api --reload
# → http://localhost:8000/docs   (Swagger UI)
```

## Docker

```bash
# CPU image (small, ~150 MB; uses FakeReve deterministic backend)
docker build -t neuroembed:cpu --target cpu .

# GPU image (CUDA 12.4; loads real REVE weights)
docker build -t neuroembed:gpu --target gpu .
```

## Usage example

```bash
curl -X POST http://localhost:8000/v1/embeddings \\
  -H "Authorization: Bearer $NEUROEMBED_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "electrode_names": ["Fp1","Fp2","C3","C4","O1","O2","F3","F4"],
    "samples": [[...], [...], ...],
    "sample_rate_hz": 200,
    "window_seconds": 4,
    "return_per_window": false
  }'
```

```json
{
  "model": "brain-bzh/reve-base",
  "window_count": 2,
  "embedding_dim": 256,
  "mean_embedding": [0.023, -0.118, ...],
  "window_embeddings": null,
  "processing_ms": 1,
  "cached": false
}
```

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /healthz` | public | Liveness |
| `GET /readyz` | public | Readiness + model_loaded flag |
| `GET /metrics` | public | Prometheus exposition |
| `GET /docs` | public | Swagger UI |
| `POST /v1/embeddings` | Bearer | 256-dim embedding per window |
| `POST /v1/cognitive` | Bearer | Embeddings + 5 cognitive scores |

All requests and responses are JSON. Errors follow RFC 7807 (`detail` is a string).

## Pricing tiers (v1 launch)

| Tier | Price | Embeddings/mo | Cognitive/mo |
|---|---|---|---|
| Free | $0 | 1,000 | 200 |
| Hobby | $29/mo | 50,000 | 10,000 |
| Pro | $299/mo | 1,000,000 | 200,000 |
| Team | $999/mo | 5,000,000 | 1,000,000 |
| Enterprise | custom | unlimited | unlimited |

## Architecture

```
Researcher / App  →  FastAPI Gateway
                       ↓
                   MinIO (file storage)    Redis Queue (Dramatiq)
                       ↓                          ↓
                  API Key Store            GPU Worker
                  (bcrypt, Postgres)        - MNE pipeline
                                              - REVE-base (HF)
                                              - Linear probes
                                              - LRU embeddings cache
                       ↓
                  Stripe (metered billing webhook)
```

See [`docs/architecture.html`](docs/architecture.html) for the full diagram.

## Neuroscience core

This is not a thin wrapper over a model card. The product *is* a deployment of
the REVE foundation model — a self-supervised transformer with a 4D positional
encoding scheme that accepts arbitrary electrode configurations. Pretrained
on 92 datasets spanning 25,000 subjects. The cognitive-state scores are
linear-probe weights fine-tuned on REVE's frozen representations, evaluated
on the public benchmarks the paper reports (TUAB, TUEV, PhysioNetMI,
BCI-IV-2a, FACED, ISRUC, Mumtaz, MAT, BCI-2020-3).

For v1 development we ship `FakeReve`, a deterministic stand-in that
generates 256-dim unit-norm embeddings from a hash of the input. This lets
the API be exercised end-to-end without downloading the gated HF weights.
The real model loads on demand when the `[model]` extra is installed.

## Privacy & safety

- **Research use only.** Output is not intended for clinical decision-making.
- **Gated-model access.** REVE is gated on HuggingFace under EDPB Opinion
  28/2024. Users must supply `NEUROEMBED_HF_TOKEN` and accept the model's
  research-use terms.
- **PHI stripped.** EDF header fields containing patient identifiers are
  removed server-side.
- **No data retention.** Uploaded recordings are not logged or stored
  beyond the cache TTL. (v1.1: presigned-URL upload + explicit retention
  policy.)

## License

MIT. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `pytest tests/ -q` and `ruff
check src tests` before opening a PR.
