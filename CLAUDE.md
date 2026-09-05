# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Educational demo: downloads three pretrained models from Hugging Face and serves each
through a different `pipeline()` task, on purpose — to show that `pipeline()` picks a
different model class and pre/postprocessing routine per task, so the response shape
differs completely between endpoints even though they're all "just" `pipeline()` calls.
Nothing is trained. Only 3 files in [src/](src/).

| endpoint | pipeline task | model | response shape |
|---|---|---|---|
| `POST /analyze` | `sentiment-analysis` | DistilBERT (SST-2) | single label + score |
| `POST /fill-mask` | `fill-mask` | `bert-base-uncased` | ranked list of candidate words |
| `POST /generate` | `text-generation` | `gpt2` | free-form generated text |

## Commands

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell); Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1 — server (MUST run from src/)
cd src && uvicorn api:app --reload

# Terminal 2 — client
cd src && python client.py
```

Interactive FastAPI docs: http://127.0.0.1:8000/docs

No tests, linter, or formatter are configured in the project.

## Details that matter

**The cwd must be `src/`.** [src/api.py](src/api.py) imports with `from model import ...`
(flat import, no package). Running `uvicorn src.api:app` from the root fails with
`ModuleNotFoundError: model`. If this gets restructured into a package, update that import
and the README command together.

**All three models load at module level**, not in a startup event or per-request:
`classifier`, `mask_filler`, and `generator` in [src/api.py](src/api.py) are all built
by calling their `load_*()` function when the module is imported. Consequences: first
launch downloads all three to `~/.cache/huggingface` (needs internet), and with
`--reload` all three reload on every file save — slower iteration than a single-model
app. Any script that imports `api` pays that cost — import from
[src/model.py](src/model.py) directly if you only need one classifier/pipeline.

**`/analyze` is English-only and binary.** SST-2 returns exclusively `POSITIVE` /
`NEGATIVE`, no neutral class. If you need reliable predictions on non-English text, swap
`SENTIMENT_MODEL` in [src/model.py](src/model.py) for a multilingual sentiment model —
that also changes the label set `/analyze` returns.

**`/fill-mask` requires the literal `[MASK]` token in the input** (BERT's mask token,
uppercase, with brackets) — e.g. `"Paris is the [MASK] of France."`. Omitting it or using
a different model's mask token (e.g. RoBERTa's `<mask>`) produces wrong or empty results.

**`/generate` output is nondeterministic-ish and unconstrained.** GPT-2 has no notion of
"correct" continuation — it just samples statistically likely next tokens. Don't expect
factual or safe-for-all-inputs output; this endpoint exists to show the third response
shape, not for production text generation.

**Naming convention:** all code, comments, and API JSON fields are in English.

**Git:** the project root is NOT a repo. The actual repo is nested inside
`api_huggingface/` (remote `github.com/jtorresj96/api_huggingface`) and is empty except
for the initial commit — the code in `src/` is not versioned there yet.
