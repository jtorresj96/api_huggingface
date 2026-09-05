# Sentiment API with Hugging Face 🤗

Demo: download pretrained models from Hugging Face, serve them with FastAPI, and
consume them with requests. No training involved — the models already come ready-made.

## What it does

Serves three different Hugging Face `pipeline` tasks, each with its own model and its
own output shape, to show that `pipeline()` is not a one-size-fits-all interface:

- **`/analyze`** — sentiment analysis (DistilBERT, task `sentiment-analysis`): is a text
  **positive** or **negative**? Returns a single label + confidence score.
- **`/fill-mask`** — mask filling (BERT, task `fill-mask`): given a text with a literal
  `[MASK]` token, returns a ranked list of candidate words.
- **`/generate`** — text generation (GPT-2, task `text-generation`): continues a prompt
  with free-form generated text.

All three are lightweight models that run on CPU (no GPU needed).

```
api-sentimiento-hf/
├── src/
│   ├── model.py      # downloads and loads the Hugging Face models
│   ├── api.py        # the FastAPI API that serves them
│   └── client.py     # consumes the API with requests
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> The first time you start the API, the models get downloaded (a few hundred MB total).
> They stay cached; subsequent times they load from disk instantly. You only need
> internet the first time.

## Usage

**Terminal 1 — start the API (the server):**
```bash
cd src
uvicorn api:app --reload
```
Wait to see: `Uvicorn running on http://127.0.0.1:8000`
(the first time it takes a bit while it downloads the models).

**Terminal 2 — consume the API (the client):**
```bash
cd src
python client.py
```

Or try the interactive page FastAPI generates automatically in your browser:
**http://127.0.0.1:8000/docs**

## Example responses

```
POST /analyze     {"text": "I loved this course"}
->  {"text": "...", "sentiment": "POSITIVE", "confidence": 0.9998}

POST /fill-mask   {"text": "Paris is the [MASK] of France."}
->  [{"token": "capital", "score": 0.9969}, {"token": "heart", "score": 0.0006}, ...]

POST /generate    {"text": "Once upon a time"}
->  {"generated_text": "Once upon a time, when she wasn't able to work full-time, ..."}
```

## The key idea

You did NOT train these models — you took them from Hugging Face. But serving them is
identical to serving your own model: each loads once at startup, and each request just
predicts. The server doesn't know (or care) where the models came from.

`pipeline("sentiment-analysis", ...)`, `pipeline("fill-mask", ...)` and
`pipeline("text-generation", ...)` each pick a different model class
(`AutoModelForSequenceClassification`, `AutoModelForMaskedLM`, `AutoModelForCausalLM`)
and a different pre/postprocessing routine — that's why the three endpoints above return
completely different response shapes even though they're all "just" calling `pipeline()`.
