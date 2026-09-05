"""API serving three Hugging Face models via different pipeline tasks."""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from model import load_classifier, load_generator, load_mask_filler

app = FastAPI(
    title="Sentiment API (Hugging Face)",
    description=(
        "Analyzes whether a text is **positive** or **negative** using a "
        "pretrained DistilBERT model (SST-2, English only). Also demos two "
        "more pipeline tasks (fill-mask, text-generation), each with a "
        "different model and a completely different output shape."
    ),
    version="1.0.0",
)

# All three models are loaded ONCE, when the API starts (not on every
# request). Loading three models eagerly means a slower startup and more
# RAM than the original single-model demo — fine here since they're all
# small CPU models, but worth knowing before adding a fourth.
classifier = load_classifier()
mask_filler = load_mask_filler()
generator = load_generator()


class Text(BaseModel):
    text: str = Field(..., examples=["I loved this course"])


class Status(BaseModel):
    status: str
    model: str


class Sentiment(BaseModel):
    text: str
    sentiment: str = Field(..., description="POSITIVE or NEGATIVE")
    confidence: float = Field(..., description="Model confidence, 0-1")


class MaskedText(BaseModel):
    text: str = Field(..., examples=["Paris is the [MASK] of France."])


class MaskPrediction(BaseModel):
    token: str = Field(..., description="Candidate word for [MASK]")
    score: float = Field(..., description="Model confidence, 0-1")


class Generation(BaseModel):
    generated_text: str


@app.get("/", tags=["health"], summary="Health check", response_model=Status)
def root():
    """Returns API status and which models are currently loaded."""
    return {
        "status": "alive",
        "model": "sentiment-analysis (DistilBERT) + fill-mask (BERT) + text-generation (GPT-2)",
    }


@app.post("/analyze", tags=["sentiment"], summary="Analyze sentiment", response_model=Sentiment)
def analyze(entry: Text):
    """Classifies a text as POSITIVE or NEGATIVE with a confidence score.

    Note: the underlying model (SST-2) was trained on English text only;
    predictions on other languages are not reliable.
    """
    # the model returns a list with one dict: [{'label': ..., 'score': ...}]
    result = classifier(entry.text)[0]
    return {
        "text": entry.text,
        "sentiment": result["label"],   # POSITIVE or NEGATIVE
        "confidence": round(float(result["score"]), 4),
    }


@app.post(
    "/fill-mask",
    tags=["fill-mask"],
    summary="Fill in a masked word",
    response_model=list[MaskPrediction],
)
def fill_mask(entry: MaskedText):
    """Predicts the top candidate words for a `[MASK]` token in the text.

    The input MUST contain the literal token `[MASK]` (BERT's mask token),
    e.g. `"Paris is the [MASK] of France."`. Unlike `/analyze`, which
    returns a single label + score, this returns a ranked list of
    candidate words — a different output shape entirely, because
    fill-mask is a different pipeline task with its own postprocessing.
    """
    # the model returns a list of dicts: [{'token_str': ..., 'score': ...}, ...]
    predictions = mask_filler(entry.text)
    return [
        {"token": p["token_str"], "score": round(float(p["score"]), 4)}
        for p in predictions
    ]


@app.post(
    "/generate",
    tags=["generation"],
    summary="Generate text",
    response_model=Generation,
)
def generate(entry: Text):
    """Continues a text with up to 40 new generated tokens.

    A third output shape: free-form generated text instead of a label
    (`/analyze`) or a list of candidate words (`/fill-mask`). GPT-2 is a
    causal language model — it has no notion of "correct" continuation,
    it just predicts the statistically likely next tokens, one at a time.
    """
    # the model returns a list of dicts: [{'generated_text': "..."}]
    result = generator(entry.text, max_new_tokens=40, num_return_sequences=1)[0]
    return {"generated_text": result["generated_text"]}
