"""Loading the Hugging Face models.

Models are downloaded automatically the FIRST time they run (they get
cached in ~/.cache/huggingface). Subsequent runs load them from disk.
"""
from transformers import pipeline

# Lightweight sentiment analysis model (DistilBERT).
# Task: text-classification. Runs on CPU, no GPU needed. ~250 MB the first time.
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# BERT base, used here for mask filling instead of classification.
# Task: fill-mask — a completely different output shape (top-k candidate
# words instead of a single label + score). Uses the literal "[MASK]" token.
MASK_MODEL = "bert-base-uncased"

# GPT-2, a causal language model.
# Task: text-generation — yet another output shape: free-form continuation
# text instead of a label or a list of candidate words.
GENERATION_MODEL = "gpt2"


def load_classifier():
    """Returns the sentiment pipeline, ready to use."""
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)


def load_mask_filler():
    """Returns the fill-mask pipeline, ready to use."""
    return pipeline("fill-mask", model=MASK_MODEL)


def load_generator():
    """Returns the text-generation pipeline, ready to use."""
    return pipeline("text-generation", model=GENERATION_MODEL)
