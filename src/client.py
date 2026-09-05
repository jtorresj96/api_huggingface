"""Client: consumes the API (sentiment + fill-mask + text-generation) with requests.

Run this in ANOTHER terminal, with the API already running.
"""
import requests

BASE_URL = "http://127.0.0.1:8000"

print("=== /analyze (sentiment-analysis) ===")
phrases = [
    "I loved this course, I learned so much",
    "The service was slow and the food was cold",
    "The movie was absolutely fantastic",
]
for phrase in phrases:
    response = requests.post(f"{BASE_URL}/analyze", json={"text": phrase})
    data = response.json()
    print(f"[{response.status_code}] '{phrase}'")
    print(f"     -> {data['sentiment']} (confidence: {data['confidence']})\n")

print("=== /fill-mask (fill-mask) ===")
masked_phrases = [
    "Paris is the [MASK] of France.",
    "The [MASK] barked at the mailman.",
]
for phrase in masked_phrases:
    response = requests.post(f"{BASE_URL}/fill-mask", json={"text": phrase})
    predictions = response.json()
    print(f"[{response.status_code}] '{phrase}'")
    for p in predictions[:3]:
        print(f"     -> {p['token']} (confidence: {p['score']})")
    print()

print("=== /generate (text-generation) ===")
prompts = [
    "Once upon a time",
    "The best way to learn programming is",
]
for prompt in prompts:
    response = requests.post(f"{BASE_URL}/generate", json={"text": prompt})
    data = response.json()
    print(f"[{response.status_code}] '{prompt}'")
    print(f"     -> {data['generated_text']!r}\n")
