"""Smoke test: hits the three live endpoints and checks response shape.

Run with the API already up (see README). Exits non-zero on any failure,
so CI can treat this as a pass/fail gate.
"""
import sys
import time

import requests

BASE_URL = "http://127.0.0.1:8000"


def wait_until_ready(timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{BASE_URL}/", timeout=2).status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise SystemExit(f"API did not become ready within {timeout}s")


def check(condition, message):
    if not condition:
        print(f"FAIL: {message}")
        sys.exit(1)
    print(f"ok: {message}")


def main():
    wait_until_ready()

    r = requests.post(f"{BASE_URL}/analyze", json={"text": "I loved this course"})
    check(r.status_code == 200, "/analyze returns 200")
    body = r.json()
    check(body["sentiment"] in ("POSITIVE", "NEGATIVE"), "/analyze sentiment is a valid label")
    check(0.0 <= body["confidence"] <= 1.0, "/analyze confidence is in [0, 1]")

    r = requests.post(f"{BASE_URL}/fill-mask", json={"text": "Paris is the [MASK] of France."})
    check(r.status_code == 200, "/fill-mask returns 200")
    predictions = r.json()
    check(isinstance(predictions, list) and len(predictions) > 0, "/fill-mask returns a non-empty list")
    check("token" in predictions[0] and "score" in predictions[0], "/fill-mask items have token + score")

    r = requests.post(f"{BASE_URL}/generate", json={"text": "Once upon a time"})
    check(r.status_code == 200, "/generate returns 200")
    check(len(r.json()["generated_text"]) > 0, "/generate returns non-empty text")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
