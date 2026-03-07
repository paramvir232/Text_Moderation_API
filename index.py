from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

moderator = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None
)

THRESHOLDS = {
    "toxic": 0.50,
    "severe_toxic": 0.40,
    "obscene": 0.50,
    "threat": 0.30,
    "insult": 0.65,
    "identity_hate": 0.40
}

@app.get("/moderate")
def moderate(text: str):

    results = moderator(text)[0]

    scores = {item["label"]: round(float(item["score"]), 2) for item in results}

    safe = True

    for key, threshold in THRESHOLDS.items():
        if scores.get(key, 0) >= threshold:
            safe = False
            break

    return {
        "safe": safe,
        "scores": scores
    }