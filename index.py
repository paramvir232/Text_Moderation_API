from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import os

app = FastAPI(title="Lightweight Toxicity Moderation API")

# Load model once (startup optimization)
moderator = pipeline(
    "text-classification",
    model="minuva/MiniLMv2-toxic-jigsaw",
    truncation=True
)

# Moderation threshold
TOXIC_THRESHOLD = 0.60


class TextRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: list[str]


@app.get("/")
def home():
    return {"status": "running", "service": "toxicity-moderation-api"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/moderate")
def moderate(data: TextRequest):

    result = moderator(data.text)[0]

    label = result["label"].lower()
    score = float(result["score"])

    safe = True
    if label == "toxic" and score >= TOXIC_THRESHOLD:
        safe = False

    return {
        "text": data.text,
        "safe": safe,
        "label": label,
        "confidence": round(score, 3)
    }


@app.post("/moderate-batch")
def moderate_batch(data: BatchRequest):

    results = moderator(data.texts)

    output = []

    for text, res in zip(data.texts, results):

        label = res[0]["label"].lower()
        score = float(res[0]["score"])

        safe = True
        if label == "toxic" and score >= TOXIC_THRESHOLD:
            safe = False

        output.append({
            "text": text,
            "safe": safe,
            "label": label,
            "confidence": round(score, 3)
        })

    return {"results": output}