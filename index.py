from fastapi import FastAPI
from detoxify import Detoxify
from fastapi.responses import JSONResponse

app = FastAPI()

model = Detoxify("original")

THRESHOLDS = {
    "toxicity": 0.60,
    "severe_toxicity": 0.40,
    "obscene": 0.50,
    "threat": 0.30,
    "insult": 0.65,
    "identity_attack": 0.40
}

@app.get("/moderate")
def moderate(text: str):

    scores = model.predict(text)

    # convert numpy → float
    scores = {k: round(float(v), 2) for k, v in scores.items()}

    safe = True

    for key, threshold in THRESHOLDS.items():
        if scores.get(key, 0) >= threshold:
            safe = False
            break

    return JSONResponse(content={
        "safe": safe,
        "scores": scores
    })