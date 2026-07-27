"""FastAPI serving layer for CrossModalFusionNet."""
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from predict import CLASS_NAMES, load_model, predict

MAX_BYTES = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()          # warm the model at startup, not on first request
    yield


app = FastAPI(
    title="CrossModalFusionNet",
    description="Multi-modal image manipulation detection.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASS_NAMES}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    raw = await file.read()

    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Not a valid image file.")

    scores = predict(image)
    top = max(scores, key=scores.get)
    return {
        "prediction": top,
        "confidence": round(scores[top], 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }