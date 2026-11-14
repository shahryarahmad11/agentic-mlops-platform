"""FastAPI inference service deployed on Kubernetes (Task 4).

Configuration comes from environment variables supplied by a ConfigMap. The model is loaded
once at startup. If loading fails the process stays alive (liveness passes) but readiness
fails, so Kubernetes stops routing traffic to the Pod instead of serving bad answers.
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_VERSION = os.getenv("APP_VERSION", "dev")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/model/model.joblib"))
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", "/app/model/metadata.json"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
FEATURES = [f"feature_{i}" for i in range(5)]

logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL, format="%(message)s")
log = logging.getLogger("inference")


class _SkipProbes(logging.Filter):
    """Probe requests every few seconds would bury real errors in the access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(p in record.getMessage() for p in ("/healthz", "/readyz"))


logging.getLogger("uvicorn.access").addFilter(_SkipProbes())

STATE: dict[str, Any] = {"model": None, "metadata": {}, "load_error": None, "started": time.time()}


def _log(event: str, level: int = logging.INFO, **fields: Any) -> None:
    log.log(level, json.dumps({"event": event, "app_version": APP_VERSION, **fields}))


def load_model() -> None:
    try:
        STATE["model"] = joblib.load(MODEL_PATH)
        STATE["metadata"] = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.is_file() else {}
        STATE["load_error"] = None
        _log("model_loaded", model_path=str(MODEL_PATH), **STATE["metadata"])
    except Exception as exc:  # noqa: BLE001 - report any load failure via readiness
        STATE["model"] = None
        STATE["load_error"] = f"{type(exc).__name__}: {exc}"
        _log("model_load_failed", logging.ERROR, model_path=str(MODEL_PATH), error=str(exc))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Prediction API", version=APP_VERSION, lifespan=lifespan)


class PredictRequest(BaseModel):
    instances: list[list[float]] = Field(min_length=1, max_length=100)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process is up and serving HTTP."""
    return {"status": "alive"}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    """Readiness: only ready when a model is loaded."""
    if STATE["model"] is None:
        raise HTTPException(503, detail={"status": "not_ready", "reason": STATE["load_error"]})
    return {"status": "ready"}


@app.get("/version")
def version() -> dict[str, Any]:
    return {
        "app_version": APP_VERSION,
        "model_loaded": STATE["model"] is not None,
        "model_path": str(MODEL_PATH),
        "model": STATE["metadata"],
        "load_error": STATE["load_error"],
        "uptime_seconds": round(time.time() - STATE["started"], 1),
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    model = STATE["model"]
    if model is None:
        raise HTTPException(503, detail="model not loaded")
    bad = [i for i, row in enumerate(request.instances) if len(row) != len(FEATURES)]
    if bad:
        raise HTTPException(422, detail=f"rows {bad} must have {len(FEATURES)} features")
    frame = pd.DataFrame(request.instances, columns=FEATURES)
    predictions = model.predict(frame).tolist()
    _log("prediction", rows=len(predictions))
    return {"predictions": predictions, "model_run_id": STATE["metadata"].get("mlflow_run_id")}
