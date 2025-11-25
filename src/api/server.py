"""FastAPI server for log risk scoring."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import LogRiskPredictor
from src.preprocessing import LogPreprocessor
from src.model import InferenceConfig


# Global instances
predictor: Optional[LogRiskPredictor] = None
preprocessor: Optional[LogPreprocessor] = None


class LogRequest(BaseModel):
    """Single log request."""
    log: str = Field(..., description="Log message to score")
    preprocess: bool = Field(True, description="Apply preprocessing")


class BatchLogRequest(BaseModel):
    """Batch log request."""
    logs: list[str] = Field(..., description="List of log messages")
    preprocess: bool = Field(True, description="Apply preprocessing")


class LogResponse(BaseModel):
    """Single log response."""
    risk_label: int = Field(..., description="Risk level 0-10")
    risk_score: float = Field(..., description="Expected risk score")
    level: str = Field(..., description="Risk level name")


class BatchLogResponse(BaseModel):
    """Batch log response."""
    results: list[LogResponse]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


RISK_LEVELS = {
    0: "trace",
    1: "debug",
    2: "info",
    3: "notice",
    4: "warning",
    5: "warning",
    6: "error",
    7: "error",
    8: "critical",
    9: "critical",
    10: "emergency",
}


def get_risk_level_name(label: int) -> str:
    """Convert risk label to level name."""
    return RISK_LEVELS.get(label, "unknown")


def create_app(
    model_path: str = "models/v2/model.onnx",
    tokenizer_path: str = "models/v2/tokenizer",
    num_threads: int = 4,
    lazy_load: bool = False,
) -> FastAPI:
    """Create FastAPI application."""
    # App-local state to avoid global variable issues
    state = {"predictor": None, "preprocessor": None}

    def get_predictor():
        if state["predictor"] is None:
            config = InferenceConfig(num_threads=num_threads)
            state["predictor"] = LogRiskPredictor(model_path, tokenizer_path, config)
            state["preprocessor"] = LogPreprocessor()
        return state["predictor"], state["preprocessor"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage model lifecycle."""
        if not lazy_load:
            get_predictor()
        yield
        state["predictor"] = None
        state["preprocessor"] = None

    app = FastAPI(
        title="K8s Log Risk Scorer",
        description="Risk scoring API for Kubernetes/System logs",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store getter in app state
    app.state.get_predictor = get_predictor

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if state["predictor"] else "unhealthy",
            model_loaded=state["predictor"] is not None,
        )

    @app.post("/predict", response_model=LogResponse)
    async def predict_single(request: LogRequest):
        """Score a single log message."""
        predictor, preprocessor = get_predictor()

        text = request.log
        if request.preprocess:
            processed = preprocessor.preprocess(text)
            text = processed["text"]

        result = predictor.predict_single(text)

        return LogResponse(
            risk_label=result["risk_label"],
            risk_score=result.get("risk_score", float(result["risk_label"])),
            level=get_risk_level_name(result["risk_label"]),
        )

    @app.post("/predict/batch", response_model=BatchLogResponse)
    async def predict_batch(request: BatchLogRequest):
        """Score multiple log messages."""
        predictor, preprocessor = get_predictor()

        if not request.logs:
            return BatchLogResponse(results=[], count=0)

        texts = request.logs
        if request.preprocess:
            texts = [preprocessor.preprocess(log)["text"] for log in texts]

        results = predictor.predict(texts)

        responses = [
            LogResponse(
                risk_label=r["risk_label"],
                risk_score=r.get("risk_score", float(r["risk_label"])),
                level=get_risk_level_name(r["risk_label"]),
            )
            for r in results
        ]

        return BatchLogResponse(results=responses, count=len(responses))

    return app


# Default app instance
app = create_app()
