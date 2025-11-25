"""FastAPI server for log risk scoring."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.inference import LogRiskPredictor
from src.preprocessing import LogPreprocessor
from src.model import InferenceConfig
from src.api.alerter import LogAlerter, create_alerter_from_env


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
    enable_alerting: bool = True,
) -> FastAPI:
    """Create FastAPI application."""
    # App-local state to avoid global variable issues
    state = {"predictor": None, "preprocessor": None, "alerter": None}
    executor = ThreadPoolExecutor(max_workers=2)

    def get_predictor():
        if state["predictor"] is None:
            config = InferenceConfig(num_threads=num_threads)
            state["predictor"] = LogRiskPredictor(model_path, tokenizer_path, config)
            state["preprocessor"] = LogPreprocessor()
            if enable_alerting:
                state["alerter"] = create_alerter_from_env()
        return state["predictor"], state["preprocessor"]

    def send_alert_background(log: str, risk_label: int, risk_score: float, level: str):
        """Send alert in background thread."""
        alerter = state.get("alerter")
        if alerter and alerter.should_alert(risk_label):
            alerter.send_alert(log, risk_label, risk_score, level)

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
    async def predict_single(request: LogRequest, background_tasks: BackgroundTasks):
        """Score a single log message."""
        predictor, preprocessor = get_predictor()

        text = request.log
        if request.preprocess:
            processed = preprocessor.preprocess(text)
            text = processed["text"]

        result = predictor.predict_single(text)

        risk_label = result["risk_label"]
        risk_score = result.get("risk_score", float(risk_label))
        level = get_risk_level_name(risk_label)

        # Send alert in background if high risk
        if state.get("alerter"):
            background_tasks.add_task(
                send_alert_background, request.log, risk_label, risk_score, level
            )

        return LogResponse(
            risk_label=risk_label,
            risk_score=risk_score,
            level=level,
        )

    @app.post("/predict/batch", response_model=BatchLogResponse)
    async def predict_batch(request: BatchLogRequest, background_tasks: BackgroundTasks):
        """Score multiple log messages."""
        predictor, preprocessor = get_predictor()

        if not request.logs:
            return BatchLogResponse(results=[], count=0)

        texts = request.logs
        if request.preprocess:
            texts = [preprocessor.preprocess(log)["text"] for log in texts]

        results = predictor.predict(texts)

        responses = []
        for i, r in enumerate(results):
            risk_label = r["risk_label"]
            risk_score = r.get("risk_score", float(risk_label))
            level = get_risk_level_name(risk_label)

            # Send alert in background if high risk
            if state.get("alerter") and risk_label >= 7:
                background_tasks.add_task(
                    send_alert_background, request.logs[i], risk_label, risk_score, level
                )

            responses.append(LogResponse(
                risk_label=risk_label,
                risk_score=risk_score,
                level=level,
            ))

        return BatchLogResponse(results=responses, count=len(responses))

    return app


# Default app instance
app = create_app()
