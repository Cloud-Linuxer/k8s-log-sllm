# API Documentation

REST API for K8s/System Log Risk Scoring.

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health Check

Check if the service is healthy and model is loaded.

```
GET /health
```

**Response**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Single Prediction

Score a single log message.

```
POST /predict
Content-Type: application/json
```

**Request Body**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| log | string | yes | - | Log message to score |
| preprocess | boolean | no | true | Apply preprocessing (token replacement) |

**Example Request**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"log": "ERROR kubelet Failed to create pod sandbox"}'
```

**Response**
```json
{
  "risk_label": 7,
  "risk_score": 7.234,
  "level": "error"
}
```

**Response Fields**
| Field | Type | Description |
|-------|------|-------------|
| risk_label | integer | Risk level 0-10 (argmax) |
| risk_score | float | Expected risk score |
| level | string | Human-readable level name |

### Batch Prediction

Score multiple log messages at once.

```
POST /predict/batch
Content-Type: application/json
```

**Request Body**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| logs | array[string] | yes | - | List of log messages |
| preprocess | boolean | no | true | Apply preprocessing |

**Example Request**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      "INFO nginx GET /health 200",
      "ERROR connection refused",
      "CRITICAL kernel panic"
    ]
  }'
```

**Response**
```json
{
  "results": [
    {"risk_label": 2, "risk_score": 2.0, "level": "info"},
    {"risk_label": 6, "risk_score": 6.0, "level": "error"},
    {"risk_label": 10, "risk_score": 9.95, "level": "emergency"}
  ],
  "count": 3
}
```

## Risk Levels

| Level | Label | Description |
|-------|-------|-------------|
| trace | 0 | Trace-level logging |
| debug | 1 | Debug information |
| info | 2 | Normal operations |
| notice | 3 | Notable events |
| warning | 4-5 | Warning conditions |
| error | 6-7 | Error conditions |
| critical | 8-9 | Critical failures |
| emergency | 10 | System unusable |

## Error Responses

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "log"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 503 Service Unavailable
```json
{
  "detail": "Model not loaded"
}
```

## Rate Limits

No rate limits are enforced by default. Configure your reverse proxy (nginx, etc.) for rate limiting in production.

## Performance Tips

1. **Use batch endpoint** for multiple logs (more efficient)
2. **Disable preprocessing** if logs are already normalized
3. **Adjust threads** via `--threads` flag based on CPU cores

## Interactive Docs

Swagger UI available at:
```
http://localhost:8000/docs
```

ReDoc available at:
```
http://localhost:8000/redoc
```
