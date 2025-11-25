# Deployment Guide

Log risk scoring pipeline with Vector integration.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Log Sources   │────▶│     Vector      │────▶│  Elasticsearch  │
│  (K8s, Docker,  │     │  (Aggregator)   │     │   (Storage)     │
│   Syslog, File) │     └────────┬────────┘     └─────────────────┘
└─────────────────┘              │
                                 │ HTTP
                                 ▼
                       ┌─────────────────┐
                       │   Log Scorer    │
                       │   (FastAPI)     │
                       └────────┬────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
           ┌───────────────┐       ┌───────────────┐
           │  Webhook/Slack│       │   Kibana      │
           │   (Alerts)    │       │ (Dashboard)   │
           └───────────────┘       └───────────────┘
```

## Quick Start

### 1. Full Stack (with Elasticsearch + Kibana)

```bash
cd deploy

# Set webhook URL (optional)
export ALERT_WEBHOOK_URL="https://hooks.slack.com/services/xxx"

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

Services:
- Log Scorer API: http://localhost:8000
- Kibana: http://localhost:5601
- Elasticsearch: http://localhost:9200
- Vector metrics: http://localhost:9598
- Syslog: UDP 514
- Webhook receiver: http://localhost:9999 (demo)

### 2. Scorer Only (standalone)

```bash
# From project root
docker build -t k8s-log-scorer .
docker run -d -p 8000:8000 \
  -e ALERT_WEBHOOK_URL="https://hooks.slack.com/services/xxx" \
  -e ALERT_CHANNEL="slack" \
  -e ALERT_MIN_RISK_LEVEL="7" \
  k8s-log-scorer
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALERT_WEBHOOK_URL` | - | Webhook URL for alerts |
| `ALERT_CHANNEL` | webhook | Alert channel: slack, discord, pagerduty, webhook |
| `ALERT_MIN_RISK_LEVEL` | 7 | Minimum risk level to trigger alert (0-10) |
| `PAGERDUTY_ROUTING_KEY` | - | PagerDuty routing key (if using pagerduty) |

### Alert Channels

**Slack**
```bash
export ALERT_WEBHOOK_URL="https://hooks.slack.com/services/T00/B00/xxx"
export ALERT_CHANNEL="slack"
```

**Discord**
```bash
export ALERT_WEBHOOK_URL="https://discord.com/api/webhooks/xxx/yyy"
export ALERT_CHANNEL="discord"
```

**PagerDuty**
```bash
export ALERT_WEBHOOK_URL="https://events.pagerduty.com/v2/enqueue"
export ALERT_CHANNEL="pagerduty"
export PAGERDUTY_ROUTING_KEY="your-routing-key"
```

## Vector Configuration

### Full Pipeline (vector.toml)

Production config with:
- Multiple sources (K8s, Docker, Syslog, files)
- Risk scoring via HTTP
- Routing by risk level
- Elasticsearch output
- Webhook alerts
- Prometheus metrics

### Simple Pipeline (vector-simple.toml)

Testing config with:
- stdin input
- Keyword-based scoring (no API)
- Console output

Test simple config:
```bash
echo "ERROR kernel panic" | docker run -i --rm \
  -v $(pwd)/vector/vector-simple.toml:/etc/vector/vector.toml \
  timberio/vector:latest
```

## Sending Logs to Vector

### Via Syslog (UDP)
```bash
echo "<190>ERROR kernel panic - not syncing" | nc -u localhost 514
```

### Via Docker (auto-collected)
Logs from containers on the same host are auto-collected.

### Via File
Place logs in `/var/log/*.log` (mounted to Vector).

## Kibana Setup

1. Open http://localhost:5601
2. Go to Management → Stack Management → Index Patterns
3. Create index pattern: `logs-scored-*`
4. Set timestamp field: `timestamp`

### Useful Visualizations

**Risk Distribution**
- Type: Pie chart
- Metric: Count
- Split by: `risk_level` terms

**High Risk Timeline**
- Type: Line chart
- Metric: Count
- Filter: `risk_label >= 7`
- X-axis: `@timestamp` histogram

**Alert Rate**
- Type: Metric
- Metric: Count
- Filter: `risk_label >= 9`

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.override.yml
services:
  log-scorer:
    deploy:
      replicas: 3

  vector:
    environment:
      - VECTOR_THREADS=4
```

### Resource Limits

```yaml
services:
  log-scorer:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
```

## Monitoring

### Prometheus Metrics

Vector exposes metrics at `:9598/metrics`:
- `vector_events_processed_total`
- `vector_events_out_total`
- `vector_http_request_duration_seconds`

### Health Checks

```bash
# Log Scorer
curl http://localhost:8000/health

# Elasticsearch
curl http://localhost:9200/_cluster/health

# Vector (via metrics)
curl http://localhost:9598/health
```

## Troubleshooting

### Logs not being scored
```bash
# Check Vector logs
docker logs vector

# Check scorer health
curl http://localhost:8000/health

# Test scoring directly
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"log": "ERROR test"}'
```

### Alerts not sending
```bash
# Check alerter config
docker exec log-scorer env | grep ALERT

# Test webhook endpoint
curl -X POST $ALERT_WEBHOOK_URL \
  -H "Content-Type: application/json" \
  -d '{"text": "test"}'
```

### High memory usage
- Reduce batch size in Vector
- Limit Elasticsearch heap: `ES_JAVA_OPTS=-Xms512m -Xmx512m`
- Use fewer scorer threads: `--threads 2`
