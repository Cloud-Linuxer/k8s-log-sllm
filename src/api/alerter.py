"""Webhook alerter for high-risk logs."""

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import urllib.request
import urllib.error


class AlertChannel(Enum):
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"  # Generic webhook
    PAGERDUTY = "pagerduty"


@dataclass
class AlertConfig:
    """Alert configuration."""
    channel: AlertChannel
    webhook_url: str
    min_risk_level: int = 7  # Minimum risk to trigger alert
    include_probabilities: bool = False


class LogAlerter:
    """Send alerts for high-risk logs."""

    def __init__(self, config: AlertConfig):
        self.config = config

    def should_alert(self, risk_label: int) -> bool:
        """Check if log should trigger alert."""
        return risk_label >= self.config.min_risk_level

    def send_alert(
        self,
        log_message: str,
        risk_label: int,
        risk_score: float,
        risk_level: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Send alert to configured channel."""
        if not self.should_alert(risk_label):
            return False

        payload = self._build_payload(
            log_message, risk_label, risk_score, risk_level, metadata
        )

        try:
            self._send_webhook(payload)
            return True
        except Exception as e:
            print(f"Alert failed: {e}")
            return False

    def _build_payload(
        self,
        log_message: str,
        risk_label: int,
        risk_score: float,
        risk_level: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Build payload for specific channel."""

        # Truncate long messages
        truncated = log_message[:500] + "..." if len(log_message) > 500 else log_message

        # Emoji based on severity
        emoji = "🚨" if risk_label >= 9 else "⚠️" if risk_label >= 7 else "📋"

        # Color based on severity
        color = "#FF0000" if risk_label >= 9 else "#FFA500" if risk_label >= 7 else "#FFFF00"

        if self.config.channel == AlertChannel.SLACK:
            return {
                "attachments": [{
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{emoji} High Risk Log Detected",
                            }
                        },
                        {
                            "type": "section",
                            "fields": [
                                {"type": "mrkdwn", "text": f"*Risk Level:*\n{risk_level.upper()}"},
                                {"type": "mrkdwn", "text": f"*Risk Score:*\n{risk_label} ({risk_score:.2f})"},
                            ]
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Log Message:*\n```{truncated}```"
                            }
                        },
                    ]
                }]
            }

        elif self.config.channel == AlertChannel.DISCORD:
            return {
                "embeds": [{
                    "title": f"{emoji} High Risk Log Detected",
                    "color": int(color.replace("#", ""), 16),
                    "fields": [
                        {"name": "Risk Level", "value": risk_level.upper(), "inline": True},
                        {"name": "Risk Score", "value": f"{risk_label} ({risk_score:.2f})", "inline": True},
                        {"name": "Log Message", "value": f"```{truncated}```", "inline": False},
                    ]
                }]
            }

        elif self.config.channel == AlertChannel.PAGERDUTY:
            return {
                "routing_key": os.getenv("PAGERDUTY_ROUTING_KEY", ""),
                "event_action": "trigger",
                "payload": {
                    "summary": f"High Risk Log: {risk_level} (score: {risk_label})",
                    "severity": "critical" if risk_label >= 9 else "error",
                    "source": "k8s-log-scorer",
                    "custom_details": {
                        "risk_label": risk_label,
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "message": truncated,
                        **(metadata or {}),
                    }
                }
            }

        else:  # Generic webhook
            return {
                "alert_type": "high_risk_log",
                "risk_label": risk_label,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "message": truncated,
                "metadata": metadata or {},
            }

    def _send_webhook(self, payload: dict) -> None:
        """Send webhook request."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.config.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status >= 400:
                raise urllib.error.HTTPError(
                    self.config.webhook_url,
                    response.status,
                    "Webhook request failed",
                    response.headers,
                    None,
                )


def create_alerter_from_env() -> Optional[LogAlerter]:
    """Create alerter from environment variables."""
    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if not webhook_url:
        return None

    channel_str = os.getenv("ALERT_CHANNEL", "webhook").lower()
    try:
        channel = AlertChannel(channel_str)
    except ValueError:
        channel = AlertChannel.WEBHOOK

    min_level = int(os.getenv("ALERT_MIN_RISK_LEVEL", "7"))

    config = AlertConfig(
        channel=channel,
        webhook_url=webhook_url,
        min_risk_level=min_level,
    )

    return LogAlerter(config)
