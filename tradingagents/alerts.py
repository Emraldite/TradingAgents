from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)


class AlertManager:
    """Small webhook alert layer for critical bot events."""

    def __init__(
        self,
        discord_webhook_url: str | None = None,
        slack_webhook_url: str | None = None,
    ):
        self.discord_webhook_url = discord_webhook_url or os.environ.get(
            "DISCORD_WEBHOOK_URL"
        )
        self.slack_webhook_url = slack_webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

    def critical(self, title: str, message: str, context: dict[str, Any] | None = None) -> None:
        text = self._format(title, message, context)
        self._post_discord(text)
        self._post_slack(text)

    def _format(
        self,
        title: str,
        message: str,
        context: dict[str, Any] | None,
    ) -> str:
        lines = [f"**{title}**", message]
        if context:
            for key, value in context.items():
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _post_discord(self, text: str) -> None:
        if not self.discord_webhook_url:
            return
        try:
            response = requests.post(
                self.discord_webhook_url,
                json={"content": text},
                timeout=10,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.error("Failed to send Discord alert: %s", exc)

    def _post_slack(self, text: str) -> None:
        if not self.slack_webhook_url:
            return
        try:
            response = requests.post(
                self.slack_webhook_url,
                json={"text": text},
                timeout=10,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.error("Failed to send Slack alert: %s", exc)
