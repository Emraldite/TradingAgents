from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from tradingagents.execution.alpaca_executor import ALPACA_BASE, _order_summary
from tradingagents.state_store import StrategyStateStore

logger = logging.getLogger(__name__)


class OrderUpdateMonitor:
    """Listen for Alpaca order events; REST reconciliation remains the fallback."""

    def __init__(self, store: StrategyStateStore, mode: str):
        self.store = store
        self.mode = mode
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=self.run_forever,
            name="alpaca-order-updates",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def run_forever(self) -> None:
        delay = 1
        while not self.stop_event.is_set():
            try:
                self._listen_once()
                delay = 1
            except Exception as exc:
                logger.error("Alpaca trade-update stream disconnected: %s", exc)
                self.store.record_health_event(
                    "error", "trade_updates", "stream disconnected", {"error": str(exc)}
                )
                self.stop_event.wait(delay)
                delay = min(delay * 2, 60)

    def _listen_once(self) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError("The installed runtime does not provide websocket support") from exc

        stream_url = ALPACA_BASE.replace("https://", "wss://").replace("http://", "ws://") + "/stream"
        with connect(stream_url, open_timeout=10, close_timeout=5) as websocket:
            websocket.send(
                json.dumps(
                    {
                        "action": "authenticate",
                        "data": {
                            "key_id": os.environ.get("ALPACA_API_KEY", ""),
                            "secret_key": os.environ.get("ALPACA_SECRET_KEY", ""),
                        },
                    }
                )
            )
            self._receive_json(websocket)
            websocket.send(
                json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}})
            )
            self._receive_json(websocket)
            self.store.record_health_event("info", "trade_updates", "stream connected")

            while not self.stop_event.is_set():
                try:
                    payload = self._receive_json(websocket)
                except TimeoutError:
                    # An idle stream is healthy; keep the authenticated socket open.
                    continue
                if payload.get("stream") != "trade_updates":
                    continue
                data = payload.get("data") or {}
                raw_order = data.get("order") or {}
                if not raw_order.get("id"):
                    continue
                order = _order_summary(raw_order)
                reason = f"trade_update:{data.get('event', 'unknown')}"
                self.store.record_order_tree(order, self.mode, reason=reason)

    @staticmethod
    def _receive_json(websocket: Any) -> dict[str, Any]:
        message = websocket.recv(timeout=30)
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        payload = json.loads(message)
        if isinstance(payload, list):
            return payload[0] if payload else {}
        return payload
