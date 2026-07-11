import json

import pytest

from tradingagents.execution.order_monitor import OrderUpdateMonitor
from tradingagents.runtime import SingleInstanceLock


class _Socket:
    def __init__(self, payload):
        self.payload = payload

    def recv(self, timeout=None):
        return self.payload


def test_single_instance_lock_rejects_second_process(tmp_path):
    path = tmp_path / "bot.lock"
    first = SingleInstanceLock(path)
    second = SingleInstanceLock(path)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="Another trading bot"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_single_instance_lock_recovers_invalid_stale_file(tmp_path):
    path = tmp_path / "bot.lock"
    path.write_text("not-json", encoding="utf-8")

    lock = SingleInstanceLock(path)
    lock.acquire()

    assert json.loads(path.read_text(encoding="utf-8"))["pid"] > 0
    lock.release()


def test_order_monitor_accepts_binary_json_frames():
    payload = OrderUpdateMonitor._receive_json(
        _Socket(b'{"stream":"trade_updates","data":{"event":"fill"}}')
    )

    assert payload["data"]["event"] == "fill"


def test_order_monitor_reconnects_after_disconnect(monkeypatch):
    events = []

    class Store:
        def record_health_event(self, severity, component, message, context=None):
            events.append((severity, component, message, context))

    monitor = OrderUpdateMonitor(Store(), "paper")
    calls = 0

    def listen_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("network down")
        monitor.stop_event.set()

    monkeypatch.setattr(monitor, "_listen_once", listen_once)
    monkeypatch.setattr(monitor.stop_event, "wait", lambda delay: False)

    monitor.run_forever()

    assert calls == 2
    assert events[0][0:3] == ("error", "trade_updates", "stream disconnected")
