import sys
from types import SimpleNamespace

import pytest

from tradingagents.scheduler import runner


@pytest.fixture(autouse=True)
def isolate_scheduler_runtime(monkeypatch, tmp_path):
    monkeypatch.setitem(runner.DEFAULT_CONFIG, "data_cache_dir", str(tmp_path))


class _FakeJob:
    def __init__(self):
        self.weekday = None
        self.at_args = None

    @property
    def minutes(self):
        return self

    def __getattr__(self, name):
        if name in {"monday", "tuesday", "wednesday", "thursday", "friday"}:
            self.weekday = name
            return self
        raise AttributeError(name)

    def at(self, *args):
        self.at_args = args
        return self

    def do(self, callback):
        self.callback = callback
        return self


class _FakeScheduler:
    instances = []

    def __init__(self):
        self.jobs = []
        self.run_pending_calls = 0
        self.__class__.instances.append(self)

    def every(self, interval=None):
        self.interval = interval
        job = _FakeJob()
        self.jobs.append(job)
        return job

    def run_pending(self):
        self.run_pending_calls += 1
        raise KeyboardInterrupt


def test_start_scheduler_runs_once_immediately(monkeypatch):
    _FakeScheduler.instances.clear()
    monkeypatch.setitem(sys.modules, "schedule", SimpleNamespace(Scheduler=_FakeScheduler))
    cycles = []
    monkeypatch.setattr(runner, "run_cycle", lambda **kwargs: cycles.append(kwargs))

    with pytest.raises(KeyboardInterrupt):
        runner.start_scheduler(
            interval_minutes=60,
            mode="dry-run",
            tickers=["AAPL"],
            discover=True,
        )

    assert len(cycles) == 1
    assert cycles[0]["tickers"] == ["AAPL"]
    assert cycles[0]["discover"] is True
    assert _FakeScheduler.instances[0].interval == 60


def test_screener_ranking_rewards_names_present_in_multiple_lists(monkeypatch):
    monkeypatch.setattr(
        runner.executor,
        "get_stock_screener_checked",
        lambda top: (
            {
                "most_actives": [{"symbol": "AAPL"}, {"symbol": "PLTR"}],
                "gainers": [{"symbol": "PLTR", "percent_change": 8}],
                "losers": [
                    {"symbol": "QQQ", "percent_change": -20},
                    {"symbol": "XYZ", "percent_change": -12},
                ],
            },
            None,
        ),
    )
    monkeypatch.setattr(
        runner.executor,
        "get_discovery_assets_checked",
        lambda symbols: (
            {
                symbol: {
                    "tradable": True,
                    "marginable": True,
                    "fractionable": True,
                    "attributes": ["has_options"],
                    "exchange": "NASDAQ",
                    "name": "Invesco QQQ Trust" if symbol == "QQQ" else f"{symbol} Common Stock",
                }
                for symbol in symbols
            },
            None,
        ),
    )

    candidates = runner._screener_candidates()
    assert candidates[0] == "PLTR"
    assert "QQQ" not in candidates


def test_dynamic_universe_keeps_watchlist_and_caps_discovery(monkeypatch):
    monkeypatch.setattr(runner, "_screener_candidates", lambda: ["PLTR", "AMD", "META"])
    monkeypatch.setattr(runner, "_hard_exclusion_reason", lambda ticker: None)

    assert runner._build_dynamic_universe(["AAPL"], max_size=3) == [
        "AAPL", "PLTR", "AMD"
    ]


def test_dynamic_universe_falls_back_when_alpaca_has_no_candidates(monkeypatch):
    monkeypatch.setattr(runner, "_screener_candidates", lambda: [])
    monkeypatch.setattr(
        runner, "_technical_candidates", lambda max_candidates: ["PLTR", "AMD"]
    )
    monkeypatch.setattr(runner, "_hard_exclusion_reason", lambda ticker: None)

    assert runner._build_dynamic_universe([], max_size=1) == ["PLTR"]


def test_start_scheduler_can_wait_for_first_interval(monkeypatch):
    _FakeScheduler.instances.clear()
    monkeypatch.setitem(sys.modules, "schedule", SimpleNamespace(Scheduler=_FakeScheduler))
    cycles = []
    monkeypatch.setattr(runner, "run_cycle", lambda **kwargs: cycles.append(kwargs))

    with pytest.raises(KeyboardInterrupt):
        runner.start_scheduler(run_immediately=False)

    assert cycles == []
    scheduler = _FakeScheduler.instances[0]
    assert [job.weekday for job in scheduler.jobs] == [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
    ]
    assert all(job.at_args == ("08:45", "America/Chicago") for job in scheduler.jobs)


def test_start_scheduler_rejects_invalid_interval():
    with pytest.raises(ValueError, match="at least 1"):
        runner.start_scheduler(interval_minutes=0)


def test_start_scheduler_rejects_invalid_daily_time():
    with pytest.raises(ValueError, match="HH:MM"):
        runner.start_scheduler(daily_at="tomorrow morning")
