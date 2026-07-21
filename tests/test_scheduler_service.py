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


def test_dynamic_universe_keeps_watchlist_and_adds_candidates(monkeypatch):
    monkeypatch.setattr(runner, "_technical_candidates", lambda: ["NVDA", "AMD"])
    monkeypatch.setattr(
        runner, "_sector_expansion_candidates", lambda tickers: ["META"]
    )
    monkeypatch.setattr(runner, "_hard_exclusion_reason", lambda ticker: None)

    assert runner._build_dynamic_universe(["AAPL", "MSFT", "NVDA"]) == [
        "AAPL", "MSFT", "NVDA", "AMD", "META"
    ]


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
