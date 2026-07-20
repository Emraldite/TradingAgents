import pandas as pd
import pytest

from tradingagents.risk.scorecard import ScorecardGate
from tradingagents.scheduler import runner
from tradingagents.state_store import StrategyStateStore


@pytest.fixture(autouse=True)
def free_gemini_config(monkeypatch):
    monkeypatch.setitem(runner.DEFAULT_CONFIG, "llm_provider", "google")
    monkeypatch.setitem(runner.DEFAULT_CONFIG, "quick_think_llm", "gemini-3.1-flash-lite")
    monkeypatch.setitem(runner.DEFAULT_CONFIG, "deep_think_llm", "gemini-3.5-flash")


class FakeScorecard:
    def __init__(self, allowed_position_pct: float = 0.005, status: str = "warming_up"):
        self.gate = ScorecardGate(
            allowed_position_pct=allowed_position_pct,
            status=status,
            reason=status,
            resolved_decisions=0,
            win_rate_pct=0.0,
            avg_alpha_pct=0.0,
            max_drawdown_pct=0.0,
        )

    def resolve_due_outcomes(self):
        return 0

    def record_decision(self, **kwargs):
        return 1

    def gate_for_strategy(self, strategy_key):
        return self.gate


def _price_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "Volume": [1_000_000, 1_100_000],
        }
    )
    frame.index = pd.to_datetime([pd.Timestamp.now().date(), pd.Timestamp.now().date()])
    return frame


def test_run_cycle_skips_when_graph_rating_is_hold(monkeypatch):
    monkeypatch.setattr(runner, "_create_analysis_graph", lambda: object())
    monkeypatch.setattr(runner, "_create_scorecard", lambda: FakeScorecard())
    monkeypatch.setattr(
        runner,
        "_run_graph_analysis",
        lambda graph, ticker, trade_date: ("Hold", {"final_trade_decision": "**Rating**: Hold"}),
    )
    monkeypatch.setattr(
        runner,
        "detect_manipulation",
        lambda ticker: {"recommendation": "ALLOW"},
    )
    monkeypatch.setattr(runner.yf, "download", lambda *args, **kwargs: _price_frame())
    monkeypatch.setattr(runner.executor, "get_account_info", lambda: None)
    monkeypatch.setattr(runner.executor, "get_portfolio", lambda: [])
    monkeypatch.setattr(runner, "validate_trade", lambda **kwargs: (True, "ok"))

    summary = runner.run_cycle(tickers=["AAPL"], dry_run=True)

    assert summary["signals"] == 0
    assert summary["simulated"] == 0
    assert summary["decisions"][0]["ticker"] == "AAPL"
    assert summary["decisions"][0]["reason"] == "graph_rating=Hold"


def test_run_cycle_uses_graph_buy_rating_for_entry(monkeypatch):
    monkeypatch.setattr(runner, "_create_analysis_graph", lambda: object())
    monkeypatch.setattr(runner, "_create_scorecard", lambda: FakeScorecard(allowed_position_pct=0.02))
    monkeypatch.setattr(
        runner,
        "_run_graph_analysis",
        lambda graph, ticker, trade_date: ("Buy", {"final_trade_decision": "**Rating**: Buy"}),
    )
    monkeypatch.setattr(
        runner,
        "detect_manipulation",
        lambda ticker: {"recommendation": "ALLOW"},
    )
    monkeypatch.setattr(runner.yf, "download", lambda *args, **kwargs: _price_frame())
    monkeypatch.setattr(runner.executor, "get_account_info", lambda: None)
    monkeypatch.setattr(runner.executor, "get_portfolio", lambda: [])
    monkeypatch.setattr(runner, "validate_trade", lambda **kwargs: (True, "ok"))

    summary = runner.run_cycle(tickers=["AAPL"], dry_run=True)

    assert summary["signals"] == 1
    assert summary["simulated"] == 1
    assert summary["decisions"][0]["decision"] == "buy-signal"
    assert summary["decisions"][0]["rating"] == "Buy"
    assert summary["decisions"][0]["notional"] == 100.0


def test_run_cycle_applies_scorecard_warmup_cap(monkeypatch):
    monkeypatch.setattr(runner, "_create_analysis_graph", lambda: object())
    monkeypatch.setattr(runner, "_create_scorecard", lambda: FakeScorecard(allowed_position_pct=0.005))
    monkeypatch.setattr(
        runner,
        "_run_graph_analysis",
        lambda graph, ticker, trade_date: ("Buy", {"final_trade_decision": "**Rating**: Buy"}),
    )
    monkeypatch.setattr(
        runner,
        "detect_manipulation",
        lambda ticker: {"recommendation": "ALLOW"},
    )
    monkeypatch.setattr(runner.yf, "download", lambda *args, **kwargs: _price_frame())
    monkeypatch.setattr(runner.executor, "get_account_info", lambda: None)
    monkeypatch.setattr(runner.executor, "get_portfolio", lambda: [])
    monkeypatch.setattr(runner, "validate_trade", lambda **kwargs: (True, "ok"))

    summary = runner.run_cycle(tickers=["AAPL"], dry_run=True)

    assert summary["signals"] == 1
    assert summary["simulated"] == 1
    assert summary["decisions"][0]["notional"] == 50.0
    assert summary["decisions"][0]["position_pct"] == 0.005


def test_graph_provider_failure_marks_cycle_failed_instead_of_complete(
    monkeypatch, tmp_path
):
    store = StrategyStateStore(tmp_path / "state.db")
    monkeypatch.setattr(runner, "state_store", store)
    monkeypatch.setattr(runner, "_create_analysis_graph", lambda: object())
    monkeypatch.setattr(runner, "_create_scorecard", lambda: FakeScorecard())
    monkeypatch.setattr(
        runner,
        "_run_graph_analysis",
        lambda *args: (_ for _ in ()).throw(
            RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
        ),
    )
    monkeypatch.setattr(runner, "detect_manipulation", lambda ticker: {"recommendation": "ALLOW"})
    monkeypatch.setattr(runner.yf, "download", lambda *args, **kwargs: _price_frame())
    monkeypatch.setattr(runner.executor, "get_account_info", lambda: None)
    monkeypatch.setattr(runner.executor, "get_portfolio", lambda: [])

    summary = runner.run_cycle(tickers=["AAPL"], mode="dry-run")

    assert summary["status"] == "failed"
    assert summary["analysis_failures"] == 1
    assert summary["submitted"] == 0
    assert summary["executed"] == 0
    assert summary["simulated"] == 0
    assert "model access and quota" in summary["reason"]
    assert summary["decisions"][0]["decision"] == "skip"
    assert store.health_snapshot()["recent_events"][0]["component"] == "llm_graph"


def test_unapproved_model_fails_before_data_or_graph(monkeypatch, tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    monkeypatch.setattr(runner, "state_store", store)
    monkeypatch.setitem(runner.DEFAULT_CONFIG, "llm_provider", "google")
    monkeypatch.setitem(runner.DEFAULT_CONFIG, "deep_think_llm", "gemini-3.1-pro-preview")
    monkeypatch.setattr(
        runner,
        "_create_analysis_graph",
        lambda: pytest.fail("graph should not be created"),
    )

    summary = runner.run_cycle(tickers=["AAPL"], mode="dry-run")

    assert summary["status"] == "failed"
    assert summary["submitted"] == 0
    assert summary["executed"] == 0
    assert summary["simulated"] == 0
    assert "Free-only mode rejected" in summary["reason"]
    assert store.health_snapshot()["recent_events"][0]["component"] == "llm_config"


def test_paper_cycle_prices_from_iex_and_does_not_count_unfilled_as_execution(
    monkeypatch, tmp_path
):
    store = StrategyStateStore(tmp_path / "state.db")
    captured = {}
    monkeypatch.setattr(runner, "state_store", store)
    monkeypatch.setattr(runner, "_create_analysis_graph", lambda: object())
    monkeypatch.setattr(
        runner, "_create_scorecard", lambda: FakeScorecard(allowed_position_pct=0.02)
    )
    monkeypatch.setattr(
        runner,
        "_run_graph_analysis",
        lambda graph, ticker, trade_date: (
            "Buy",
            {"final_trade_decision": "**Rating**: Buy"},
        ),
    )
    monkeypatch.setattr(runner, "detect_manipulation", lambda ticker: {"recommendation": "ALLOW"})
    monkeypatch.setattr(runner.yf, "download", lambda *args, **kwargs: _price_frame())
    monkeypatch.setattr(runner, "validate_trade", lambda **kwargs: (True, "ok"))
    monkeypatch.setattr(
        runner.executor,
        "require_market_open",
        lambda: (True, {"is_open": True}, None),
    )
    monkeypatch.setattr(runner.executor, "get_portfolio_checked", lambda: ([], None))
    monkeypatch.setattr(runner.executor, "get_recent_orders", lambda **kwargs: [])
    monkeypatch.setattr(
        runner.executor, "get_account_activities_checked", lambda: ([], None)
    )
    monkeypatch.setattr(
        runner.executor,
        "get_account_info",
        lambda: {
            "account_id": "paper-account",
            "cash": 10_000,
            "portfolio_value": 10_000,
            "buying_power": 10_000,
            "account_blocked": False,
            "trading_blocked": False,
            "trade_suspended_by_user": False,
        },
    )
    monkeypatch.setattr(runner.executor, "is_official_endpoint", lambda: True)
    monkeypatch.setattr(runner.executor, "is_paper_endpoint", lambda: True)
    monkeypatch.setattr(
        runner.executor,
        "get_stock_snapshot_checked",
        lambda ticker: (
            {
                "bid": 100.0,
                "ask": 100.2,
                "last": 100.1,
                "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            },
            None,
        ),
    )

    def submit_bracket(ticker, notional, limit_price, stop_price, target_price, client_id):
        captured.update(
            limit_price=limit_price,
            stop_price=stop_price,
            target_price=target_price,
        )
        return {
            "order_id": "paper-buy-1",
            "client_order_id": client_id,
            "ticker": ticker,
            "side": "buy",
            "type": "limit",
            "order_class": "bracket",
            "status": "accepted",
            "notional": notional,
            "qty": 0,
            "filled_qty": 0,
            "filled_avg_price": 0,
            "legs": [],
        }

    monkeypatch.setattr(runner.executor, "execute_buy_bracket", submit_bracket)

    summary = runner.run_cycle(tickers=["AAPL"], mode="paper")

    assert summary["status"] == "complete"
    assert summary["submitted"] == 1
    assert summary["executed"] == 0
    assert captured["limit_price"] > 100.2
    assert store.open_positions() == []
    assert store.reserved_open_position_count() == 1
