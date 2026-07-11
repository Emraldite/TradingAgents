import json
from datetime import datetime, timedelta, timezone

from tradingagents.risk.release_gate import REQUIRED_RELEASE_CHECKS, build_release_report
from tradingagents.risk.scorecard import Scorecard
from tradingagents.state_store import StrategyStateStore


def test_release_report_remains_locked_without_evidence(tmp_path):
    report = build_release_report(
        store=StrategyStateStore(tmp_path / "state.db"),
        scorecard=Scorecard(tmp_path / "scorecard.db"),
        strategy_key="test",
        account_id="",
        output_path=tmp_path / "release.json",
    )

    assert report["approved"] is False
    assert report["checks"]["resolved_decisions"] is False
    assert report["checks"]["account_id_configured"] is False


def test_release_report_approves_only_complete_paper_evidence(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    scorecard = Scorecard(tmp_path / "scorecard.db")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(100):
        decision_id = scorecard.record_decision(
            strategy_key="test",
            ticker=f"T{index}",
            trade_date=(start + timedelta(days=index)).date().isoformat(),
            rating="Buy",
            model_provider="google",
            quick_model="quick",
            deep_model="deep",
            mode="paper",
            entry_price=100,
            final_trade_decision="Buy",
        )
        scorecard.record_outcome(
            int(decision_id),
            exit_price=103,
            return_pct=0.03,
            benchmark_return_pct=0.01,
            max_drawdown_pct=-0.01,
            stop_triggered=False,
        )
        cycle_id = store.start_cycle("paper", [f"T{index}"])
        store.complete_cycle(cycle_id, decisions=[], portfolio=[])

    with store._connect() as conn:
        conn.execute(
            "UPDATE strategy_cycles SET started_at=? WHERE cycle_id=1",
            (start.isoformat(),),
        )
        conn.execute(
            "UPDATE strategy_cycles SET completed_at=? WHERE cycle_id=100",
            ((start + timedelta(days=99)).isoformat(),),
        )

    backtest_report = tmp_path / "backtest.json"
    backtest_report.write_text(
        json.dumps(
            {
                "strategy_key": "test",
                "result": {
                    "ticker_count": 5,
                    "num_trades": 30,
                    "alpha_pct": 2.0,
                    "max_drawdown_pct": -5.0,
                    "start_date": "2020-01-01",
                    "end_date": "2025-01-01",
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_release_report(
        store=store,
        scorecard=scorecard,
        strategy_key="test",
        account_id="real-account",
        output_path=tmp_path / "release.json",
        backtest_validation_path=backtest_report,
    )

    assert report["approved"] is True
    assert all(report["checks"].values())
    assert set(report["checks"]) == REQUIRED_RELEASE_CHECKS
