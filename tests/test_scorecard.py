import pytest

from tradingagents.risk.scorecard import Scorecard


def _decision(scorecard: Scorecard, ticker: str = "AAPL", rating: str = "Buy") -> int:
    decision_id = scorecard.record_decision(
        strategy_key="test",
        ticker=ticker,
        trade_date="2025-01-01",
        rating=rating,
        model_provider="openai",
        quick_model="quick",
        deep_model="deep",
        mode="dry-run",
        entry_price=100.0,
        final_trade_decision="Buy",
    )
    assert decision_id is not None
    return decision_id


def test_scorecard_warmup_caps_at_half_percent(tmp_path):
    scorecard = Scorecard(tmp_path / "scorecard.db")

    gate = scorecard.gate_for_strategy("test")

    assert gate.status == "warming_up"
    assert gate.allowed_position_pct == 0.005


def test_scorecard_blocks_bad_alpha_after_warmup(tmp_path):
    scorecard = Scorecard(tmp_path / "scorecard.db")
    for index in range(30):
        decision_id = _decision(scorecard, ticker=f"BAD{index}")
        scorecard.record_outcome(
            decision_id,
            exit_price=98.0,
            return_pct=-0.02,
            benchmark_return_pct=0.01,
            max_drawdown_pct=-0.06,
            stop_triggered=True,
        )

    gate = scorecard.gate_for_strategy("test")

    assert gate.status == "blocked"
    assert gate.allowed_position_pct == 0.0


def test_scorecard_allows_one_percent_after_positive_alpha(tmp_path):
    scorecard = Scorecard(tmp_path / "scorecard.db")
    for index in range(30):
        decision_id = _decision(scorecard, ticker=f"GOOD{index}")
        scorecard.record_outcome(
            decision_id,
            exit_price=103.0,
            return_pct=0.03,
            benchmark_return_pct=0.01,
            max_drawdown_pct=-0.02,
            stop_triggered=False,
        )
    _decision(scorecard, ticker="PENDING")

    gate = scorecard.gate_for_strategy("test")
    summary = scorecard.strategy_summary("test")

    assert gate.status == "tier1"
    assert gate.allowed_position_pct == 0.01
    assert summary["win_rate_pct"] == 100.0


def test_scorecard_allows_two_percent_after_large_controlled_sample(tmp_path):
    scorecard = Scorecard(tmp_path / "scorecard.db")
    for index in range(60):
        decision_id = _decision(scorecard, ticker=f"BEST{index}")
        scorecard.record_outcome(
            decision_id,
            exit_price=104.0,
            return_pct=0.04,
            benchmark_return_pct=0.01,
            max_drawdown_pct=-0.02,
            stop_triggered=False,
        )

    gate = scorecard.gate_for_strategy("test")

    assert gate.status == "tier2"
    assert gate.allowed_position_pct == 0.02


def test_sell_decision_is_rewarded_when_stock_underperforms(tmp_path):
    scorecard = Scorecard(tmp_path / "scorecard.db")
    decision_id = _decision(scorecard, ticker="FALL", rating="Sell")

    scorecard.record_outcome(
        decision_id,
        exit_price=90,
        return_pct=-0.10,
        benchmark_return_pct=0.01,
        max_drawdown_pct=-0.01,
        stop_triggered=False,
    )

    summary = scorecard.strategy_summary("test")
    assert summary["win_rate_pct"] == 100
    assert summary["avg_alpha_pct"] == pytest.approx(11)


def test_hold_is_excluded_from_directional_gate_sample(tmp_path):
    scorecard = Scorecard(tmp_path / "scorecard.db")
    decision_id = _decision(scorecard, ticker="FLAT", rating="Hold")

    scorecard.record_outcome(
        decision_id,
        exit_price=80,
        return_pct=-0.20,
        benchmark_return_pct=0.01,
        max_drawdown_pct=-0.25,
        stop_triggered=True,
    )

    summary = scorecard.strategy_summary("test")
    assert summary["total_decisions"] == 1
    assert summary["resolved_decisions"] == 0
    assert scorecard.gate_for_strategy("test").status == "warming_up"


def test_decision_artifact_preserves_evidence_and_config(tmp_path):
    scorecard = Scorecard(tmp_path / "scorecard.db")
    decision_id = scorecard.record_decision(
        strategy_key="test",
        ticker="AAPL",
        trade_date="2026-07-22",
        rating="Buy",
        model_provider="groq",
        quick_model="quick",
        deep_model="deep",
        mode="paper",
        entry_price=200,
        final_trade_decision="Rating: Buy",
        evidence={"market_report": "Momentum is positive."},
        config={"rules": {"stop_loss_pct": 0.05}},
    )

    artifact = scorecard.decision_artifact(decision_id)

    assert artifact is not None
    assert artifact["evidence"]["market_report"] == "Momentum is positive."
    assert artifact["config"]["rules"]["stop_loss_pct"] == 0.05


def test_experiment_records_and_leaderboard_include_spy_control(tmp_path):
    scorecard = Scorecard(tmp_path / "scorecard.db", min_resolved_decisions=1)
    decision_id = _decision(scorecard)
    scorecard.record_outcome(
        decision_id,
        exit_price=103,
        return_pct=0.03,
        benchmark_return_pct=0.01,
        max_drawdown_pct=-0.01,
        stop_triggered=False,
    )
    experiment_id = scorecard.record_experiment(
        kind="graph-decision-replay",
        strategy_key="test",
        config={"initial_cash": 10_000},
        data={"tickers": ["AAPL"]},
        metrics={"alpha_pct": 2.0},
        artifact_path="report.json",
    )

    experiments = scorecard.experiments()
    leaderboard = scorecard.leaderboard()

    assert experiments[0]["id"] == experiment_id
    assert experiments[0]["config"]["initial_cash"] == 10_000
    assert leaderboard[0]["strategy_key"] == "test"
    assert leaderboard[0]["role"] == "champion"
    assert leaderboard[1]["strategy_key"] == "SPY"
