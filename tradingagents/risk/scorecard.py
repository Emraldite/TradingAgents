from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScorecardGate:
    allowed_position_pct: float
    status: str
    reason: str
    resolved_decisions: int
    win_rate_pct: float
    avg_alpha_pct: float
    max_drawdown_pct: float


class Scorecard:
    """SQLite scorecard for measuring whether AI trade decisions deserve size."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        horizon_days: int = 10,
        stop_loss_pct: float = -0.05,
        warmup_position_pct: float = 0.005,
        tier1_position_pct: float = 0.01,
        tier2_position_pct: float = 0.02,
        min_resolved_decisions: int = 30,
        tier2_min_decisions: int = 60,
        benchmark_ticker: str = "SPY",
    ) -> None:
        self.db_path = Path(db_path)
        self.horizon_days = horizon_days
        self.stop_loss_pct = stop_loss_pct
        self.warmup_position_pct = warmup_position_pct
        self.tier1_position_pct = tier1_position_pct
        self.tier2_position_pct = tier2_position_pct
        self.min_resolved_decisions = min_resolved_decisions
        self.tier2_min_decisions = tier2_min_decisions
        self.benchmark_ticker = benchmark_ticker
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scorecard_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_key TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    model_provider TEXT,
                    quick_model TEXT,
                    deep_model TEXT,
                    mode TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    final_trade_decision TEXT,
                    evidence_json TEXT,
                    strategy_version TEXT,
                    config_json TEXT,
                    confidence REAL,
                    horizon_days INTEGER NOT NULL,
                    benchmark_ticker TEXT NOT NULL,
                    resolved_at TEXT,
                    exit_price REAL,
                    return_pct REAL,
                    benchmark_return_pct REAL,
                    alpha_pct REAL,
                    directional_return_pct REAL,
                    directional_alpha_pct REAL,
                    directional_drawdown_pct REAL,
                    max_drawdown_pct REAL,
                    stop_triggered INTEGER,
                    score INTEGER,
                    created_at TEXT NOT NULL,
                    UNIQUE(strategy_key, ticker, trade_date, mode)
                )
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(scorecard_decisions)")}
            for column in (
                "directional_return_pct",
                "directional_alpha_pct",
                "directional_drawdown_pct",
                "strategy_version",
                "config_json",
                "confidence",
                "evidence_json",
            ):
                if column not in columns:
                    definition = (
                        "TEXT"
                        if column in {"strategy_version", "config_json", "evidence_json"}
                        else "REAL"
                    )
                    conn.execute(
                        f"ALTER TABLE scorecard_decisions ADD COLUMN {column} {definition}"
                    )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scorecard_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    strategy_key TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    artifact_path TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def record_decision(
        self,
        *,
        strategy_key: str,
        ticker: str,
        trade_date: str,
        rating: str,
        model_provider: str | None,
        quick_model: str | None,
        deep_model: str | None,
        mode: str,
        entry_price: float,
        final_trade_decision: str,
        evidence: dict[str, Any] | None = None,
        strategy_version: str | None = None,
        config: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> int | None:
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")

        now = datetime.utcnow().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO scorecard_decisions (
                    strategy_key, ticker, trade_date, rating, model_provider,
                    quick_model, deep_model, mode, entry_price,
                    final_trade_decision, evidence_json, strategy_version,
                    config_json, confidence, horizon_days, benchmark_ticker, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_key,
                    ticker.upper(),
                    trade_date,
                    rating,
                    model_provider,
                    quick_model,
                    deep_model,
                    mode,
                    entry_price,
                    final_trade_decision,
                    json.dumps(evidence, sort_keys=True) if evidence else None,
                    strategy_version,
                    json.dumps(config, sort_keys=True) if config else None,
                    confidence,
                    self.horizon_days,
                    self.benchmark_ticker,
                    now,
                ),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    """
                    SELECT id FROM scorecard_decisions
                    WHERE strategy_key = ? AND ticker = ? AND trade_date = ? AND mode = ?
                    """,
                    (strategy_key, ticker.upper(), trade_date, mode),
                ).fetchone()
                return int(row["id"]) if row else None
            return int(cursor.lastrowid)

    def decision_artifact(self, decision_id: int) -> dict[str, Any] | None:
        """Return one stored decision and its exact evidence without live API calls."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scorecard_decisions WHERE id=?", (decision_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        for column in ("config_json", "evidence_json"):
            raw = result.pop(column, None)
            result[column.removesuffix("_json")] = json.loads(raw) if raw else {}
        return result

    def record_experiment(
        self,
        *,
        kind: str,
        strategy_key: str,
        config: dict[str, Any],
        data: dict[str, Any],
        metrics: dict[str, Any],
        artifact_path: str | None = None,
    ) -> int:
        """Persist enough metadata to reproduce and compare one validation run."""
        if not kind.strip() or not strategy_key.strip():
            raise ValueError("kind and strategy_key are required")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scorecard_experiments (
                    kind, strategy_key, config_json, data_json, metrics_json,
                    artifact_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    strategy_key,
                    json.dumps(config, sort_keys=True),
                    json.dumps(data, sort_keys=True),
                    json.dumps(metrics, sort_keys=True),
                    artifact_path,
                    datetime.utcnow().isoformat(timespec="seconds"),
                ),
            )
            return int(cursor.lastrowid)

    def experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM scorecard_experiments
                ORDER BY id DESC LIMIT ?
                """,
                (max(1, min(limit, 100)),),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            for column in ("config_json", "data_json", "metrics_json"):
                raw = item.pop(column)
                item[column.removesuffix("_json")] = json.loads(raw)
            results.append(item)
        return results

    def record_outcome(
        self,
        decision_id: int,
        *,
        exit_price: float,
        return_pct: float,
        benchmark_return_pct: float,
        max_drawdown_pct: float,
        stop_triggered: bool,
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT rating FROM scorecard_decisions WHERE id=?", (decision_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Decision {decision_id} not found")
            rating = str(row["rating"])
            direction = self._rating_direction(rating)
            directional_return = direction * return_pct if direction else None
            directional_alpha = (
                direction * (return_pct - benchmark_return_pct) if direction else None
            )
            directional_drawdown = max_drawdown_pct if direction else None
            score = (
                self._score(
                    directional_return_pct=directional_return,
                    directional_alpha_pct=directional_alpha,
                    directional_drawdown_pct=directional_drawdown,
                    stop_triggered=stop_triggered,
                )
                if direction
                else None
            )
            conn.execute(
                """
                UPDATE scorecard_decisions
                SET resolved_at = ?, exit_price = ?, return_pct = ?,
                    benchmark_return_pct = ?, alpha_pct = ?,
                    max_drawdown_pct = ?, directional_return_pct = ?,
                    directional_alpha_pct = ?, directional_drawdown_pct = ?,
                    stop_triggered = ?, score = ?
                WHERE id = ?
                """,
                (
                    datetime.utcnow().isoformat(timespec="seconds"),
                    exit_price,
                    return_pct,
                    benchmark_return_pct,
                    directional_alpha,
                    max_drawdown_pct,
                    directional_return,
                    directional_alpha,
                    directional_drawdown,
                    1 if stop_triggered else 0,
                    score,
                    decision_id,
                ),
            )

    def resolve_due_outcomes(self, *, as_of: date | None = None) -> int:
        """Resolve decisions once enough market bars exist after the trade date."""
        import yfinance as yf

        as_of = as_of or date.today()
        start_cutoff = (as_of - timedelta(days=max(self.horizon_days * 3, 21))).isoformat()
        end = (as_of + timedelta(days=1)).isoformat()
        resolved = 0

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM scorecard_decisions
                WHERE resolved_at IS NULL AND trade_date <= ?
                ORDER BY trade_date ASC
                """,
                (start_cutoff,),
            ).fetchall()

        for row in rows:
            try:
                stock = yf.download(
                    row["ticker"],
                    start=row["trade_date"],
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    multi_level_index=False,
                )
                benchmark = yf.download(
                    row["benchmark_ticker"],
                    start=row["trade_date"],
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    multi_level_index=False,
                )
                outcome = self._calculate_outcome(row, stock, benchmark)
                if outcome is None:
                    continue
                self.record_outcome(int(row["id"]), **outcome)
                resolved += 1
            except Exception as exc:
                # One bad ticker/data response should not stop the trading cycle.
                logger.warning(
                    "Could not resolve scorecard decision %s for %s: %s",
                    row["id"],
                    row["ticker"],
                    exc,
                )
                continue
        return resolved

    def _calculate_outcome(
        self,
        row: sqlite3.Row,
        stock: Any,
        benchmark: Any,
    ) -> dict[str, float | bool] | None:
        close = self._series(stock, "Close")
        low = self._series(stock, "Low")
        high = self._series(stock, "High")
        bench_close = self._series(benchmark, "Close")
        if low is None:
            low = close
        if high is None:
            high = close
        if close is None or low is None or high is None or bench_close is None:
            return None
        if len(close) <= self.horizon_days or len(bench_close) <= self.horizon_days:
            return None

        entry_price = float(row["entry_price"])
        window_close = close.iloc[1 : self.horizon_days + 1]
        window_low = low.iloc[1 : self.horizon_days + 1]
        window_high = high.iloc[1 : self.horizon_days + 1]
        if window_close.empty or window_low.empty:
            return None

        exit_price = float(window_close.iloc[-1])
        return_pct = (exit_price / entry_price) - 1
        direction = self._rating_direction(str(row["rating"]))
        if direction < 0:
            max_drawdown_pct = -((float(window_high.max()) / entry_price) - 1)
        else:
            max_drawdown_pct = (float(window_low.min()) / entry_price) - 1
        benchmark_return_pct = (
            float(bench_close.iloc[self.horizon_days]) / float(bench_close.iloc[0])
        ) - 1
        return {
            "exit_price": exit_price,
            "return_pct": return_pct,
            "benchmark_return_pct": benchmark_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "stop_triggered": max_drawdown_pct <= self.stop_loss_pct,
        }

    @staticmethod
    def _series(frame: Any, column: str) -> Any | None:
        if frame is None or getattr(frame, "empty", True):
            return None
        if column in frame:
            return frame[column].dropna()
        if hasattr(frame.columns, "levels"):
            matches = [col for col in frame.columns if column in col]
            if matches:
                return frame[matches[0]].dropna()
        return None

    @staticmethod
    def _rating_direction(rating: str) -> int:
        if rating in {"Buy", "Overweight"}:
            return 1
        if rating in {"Sell", "Underweight"}:
            return -1
        return 0

    @staticmethod
    def _score(
        *,
        directional_return_pct: float,
        directional_alpha_pct: float,
        directional_drawdown_pct: float,
        stop_triggered: bool,
    ) -> int:
        score = 0
        if directional_return_pct > 0:
            score += 1
        if directional_alpha_pct > 0:
            score += 1
        if directional_drawdown_pct > -0.03:
            score += 1
        if directional_return_pct < -0.02:
            score -= 1
        if stop_triggered:
            score -= 1
        return score

    def gate_for_strategy(self, strategy_key: str) -> ScorecardGate:
        stats = self.strategy_summary(strategy_key)
        resolved = int(stats["resolved_decisions"])
        avg_alpha_pct = float(stats["avg_alpha_pct"])
        win_rate_pct = float(stats["win_rate_pct"])
        max_drawdown_pct = float(stats["max_drawdown_pct"])
        avg_score = float(stats["avg_score"])

        if resolved < self.min_resolved_decisions:
            return ScorecardGate(
                self.warmup_position_pct,
                "warming_up",
                f"fewer than {self.min_resolved_decisions} resolved decisions",
                resolved,
                win_rate_pct,
                avg_alpha_pct,
                max_drawdown_pct,
            )

        if avg_alpha_pct <= 0 or avg_score < 1 or max_drawdown_pct <= -5:
            return ScorecardGate(
                0.0,
                "blocked",
                "scorecard performance is not good enough for new buys",
                resolved,
                win_rate_pct,
                avg_alpha_pct,
                max_drawdown_pct,
            )

        if resolved >= self.tier2_min_decisions and max_drawdown_pct > -3:
            return ScorecardGate(
                self.tier2_position_pct,
                "tier2",
                "positive alpha with controlled drawdown",
                resolved,
                win_rate_pct,
                avg_alpha_pct,
                max_drawdown_pct,
            )

        return ScorecardGate(
            self.tier1_position_pct,
            "tier1",
            "positive alpha after warmup",
            resolved,
            win_rate_pct,
            avg_alpha_pct,
            max_drawdown_pct,
        )

    def strategy_summary(self, strategy_key: str) -> dict[str, float | int | str]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_decisions,
                    SUM(CASE WHEN resolved_at IS NULL THEN 1 ELSE 0 END) AS pending_decisions,
                    SUM(CASE WHEN resolved_at IS NOT NULL AND directional_return_pct IS NOT NULL THEN 1 ELSE 0 END) AS resolved_decisions,
                    AVG(
                        CASE
                            WHEN resolved_at IS NOT NULL AND directional_return_pct IS NOT NULL THEN
                                CASE WHEN directional_return_pct > 0 THEN 1.0 ELSE 0.0 END
                        END
                    ) AS win_rate,
                    AVG(directional_alpha_pct) AS avg_alpha,
                    MIN(directional_drawdown_pct) AS max_drawdown,
                    AVG(score) AS avg_score
                FROM scorecard_decisions
                WHERE strategy_key = ?
                """,
                (strategy_key,),
            ).fetchone()

        return self._format_summary_row(strategy_key, row)

    def summaries(self) -> list[dict[str, float | int | str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    strategy_key,
                    COUNT(*) AS total_decisions,
                    SUM(CASE WHEN resolved_at IS NULL THEN 1 ELSE 0 END) AS pending_decisions,
                    SUM(CASE WHEN resolved_at IS NOT NULL AND directional_return_pct IS NOT NULL THEN 1 ELSE 0 END) AS resolved_decisions,
                    AVG(
                        CASE
                            WHEN resolved_at IS NOT NULL AND directional_return_pct IS NOT NULL THEN
                                CASE WHEN directional_return_pct > 0 THEN 1.0 ELSE 0.0 END
                        END
                    ) AS win_rate,
                    AVG(directional_alpha_pct) AS avg_alpha,
                    MIN(directional_drawdown_pct) AS max_drawdown,
                    AVG(score) AS avg_score
                FROM scorecard_decisions
                GROUP BY strategy_key
                ORDER BY strategy_key
                """
            ).fetchall()

        return [self._format_summary_row(str(row["strategy_key"]), row) for row in rows]

    def leaderboard(self) -> list[dict[str, float | int | str]]:
        """Rank strategy versions against SPY's zero-alpha control."""
        rows = self.summaries()
        resolved = max((int(row["resolved_decisions"]) for row in rows), default=0)
        rows.append(
            {
                "strategy_key": self.benchmark_ticker,
                "total_decisions": resolved,
                "pending_decisions": 0,
                "resolved_decisions": resolved,
                "win_rate_pct": 0.0,
                "avg_alpha_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "avg_score": 0.0,
                "eligible": True,
            }
        )
        for row in rows:
            row.setdefault(
                "eligible",
                int(row["resolved_decisions"]) >= self.min_resolved_decisions,
            )
        ranked = sorted(
            rows,
            key=lambda row: (
                bool(row["eligible"]),
                float(row["avg_alpha_pct"]),
                row["strategy_key"] == self.benchmark_ticker,
                float(row["max_drawdown_pct"]),
                int(row["resolved_decisions"]),
            ),
            reverse=True,
        )
        return [
            {**row, "rank": index, "role": "champion" if index == 1 else "challenger"}
            for index, row in enumerate(ranked, start=1)
        ]

    def decisions_for_ticker(
        self,
        ticker: str,
        strategy_key: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """SELECT id, strategy_key, ticker, trade_date, rating, mode,
                          entry_price, final_trade_decision, resolved_at,
                          directional_return_pct, directional_alpha_pct
                   FROM scorecard_decisions WHERE ticker=?"""
        params: list[Any] = [ticker.upper()]
        if strategy_key:
            query += " AND strategy_key=?"
            params.append(strategy_key)
        query += " ORDER BY trade_date, id"
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, tuple(params)).fetchall()]

    def _format_summary_row(
        self,
        strategy_key: str,
        row: sqlite3.Row | None,
    ) -> dict[str, float | int | str]:
        if row is None:
            return {
                "strategy_key": strategy_key,
                "total_decisions": 0,
                "pending_decisions": 0,
                "resolved_decisions": 0,
                "win_rate_pct": 0.0,
                "avg_alpha_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "avg_score": 0.0,
            }

        return {
            "strategy_key": strategy_key,
            "total_decisions": int(row["total_decisions"] or 0),
            "pending_decisions": int(row["pending_decisions"] or 0),
            "resolved_decisions": int(row["resolved_decisions"] or 0),
            "win_rate_pct": float(row["win_rate"] or 0.0) * 100,
            "avg_alpha_pct": float(row["avg_alpha"] or 0.0) * 100,
            "max_drawdown_pct": float(row["max_drawdown"] or 0.0) * 100,
            "avg_score": float(row["avg_score"] or 0.0),
        }
