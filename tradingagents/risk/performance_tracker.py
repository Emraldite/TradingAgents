from __future__ import annotations

import csv
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)


class PerformanceTracker:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                entry_date TEXT,
                exit_date TEXT,
                entry_price REAL,
                exit_price REAL,
                position_size REAL,
                return_pct REAL,
                hold_days INTEGER,
                exit_reason TEXT,
                conviction_score REAL,
                signal_scores TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                tickers_scanned INTEGER,
                signals_fired INTEGER,
                trades_executed INTEGER,
                portfolio_value REAL,
                daily_pnl REAL
            )"""
        )
        conn.commit()
        conn.close()

    def log_trade_entry(
        self,
        ticker: str,
        entry_date: str,
        entry_price: float,
        position_size: float,
        conviction_score: float | None = None,
        signal_scores: dict | None = None,
    ) -> int:
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.execute(
            """INSERT INTO trades (ticker, entry_date, entry_price, position_size, conviction_score, signal_scores)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                ticker,
                entry_date,
                entry_price,
                position_size,
                conviction_score,
                str(signal_scores) if signal_scores else None,
            ),
        )
        trade_id = cur.lastrowid
        conn.commit()
        conn.close()
        logger.info("Trade logged: %s entry at %.2f", ticker, entry_price)
        return trade_id

    def log_trade_exit(
        self,
        trade_id: int,
        exit_date: str,
        exit_price: float,
        exit_reason: str = "",
    ) -> None:
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT entry_price, entry_date FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
        if not row:
            logger.warning("Trade %d not found for exit", trade_id)
            conn.close()
            return
        entry_price, entry_date = row
        return_pct = (exit_price - entry_price) / entry_price
        entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
        exit_dt = datetime.strptime(exit_date, "%Y-%m-%d")
        hold_days = (exit_dt - entry_dt).days

        conn.execute(
            """UPDATE trades SET exit_date=?, exit_price=?, return_pct=?, hold_days=?, exit_reason=?
               WHERE id=?""",
            (exit_date, exit_price, return_pct, hold_days, exit_reason, trade_id),
        )
        conn.commit()
        conn.close()
        logger.info(
            "Trade %d closed: return %.2f%% over %d days",
            trade_id,
            return_pct * 100,
            hold_days,
        )

    def log_cycle(
        self,
        tickers_scanned: int,
        signals_fired: int,
        trades_executed: int,
        portfolio_value: float,
        daily_pnl: float,
    ) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO cycles (timestamp, tickers_scanned, signals_fired, trades_executed, portfolio_value, daily_pnl)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                tickers_scanned,
                signals_fired,
                trades_executed,
                portfolio_value,
                daily_pnl,
            ),
        )
        conn.commit()
        conn.close()

    def get_summary(self) -> dict[str, Any]:
        conn = sqlite3.connect(str(self.db_path))
        trades = conn.execute(
            "SELECT return_pct, hold_days FROM trades WHERE exit_price IS NOT NULL"
        ).fetchall()
        total_trades = len(trades)
        if total_trades == 0:
            conn.close()
            return {"total_trades": 0}

        returns = [t[0] for t in trades if t[0] is not None]
        total_return = sum(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        avg_return = sum(returns) / len(returns) if returns else 0
        max_drawdown = min(returns) if returns else 0
        hold_days = [t[1] for t in trades if t[1] is not None]
        avg_hold = sum(hold_days) / len(hold_days) if hold_days else 0

        last_value = conn.execute(
            "SELECT portfolio_value FROM cycles ORDER BY id DESC LIMIT 1"
        ).fetchone()

        conn.close()

        spy = yf.download("SPY", period="1mo", progress=False)
        spy_return = 0
        if not spy.empty:
            spy_close = spy["Close"]
            spy_return = (spy_close.iloc[-1] - spy_close.iloc[0]) / spy_close.iloc[0]

        return {
            "total_trades": total_trades,
            "total_return_pct": round(total_return * 100, 2),
            "win_rate": round(win_rate * 100, 1),
            "avg_return_pct": round(avg_return * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "avg_hold_days": round(avg_hold, 1),
            "last_portfolio_value": round(last_value[0], 2) if last_value else None,
            "spy_return_pct": round(float(spy_return) * 100, 2),
        }

    def export_csv(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        trades = conn.execute("SELECT * FROM trades").fetchall()
        if trades:
            cols = [d[0] for d in conn.execute("PRAGMA table_info(trades)").fetchall()]
            path = output_dir / "trades_export.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                w.writerows(trades)
            logger.info("Trades exported to %s", path)

        cycles = conn.execute("SELECT * FROM cycles").fetchall()
        if cycles:
            cols = [d[0] for d in conn.execute("PRAGMA table_info(cycles)").fetchall()]
            path = output_dir / "cycles_export.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                w.writerows(cycles)
            logger.info("Cycles exported to %s", path)

        conn.close()
