from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ACTIVE_ORDER_STATUSES = {
    "new",
    "accepted",
    "pending_new",
    "partially_filled",
    "accepted_for_bidding",
    "calculated",
    "held",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrategyStateStore:
    """SQLite state for strategy decisions, broker reconciliation, and audit logs."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS strategy_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    entry_date TEXT,
                    entry_price REAL,
                    quantity REAL NOT NULL DEFAULT 0,
                    broker_quantity REAL NOT NULL DEFAULT 0,
                    entry_signal_score REAL,
                    thesis TEXT,
                    strategy_version TEXT,
                    order_id TEXT,
                    stop_order_id TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    mode TEXT NOT NULL DEFAULT 'live',
                    last_reconciled_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_strategy_positions_ticker_status
                   ON strategy_positions (ticker, status)"""
            )
            self._ensure_column(conn, "strategy_positions", "stop_order_id", "TEXT")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS strategy_orders (
                    order_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT,
                    status TEXT NOT NULL,
                    qty REAL NOT NULL DEFAULT 0,
                    notional REAL NOT NULL DEFAULT 0,
                    filled_qty REAL NOT NULL DEFAULT 0,
                    filled_avg_price REAL NOT NULL DEFAULT 0,
                    submitted_at TEXT,
                    filled_at TEXT,
                    mode TEXT NOT NULL DEFAULT 'live',
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS strategy_trades (
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    fill_price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    reason TEXT,
                    order_id TEXT,
                    mode TEXT NOT NULL DEFAULT 'live'
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS strategy_cooldowns (
                    ticker TEXT PRIMARY KEY,
                    cooldown_until TEXT NOT NULL,
                    reason TEXT,
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS strategy_cycles (
                    cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    mode TEXT NOT NULL,
                    tickers_json TEXT,
                    decisions_json TEXT,
                    portfolio_json TEXT,
                    status TEXT NOT NULL,
                    error TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS strategy_corporate_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    ex_date TEXT NOT NULL,
                    factor REAL NOT NULL,
                    applied_at TEXT NOT NULL,
                    UNIQUE(ticker, action_type, ex_date)
                )"""
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('strategy_schema_version', '3')"
            )

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def start_cycle(self, mode: str, tickers: list[str]) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO strategy_cycles (started_at, mode, tickers_json, status)
                   VALUES (?, ?, ?, ?)""",
                (utc_now_iso(), mode, json.dumps(tickers), "running"),
            )
            return int(cur.lastrowid)

    def complete_cycle(
        self,
        cycle_id: int,
        decisions: list[dict[str, Any]],
        portfolio: list[dict[str, Any]],
        status: str = "complete",
        error: str | None = None,
    ) -> int:
        with self._connect() as conn:
            conn.execute(
                """UPDATE strategy_cycles
                   SET completed_at=?, decisions_json=?, portfolio_json=?, status=?, error=?
                   WHERE cycle_id=?""",
                (
                    utc_now_iso(),
                    json.dumps(decisions),
                    json.dumps(portfolio),
                    status,
                    error,
                    cycle_id,
                ),
            )

    def upsert_order(self, order: dict[str, Any], mode: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO strategy_orders (
                    order_id, ticker, side, order_type, status, qty, notional,
                    filled_qty, filled_avg_price, submitted_at, filled_at, mode, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    status=excluded.status,
                    qty=excluded.qty,
                    notional=excluded.notional,
                    filled_qty=excluded.filled_qty,
                    filled_avg_price=excluded.filled_avg_price,
                    filled_at=excluded.filled_at,
                    updated_at=excluded.updated_at""",
                (
                    order["order_id"],
                    order["ticker"],
                    order.get("side", ""),
                    order.get("type", order.get("order_type", "")),
                    order.get("status", ""),
                    float(order.get("qty", 0) or 0),
                    float(order.get("notional", 0) or 0),
                    float(order.get("filled_qty", 0) or 0),
                    float(order.get("filled_avg_price", 0) or 0),
                    order.get("submitted_at"),
                    order.get("filled_at"),
                    mode,
                    utc_now_iso(),
                ),
            )

    def record_buy_signal(
        self,
        ticker: str,
        entry_price: float,
        position_size: float,
        mode: str,
        order: dict[str, Any] | None = None,
        reason: str = "entry_signal",
    ) -> None:
        now = utc_now_iso()
        filled_qty = float(order.get("filled_qty", 0) if order else 0)
        avg_price = float(order.get("filled_avg_price", 0) if order else 0)
        order_id = order.get("order_id") if order else None
        entry = avg_price or entry_price
        quantity = filled_qty or (position_size / entry if entry else 0)
        status = "open" if quantity > 0 else "pending"

        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO strategy_positions (
                    ticker, entry_date, entry_price, quantity, broker_quantity,
                    order_id, status, mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, now, entry, quantity, filled_qty, order_id, status, mode, now, now),
            )
            conn.execute(
                """INSERT INTO strategy_trades (
                    ticker, side, quantity, fill_price, timestamp, reason, order_id, mode
                ) VALUES (?, 'buy', ?, ?, ?, ?, ?, ?)""",
                (ticker, quantity, entry, now, reason, order_id, mode),
            )
            return int(cur.lastrowid)

    def open_positions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM strategy_positions
                   WHERE status IN ('open', 'broker_only')
                   ORDER BY id"""
            ).fetchall()
            return [dict(row) for row in rows]

    def active_positions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM strategy_positions
                   WHERE status IN ('open', 'pending', 'broker_only', 'closing')
                   ORDER BY id"""
            ).fetchall()
            return [dict(row) for row in rows]

    def active_cooldowns(self) -> list[dict[str, Any]]:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("DELETE FROM strategy_cooldowns WHERE cooldown_until <= ?", (now,))
            rows = conn.execute(
                """SELECT ticker, cooldown_until, reason, updated_at
                   FROM strategy_cooldowns
                   ORDER BY cooldown_until"""
            ).fetchall()
            return [dict(row) for row in rows]

    def recent_trades(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT trade_id, ticker, side, quantity, fill_price, timestamp,
                          reason, order_id, mode
                   FROM strategy_trades
                   ORDER BY trade_id DESC
                   LIMIT ?""",
                (max(1, min(limit, 100)),),
            ).fetchall()
            return [dict(row) for row in rows]

    def recent_cycles(self, limit: int = 5) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT cycle_id, started_at, completed_at, mode, tickers_json,
                          decisions_json, status, error
                   FROM strategy_cycles
                   ORDER BY cycle_id DESC
                   LIMIT ?""",
                (max(1, min(limit, 100)),),
            ).fetchall()

        cycles = []
        for row in rows:
            cycle = dict(row)
            tickers = json.loads(cycle["tickers_json"]) if cycle.get("tickers_json") else []
            decisions = json.loads(cycle["decisions_json"]) if cycle.get("decisions_json") else []
            cycle["ticker_count"] = len(tickers)
            cycle["action_summary"] = summarize_decisions(decisions)
            cycles.append(cycle)
        return cycles

    def record_sell(
        self,
        position_id: int,
        ticker: str,
        quantity: float,
        fill_price: float,
        reason: str,
        mode: str,
        order: dict[str, Any] | None = None,
        cooldown_hours: int = 24,
    ) -> None:
        now = utc_now_iso()
        order_id = order.get("order_id") if order else None
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO strategy_trades (
                    ticker, side, quantity, fill_price, timestamp, reason, order_id, mode
                ) VALUES (?, 'sell', ?, ?, ?, ?, ?, ?)""",
                (ticker, quantity, fill_price, now, reason, order_id, mode),
            )
            conn.execute(
                """UPDATE strategy_positions
                   SET status='closed', quantity=0, broker_quantity=0, updated_at=?
                   WHERE id=?""",
                (now, position_id),
            )
            conn.execute(
                """INSERT OR REPLACE INTO strategy_cooldowns (
                    ticker, cooldown_until, reason, updated_at
                ) VALUES (?, ?, ?, ?)""",
                (
                    ticker,
                    (datetime.now(timezone.utc) + timedelta(hours=cooldown_hours)).isoformat(),
                    f"sold: {reason}",
                    now,
                ),
            )

    def mark_position_closing(self, position_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE strategy_positions SET status='closing', updated_at=? WHERE id=?",
                (utc_now_iso(), position_id),
            )

    def set_stop_order(self, position_id: int, stop_order_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE strategy_positions
                   SET stop_order_id=?, updated_at=?
                   WHERE id=?""",
                (stop_order_id, utc_now_iso(), position_id),
            )

    def positions_needing_stop_orders(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM strategy_positions
                   WHERE status IN ('open', 'broker_only')
                     AND mode='live'
                     AND broker_quantity > 0
                     AND entry_price > 0
                     AND (stop_order_id IS NULL OR stop_order_id = '')
                   ORDER BY id"""
            ).fetchall()
            return [dict(row) for row in rows]

    def has_corporate_action(self, ticker: str, action_type: str, ex_date: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT 1 FROM strategy_corporate_actions
                   WHERE ticker=? AND action_type=? AND ex_date=?""",
                (ticker, action_type, ex_date),
            ).fetchone()
            return row is not None

    def apply_split(
        self,
        position_id: int,
        ticker: str,
        ex_date: str,
        factor: float,
    ) -> bool:
        now = utc_now_iso()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT entry_price, quantity, broker_quantity FROM strategy_positions WHERE id=?",
                (position_id,),
            ).fetchone()
            if row is None or factor <= 0:
                return False

            new_entry = float(row["entry_price"] or 0) / factor
            broker_qty = float(row["broker_quantity"] or 0)
            current_qty = float(row["quantity"] or 0)
            new_qty = broker_qty if broker_qty > 0 else current_qty * factor
            try:
                conn.execute(
                    """INSERT INTO strategy_corporate_actions (
                        ticker, action_type, ex_date, factor, applied_at
                    ) VALUES (?, 'split', ?, ?, ?)""",
                    (ticker, ex_date, factor, now),
                )
            except sqlite3.IntegrityError:
                return False

            conn.execute(
                """UPDATE strategy_positions
                   SET entry_price=?, quantity=?, updated_at=?
                   WHERE id=?""",
                (new_entry, new_qty, now, position_id),
            )
            return True

    def get_cycle(self, cycle_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM strategy_cycles WHERE cycle_id=?", (cycle_id,)
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            for key in ("tickers_json", "decisions_json", "portfolio_json"):
                raw = result.get(key)
                result[key.replace("_json", "")] = json.loads(raw) if raw else []
            return result

    def reconcile_broker_state(
        self,
        positions: list[dict[str, Any]],
        orders: list[dict[str, Any]],
        mode: str,
    ) -> set[str]:
        now = utc_now_iso()
        broker_tickers = set()
        for order in orders:
            self.upsert_order(order, mode)

        with self._connect() as conn:
            for position in positions:
                ticker = position["ticker"]
                broker_tickers.add(ticker)
                qty = float(position.get("qty", 0) or 0)
                cost_basis = float(position.get("cost_basis", 0) or 0)
                entry_price = cost_basis / qty if qty else 0
                row = conn.execute(
                    """SELECT id FROM strategy_positions
                       WHERE ticker=? AND status IN ('open', 'pending', 'broker_only')
                       ORDER BY id DESC LIMIT 1""",
                    (ticker,),
                ).fetchone()
                if row:
                    conn.execute(
                        """UPDATE strategy_positions
                           SET broker_quantity=?, quantity=?, last_reconciled_at=?,
                               status='open', updated_at=?
                           WHERE id=?""",
                        (qty, qty, now, now, row["id"]),
                    )
                else:
                    conn.execute(
                        """INSERT INTO strategy_positions (
                            ticker, entry_date, entry_price, quantity, broker_quantity,
                            status, mode, last_reconciled_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 'broker_only', ?, ?, ?, ?)""",
                        (ticker, now, entry_price, qty, qty, mode, now, now, now),
                    )

            rows = conn.execute(
                "SELECT id, ticker FROM strategy_positions WHERE status IN ('open', 'pending', 'broker_only')"
            ).fetchall()
            for row in rows:
                if row["ticker"] not in broker_tickers:
                    conn.execute(
                        """UPDATE strategy_positions
                           SET broker_quantity=0, status='closed', last_reconciled_at=?, updated_at=?
                           WHERE id=?""",
                        (now, now, row["id"]),
                    )

        return broker_tickers

    def active_blocked_tickers(self) -> dict[str, str]:
        now = utc_now_iso()
        blocked: dict[str, str] = {}
        with self._connect() as conn:
            conn.execute("DELETE FROM strategy_cooldowns WHERE cooldown_until <= ?", (now,))
            for row in conn.execute(
                "SELECT ticker FROM strategy_positions WHERE status IN ('open', 'pending', 'broker_only', 'closing')"
            ):
                blocked[row["ticker"]] = "already held"
            for row in conn.execute(
                "SELECT ticker, status FROM strategy_orders WHERE status IN ({})".format(
                    ",".join("?" for _ in ACTIVE_ORDER_STATUSES)
                ),
                tuple(ACTIVE_ORDER_STATUSES),
            ):
                blocked[row["ticker"]] = f"pending order: {row['status']}"
            for row in conn.execute("SELECT ticker, reason FROM strategy_cooldowns"):
                blocked[row["ticker"]] = row["reason"] or "cooldown"
        return blocked


def summarize_decisions(decisions: list[dict[str, Any]]) -> str:
    if not decisions:
        return "-"
    counts: dict[str, int] = {}
    for decision in decisions:
        name = str(decision.get("decision") or "unknown")
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{name}:{count}" for name, count in sorted(counts.items()))
