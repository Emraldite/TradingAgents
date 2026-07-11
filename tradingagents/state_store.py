from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ACTIVE_ORDER_STATUSES = {
    "new",
    "accepted",
    "pending_new",
    "partially_filled",
    "accepted_for_bidding",
    "calculated",
    "held",
}
TERMINAL_ORDER_STATUSES = {
    "filled",
    "canceled",
    "expired",
    "rejected",
    "replaced",
    "stopped",
    "suspended",
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
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA foreign_keys=ON")
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
                    client_order_id TEXT,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT,
                    order_class TEXT,
                    parent_order_id TEXT,
                    status TEXT NOT NULL,
                    qty REAL NOT NULL DEFAULT 0,
                    notional REAL NOT NULL DEFAULT 0,
                    filled_qty REAL NOT NULL DEFAULT 0,
                    filled_avg_price REAL NOT NULL DEFAULT 0,
                    limit_price REAL NOT NULL DEFAULT 0,
                    stop_price REAL NOT NULL DEFAULT 0,
                    reason TEXT,
                    raw_json TEXT,
                    submitted_at TEXT,
                    filled_at TEXT,
                    mode TEXT NOT NULL DEFAULT 'live',
                    updated_at TEXT NOT NULL
                )"""
            )
            for column, definition in (
                ("client_order_id", "TEXT"),
                ("order_class", "TEXT"),
                ("parent_order_id", "TEXT"),
                ("limit_price", "REAL NOT NULL DEFAULT 0"),
                ("stop_price", "REAL NOT NULL DEFAULT 0"),
                ("reason", "TEXT"),
                ("raw_json", "TEXT"),
            ):
                self._ensure_column(conn, "strategy_orders", column, definition)
            conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_orders_client_id
                   ON strategy_orders (client_order_id)
                   WHERE client_order_id IS NOT NULL AND client_order_id != ''"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS strategy_order_events (
                    event_key TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cumulative_filled_qty REAL NOT NULL DEFAULT 0,
                    event_at TEXT NOT NULL,
                    raw_json TEXT
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
                    execution_id TEXT,
                    mode TEXT NOT NULL DEFAULT 'live'
                )"""
            )
            self._ensure_column(conn, "strategy_trades", "execution_id", "TEXT")
            conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_trades_execution_id
                   ON strategy_trades (execution_id)
                   WHERE execution_id IS NOT NULL AND execution_id != ''"""
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
                """CREATE TABLE IF NOT EXISTS strategy_risk_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    trading_date TEXT NOT NULL,
                    week_key TEXT NOT NULL,
                    daily_start_equity REAL NOT NULL,
                    weekly_start_equity REAL NOT NULL,
                    peak_equity REAL NOT NULL,
                    halted_until TEXT,
                    halt_reason TEXT,
                    manual_halt INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS strategy_equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL DEFAULT 0,
                    buying_power REAL NOT NULL DEFAULT 0,
                    daily_drawdown_pct REAL NOT NULL DEFAULT 0,
                    weekly_drawdown_pct REAL NOT NULL DEFAULT 0,
                    total_drawdown_pct REAL NOT NULL DEFAULT 0,
                    mode TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS strategy_health_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context_json TEXT,
                    acknowledged_at TEXT,
                    acknowledgement TEXT
                )"""
            )
            self._ensure_column(conn, "strategy_health_events", "acknowledged_at", "TEXT")
            self._ensure_column(conn, "strategy_health_events", "acknowledgement", "TEXT")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS strategy_account_activities (
                    activity_id TEXT PRIMARY KEY,
                    activity_type TEXT NOT NULL,
                    ticker TEXT,
                    quantity REAL NOT NULL DEFAULT 0,
                    price REAL NOT NULL DEFAULT 0,
                    net_amount REAL NOT NULL DEFAULT 0,
                    activity_at TEXT,
                    raw_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('strategy_schema_version', '4')"
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
        self.record_order_update(order, mode)

    def record_order_tree(
        self,
        order: dict[str, Any],
        mode: str,
        reason: str = "broker_update",
    ) -> list[dict[str, Any]]:
        results = [self.record_order_update(order, mode, reason=reason)]
        for leg in order.get("legs") or []:
            leg_data = dict(leg)
            leg_data.setdefault("parent_order_id", order.get("order_id"))
            results.extend(self.record_order_tree(leg_data, mode, reason=reason))
        return results

    def record_order_update(
        self,
        order: dict[str, Any],
        mode: str,
        reason: str = "broker_update",
    ) -> dict[str, Any]:
        """Persist an order snapshot and apply only newly confirmed fill quantity."""
        order_id = str(order["order_id"])
        status = str(order.get("status", ""))
        filled_qty = float(order.get("filled_qty", 0) or 0)
        filled_avg_price = float(order.get("filled_avg_price", 0) or 0)
        now = utc_now_iso()
        position_id: int | None = None
        fill_delta = 0.0

        with self._connect() as conn:
            previous = conn.execute(
                "SELECT filled_qty, filled_avg_price, status FROM strategy_orders WHERE order_id=?",
                (order_id,),
            ).fetchone()
            previous_qty = float(previous["filled_qty"] or 0) if previous else 0.0
            previous_avg = float(previous["filled_avg_price"] or 0) if previous else 0.0
            previous_status = str(previous["status"] or "") if previous else ""
            if previous:
                if filled_qty + 1e-9 < previous_qty:
                    raise ValueError(
                        f"Order {order_id} cumulative fill regressed from {previous_qty} to {filled_qty}"
                    )
                if (
                    previous_status in TERMINAL_ORDER_STATUSES
                    and status != previous_status
                ):
                    raise ValueError(
                        f"Order {order_id} regressed from terminal status {previous_status} to {status}"
                    )
            conn.execute(
                """INSERT INTO strategy_orders (
                    order_id, client_order_id, ticker, side, order_type, order_class,
                    parent_order_id, status, qty, notional, filled_qty, filled_avg_price,
                    limit_price, stop_price, reason, raw_json, submitted_at, filled_at,
                    mode, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    client_order_id=COALESCE(NULLIF(excluded.client_order_id, ''), strategy_orders.client_order_id),
                    status=excluded.status,
                    qty=excluded.qty,
                    notional=excluded.notional,
                    filled_qty=excluded.filled_qty,
                    filled_avg_price=excluded.filled_avg_price,
                    limit_price=excluded.limit_price,
                    stop_price=excluded.stop_price,
                    reason=excluded.reason,
                    raw_json=excluded.raw_json,
                    filled_at=excluded.filled_at,
                    updated_at=excluded.updated_at""",
                (
                    order_id,
                    order.get("client_order_id") or None,
                    order["ticker"],
                    order.get("side", ""),
                    order.get("type", order.get("order_type", "")),
                    order.get("order_class", ""),
                    order.get("parent_order_id"),
                    status,
                    float(order.get("qty", 0) or 0),
                    float(order.get("notional", 0) or 0),
                    filled_qty,
                    filled_avg_price,
                    float(order.get("limit_price", 0) or 0),
                    float(order.get("stop_price", 0) or 0),
                    reason,
                    json.dumps(order, default=str),
                    order.get("submitted_at"),
                    order.get("filled_at"),
                    mode,
                    now,
                ),
            )

            event_key = f"{order_id}:{status}:{filled_qty:.9f}"
            conn.execute(
                """INSERT OR IGNORE INTO strategy_order_events (
                    event_key, order_id, status, cumulative_filled_qty, event_at, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (event_key, order_id, status, filled_qty, now, json.dumps(order, default=str)),
            )

            fill_delta = max(0.0, filled_qty - previous_qty)
            if fill_delta > 1e-9:
                if filled_avg_price > 0:
                    cumulative_cost = filled_qty * filled_avg_price
                    previous_cost = previous_qty * previous_avg
                    fill_price = max(0.0, (cumulative_cost - previous_cost) / fill_delta)
                else:
                    fill_price = float(order.get("limit_price", 0) or 0)
                execution_id = f"{order_id}:{filled_qty:.9f}"
                conn.execute(
                    """INSERT OR IGNORE INTO strategy_trades (
                        ticker, side, quantity, fill_price, timestamp, reason,
                        order_id, execution_id, mode
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        order["ticker"],
                        order.get("side", ""),
                        fill_delta,
                        fill_price,
                        order.get("filled_at") or now,
                        reason,
                        order_id,
                        execution_id,
                        mode,
                    ),
                )

                if order.get("side") == "buy":
                    row = conn.execute(
                        """SELECT id FROM strategy_positions
                           WHERE order_id=? AND status IN ('open', 'pending', 'broker_only')
                           ORDER BY id DESC LIMIT 1""",
                        (order_id,),
                    ).fetchone()
                    if row:
                        position_id = int(row["id"])
                        conn.execute(
                            """UPDATE strategy_positions
                               SET entry_date=COALESCE(entry_date, ?), entry_price=?,
                                   quantity=?, broker_quantity=?, status='open', updated_at=?
                               WHERE id=?""",
                            (
                                order.get("filled_at") or now,
                                filled_avg_price or fill_price,
                                filled_qty,
                                filled_qty,
                                now,
                                position_id,
                            ),
                        )
                    else:
                        cur = conn.execute(
                            """INSERT INTO strategy_positions (
                                ticker, entry_date, entry_price, quantity, broker_quantity,
                                order_id, status, mode, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)""",
                            (
                                order["ticker"],
                                order.get("filled_at") or now,
                                filled_avg_price or fill_price,
                                filled_qty,
                                filled_qty,
                                order_id,
                                mode,
                                now,
                                now,
                            ),
                        )
                        position_id = int(cur.lastrowid)
                elif order.get("side") == "sell":
                    row = conn.execute(
                        """SELECT id, quantity FROM strategy_positions
                           WHERE ticker=? AND status IN ('open', 'broker_only', 'closing')
                           ORDER BY id DESC LIMIT 1""",
                        (order["ticker"],),
                    ).fetchone()
                    if row:
                        position_id = int(row["id"])
                        remaining = max(0.0, float(row["quantity"] or 0) - fill_delta)
                        next_status = "closed" if remaining <= 1e-9 else "open"
                        conn.execute(
                            """UPDATE strategy_positions
                               SET quantity=?, broker_quantity=?, status=?, updated_at=?
                               WHERE id=?""",
                            (remaining, remaining, next_status, now, position_id),
                        )
                        if next_status == "closed":
                            conn.execute(
                                """INSERT OR REPLACE INTO strategy_cooldowns (
                                    ticker, cooldown_until, reason, updated_at
                                ) VALUES (?, ?, ?, ?)""",
                                (
                                    order["ticker"],
                                    (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
                                    f"sold: {reason}",
                                    now,
                                ),
                            )

        return {
            "order_id": order_id,
            "status": status,
            "fill_delta": fill_delta,
            "position_id": position_id,
        }

    def record_buy_signal(
        self,
        ticker: str,
        entry_price: float,
        position_size: float,
        mode: str,
        order: dict[str, Any] | None = None,
        reason: str = "entry_signal",
    ) -> int:
        if order is not None:
            result = self.record_order_update(order, mode, reason=reason)
            return int(result["position_id"] or 0)
        if mode != "shadow":
            raise ValueError("Broker-backed positions require a confirmed order fill")

        now = utc_now_iso()
        order_id = None
        entry = entry_price
        quantity = position_size / entry if entry else 0
        status = "open" if quantity > 0 else "pending"

        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO strategy_positions (
                    ticker, entry_date, entry_price, quantity, broker_quantity,
                    order_id, status, mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, now, entry, quantity, 0, order_id, status, mode, now, now),
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

    def reserved_open_position_count(self) -> int:
        """Count held tickers plus active buy orders that reserve a future slot."""
        placeholders = ",".join("?" for _ in ACTIVE_ORDER_STATUSES)
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT ticker FROM strategy_positions
                    WHERE status IN ('open', 'broker_only', 'closing')
                    UNION
                    SELECT ticker FROM strategy_orders
                    WHERE side='buy' AND status IN ({placeholders})""",
                tuple(ACTIVE_ORDER_STATUSES),
            ).fetchall()
            return len(rows)

    def reserved_buy_notional(self) -> float:
        placeholders = ",".join("?" for _ in ACTIVE_ORDER_STATUSES)
        with self._connect() as conn:
            row = conn.execute(
                f"""SELECT SUM(
                        CASE WHEN notional > 0 THEN notional ELSE qty * limit_price END
                    ) AS reserved
                    FROM strategy_orders
                    WHERE side='buy' AND status IN ({placeholders})""",
                tuple(ACTIVE_ORDER_STATUSES),
            ).fetchone()
            return float(row["reserved"] or 0.0)

    def evaluate_persistent_risk(
        self,
        *,
        equity: float,
        cash: float = 0.0,
        buying_power: float = 0.0,
        mode: str,
        daily_limit_pct: float = 0.005,
        weekly_limit_pct: float = 0.01,
        total_limit_pct: float = 0.03,
        now: datetime | None = None,
        timezone_name: str = "America/Chicago",
    ) -> dict[str, Any]:
        if equity <= 0:
            raise ValueError("equity must be positive")
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        local = current.astimezone(ZoneInfo(timezone_name))
        trading_date = local.date().isoformat()
        iso_year, iso_week, _ = local.isocalendar()
        week_key = f"{iso_year}-W{iso_week:02d}"
        now_iso = current.astimezone(timezone.utc).isoformat()

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM strategy_risk_state WHERE id=1").fetchone()
            if row is None:
                daily_start = weekly_start = peak = equity
                halted_until = None
                halt_reason = None
                manual_halt = False
            else:
                daily_start = float(row["daily_start_equity"])
                weekly_start = float(row["weekly_start_equity"])
                peak = max(float(row["peak_equity"]), equity)
                halted_until = row["halted_until"]
                halt_reason = row["halt_reason"]
                manual_halt = bool(row["manual_halt"])
                # A manual halt can be created before the first broker snapshot.
                if daily_start <= 0:
                    daily_start = equity
                if weekly_start <= 0:
                    weekly_start = equity
                if row["trading_date"] != trading_date:
                    daily_start = equity
                    if halt_reason and str(halt_reason).startswith("daily"):
                        halted_until = None
                        halt_reason = None
                if row["week_key"] != week_key:
                    weekly_start = equity
                    if halt_reason and str(halt_reason).startswith("weekly"):
                        halted_until = None
                        halt_reason = None

            daily_drawdown = (equity / daily_start) - 1 if daily_start else 0.0
            weekly_drawdown = (equity / weekly_start) - 1 if weekly_start else 0.0
            total_drawdown = (equity / peak) - 1 if peak else 0.0

            if manual_halt:
                halt_reason = halt_reason or "manual halt"
            elif total_drawdown <= -abs(total_limit_pct):
                halted_until = "9999-12-31T23:59:59+00:00"
                halt_reason = f"total drawdown {total_drawdown:.2%}"
            elif weekly_drawdown <= -abs(weekly_limit_pct):
                halted_until = (current + timedelta(hours=72)).astimezone(timezone.utc).isoformat()
                halt_reason = f"weekly drawdown {weekly_drawdown:.2%}"
            elif daily_drawdown <= -abs(daily_limit_pct):
                halted_until = (current + timedelta(hours=24)).astimezone(timezone.utc).isoformat()
                halt_reason = f"daily drawdown {daily_drawdown:.2%}"

            active_halt = manual_halt
            if halted_until:
                try:
                    active_halt = active_halt or datetime.fromisoformat(halted_until) > current
                except ValueError:
                    active_halt = True

            conn.execute(
                """INSERT INTO strategy_risk_state (
                    id, trading_date, week_key, daily_start_equity,
                    weekly_start_equity, peak_equity, halted_until,
                    halt_reason, manual_halt, updated_at
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    trading_date=excluded.trading_date,
                    week_key=excluded.week_key,
                    daily_start_equity=excluded.daily_start_equity,
                    weekly_start_equity=excluded.weekly_start_equity,
                    peak_equity=excluded.peak_equity,
                    halted_until=excluded.halted_until,
                    halt_reason=excluded.halt_reason,
                    manual_halt=excluded.manual_halt,
                    updated_at=excluded.updated_at""",
                (
                    trading_date,
                    week_key,
                    daily_start,
                    weekly_start,
                    peak,
                    halted_until,
                    halt_reason,
                    1 if manual_halt else 0,
                    now_iso,
                ),
            )
            conn.execute(
                """INSERT INTO strategy_equity_snapshots (
                    timestamp, equity, cash, buying_power, daily_drawdown_pct,
                    weekly_drawdown_pct, total_drawdown_pct, mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now_iso,
                    equity,
                    cash,
                    buying_power,
                    daily_drawdown,
                    weekly_drawdown,
                    total_drawdown,
                    mode,
                ),
            )

        return {
            "halted": active_halt,
            "reason": halt_reason,
            "halted_until": halted_until,
            "daily_drawdown_pct": daily_drawdown * 100,
            "weekly_drawdown_pct": weekly_drawdown * 100,
            "total_drawdown_pct": total_drawdown * 100,
            "peak_equity": peak,
        }

    def set_manual_halt(self, reason: str = "manual halt") -> None:
        if not reason.strip():
            raise ValueError("A halt reason is required")
        now = utc_now_iso()
        current = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Chicago"))
        iso_year, iso_week, _ = current.isocalendar()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO strategy_risk_state (
                       id, trading_date, week_key, daily_start_equity,
                       weekly_start_equity, peak_equity, halted_until,
                       halt_reason, manual_halt, updated_at
                   ) VALUES (1, ?, ?, 0, 0, 0, NULL, ?, 1, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       manual_halt=1,
                       halt_reason=excluded.halt_reason,
                       halted_until=NULL,
                       updated_at=excluded.updated_at""",
                (current.date().isoformat(), f"{iso_year}-W{iso_week:02d}", reason.strip(), now),
            )

    def clear_manual_halt(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE strategy_risk_state
                   SET manual_halt=0, halt_reason=NULL, halted_until=NULL, updated_at=?
                   WHERE id=1""",
                (utc_now_iso(),),
            )

    def risk_status(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM strategy_risk_state WHERE id=1").fetchone()
            return dict(row) if row else None

    def record_health_event(
        self,
        severity: str,
        component: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO strategy_health_events (
                    timestamp, severity, component, message, context_json
                ) VALUES (?, ?, ?, ?, ?)""",
                (utc_now_iso(), severity, component, message, json.dumps(context or {})),
            )

    def acknowledge_health_events(self, note: str) -> int:
        if not note.strip():
            raise ValueError("An acknowledgement note is required")
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE strategy_health_events
                   SET acknowledged_at=?, acknowledgement=?
                   WHERE acknowledged_at IS NULL""",
                (utc_now_iso(), note.strip()),
            )
            return int(cursor.rowcount)

    def reconcile_account_activities(self, activities: list[dict[str, Any]]) -> int:
        recorded = 0
        with self._connect() as conn:
            for activity in activities:
                activity_id = str(
                    activity.get("id")
                    or activity.get("activity_id")
                    or activity.get("transaction_id")
                    or ""
                )
                if not activity_id:
                    continue
                cursor = conn.execute(
                    """INSERT OR IGNORE INTO strategy_account_activities (
                        activity_id, activity_type, ticker, quantity, price,
                        net_amount, activity_at, raw_json, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        activity_id,
                        str(activity.get("activity_type") or activity.get("type") or "unknown"),
                        activity.get("symbol"),
                        float(activity.get("qty", 0) or 0),
                        float(activity.get("price", 0) or 0),
                        float(activity.get("net_amount", 0) or 0),
                        activity.get("transaction_time") or activity.get("date"),
                        json.dumps(activity, default=str),
                        utc_now_iso(),
                    ),
                )
                recorded += int(cursor.rowcount > 0)
        return recorded

    def performance_summary(self) -> dict[str, Any]:
        """Calculate realized P&L from confirmed fill rows using average cost."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT ticker, side, quantity, fill_price
                   FROM strategy_trades ORDER BY trade_id"""
            ).fetchall()
            latest_equity = conn.execute(
                """SELECT equity, timestamp FROM strategy_equity_snapshots
                   ORDER BY id DESC LIMIT 1"""
            ).fetchone()
            activity_fees = conn.execute(
                """SELECT SUM(CASE WHEN net_amount < 0 THEN -net_amount ELSE 0 END) AS fees
                   FROM strategy_account_activities
                   WHERE UPPER(activity_type) IN ('FEE', 'REGULATORY_FEE')"""
            ).fetchone()

        inventory: dict[str, dict[str, float]] = {}
        realized = 0.0
        closed_quantity = 0.0
        for row in rows:
            ticker = str(row["ticker"])
            side = str(row["side"])
            qty = float(row["quantity"] or 0)
            price = float(row["fill_price"] or 0)
            position = inventory.setdefault(ticker, {"qty": 0.0, "cost": 0.0})
            if side == "buy":
                position["cost"] += qty * price
                position["qty"] += qty
            elif side == "sell" and qty > 0:
                sell_qty = min(qty, position["qty"])
                avg_cost = position["cost"] / position["qty"] if position["qty"] else 0.0
                realized += sell_qty * (price - avg_cost)
                position["qty"] -= sell_qty
                position["cost"] = max(0.0, position["cost"] - sell_qty * avg_cost)
                closed_quantity += sell_qty

        return {
            "realized_pnl": round(realized - float(activity_fees["fees"] or 0), 2),
            "fees": round(float(activity_fees["fees"] or 0), 2),
            "fill_count": len(rows),
            "closed_quantity": closed_quantity,
            "open_cost_basis": round(sum(item["cost"] for item in inventory.values()), 2),
            "latest_equity": float(latest_equity["equity"]) if latest_equity else None,
            "latest_equity_at": str(latest_equity["timestamp"]) if latest_equity else None,
        }

    def health_snapshot(self) -> dict[str, Any]:
        placeholders = ",".join("?" for _ in ACTIVE_ORDER_STATUSES)
        with self._connect() as conn:
            last_cycle = conn.execute(
                """SELECT cycle_id, started_at, completed_at, mode, status, error
                   FROM strategy_cycles ORDER BY cycle_id DESC LIMIT 1"""
            ).fetchone()
            active_orders = conn.execute(
                f"SELECT COUNT(*) AS count FROM strategy_orders WHERE status IN ({placeholders})",
                tuple(ACTIVE_ORDER_STATUSES),
            ).fetchone()
            positions = conn.execute(
                """SELECT COUNT(*) AS count, MAX(last_reconciled_at) AS last_reconciled_at
                   FROM strategy_positions WHERE status IN ('open', 'broker_only', 'closing')"""
            ).fetchone()
            recent_events = conn.execute(
                """SELECT timestamp, severity, component, message, context_json
                   FROM strategy_health_events ORDER BY id DESC LIMIT 10"""
            ).fetchall()
        unprotected = [str(row["ticker"]) for row in self.positions_needing_stop_orders()]
        return {
            "last_cycle": dict(last_cycle) if last_cycle else None,
            "active_orders": int(active_orders["count"] or 0),
            "open_positions": int(positions["count"] or 0),
            "last_reconciled_at": positions["last_reconciled_at"],
            "unprotected_tickers": unprotected,
            "risk": self.risk_status(),
            "performance": self.performance_summary(),
            "recent_events": [dict(row) for row in recent_events],
        }

    def release_statistics(self) -> dict[str, Any]:
        with self._connect() as conn:
            cycles = conn.execute(
                """SELECT COUNT(*) AS total,
                          MIN(started_at) AS first_started,
                          MAX(COALESCE(completed_at, started_at)) AS last_completed,
                          SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                          SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS unresolved
                   FROM strategy_cycles WHERE mode IN ('paper', 'live')"""
            ).fetchone()
            critical = conn.execute(
                """SELECT COUNT(*) AS count FROM strategy_health_events
                   WHERE severity='critical' AND acknowledged_at IS NULL"""
            ).fetchone()
        first = cycles["first_started"] if cycles else None
        last = cycles["last_completed"] if cycles else None
        span_days = 0
        if first and last:
            try:
                span_days = max(
                    0,
                    (datetime.fromisoformat(last) - datetime.fromisoformat(first)).days,
                )
            except ValueError:
                span_days = 0
        return {
            "paper_cycles": int(cycles["total"] or 0),
            "failed_cycles": int(cycles["failed"] or 0),
            "unresolved_cycles": int(cycles["unresolved"] or 0),
            "paper_span_days": span_days,
            "unacknowledged_critical_events": int(critical["count"] or 0),
            "failure_rate_pct": (
                (int(cycles["failed"] or 0) / int(cycles["total"] or 1)) * 100
            ),
        }

    def backup_to(self, destination: str | Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as source, sqlite3.connect(str(target)) as backup:
            source.backup(backup)
            result = backup.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                raise RuntimeError(f"SQLite backup integrity check failed for {target}")
        return target

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
                          reason, order_id, execution_id, mode
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
                      AND mode IN ('live', 'paper', 'real')
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
            self.record_order_tree(order, mode)

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

            if mode == "live":
                conn.execute(
                    """UPDATE strategy_positions SET stop_order_id=NULL, updated_at=?
                       WHERE status IN ('open', 'broker_only')""",
                    (now,),
                )
                placeholders = ",".join("?" for _ in ACTIVE_ORDER_STATUSES)
                stop_rows = conn.execute(
                    f"""SELECT order_id, ticker FROM strategy_orders
                        WHERE side='sell'
                          AND order_type IN ('stop', 'stop_limit', 'trailing_stop')
                          AND status IN ({placeholders})""",
                    tuple(ACTIVE_ORDER_STATUSES),
                ).fetchall()
                for stop in stop_rows:
                    conn.execute(
                        """UPDATE strategy_positions SET stop_order_id=?, updated_at=?
                           WHERE id=(
                               SELECT id FROM strategy_positions
                               WHERE ticker=? AND status IN ('open', 'broker_only')
                               ORDER BY id DESC LIMIT 1
                           )""",
                        (stop["order_id"], now, stop["ticker"]),
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
