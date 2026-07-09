from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_cash: float = 10_000.0
    position_pct: float = 0.02
    stop_loss_pct: float = -0.05
    take_profit_pct: float = 0.08
    max_hold_days: int = 10
    min_daily_volume: float = 500_000
    min_score: float = 2.0
    fee_pct: float = 0.001


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))


def build_walk_forward_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build features that are shifted one day to avoid look-ahead bias."""
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    raw = pd.DataFrame(index=df.index)
    raw["close"] = df["Close"]
    raw["volume"] = df["Volume"]
    raw["sma20"] = df["Close"].rolling(20).mean()
    raw["sma50"] = df["Close"].rolling(50).mean()
    raw["avg_volume20"] = df["Volume"].rolling(20).mean()
    raw["rsi14"] = _rsi(df["Close"])

    features = raw.shift(1)
    df["trend_score"] = (features["close"] > features["sma50"]).astype(float)
    df["momentum_score"] = (features["close"] > features["sma20"]).astype(float)
    df["volume_score"] = (
        (features["avg_volume20"] > 0)
        & (features["volume"] / features["avg_volume20"] >= 1.5)
    ).astype(float)
    df["rsi_score"] = ((features["rsi14"] >= 35) & (features["rsi14"] <= 70)).astype(float)
    df["signal_score"] = (
        df["trend_score"] + df["momentum_score"] + df["volume_score"] + df["rsi_score"]
    )
    return df


def run_walk_forward_backtest(
    data: pd.DataFrame,
    config: WalkForwardConfig | None = None,
) -> dict[str, Any]:
    """Simulate a simple low-risk long strategy with next-open execution."""
    cfg = config or WalkForwardConfig()
    df = build_walk_forward_features(data)
    cash = cfg.initial_cash
    position_qty = 0.0
    entry_price = 0.0
    entry_date: pd.Timestamp | None = None
    hold_days = 0
    equity_curve: list[float] = []
    trades: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        date = row["Date"]
        if open_price <= 0 or close_price <= 0:
            continue

        if position_qty > 0:
            hold_days += 1
            return_pct = (open_price - entry_price) / entry_price
            exit_reason = None
            if return_pct <= cfg.stop_loss_pct:
                exit_reason = "stop_loss"
            elif return_pct >= cfg.take_profit_pct:
                exit_reason = "take_profit"
            elif hold_days >= cfg.max_hold_days:
                exit_reason = "time_exit"

            if exit_reason:
                proceeds = position_qty * open_price * (1 - cfg.fee_pct)
                pnl = proceeds - (position_qty * entry_price)
                cash += proceeds
                trades.append(
                    {
                        "entry_date": entry_date.strftime("%Y-%m-%d") if entry_date else "",
                        "exit_date": date.strftime("%Y-%m-%d"),
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(open_price, 4),
                        "return_pct": round(return_pct * 100, 2),
                        "pnl": round(pnl, 2),
                        "reason": exit_reason,
                    }
                )
                position_qty = 0.0
                entry_price = 0.0
                entry_date = None
                hold_days = 0

        if position_qty == 0 and row["Volume"] >= cfg.min_daily_volume:
            if float(row["signal_score"]) >= cfg.min_score:
                notional = cash * cfg.position_pct
                if notional > 0:
                    cost = notional * (1 + cfg.fee_pct)
                    if cost <= cash:
                        position_qty = notional / open_price
                        cash -= cost
                        entry_price = open_price
                        entry_date = date
                        hold_days = 0

        equity_curve.append(cash + position_qty * close_price)

    if position_qty > 0 and not df.empty:
        last = df.iloc[-1]
        final_price = float(last["Close"])
        proceeds = position_qty * final_price * (1 - cfg.fee_pct)
        return_pct = (final_price - entry_price) / entry_price if entry_price else 0
        cash += proceeds
        trades.append(
            {
                "entry_date": entry_date.strftime("%Y-%m-%d") if entry_date else "",
                "exit_date": last["Date"].strftime("%Y-%m-%d"),
                "entry_price": round(entry_price, 4),
                "exit_price": round(final_price, 4),
                "return_pct": round(return_pct * 100, 2),
                "pnl": round(proceeds - (position_qty * entry_price), 2),
                "reason": "end_of_test",
            }
        )

    equity = pd.Series(equity_curve, dtype="float64")
    returns = equity.pct_change().dropna()
    total_return = (cash / cfg.initial_cash - 1) * 100
    rolling_max = equity.cummax() if not equity.empty else pd.Series(dtype="float64")
    drawdown = ((equity - rolling_max) / rolling_max).min() * 100 if not equity.empty else 0
    sharpe = (
        returns.mean() / returns.std() * (252 ** 0.5)
        if not returns.empty and returns.std() > 0
        else 0.0
    )
    win_rate = (
        sum(1 for trade in trades if float(trade["return_pct"]) > 0) / len(trades) * 100
        if trades
        else 0.0
    )
    return {
        "total_return_pct": round(float(total_return), 2),
        "max_drawdown_pct": round(float(drawdown), 2),
        "sharpe": round(float(sharpe), 2),
        "win_rate_pct": round(float(win_rate), 2),
        "num_trades": len(trades),
        "final_equity": round(float(cash), 2),
        "trades": trades,
    }
