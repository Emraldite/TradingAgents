from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def _volume_spike_signal(data: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
    volume = data["Volume"]
    avg_volume = volume.rolling(20).mean()
    return (volume > avg_volume * threshold) & (avg_volume > 0)


def _volume_drop_signal(data: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    volume = data["Volume"]
    avg_volume = volume.rolling(20).mean()
    return (volume < avg_volume * threshold) & (avg_volume > 0)


def _rsi_signal(data: pd.DataFrame, period: int = 14, oversold: float = 30) -> pd.Series:
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("inf"))
    rsi = 100 - (100 / (1 + rs))
    return rsi < oversold


def _price_above_sma(data: pd.DataFrame, period: int = 50) -> pd.Series:
    sma = data["Close"].rolling(period).mean()
    return data["Close"] > sma


def backtest_signal(
    ticker: str,
    entry_signal_fn,
    exit_signal_fn,
    start: str = "2020-01-01",
    end: str = "2024-01-01",
    init_cash: float = 10_000,
    fees: float = 0.001,
) -> dict[str, Any]:
    import vectorbt as vbt

    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        logger.warning("No data for %s", ticker)
        return {"ticker": ticker, "error": "no data"}

    entries = entry_signal_fn(data)
    exits = exit_signal_fn(data)

    pf = vbt.Portfolio.from_signals(
        data["Close"],
        entries,
        exits,
        init_cash=init_cash,
        fees=fees,
    )

    stats = pf.stats()
    return {
        "ticker": ticker,
        "total_return": stats.get("Total Return [%]", 0),
        "sharpe": stats.get("Sharpe Ratio", 0),
        "max_dd": stats.get("Max Drawdown [%]", 0),
        "win_rate": stats.get("Win Rate [%]", 0),
        "num_trades": stats.get("Total Trades", 0),
        "avg_hold": stats.get("Avg Holding Period", 0),
    }


def backtest_spy_benchmark(
    start: str = "2020-01-01",
    end: str = "2024-01-01",
) -> dict[str, Any]:
    import vectorbt as vbt

    spy = yf.download("SPY", start=start, end=end, progress=False)
    if spy.empty:
        return {"error": "no spy data"}

    spy_returns = spy["Close"].pct_change().dropna()
    cumulative = (1 + spy_returns).cumprod()
    total_ret = (cumulative.iloc[-1] - 1) * 100
    sharpe = spy_returns.mean() / spy_returns.std() * (252**0.5) if spy_returns.std() > 0 else 0
    rolling_max = cumulative.cummax()
    dd = (cumulative - rolling_max) / rolling_max
    max_dd = dd.min() * 100

    return {
        "ticker": "SPY",
        "total_return": round(total_ret, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 2),
    }


def run_all_backtests(
    tickers: list[str] | None = None,
    start: str = "2020-01-01",
    end: str = "2024-01-01",
) -> list[dict[str, Any]]:
    if tickers is None:
        tickers = ["LMT", "CVX", "AAPL", "MSFT", "AMZN", "JPM", "XOM", "UNH"]

    benchmark = backtest_spy_benchmark(start, end)
    results: list[dict[str, Any]] = [benchmark]

    for ticker in tickers:
        result = backtest_signal(ticker, _volume_spike_signal, _volume_drop_signal, start, end)
        result["signal"] = "volume_spike"
        results.append(result)
        logger.info("Volume spike backtest for %s: return %.1f%%", ticker, result.get("total_return", 0))

    for ticker in tickers:
        result = backtest_signal(ticker, _rsi_signal, lambda d: ~_rsi_signal(d), start, end)
        result["signal"] = "rsi_oversold"
        results.append(result)
        logger.info("RSI backtest for %s: return %.1f%%", ticker, result.get("total_return", 0))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "backtest_results.csv", index=False)
    logger.info("All backtest results saved to %s", RESULTS_DIR / "backtest_results.csv")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_all_backtests()
    for r in results:
        print(f"{r.get('ticker', '?'):>6} | {r.get('signal', 'benchmark'):>16} | "
              f"Return: {r.get('total_return', 0):>6.1f}% | "
              f"Sharpe: {r.get('sharpe', 0):>5.2f} | "
              f"MaxDD: {r.get('max_dd', 0):>5.1f}%")
