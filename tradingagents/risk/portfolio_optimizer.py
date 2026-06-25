from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def optimize_portfolio(
    tickers: list[str],
    method: str = "MV",
    lookback_days: int = 252,
    min_weight: float = 0.0,
    max_weight: float = 0.3,
) -> pd.Series | None:
    if len(tickers) < 2:
        if len(tickers) == 1:
            return pd.Series({tickers[0]: 1.0})
        return None

    try:
        data = yf.download(
            tickers,
            period=f"{lookback_days}d",
            auto_adjust=True,
            progress=False,
        )
        close = data["Close"]
        returns = close.pct_change().dropna()

        import riskfolio as rp

        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu="hist", method_cov="hist")

        method_map = {
            "MV": ("Classic", "MV"),
            "CVaR": ("Classic", "CVaR"),
            "MS": ("Classic", "MS"),  # Max Sharpe
        }
        model, rm = method_map.get(method, ("Classic", "MV"))

        weights = port.optimization(
            model=model,
            rm=rm,
            obj="Sharpe",
            hist=True,
        )

        if weights is not None and not weights.empty:
            weights = weights.clip(lower=min_weight, upper=max_weight)
            weights = weights / weights.sum()

        return weights

    except ImportError:
        logger.warning("riskfolio-lib not installed; using equal-weight fallback")
        return pd.Series({t: 1.0 / len(tickers) for t in tickers})
    except Exception as exc:
        logger.error("Portfolio optimization failed: %s", exc)
        return pd.Series({t: 1.0 / len(tickers) for t in tickers})
