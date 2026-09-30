"""
Historical Backtesting Engine

Implements a simple directional strategy:
  - If predicted direction = UP → go long (buy)
  - If predicted direction = DOWN → stay flat (no position)

IMPORTANT DISCLAIMERS (enforced in responses):
  - This is an educational simulation using historical data.
  - Does NOT account for: transaction costs, slippage, bid-ask spread,
    market impact, taxes, or real-world order execution.
  - Past performance is NOT indicative of future results.
  - All chronological ordering is strictly preserved — no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date

from app.logging_config import get_logger

logger = get_logger(__name__)


def run_backtest(
    price_series: pd.Series,
    predicted_directions: list[str],  # "UP" or "DOWN" for each period
    start_capital: float = 10_000.0,
) -> dict:
    """
    Simulate a simple long-flat strategy.

    Parameters
    ----------
    price_series : pd.Series
        Daily close prices indexed by date, chronologically ordered.
    predicted_directions : list[str]
        One prediction per period. Must align with price_series.
    start_capital : float
        Starting portfolio value in dollars.

    Returns
    -------
    dict with equity_curve, trade_log, and performance metrics.
    """
    if len(price_series) != len(predicted_directions):
        raise ValueError(
            f"Length mismatch: prices={len(price_series)}, directions={len(predicted_directions)}"
        )

    prices = price_series.values
    dates = price_series.index

    equity = start_capital
    equity_curve = []
    trade_log = []
    positions_held = []
    daily_returns = []

    # Simulate day by day
    for i in range(len(prices) - 1):
        direction = predicted_directions[i]
        daily_return = (prices[i + 1] - prices[i]) / prices[i]

        if direction == "UP":
            # Long position — capture the next-day return
            period_return = daily_return
            in_trade = True
        else:
            # Flat — no exposure, 0 return
            period_return = 0.0
            in_trade = False

        equity *= (1 + period_return)
        daily_returns.append(period_return)
        positions_held.append(1 if in_trade else 0)

        equity_curve.append({
            "date": dates[i + 1].date() if hasattr(dates[i + 1], "date") else dates[i + 1],
            "equity": round(equity, 2),
            "cumulative_return": round((equity / start_capital - 1) * 100, 4),
        })

        if in_trade:
            trade_log.append({
                "date": str(dates[i].date() if hasattr(dates[i], "date") else dates[i]),
                "direction": direction,
                "entry_price": float(prices[i]),
                "exit_price": float(prices[i + 1]),
                "pnl_pct": round(daily_return * 100, 4),
                "result": "WIN" if daily_return > 0 else "LOSS",
            })

    # ── Performance Metrics ────────────────────────────────────────────────────
    total_trades = len(trade_log)
    wins = sum(1 for t in trade_log if t["result"] == "WIN")
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    cumulative_return = (equity / start_capital - 1) * 100  # as percentage

    max_drawdown = _compute_max_drawdown([e["equity"] for e in equity_curve], start_capital)

    sharpe = _compute_sharpe(daily_returns)

    logger.info(
        "Backtest complete",
        total_trades=total_trades,
        win_rate=f"{win_rate:.2%}",
        cumulative_return=f"{cumulative_return:.2f}%",
        max_drawdown=f"{max_drawdown:.2f}%",
        sharpe=f"{sharpe:.3f}",
    )

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "equity_curve": equity_curve,
        "trade_log": trade_log,
    }


def _compute_max_drawdown(equity_values: list[float], start_capital: float) -> float:
    """Maximum drawdown as a percentage."""
    if not equity_values:
        return 0.0
    peak = start_capital
    max_dd = 0.0
    for eq in equity_values:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_sharpe(daily_returns: list[float], risk_free_daily: float = 0.0) -> float:
    """Annualised Sharpe ratio (assumes 252 trading days)."""
    if len(daily_returns) < 2:
        return 0.0
    arr = np.array(daily_returns)
    excess = arr - risk_free_daily
    std = excess.std()
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(252))
