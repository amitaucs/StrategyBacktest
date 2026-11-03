"""
Performance metrics calculation engine.

Computes all 12 institutional trading ratios, drawdown metrics, period aggregations,
and classification ratings from completed trade logs.
"""

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd

from .classifier import (
    classify_cagr,
    classify_max_dd,
    classify_calmar,
    classify_sharpe,
    classify_sortino,
    classify_profit_factor,
    classify_expectancy,
    classify_recovery_factor,
    classify_consec_losses,
    classify_dd_duration,
    classify_win_rate,
    classify_win_loss_ratio,
)


def calculate_ratios(
    trades_df: pd.DataFrame,
    all_dates: pd.Series,
    initial_capital: float = 1_000_000.0,
    risk_free_rate: float = 0.065,
    frequency: str = "auto",
    equity_curve: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Computes all 12 institutional trading ratios and performance metrics.

    Args:
        trades_df: DataFrame containing trade records with columns:
                   'date', 'pnl_inr', 'stat_cost', 'exit_reason', and optional 'mae_inr'.
        all_dates: Series or DatetimeIndex of all valid market trading dates in backtest period.
        initial_capital: Starting portfolio capital in ₹ (default: 1,000,000.0).
        risk_free_rate: Annual risk-free interest rate for Sharpe/Sortino (default: 6.5% p.a.).
        frequency: Observation frequency for return annualization: 'auto', 'daily', 'monthly', or 'weekly'.
        equity_curve: Optional explicit Mark-to-Market equity series. If provided, CAGR, drawdowns,
                      and return streams will be computed directly from this series rather than
                      inferring them solely from realized trade exit events.

    Returns:
        Dictionary containing all metrics, series, ratings, and summary classifications.
    """
    if not isinstance(all_dates, pd.Series):
        all_dates = pd.Series(list(all_dates))

    if trades_df.empty and (equity_curve is None or len(equity_curve) == 0):
        return {
            "empty": True,
            "total_trades": 0,
            "total_days": all_dates.nunique() if not all_dates.empty else 0,
            "frequency": frequency,
        }

    start_date = pd.to_datetime(all_dates.min())
    end_date = pd.to_datetime(all_dates.max())
    total_days = all_dates.nunique()
    total_trades = len(trades_df)

    # 1. Determine Observation Frequency and Annualization Factors
    sorted_dates = sorted(all_dates.unique())
    freq_lower = frequency.lower()
    if freq_lower == "auto":
        if len(sorted_dates) > 1:
            diffs = pd.to_datetime(pd.Series(sorted_dates)).diff().dt.days.dropna()
            median_gap = diffs.median()
            if median_gap >= 20:
                detected_freq = "monthly"
            elif median_gap >= 5:
                detected_freq = "weekly"
            else:
                detected_freq = "daily"
        else:
            detected_freq = "daily"
    else:
        detected_freq = freq_lower

    if detected_freq == "monthly":
        periods_per_year = 12.0
        bars_per_month = 1.0
        t0_delta = pd.Timedelta(days=30)
    elif detected_freq == "weekly":
        periods_per_year = 52.0
        bars_per_month = 4.33
        t0_delta = pd.Timedelta(days=7)
    else:  # daily
        periods_per_year = 252.0
        bars_per_month = 21.0
        t0_delta = pd.Timedelta(days=1)

    # 2. Baseline & Equity Curve Construction (Mark-to-Market or Realized Fallback)
    if equity_curve is not None and not equity_curve.empty:
        user_eq = pd.Series(equity_curve).copy()
        user_eq.index = pd.to_datetime(user_eq.index)
        user_eq = user_eq.sort_index()
        t0_date = user_eq.index[0] - t0_delta
        if float(user_eq.iloc[0]) != float(initial_capital):
            full_equity = pd.concat([pd.Series([initial_capital], index=[t0_date]), user_eq])
        else:
            full_equity = user_eq
        equity_curve = full_equity
        period_returns = equity_curve.pct_change().dropna()
    else:
        daily_pnl = trades_df.groupby("date")["pnl_inr"].sum().reindex(sorted_dates, fill_value=0.0)
        t0_date = pd.to_datetime(sorted_dates[0]) - t0_delta
        equity_curve = pd.concat([pd.Series([initial_capital], index=[t0_date]), initial_capital + daily_pnl.cumsum()])
        period_returns = equity_curve.pct_change().dropna()

    # 3. CAGR (Annual Return)
    years = max((end_date - start_date).days / 365.25, 0.01)
    ending_capital = equity_curve.iloc[-1]
    cagr = ((ending_capital / initial_capital) ** (1.0 / years) - 1.0) * 100.0

    # 4. Maximum Drawdown (% and ₹) — Conservative composite of Period, Trade-level, & MAE
    running_peak = equity_curve.cummax()
    dd_inr_series = running_peak - equity_curve
    dd_pct_series = (dd_inr_series / running_peak) * 100.0
    period_max_dd_pct = dd_pct_series.max()
    period_max_dd_inr = dd_inr_series.max()

    if not trades_df.empty:
        trade_equity = pd.concat([pd.Series([initial_capital]), initial_capital + trades_df["pnl_inr"].cumsum()])
        trade_peak = trade_equity.cummax()
        trade_max_dd_pct = ((trade_peak - trade_equity) / trade_peak).max() * 100.0
        trade_max_dd_inr = (trade_peak - trade_equity).max()

        mae_series = trades_df["mae_inr"] if "mae_inr" in trades_df else pd.Series([0.0] * len(trades_df))
        intratrade_troughs = (trade_peak.iloc[:-1].values - trade_equity.iloc[:-1].values) + mae_series.values
        intratrade_max_dd_inr = float(intratrade_troughs.max()) if len(intratrade_troughs) > 0 else 0.0

        # Evaluate intra-trade drawdown % against contemporaneous peak rather than initial capital
        contemporaneous_peaks = trade_peak.iloc[:-1].values
        intratrade_dd_pct_series = np.where(
            contemporaneous_peaks > 0,
            (intratrade_troughs / contemporaneous_peaks) * 100.0,
            0.0
        )
        intratrade_max_dd_pct = float(intratrade_dd_pct_series.max()) if len(intratrade_dd_pct_series) > 0 else 0.0

        max_dd_pct = max(period_max_dd_pct, trade_max_dd_pct, intratrade_max_dd_pct)
        max_dd_inr = max(period_max_dd_inr, trade_max_dd_inr, intratrade_max_dd_inr)
    else:
        max_dd_pct = period_max_dd_pct
        max_dd_inr = period_max_dd_inr

    # 5. Calmar Ratio
    calmar_ratio = cagr / max_dd_pct if max_dd_pct > 0 else np.nan

    # 6. Sharpe Ratio (Correctly Annualized by Observation Frequency)
    rf_period = risk_free_rate / periods_per_year
    excess_returns = period_returns - rf_period
    period_std = period_returns.std()
    sharpe_ratio = (excess_returns.mean() / period_std) * np.sqrt(periods_per_year) if period_std > 0 else 0.0

    # 7. Sortino Ratio (Correctly Annualized by Observation Frequency)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0.0
    sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

    # 8. Trade Metrics & P&L Aggregations
    if not trades_df.empty:
        winners = trades_df[trades_df["pnl_inr"] > 0]["pnl_inr"]
        losers = trades_df[trades_df["pnl_inr"] <= 0]["pnl_inr"]
        gross_profit = winners.sum()
        gross_loss = abs(losers.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        combined_total_pnl = trades_df["pnl_inr"].sum()
        expectancy_per_trade = combined_total_pnl / total_trades if total_trades > 0 else 0.0
        recovery_factor = combined_total_pnl / max_dd_inr if max_dd_inr > 0 else np.inf

        win_rate = (len(winners) / total_trades) * 100.0 if total_trades > 0 else 0.0
        avg_win = winners.mean() if len(winners) > 0 else 0.0
        avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else (np.inf if avg_win > 0 else 0.0)

        is_loss = (trades_df["pnl_inr"] <= 0).astype(int)
        loss_streaks = is_loss.groupby((is_loss != is_loss.shift()).cumsum()).sum()
        max_consecutive_losses = int(loss_streaks.max()) if not loss_streaks.empty else 0

        trades_df_copy = trades_df.copy()
        trades_df_copy["year"] = pd.to_datetime(trades_df_copy["date"]).dt.year
        trades_df_copy["ym"] = pd.to_datetime(trades_df_copy["date"]).dt.strftime("%b %Y")
        yearly_pnl = trades_df_copy.groupby("year")["pnl_inr"].sum()
        monthly_pnl = trades_df_copy.groupby("ym")["pnl_inr"].sum()

        best_year = f"{yearly_pnl.idxmax()} (₹{yearly_pnl.max():,.2f})" if not yearly_pnl.empty else "N/A"
        best_month = f"{monthly_pnl.idxmax()} (₹{monthly_pnl.max():,.2f})" if not monthly_pnl.empty else "N/A"
        worst_month = f"{monthly_pnl.idxmin()} (₹{monthly_pnl.min():,.2f})" if not monthly_pnl.empty else "N/A"

        total_stat_costs = trades_df["stat_cost"].sum() if "stat_cost" in trades_df else 0.0
        exit_counts = trades_df["exit_reason"].value_counts().to_dict() if "exit_reason" in trades_df else {}

        pf_rating = classify_profit_factor(profit_factor)
        expectancy_rating = classify_expectancy(expectancy_per_trade)
        rf_rating = classify_recovery_factor(recovery_factor)
        win_rate_rating = classify_win_rate(win_rate)
        win_loss_rating = classify_win_loss_ratio(avg_win_loss_ratio)
    else:
        combined_total_pnl = float(ending_capital - initial_capital)
        profit_factor = np.nan
        expectancy_per_trade = 0.0
        recovery_factor = combined_total_pnl / max_dd_inr if max_dd_inr > 0 else np.nan
        win_rate = 0.0
        avg_win_loss_ratio = 0.0
        max_consecutive_losses = 0
        best_year = "N/A"
        best_month = "N/A"
        worst_month = "N/A"
        total_stat_costs = 0.0
        exit_counts = {}

        pf_rating = "N/A (OPEN)"
        expectancy_rating = "N/A (OPEN)"
        rf_rating = classify_recovery_factor(recovery_factor) if not np.isnan(recovery_factor) else "N/A (OPEN)"
        win_rate_rating = "N/A (OPEN)"
        win_loss_rating = "N/A (OPEN)"

    # 13. Drawdown Duration (in Months, Corrected for Frequency)
    in_dd = equity_curve < running_peak
    dd_groups = (~in_dd).cumsum()[in_dd]
    if not dd_groups.empty:
        max_dd_bars = dd_groups.value_counts().max()
        avg_dd_bars = dd_groups.value_counts().mean()
        max_dd_duration_months = max_dd_bars / bars_per_month
        avg_dd_duration_months = avg_dd_bars / bars_per_month
        dd_duration_months = max_dd_duration_months
    else:
        max_dd_duration_months = 0.0
        avg_dd_duration_months = 0.0
        dd_duration_months = 0.0

    # Standard Ratings
    cagr_rating = classify_cagr(cagr)
    max_dd_rating = classify_max_dd(max_dd_pct)
    calmar_rating = classify_calmar(calmar_ratio)
    sharpe_rating = classify_sharpe(sharpe_ratio)
    sortino_rating = classify_sortino(sortino_ratio)
    consec_losses_rating = classify_consec_losses(max_consecutive_losses)
    dd_duration_rating = classify_dd_duration(dd_duration_months)

    all_ratings = [
        cagr_rating, max_dd_rating, calmar_rating, sharpe_rating,
        sortino_rating, pf_rating, expectancy_rating, rf_rating,
        win_rate_rating, win_loss_rating, consec_losses_rating, dd_duration_rating
    ]

    score_map = {
        "EXCELLENT": 4, "GOOD": 3, "ACCEPTABLE": 2, "MODERATE": 2,
        "WEAK": 1, "RISKY": 0, "UNACCEPTABLE": 0, "LOSS MAKING": 0, "REJECT": -5
    }
    avg_score = sum(score_map.get(r, 1) for r in all_ratings) / len(all_ratings)

    if expectancy_rating == "REJECT" or pf_rating == "LOSS MAKING":
        overall_outcome = "REJECT / UNVIABLE"
    elif "UNACCEPTABLE" in all_ratings or "RISKY" in all_ratings:
        overall_outcome = "ACCEPTABLE (HIGH DRAWDOWN RISK)"
    elif avg_score >= 3.0:
        overall_outcome = "EXCELLENT"
    elif avg_score >= 2.2:
        overall_outcome = "GOOD"
    elif avg_score >= 1.5:
        overall_outcome = "ACCEPTABLE"
    else:
        overall_outcome = "WEAK"

    return {
        "empty": False,
        "start_date": start_date,
        "end_date": end_date,
        "total_days": total_days,
        "frequency": detected_freq,
        "total_trades": total_trades,
        "initial_capital": initial_capital,
        "ending_capital": ending_capital,
        "equity_curve": equity_curve,
        "period_returns": period_returns,
        "daily_returns": period_returns,
        # 12 Core Ratios
        "cagr": cagr,
        "cagr_rating": cagr_rating,
        "max_dd_pct": max_dd_pct,
        "max_dd_inr": max_dd_inr,
        "max_dd_rating": max_dd_rating,
        "calmar_ratio": calmar_ratio,
        "calmar_rating": calmar_rating,
        "sharpe_ratio": sharpe_ratio,
        "sharpe_rating": sharpe_rating,
        "sortino_ratio": sortino_ratio,
        "sortino_rating": sortino_rating,
        "profit_factor": profit_factor,
        "profit_factor_rating": pf_rating,
        "expectancy_per_trade": expectancy_per_trade,
        "expectancy_rating": expectancy_rating,
        "recovery_factor": recovery_factor,
        "recovery_factor_rating": rf_rating,
        "win_rate": win_rate,
        "win_rate_rating": win_rate_rating,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "win_loss_rating": win_loss_rating,
        "max_consecutive_losses": max_consecutive_losses,
        "consec_losses_rating": consec_losses_rating,
        "max_dd_duration_months": max_dd_duration_months,
        "avg_dd_duration_months": avg_dd_duration_months,
        "dd_duration_rating": dd_duration_rating,
        # Financial Totals & Exits
        "combined_total_pnl": combined_total_pnl,
        "total_stat_costs": total_stat_costs,
        "exit_counts": exit_counts,
        "best_year": best_year,
        "best_month": best_month,
        "worst_month": worst_month,
        "all_ratings": all_ratings,
        "overall_outcome": overall_outcome,
    }


__all__ = ["calculate_ratios"]

