"""
Performance reporting dashboard module.

Formats and prints the standardized 12-ratio institutional scorecard,
exit breakdowns, and threshold compliance summaries using tabulate.
"""

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from tabulate import tabulate

from .classifier import (
    CAGR_BAD_BELOW,
    CAGR_ACCEPTABLE_MAX,
    CAGR_GOOD_MIN,
    CAGR_EXCELLENT_MIN,
    MAX_DD_EXCELLENT_BELOW,
    MAX_DD_GOOD_MAX,
    MAX_DD_ACCEPTABLE_MAX,
    CALMAR_WEAK_BELOW,
    CALMAR_ACCEPTABLE_MAX,
    CALMAR_GOOD_MAX,
    SHARPE_WEAK_BELOW,
    SHARPE_ACCEPTABLE_MAX,
    SHARPE_EXCELLENT_MIN,
    SORTINO_WEAK_BELOW,
    SORTINO_ACCEPTABLE_MAX,
    SORTINO_EXCELLENT_MIN,
    PROFIT_FACTOR_LOSS_BELOW,
    PROFIT_FACTOR_WEAK_MAX,
    PROFIT_FACTOR_ACCEPTABLE_MAX,
    PROFIT_FACTOR_EXCELLENT_MIN,
    EXPECTANCY_REJECT_BELOW,
    EXPECTANCY_WEAK_MAX,
    EXPECTANCY_EXCELLENT_MIN,
    RECOVERY_FACTOR_WEAK_BELOW,
    RECOVERY_FACTOR_ACCEPTABLE_MAX,
    RECOVERY_FACTOR_GOOD_MIN,
    RECOVERY_FACTOR_EXCELLENT_MIN,
    CONSEC_LOSSES_EXCELLENT_BELOW,
    CONSEC_LOSSES_GOOD_MAX,
    CONSEC_LOSSES_ACCEPTABLE_MAX,
    CONSEC_LOSSES_UNACCEPTABLE_ABOVE,
    DD_DURATION_EXCELLENT_BELOW,
    DD_DURATION_GOOD_MAX,
    DD_DURATION_ACCEPTABLE_MAX,
    DD_DURATION_WEAK_ABOVE,
    WIN_RATE_WEAK_BELOW,
    WIN_RATE_ACCEPTABLE_MAX,
    WIN_RATE_GOOD_MIN,
    WIN_RATE_EXCELLENT_MIN,
    WIN_LOSS_RATIO_WEAK_BELOW,
    WIN_LOSS_RATIO_ACCEPTABLE_MAX,
    WIN_LOSS_RATIO_GOOD_MIN,
    WIN_LOSS_RATIO_EXCELLENT_MIN,
)
from .calculator import calculate_ratios


def generate_performance_summary(
    trades_df: pd.DataFrame,
    all_dates: pd.Series,
    initial_capital: float = 1_000_000.0,
    strategy_title: str = "NIFTY 15M ST05 STRATEGY",
    risk_free_rate: float = 0.065,
    frequency: str = "auto",
    equity_curve: Optional[pd.Series] = None,
    open_positions: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Computes and prints the full institutional performance dashboard in the terminal.

    Args:
        trades_df: DataFrame of executed trades.
        all_dates: Series or DatetimeIndex of trading session dates.
        initial_capital: Starting capital in ₹.
        strategy_title: Title header for the terminal report.
        risk_free_rate: Annual risk-free rate (default: 6.5% p.a.).
        frequency: Observation frequency for return annualization: 'auto', 'daily', 'monthly', or 'weekly'.
        equity_curve: Optional explicit Mark-to-Market equity series.
        open_positions: Optional list of active open holding dictionaries.

    Returns:
        The metrics dictionary computed by calculate_ratios, or None if trades_df is empty.
    """
    if trades_df.empty and (equity_curve is None or len(equity_curve) == 0):
        print(f"\nNo trades or positions generated for {strategy_title} during the backtest period.\n")
        return None

    m = calculate_ratios(
        trades_df=trades_df,
        all_dates=all_dates,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        frequency=frequency,
        equity_curve=equity_curve,
    )

    if m.get("empty", False):
        print(f"\nNo trades or positions generated for {strategy_title} during the backtest period.\n")
        return None

    # Print Dashboard Header
    obs_label = "MONTHS" if m.get("frequency") == "monthly" else ("WEEKS" if m.get("frequency") == "weekly" else "DAYS")
    print("\n" + "=" * 85)
    print(f" BACKTEST PERFORMANCE SUMMARY — {strategy_title.upper()}")
    print(f" PERIOD: {m['start_date'].strftime('%Y-%m-%d')} to {m['end_date'].strftime('%Y-%m-%d')} | "
          f"TOTAL TRADES: {m['total_trades']} | TOTAL {obs_label}: {m['total_days']} | CAPITAL: ₹{m['initial_capital']:,.0f}")
    print("=" * 85)

    # 12-Metric Scorecard Grid
    pf_display = f"{m['profit_factor']:.2f} ({m['profit_factor_rating']})" if not pd.isna(m.get('profit_factor', np.nan)) else f"{m['profit_factor_rating']}"
    exp_display = f"₹{m['expectancy_per_trade']:,.2f} ({m['expectancy_rating']})" if m['total_trades'] > 0 else f"{m['expectancy_rating']}"
    rf_display = f"{m['recovery_factor']:.2f} ({m['recovery_factor_rating']})" if not pd.isna(m.get('recovery_factor', np.nan)) else f"{m['recovery_factor_rating']}"
    wr_display = f"{m['win_rate']:.2f}% ({m['win_rate_rating']})" if m['total_trades'] > 0 else f"{m['win_rate_rating']}"
    wl_display = f"{m['avg_win_loss_ratio']:.2f} ({m['win_loss_rating']})" if m['total_trades'] > 0 else f"{m['win_loss_rating']}"

    summary_table = [
        ["1. CAGR (Annual Return)", f"{m['cagr']:.2f}% ({m['cagr_rating']})", "2. Maximum Drawdown", f"{m['max_dd_pct']:.2f}% ({m['max_dd_rating']})"],
        ["3. Calmar Ratio", f"{m['calmar_ratio']:.2f} ({m['calmar_rating']})", "4. Sharpe Ratio", f"{m['sharpe_ratio']:.2f} ({m['sharpe_rating']})"],
        ["5. Sortino Ratio", f"{m['sortino_ratio']:.2f} ({m['sortino_rating']})", "6. Profit Factor", pf_display],
        ["7. Expectancy (Per Trade)", exp_display, "8. Recovery Factor", rf_display],
        ["9. Win Rate", wr_display, "10. Avg Win / Avg Loss", wl_display],
        ["11. Consecutive Losing Trades", f"{m['max_consecutive_losses']} ({m['consec_losses_rating']})", "12. Max DD Duration", f"{m['max_dd_duration_months']:.1f}M [Avg: {m['avg_dd_duration_months']:.1f}M] ({m['dd_duration_rating']})"]
    ]
    print(tabulate(summary_table, tablefmt="fancy_grid"))

    # Active Open Holdings (Marked-to-Market)
    if open_positions:
        print("\n ACTIVE OPEN HOLDINGS (MARKED TO MARKET):")
        open_table = []
        for op in open_positions:
            open_table.append([
                op.get("symbol", "N/A"),
                str(op.get("entry_date", "N/A")),
                f"₹{op.get('entry', 0.0):,.2f}",
                f"₹{op.get('current_price', 0.0):,.2f}",
                f"{op.get('qty', 0):,}",
                f"₹{op.get('unrealized_pnl', 0.0):,.2f}",
                f"{op.get('unrealized_return_pct', 0.0):+.2f}%",
                f"₹{op.get('sl', 0.0):,.2f}",
            ])
        headers = ["Symbol", "Entry Date", "Entry Price", "Current CMP", "Qty", "Unrealized P&L", "Return (%)", "Stop Loss"]
        print(tabulate(open_table, headers=headers, tablefmt="fancy_grid"))

    # Trade Exits & Monetary Breakdown
    exit_breakdown_str = " | ".join([f"{k}: {v}" for k, v in m["exit_counts"].items()]) if m["exit_counts"] else "None (Holdings Open)"
    print(f"\n REALIZED EXITS     : {exit_breakdown_str}")
    if open_positions:
        realized_pnl = trades_df["pnl_inr"].sum() if not trades_df.empty else 0.0
        unrealized_pnl = sum(op.get("unrealized_pnl", 0.0) for op in open_positions)
        print(f" REALIZED PNL       : ₹{realized_pnl:,.2f}")
        print(f" UNREALIZED MTM PNL : ₹{unrealized_pnl:,.2f}")
        print(f" COMBINED TOTAL PNL : ₹{realized_pnl + unrealized_pnl:,.2f}")
    else:
        print(f" COMBINED TOTAL PNL : ₹{m['combined_total_pnl']:,.2f}")
    total_stat_fees = m['total_stat_costs']
    if open_positions:
        total_stat_fees += sum(op.get("entry_charges", 0.0) for op in open_positions)
    print(f" TOTAL STATUTORY FEES: ₹{total_stat_fees:,.2f} (STT, GST, Exch, Stamp Duty, Brokerage)")
    print(f" BEST YEAR          : {m['best_year']}")
    print(f" BEST MONTH         : {m['best_month']}")
    print(f" WORST MONTH        : {m['worst_month']}")

    # Detailed Benchmark Breakdown
    print(f" CAGR RATING        : {m['cagr_rating']} ({m['cagr']:.2f}% vs thresholds: <{CAGR_BAD_BELOW:.0f}% Bad, {CAGR_BAD_BELOW:.0f}-{CAGR_ACCEPTABLE_MAX:.0f}% Acceptable, {CAGR_GOOD_MIN:.0f}-{CAGR_EXCELLENT_MIN:.0f}% Good, >{CAGR_EXCELLENT_MIN:.0f}% Excellent)")
    print(f" MAX DD RATING      : {m['max_dd_rating']} ({m['max_dd_pct']:.2f}% vs thresholds: <{MAX_DD_EXCELLENT_BELOW:.0f}% Excellent, {MAX_DD_EXCELLENT_BELOW:.0f}-{MAX_DD_GOOD_MAX:.0f}% Good, {MAX_DD_GOOD_MAX:.0f}-{MAX_DD_ACCEPTABLE_MAX:.0f}% Acceptable, >{MAX_DD_ACCEPTABLE_MAX:.0f}% Risky)")
    print(f" CALMAR RATING      : {m['calmar_rating']} ({m['calmar_ratio']:.2f} vs thresholds: <{CALMAR_WEAK_BELOW:.1f} Weak, {CALMAR_WEAK_BELOW:.1f}-{CALMAR_ACCEPTABLE_MAX:.1f} Acceptable, {CALMAR_ACCEPTABLE_MAX:.1f}-{CALMAR_GOOD_MAX:.1f} Good, >{CALMAR_GOOD_MAX:.1f} Excellent)")
    print(f" SHARPE RATING      : {m['sharpe_rating']} ({m['sharpe_ratio']:.2f} vs thresholds: <{SHARPE_WEAK_BELOW:.1f} Weak, {SHARPE_WEAK_BELOW:.1f}-{SHARPE_ACCEPTABLE_MAX:.1f} Acceptable, {SHARPE_ACCEPTABLE_MAX:.1f}-{SHARPE_EXCELLENT_MIN:.1f} Good, >{SHARPE_EXCELLENT_MIN:.1f} Excellent)")
    print(f" SORTINO RATING     : {m['sortino_rating']} ({m['sortino_ratio']:.2f} vs thresholds: <{SORTINO_WEAK_BELOW:.1f} Weak, {SORTINO_WEAK_BELOW:.1f}-{SORTINO_ACCEPTABLE_MAX:.1f} Acceptable, {SORTINO_ACCEPTABLE_MAX:.1f}-{SORTINO_EXCELLENT_MIN:.1f} Good, >{SORTINO_EXCELLENT_MIN:.1f} Excellent)")
    print(f" PROFIT FACTOR RATING: {m['profit_factor_rating']} ({m['profit_factor']:.2f} vs thresholds: <{PROFIT_FACTOR_LOSS_BELOW:.1f} Loss Making, {PROFIT_FACTOR_LOSS_BELOW:.1f}-{PROFIT_FACTOR_WEAK_MAX:.1f} Weak, {PROFIT_FACTOR_WEAK_MAX:.1f}-{PROFIT_FACTOR_ACCEPTABLE_MAX:.1f} Acceptable, {PROFIT_FACTOR_ACCEPTABLE_MAX:.1f}-{PROFIT_FACTOR_EXCELLENT_MIN:.1f} Good, >{PROFIT_FACTOR_EXCELLENT_MIN:.1f} Excellent)")
    print(f" EXPECTANCY RATING  : {m['expectancy_rating']} (₹{m['expectancy_per_trade']:,.2f} vs thresholds: <₹{EXPECTANCY_REJECT_BELOW:.0f} Reject, ₹{EXPECTANCY_REJECT_BELOW:.0f}-₹{EXPECTANCY_WEAK_MAX:.0f} Weak, ₹{EXPECTANCY_WEAK_MAX:.0f}-₹{EXPECTANCY_EXCELLENT_MIN:.0f} Good, >₹{EXPECTANCY_EXCELLENT_MIN:.0f} Excellent)")
    print(f" RECOVERY FACTOR RATING: {m['recovery_factor_rating']} ({m['recovery_factor']:.2f} vs thresholds: <{RECOVERY_FACTOR_WEAK_BELOW:.1f} Weak, {RECOVERY_FACTOR_WEAK_BELOW:.1f}-{RECOVERY_FACTOR_ACCEPTABLE_MAX:.1f} Acceptable, {RECOVERY_FACTOR_GOOD_MIN:.1f}-{RECOVERY_FACTOR_EXCELLENT_MIN:.1f} Good, >{RECOVERY_FACTOR_EXCELLENT_MIN:.1f} Excellent)")
    print(f" CONSEC LOSSES RATING: {m['consec_losses_rating']} ({m['max_consecutive_losses']} vs thresholds: <{CONSEC_LOSSES_EXCELLENT_BELOW} Excellent, {CONSEC_LOSSES_EXCELLENT_BELOW}-{CONSEC_LOSSES_GOOD_MAX} Good, {CONSEC_LOSSES_GOOD_MAX}-{CONSEC_LOSSES_ACCEPTABLE_MAX} Acceptable, >{CONSEC_LOSSES_UNACCEPTABLE_ABOVE} Unacceptable)")
    print(f" DD DURATION RATING : {m['dd_duration_rating']} ({m['max_dd_duration_months']:.1f}M vs thresholds: <{DD_DURATION_EXCELLENT_BELOW:.0f}M Excellent, {DD_DURATION_EXCELLENT_BELOW:.0f}-{DD_DURATION_GOOD_MAX:.0f}M Good, {DD_DURATION_GOOD_MAX:.0f}-{DD_DURATION_ACCEPTABLE_MAX:.0f}M Acceptable, >{DD_DURATION_WEAK_ABOVE:.0f}M Weak)")
    print(f" WIN RATE RATING    : {m['win_rate_rating']} ({m['win_rate']:.2f}% vs thresholds: <{WIN_RATE_WEAK_BELOW:.0f}% Weak, {WIN_RATE_WEAK_BELOW:.0f}-{WIN_RATE_ACCEPTABLE_MAX:.0f}% Acceptable, {WIN_RATE_GOOD_MIN:.0f}-{WIN_RATE_EXCELLENT_MIN:.0f}% Good, >{WIN_RATE_EXCELLENT_MIN:.0f}% Excellent)")
    print(f" WIN/LOSS RATING    : {m['win_loss_rating']} ({m['avg_win_loss_ratio']:.2f} vs thresholds: <{WIN_LOSS_RATIO_WEAK_BELOW:.1f} Weak, {WIN_LOSS_RATIO_WEAK_BELOW:.1f}-{WIN_LOSS_RATIO_ACCEPTABLE_MAX:.1f} Acceptable, {WIN_LOSS_RATIO_GOOD_MIN:.1f}-{WIN_LOSS_RATIO_EXCELLENT_MIN:.1f} Good, >{WIN_LOSS_RATIO_EXCELLENT_MIN:.1f} Excellent)")
    print("-" * 85)
    print(f" ★ OVERALL TEST OUTCOME : {m['overall_outcome']}")
    all_ratings = m["all_ratings"]
    print(f" SCORECARD SUMMARY     : {all_ratings.count('EXCELLENT')} Excellent | {all_ratings.count('GOOD')} Good | {all_ratings.count('ACCEPTABLE') + all_ratings.count('MODERATE')} Acceptable | {all_ratings.count('WEAK')} Weak | {all_ratings.count('RISKY') + all_ratings.count('UNACCEPTABLE') + all_ratings.count('LOSS MAKING') + all_ratings.count('REJECT')} Critical")
    print("=" * 85 + "\n")

    return m


__all__ = ["generate_performance_summary"]

