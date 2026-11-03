"""Unit tests for the firm-wide ratios package using standard library unittest."""

import unittest
import pandas as pd

from ratios import (
    calculate_ratios,
    generate_performance_summary,
    classify_cagr,
    classify_max_dd,
    classify_sharpe,
    classify_sortino,
    classify_profit_factor,
    classify_win_rate,
)


class TestRatiosPackage(unittest.TestCase):

    def test_classifiers(self):
        """Verify standard ratings across key thresholds."""
        # CAGR
        self.assertEqual(classify_cagr(35.0), "EXCELLENT")
        self.assertEqual(classify_cagr(25.0), "GOOD")
        self.assertEqual(classify_cagr(18.0), "MODERATE")
        self.assertEqual(classify_cagr(14.0), "ACCEPTABLE")
        self.assertEqual(classify_cagr(8.0), "BAD")

        # Max DD
        self.assertEqual(classify_max_dd(8.0), "EXCELLENT")
        self.assertEqual(classify_max_dd(12.0), "GOOD")
        self.assertEqual(classify_max_dd(20.0), "ACCEPTABLE")
        self.assertEqual(classify_max_dd(30.0), "RISKY")

        # Sharpe
        self.assertEqual(classify_sharpe(2.5), "EXCELLENT")
        self.assertEqual(classify_sharpe(1.7), "GOOD")
        self.assertEqual(classify_sharpe(1.2), "ACCEPTABLE")
        self.assertEqual(classify_sharpe(0.8), "WEAK")

        # Profit Factor
        self.assertEqual(classify_profit_factor(2.5), "EXCELLENT")
        self.assertEqual(classify_profit_factor(1.8), "GOOD")
        self.assertEqual(classify_profit_factor(1.4), "ACCEPTABLE")
        self.assertEqual(classify_profit_factor(1.1), "WEAK")
        self.assertEqual(classify_profit_factor(0.9), "LOSS MAKING")

    def test_calculate_ratios_empty(self):
        """Ensure graceful handling of empty trade sets."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        empty_trades = pd.DataFrame()
        res = calculate_ratios(empty_trades, pd.Series(dates.date))
        self.assertTrue(res["empty"])
        self.assertEqual(res["total_trades"], 0)

    def test_calculate_ratios_sample_data(self):
        """Verify mathematical computation of ratios."""
        dates = pd.date_range("2023-01-01", periods=250, freq="B")
        trades = pd.DataFrame([
            {"date": dates[10].date(), "pnl_inr": 20000.0, "stat_cost": 500.0, "exit_reason": "TARGET", "mae_inr": 2000.0},
            {"date": dates[30].date(), "pnl_inr": -10000.0, "stat_cost": 500.0, "exit_reason": "SL", "mae_inr": 10000.0},
            {"date": dates[60].date(), "pnl_inr": 30000.0, "stat_cost": 500.0, "exit_reason": "TARGET", "mae_inr": 3000.0},
            {"date": dates[90].date(), "pnl_inr": -10000.0, "stat_cost": 500.0, "exit_reason": "SL", "mae_inr": 10000.0},
        ])

        res = calculate_ratios(trades, pd.Series(dates.date), initial_capital=1000000.0)
        self.assertFalse(res["empty"])
        self.assertEqual(res["total_trades"], 4)
        self.assertEqual(res["combined_total_pnl"], 30000.0)
        self.assertEqual(res["win_rate"], 50.0)
        self.assertEqual(res["profit_factor"], 2.5)
        self.assertEqual(res["expectancy_per_trade"], 7500.0)
        self.assertEqual(res["total_stat_costs"], 2000.0)

    def test_generate_performance_summary_execution(self):
        """Verify terminal dashboard executes without error."""
        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        trades = pd.DataFrame([
            {"date": dates[10].date(), "pnl_inr": 10000.0, "stat_cost": 300.0, "exit_reason": "TARGET", "mae_inr": 1000.0},
            {"date": dates[20].date(), "pnl_inr": -5000.0, "stat_cost": 300.0, "exit_reason": "SL", "mae_inr": 5000.0},
        ])

        res = generate_performance_summary(
            trades_df=trades,
            all_dates=pd.Series(dates.date),
            initial_capital=500000.0,
            strategy_title="TEST SUITE"
        )
        self.assertIsNotNone(res)
        self.assertIn("cagr", res)
        self.assertIn("max_dd_pct", res)
        self.assertIn("overall_outcome", res)

    def test_monthly_frequency_auto_detection(self):
        """Verify that monthly dates are correctly auto-detected and annualized with 12 periods."""
        dates = pd.date_range("2020-01-31", periods=24, freq="ME")
        trades = pd.DataFrame([
            {"date": dates[5].date(), "pnl_inr": 50000.0, "stat_cost": 500.0, "exit_reason": "EXIT_21EMA", "mae_inr": 2000.0},
            {"date": dates[12].date(), "pnl_inr": -10000.0, "stat_cost": 500.0, "exit_reason": "SL", "mae_inr": 10000.0},
        ])
        res = calculate_ratios(trades, pd.Series(dates.date), initial_capital=1000000.0)
        self.assertEqual(res["frequency"], "monthly")
        # 24 monthly periods
        self.assertEqual(res["total_days"], 24)
        # Verify Sharpe ratio is computed without error
        self.assertFalse(pd.isna(res["sharpe_ratio"]))

    def test_mae_evaluated_against_contemporaneous_peak(self):
        """Verify that intra-trade MAE drawdown is evaluated against contemporaneous equity peak rather than initial capital."""
        dates = pd.date_range("2023-01-01", periods=20, freq="B")
        # Trade 1: initial capital doubles from 1,000,000 to 2,000,000 (contemporaneous peak = 2,000,000)
        # Trade 2: suffers 200,000 MAE drawdown.
        # Against contemporaneous peak (2,000,000): 200,000 / 2,000,000 = 10.0%.
        # Against initial capital (1,000,000): 200,000 / 1,000,000 = 20.0% (incorrect).
        trades = pd.DataFrame([
            {"date": dates[5].date(), "pnl_inr": 1000000.0, "stat_cost": 500.0, "exit_reason": "TARGET", "mae_inr": 0.0},
            {"date": dates[10].date(), "pnl_inr": 0.0, "stat_cost": 500.0, "exit_reason": "EXIT", "mae_inr": 200000.0},
        ])
        res = calculate_ratios(trades, pd.Series(dates.date), initial_capital=1000000.0)
        self.assertAlmostEqual(res["max_dd_pct"], 10.0, places=2)
        self.assertEqual(res["max_dd_inr"], 200000.0)


if __name__ == "__main__":
    unittest.main()
