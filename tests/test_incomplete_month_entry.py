"""Unit tests verifying that incomplete current month entries are triggered properly."""

import unittest
import pandas as pd
import datetime as dt
from strategies.st07.st07 import run_st07_portfolio_backtest, run_st07_backtest


class TestIncompleteMonthEntry(unittest.TestCase):

    def test_incomplete_month_triggers_entry_from_prior_completed_setup(self):
        """
        Verify that when October completes with setup_valid=True,
        and November is an incomplete month with its first trading day,
        the entry is triggered on the first trading day of November.
        """
        dates_oct = pd.date_range("2026-10-01", "2026-10-31", freq="B")
        dates_nov = pd.date_range("2026-11-01", "2026-11-05", freq="B")

        df_d = pd.DataFrame({
            "date": list(dates_oct.date) + list(dates_nov.date),
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000,
        })
        df_d["ym"] = pd.to_datetime(df_d["date"]).dt.to_period("M")

        # Completed monthly data: ONLY October is complete; November is excluded
        df_m = pd.DataFrame([{
            "date": dates_oct[-1].date(),
            "ym": pd.Period("2026-10", "M"),
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "ha_close": 105.0,
            "ema21": 100.0,
            "setup_valid": True,
            "is_warmup": False,
        }])

        # 1. Test Portfolio Engine
        trades, mtm, open_holdings = run_st07_portfolio_backtest({"TEST": df_d}, {"TEST": df_m})
        self.assertEqual(len(trades), 0)
        self.assertEqual(len(open_holdings), 1, "Portfolio failed to trigger entry in incomplete November!")
        self.assertEqual(open_holdings[0]["symbol"], "TEST")
        self.assertEqual(open_holdings[0]["entry_date"], dates_nov[0].date())

        # 2. Test Standalone Engine (should yield identical result via unified implementation)
        trades_sa, mtm_sa, open_holdings_sa = run_st07_backtest(df_d, df_m, symbol="TEST")
        self.assertEqual(len(trades_sa), 0)
        self.assertEqual(len(open_holdings_sa), 1, "Standalone failed to trigger entry in incomplete November!")
        self.assertEqual(open_holdings_sa[0]["entry_date"], dates_nov[0].date())
        # Both should match
        self.assertEqual(open_holdings[0]["entry"], open_holdings_sa[0]["entry"])


if __name__ == "__main__":
    unittest.main()

