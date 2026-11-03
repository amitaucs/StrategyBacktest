"""Unit tests for ST07 position sizing and entry costs reservation."""

import unittest
from strategies.st07.st07 import (
    calculate_entry_charges,
    calculate_exit_charges,
    calculate_statutory_costs_equity,
    calculate_max_affordable_qty,
)


class TestST07PositionSizing(unittest.TestCase):

    def test_entry_charges_and_exit_charges_sum_to_roundtrip(self):
        """Verify buy and sell charges reconcile to the total roundtrip statutory costs."""
        entry_px = 500.0
        exit_px = 550.0
        qty = 200
        buy_charges = calculate_entry_charges(entry_px, qty)
        sell_charges = calculate_exit_charges(exit_px, qty)
        total_roundtrip = calculate_statutory_costs_equity(entry_px, exit_px, qty)

        self.assertAlmostEqual(buy_charges + sell_charges, total_roundtrip, places=6)

    def test_max_affordable_qty_respects_cash_budget(self):
        """Ensure entry value + entry charges strictly <= available cash."""
        test_cases = [
            (250000.0, 275.04),   # SBIN case where naive int(250000/275.04) = 908 causes deficit
            (1000000.0, 1283.64), # LT case
            (50000.0, 222.61),    # Small capital
            (10000.0, 3500.0),    # High priced stock, low cash
            (500.0, 1000.0),      # Cannot afford even 1 share
            (0.0, 100.0),         # Zero cash
        ]

        for cash, price in test_cases:
            qty = calculate_max_affordable_qty(cash, price)
            if qty > 0:
                entry_val = qty * price
                entry_charges = calculate_entry_charges(price, qty)
                total_cost = entry_val + entry_charges
                self.assertLessEqual(
                    total_cost,
                    cash,
                    f"Deficit detected! cash={cash}, price={price}, qty={qty}, total_cost={total_cost}"
                )
                # Verify that qty + 1 would breach available cash (maximality)
                higher_val = (qty + 1) * price
                higher_charges = calculate_entry_charges(price, qty + 1)
                self.assertGreater(
                    higher_val + higher_charges,
                    cash,
                    f"Non-maximal sizing! Could have afforded qty={qty + 1}"
                )
            else:
                # If qty == 0, 1 share must exceed cash
                if price > 0 and cash > 0:
                    self.assertGreater(price + calculate_entry_charges(price, 1), cash)


if __name__ == "__main__":
    unittest.main()

