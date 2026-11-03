"""
ST07: Monthly Heikin Ashi + 89 EMA Crossover Strategy
NSE Equity — Rules-Based Positional Trading Playbook

Core Strategy Rules:
- Timeframe: Monthly (resampled from 10-year daily historical OHLCV).
- Chart: Heikin Ashi candles calculated recursively.
- Indicators: 89-period EMA (trend/entry) and 21-period EMA (exit reference) on Monthly close.
- Setup: 89 EMA must be uptrending (slope > 0); HA candle opens below 89 EMA and closes above 89 EMA.
- Entry: Enter 100% long on confirmation at next month open using real market price.
- Stop-Loss: Low of the crossover/setup candle (real market low).
- Exit: 100% exit when price falls below/hits setup candle low OR Monthly HA candle closes below 21 EMA.
- Execution Realism: Signals from Heikin Ashi; fills, slippage, and P&L strictly on real market prices.
- Reporting: Integrated with firm-wide 'ratios' package for institutional 12-metric scorecard.
"""

import os
import sys
import time
import datetime as dt
from pathlib import Path
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from dhanhq import DhanContext, dhanhq

# Add project root to sys.path to import shared packages (ratios, etc.)
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ratios import generate_performance_summary

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# 1. CONFIGURATION & PARAMETERS
# ==============================================================================
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "YOUR_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000000.0"))
ST07_UNIVERSE_RAW = os.getenv("ST07_UNIVERSE", "RELIANCE")
ST07_BACKTEST_YEARS = int(os.getenv("ST07_BACKTEST_YEARS", "10"))
ST07_EMA_TREND = int(os.getenv("ST07_EMA_TREND", "89"))
ST07_EMA_EXIT = int(os.getenv("ST07_EMA_EXIT", "21"))
ST07_WARMUP_BARS = int(os.getenv("ST07_WARMUP_BARS", "24"))  # 24 monthly bars (~2 years) for EMA stabilization
ST07_SLIPPAGE_PCT = float(os.getenv("ST07_SLIPPAGE_PCT", "0.05"))  # 0.05% slippage on delivery

# Pre-mapped security database for top liquid NSE equities
SECURITY_LOOKUP = {
    "RELIANCE": {"security_id": "2885", "exchange_segment": "NSE_EQ", "instrument_type": "EQUITY", "symbol": "RELIANCE"},
    "TCS": {"security_id": "11536", "exchange_segment": "NSE_EQ", "instrument_type": "EQUITY", "symbol": "TCS"},
    "INFY": {"security_id": "1594", "exchange_segment": "NSE_EQ", "instrument_type": "EQUITY", "symbol": "INFY"},
    "HDFCBANK": {"security_id": "1333", "exchange_segment": "NSE_EQ", "instrument_type": "EQUITY", "symbol": "HDFCBANK"},
    "ICICIBANK": {"security_id": "4963", "exchange_segment": "NSE_EQ", "instrument_type": "EQUITY", "symbol": "ICICIBANK"},
    "LT": {"security_id": "11483", "exchange_segment": "NSE_EQ", "instrument_type": "EQUITY", "symbol": "LT"},
    "SBIN": {"security_id": "3045", "exchange_segment": "NSE_EQ", "instrument_type": "EQUITY", "symbol": "SBIN"},
    "BHARTIARTL": {"security_id": "10604", "exchange_segment": "NSE_EQ", "instrument_type": "EQUITY", "symbol": "BHARTIARTL"},
    "ITC": {"security_id": "1660", "exchange_segment": "NSE_EQ", "instrument_type": "EQUITY", "symbol": "ITC"},
    "KOTAKBANK": {"security_id": "1922", "exchange_segment": "NSE_EQ", "instrument_type": "EQUITY", "symbol": "KOTAKBANK"},
    "NIFTY50": {"security_id": "13", "exchange_segment": "IDX_I", "instrument_type": "INDEX", "symbol": "NIFTY50"},
}

def parse_universe(universe_str: str) -> List[Dict[str, str]]:
    """
    Parses comma-separated universe string into security configuration dictionaries.
    Supports either pre-mapped symbol names (e.g. 'RELIANCE') or custom formatted
    strings: 'SYMBOL:SECURITY_ID:SEGMENT:INSTRUMENT'.
    Handles square brackets (e.g. [SBIN, LT, ITC, TCS]) and quotes gracefully.
    """
    # Clean brackets and wrapping quotes if user passed list syntax like [SBIN, LT, ITC, TCS]
    clean_str = universe_str.strip().strip("[]\"'")
    raw_items = [x.strip().strip("\"' ") for x in clean_str.split(",") if x.strip().strip("\"' ")]
    configs = []
    for item in raw_items:
        if ":" in item:
            parts = item.split(":")
            configs.append({
                "symbol": parts[0].strip().upper(),
                "security_id": parts[1].strip(),
                "exchange_segment": parts[2].strip() if len(parts) > 2 else "NSE_EQ",
                "instrument_type": parts[3].strip() if len(parts) > 3 else "EQUITY",
            })
        else:
            sym = item.upper()
            if sym in SECURITY_LOOKUP:
                configs.append(SECURITY_LOOKUP[sym])
            else:
                print(f"Warning: Symbol '{sym}' not found in SECURITY_LOOKUP. Using default NSE_EQ mapping.")
                configs.append({
                    "symbol": sym,
                    "security_id": sym,
                    "exchange_segment": "NSE_EQ",
                    "instrument_type": "EQUITY",
                })
    return configs

# ==============================================================================
# 2. NSE CASH EQUITY STATUTORY COST CALCULATOR (DELIVERY)
# ==============================================================================
def calculate_statutory_costs_equity(entry_px: float, exit_px: float, qty: int) -> float:
    """
    Calculates exact statutory taxes & regulatory fees for NSE Cash Equity Delivery:
    - STT (Securities Transaction Tax): 0.1% on Buy leg + 0.1% on Sell leg
    - Exchange Turnover Charges: 0.00297% on both legs + 18% GST (0.0035046% total)
    - Stamp Duty: 0.015% on Buy turnover
    - SEBI Turnover Charges: ₹10 / crore + 18% GST on both legs
    - Brokerage: ₹0 (standard discount broker delivery rate)
    """
    buy_turnover = entry_px * qty
    sell_turnover = exit_px * qty
    total_turnover = buy_turnover + sell_turnover

    stt = 0.001 * buy_turnover + 0.001 * sell_turnover
    exchange_charges = 0.000035046 * total_turnover
    stamp_duty = 0.00015 * buy_turnover
    sebi_charges = (10.0 / 10000000.0) * total_turnover * 1.18
    brokerage = 0.0  # Zero brokerage on delivery

    return stt + exchange_charges + stamp_duty + sebi_charges + brokerage

# ==============================================================================
# 3. HISTORICAL DATA INGESTION & MONTHLY RESAMPLING
# ==============================================================================
def fetch_dhan_daily_data(dhan: dhanhq, sec_info: Dict[str, str], years: int = ST07_BACKTEST_YEARS) -> pd.DataFrame:
    """
    Fetches daily historical data for the instrument from DhanHQ and resamples to Monthly OHLCV.
    """
    end_date = dt.date.today()
    start_date = end_date - relativedelta(years=years)

    symbol = sec_info["symbol"]
    sec_id = sec_info["security_id"]
    segment = sec_info["exchange_segment"]
    inst_type = sec_info["instrument_type"]

    print(f"Fetching {years}-year daily data for {symbol} ({sec_id}) from {start_date} to {end_date} via DhanHQ...")
    time.sleep(1.0)
    resp = dhan.historical_daily_data(
        security_id=sec_id,
        exchange_segment=segment,
        instrument_type=inst_type,
        from_date=start_date.strftime("%Y-%m-%d"),
        to_date=end_date.strftime("%Y-%m-%d"),
    )

    if not (isinstance(resp, dict) and resp.get("status") == "success" and resp.get("data")):
        err_msg = resp.get("error_message") or resp.get("remarks") or str(resp) if isinstance(resp, dict) else str(resp)
        raise RuntimeError(f"DhanHQ Daily Fetch Error for {symbol}: {err_msg}")

    data = resp["data"]
    timestamps = (
        pd.to_datetime(data["timestamp"], unit="s", utc=True)
        .tz_convert("Asia/Kolkata")
        .tz_localize(None)
    )

    df_daily = pd.DataFrame({
        "timestamp": timestamps,
        "date": timestamps.date,
        "open": data["open"],
        "high": data["high"],
        "low": data["low"],
        "close": data["close"],
        "volume": data.get("volume", [0] * len(data["close"]))
    })

    # Data integrity & sanity filter
    df_daily = df_daily.dropna().drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df_daily = df_daily[(df_daily["high"] >= df_daily["low"]) & (df_daily["open"] > 0) & (df_daily["close"] > 0)]

    print(f"  ✓ Fetched {len(df_daily)} daily candles for {symbol} ({df_daily['date'].iloc[0]} to {df_daily['date'].iloc[-1]})")

    # Resample daily data to Monthly OHLCV
    df_daily["dt_index"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.set_index("dt_index")

    # Monthly aggregation
    monthly_df = df_daily.resample("ME").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "date": "last"
    }).dropna().reset_index(drop=True)

    print(f"  ✓ Resampled into {len(monthly_df)} monthly candles for {symbol}")
    return monthly_df

# ==============================================================================
# 4. INDICATOR ENGINE (HEIKIN ASHI, 89 EMA & 21 EMA)
# ==============================================================================
def compute_indicators(
    df: pd.DataFrame,
    ema_trend_period: int = ST07_EMA_TREND,
    ema_exit_period: int = ST07_EMA_EXIT,
    warmup_bars: int = ST07_WARMUP_BARS
) -> pd.DataFrame:
    """
    Computes Heikin Ashi candles and 89 / 21 EMAs on monthly closes.
    """
    df = df.copy()

    # 1. Recursive Heikin Ashi Calculation
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = np.zeros(len(df))
    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0

    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0

    ha_high = np.maximum(df["high"], np.maximum(ha_open, ha_close))
    ha_low = np.minimum(df["low"], np.minimum(ha_open, ha_close))

    df["ha_open"] = ha_open
    df["ha_high"] = ha_high
    df["ha_low"] = ha_low
    df["ha_close"] = ha_close

    # 2. EMAs on Monthly Closes
    df["ema89"] = df["close"].ewm(span=ema_trend_period, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=ema_exit_period, adjust=False).mean()

    # 3. 89 EMA Slope Filter (Uptrending)
    df["ema89_up"] = df["ema89"] > df["ema89"].shift(1)

    # 4. Setup Trigger: HA candle opens below 89 EMA and closes above 89 EMA
    df["ha_cross_89"] = (df["ha_open"] < df["ema89"]) & (df["ha_close"] > df["ema89"])
    df["setup_valid"] = df["ha_cross_89"] & df["ema89_up"]

    # 5. Warm-up Period (Configurable bars excluded to allow EMA stabilization)
    df["is_warmup"] = np.arange(len(df)) < warmup_bars

    return df

# ==============================================================================
# 5. ST07 INTRADAY / POSITIONAL BACKTEST ENGINE
# ==============================================================================
def run_st07_backtest(df: pd.DataFrame, symbol: str = "RELIANCE") -> pd.DataFrame:
    """
    Executes ST07 Monthly Strategy Backtest:
    - Long Entry: Entered on next month open after a confirmed monthly HA crossover with 89 EMA rising.
    - Initial SL: Set to the real market Low of the crossover/setup candle.
    - Exit 1 (SL Breach): If intra-month price drops to/below SL -> exit 100%.
    - Exit 2 (Trend Break): If monthly HA close drops below 21 EMA -> exit 100%.
    - Realism: Fills, slippage, and P&L strictly calculated on actual exchange prices.
    """
    trades = []
    position = None
    current_capital = INITIAL_CAPITAL
    slippage_mult = ST07_SLIPPAGE_PCT / 100.0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # ----------------------------------------------------------------------
        # 1. MANAGE EXISTING OPEN POSITION
        # ----------------------------------------------------------------------
        if position is not None:
            # Update intra-trade Maximum Adverse Excursion (MAE)
            adverse = max(0.0, position["entry"] - row["low"])
            position["mae_pts"] = max(position.get("mae_pts", 0.0), adverse)

            # Exit Condition 1: Stop-Loss breach intra-month
            hit_sl = row["low"] <= position["sl"]

            # Exit Condition 2: Trend break (Monthly HA candle closes below 21 EMA)
            ha_close_below_21 = row["ha_close"] < row["ema21"]

            if hit_sl:
                # Filled at stop price or open if gapped down below stop
                sl_fill = min(row["open"], position["sl"])
                exit_price = sl_fill * (1.0 - slippage_mult)
                stat_cost = calculate_statutory_costs_equity(position["entry"], exit_price, position["qty"])
                pnl_inr_gross = (exit_price - position["entry"]) * position["qty"]
                pnl_inr_net = pnl_inr_gross - stat_cost
                current_capital += (exit_price * position["qty"]) - stat_cost

                trades.append({
                    "symbol": symbol,
                    "date": row["date"],
                    "entry_date": position["entry_date"],
                    "exit_date": row["date"],
                    "side": "LONG",
                    "entry": position["entry"],
                    "exit": exit_price,
                    "sl": position["sl"],
                    "qty": position["qty"],
                    "pnl_inr": pnl_inr_net,
                    "pnl_pts": pnl_inr_net / position["qty"],
                    "exit_reason": "SL_ENTRY_CANDLE",
                    "stat_cost": stat_cost,
                    "mae_inr": position["mae_pts"] * position["qty"],
                })
                position = None

            elif ha_close_below_21:
                # Filled at confirmed month-end close with slippage
                exit_price = row["close"] * (1.0 - slippage_mult)
                stat_cost = calculate_statutory_costs_equity(position["entry"], exit_price, position["qty"])
                pnl_inr_gross = (exit_price - position["entry"]) * position["qty"]
                pnl_inr_net = pnl_inr_gross - stat_cost
                current_capital += (exit_price * position["qty"]) - stat_cost

                trades.append({
                    "symbol": symbol,
                    "date": row["date"],
                    "entry_date": position["entry_date"],
                    "exit_date": row["date"],
                    "side": "LONG",
                    "entry": position["entry"],
                    "exit": exit_price,
                    "sl": position["sl"],
                    "qty": position["qty"],
                    "pnl_inr": pnl_inr_net,
                    "pnl_pts": pnl_inr_net / position["qty"],
                    "exit_reason": "EXIT_21EMA",
                    "stat_cost": stat_cost,
                    "mae_inr": position["mae_pts"] * position["qty"],
                })
                position = None

        # ----------------------------------------------------------------------
        # 2. TRIGGER NEW ENTRY (If setup completed on previous month)
        # ----------------------------------------------------------------------
        if position is None and not prev_row["is_warmup"]:
            if prev_row["setup_valid"]:
                entry_price = row["open"] * (1.0 + slippage_mult)
                sl_price = prev_row["low"]  # Real market low of the setup/crossover candle
                risk_pts = entry_price - sl_price

                if risk_pts > 0:
                    # Allocate 100% available capital to the positional trade
                    qty = int(current_capital / entry_price)
                    if qty > 0:
                        position = {
                            "symbol": symbol,
                            "entry_date": row["date"],
                            "entry": entry_price,
                            "sl": sl_price,
                            "risk_pts": risk_pts,
                            "qty": qty,
                            "mae_pts": max(0.0, entry_price - row["low"]),
                        }
                        current_capital -= (entry_price * qty)

                        # Immediate evaluation on the entry month bar
                        hit_sl_entry = row["low"] <= position["sl"]
                        ha_close_entry = row["ha_close"] < row["ema21"]

                        if hit_sl_entry:
                            sl_fill = min(row["open"], position["sl"])
                            exit_price = sl_fill * (1.0 - slippage_mult)
                            stat_cost = calculate_statutory_costs_equity(position["entry"], exit_price, position["qty"])
                            pnl_inr_gross = (exit_price - position["entry"]) * position["qty"]
                            pnl_inr_net = pnl_inr_gross - stat_cost
                            current_capital += (exit_price * position["qty"]) - stat_cost

                            trades.append({
                                "symbol": symbol,
                                "date": row["date"],
                                "entry_date": position["entry_date"],
                                "exit_date": row["date"],
                                "side": "LONG",
                                "entry": position["entry"],
                                "exit": exit_price,
                                "sl": position["sl"],
                                "qty": position["qty"],
                                "pnl_inr": pnl_inr_net,
                                "pnl_pts": pnl_inr_net / position["qty"],
                                "exit_reason": "SL_ENTRY_CANDLE",
                                "stat_cost": stat_cost,
                                "mae_inr": position["mae_pts"] * position["qty"],
                            })
                            position = None
                        elif ha_close_entry:
                            exit_price = row["close"] * (1.0 - slippage_mult)
                            stat_cost = calculate_statutory_costs_equity(position["entry"], exit_price, position["qty"])
                            pnl_inr_gross = (exit_price - position["entry"]) * position["qty"]
                            pnl_inr_net = pnl_inr_gross - stat_cost
                            current_capital += (exit_price * position["qty"]) - stat_cost

                            trades.append({
                                "symbol": symbol,
                                "date": row["date"],
                                "entry_date": position["entry_date"],
                                "exit_date": row["date"],
                                "side": "LONG",
                                "entry": position["entry"],
                                "exit": exit_price,
                                "sl": position["sl"],
                                "qty": position["qty"],
                                "pnl_inr": pnl_inr_net,
                                "pnl_pts": pnl_inr_net / position["qty"],
                                "exit_reason": "EXIT_21EMA",
                                "stat_cost": stat_cost,
                                "mae_inr": position["mae_pts"] * position["qty"],
                            })
                            position = None

    # End-of-dataset open position square-off
    if position is not None:
        last_row = df.iloc[-1]
        exit_price = last_row["close"] * (1.0 - slippage_mult)
        stat_cost = calculate_statutory_costs_equity(position["entry"], exit_price, position["qty"])
        pnl_inr_gross = (exit_price - position["entry"]) * position["qty"]
        pnl_inr_net = pnl_inr_gross - stat_cost
        current_capital += (exit_price * position["qty"]) - stat_cost

        trades.append({
            "symbol": symbol,
            "date": last_row["date"],
            "entry_date": position["entry_date"],
            "exit_date": last_row["date"],
            "side": "LONG",
            "entry": position["entry"],
            "exit": exit_price,
            "sl": position["sl"],
            "qty": position["qty"],
            "pnl_inr": pnl_inr_net,
            "pnl_pts": pnl_inr_net / position["qty"],
            "exit_reason": "END_OF_DATA",
            "stat_cost": stat_cost,
            "mae_inr": position.get("mae_pts", 0.0) * position["qty"],
        })
        position = None

    return pd.DataFrame(trades)

# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    if DHAN_CLIENT_ID == "YOUR_CLIENT_ID" or DHAN_ACCESS_TOKEN == "YOUR_ACCESS_TOKEN":
        print("Warning: Please set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables in .env.")

    dhan_context = DhanContext(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
    dhan_client = dhanhq(dhan_context)

    universe = parse_universe(ST07_UNIVERSE_RAW)
    print(f"\nInitialized ST07 Backtest Engine for universe: {[x['symbol'] for x in universe]}")
    print(f"Strategy: Monthly Heikin Ashi + 89 EMA Crossover (Exit: 21 EMA / Entry Low)")
    print(f"Initial Capital: ₹{INITIAL_CAPITAL:,.0f} | Backtest Period: {ST07_BACKTEST_YEARS} Years\n")

    all_trades_list = []
    all_dates_set = set()

    for sec_info in universe:
        sym = sec_info["symbol"]
        try:
            monthly_df = fetch_dhan_daily_data(dhan_client, sec_info, years=ST07_BACKTEST_YEARS)
            ind_df = compute_indicators(
                monthly_df,
                ema_trend_period=ST07_EMA_TREND,
                ema_exit_period=ST07_EMA_EXIT,
                warmup_bars=ST07_WARMUP_BARS
            )
            trades_df = run_st07_backtest(ind_df, symbol=sym)

            all_dates_set.update(ind_df["date"])
            title = f"ST07 MONTHLY HEIKIN ASHI + 89 EMA — {sym}"
            generate_performance_summary(
                trades_df=trades_df,
                all_dates=ind_df["date"],
                initial_capital=INITIAL_CAPITAL,
                strategy_title=title,
            )

            if trades_df.empty:
                print(f"[DIAGNOSTIC] No trades generated for {sym}:")
                min_ha_ratio = (ind_df["ha_open"] / ind_df["ema89"]).min()
                crossovers = ind_df["ha_cross_89"].sum()
                uptrends = ind_df["ema89_up"].sum()
                print(f"  • Monthly Bars: {len(ind_df)} | Warmup excluded: {ST07_WARMUP_BARS} bars")
                print(f"  • Crossovers (HA Open < 89 EMA & HA Close > 89 EMA): {crossovers}")
                print(f"  • 89 EMA Uptrending Bars: {uptrends} / {len(ind_df)}")
                print(f"  • Min HA Open / 89 EMA Ratio: {min_ha_ratio:.4f}")
                if crossovers == 0:
                    print(f"  • Insight: {sym} traded strictly above its 89-month EMA throughout the period (secular bull run).")
                print()
            else:
                all_trades_list.append(trades_df)

        except Exception as e:
            print(f"Error backtesting symbol {sym}: {e}")

    if len(universe) > 1 and len(all_trades_list) > 0:
        combined_df = pd.concat(all_trades_list, ignore_index=True)
        generate_performance_summary(
            trades_df=combined_df,
            all_dates=sorted(list(all_dates_set)),
            initial_capital=INITIAL_CAPITAL,
            strategy_title="ST07 COMBINED PORTFOLIO SUMMARY",
        )

if __name__ == "__main__":
    main()
