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
from typing import Dict, Any, List, Tuple, Optional
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
ST07_PREHISTORY_YEARS = int(os.getenv("ST07_PREHISTORY_YEARS", "10"))  # Extended prehistory warmup (>= 89 months)
ST07_EMA_TREND = int(os.getenv("ST07_EMA_TREND", "89"))
ST07_EMA_EXIT = int(os.getenv("ST07_EMA_EXIT", "21"))
ST07_WARMUP_BARS = int(os.getenv("ST07_WARMUP_BARS", "89"))  # Minimum 89 monthly bars for EMA stabilization
ST07_MAX_POSITIONS = int(os.getenv("ST07_MAX_POSITIONS", "4"))  # Maximum concurrent positions in shared portfolio
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
def calculate_entry_charges(entry_px: float, qty: int) -> float:
    """
    Buy-side statutory charges for NSE Cash Equity Delivery:
    - STT (Securities Transaction Tax): 0.1% on Buy turnover
    - Stamp Duty: 0.015% on Buy turnover
    - Exchange Turnover Charges: 0.00297% on Buy turnover + 18% GST (0.0035046% total)
    - SEBI Turnover Charges: ₹10 / crore + 18% GST on Buy turnover (0.000118% total)
    - Brokerage: ₹0 (standard discount broker delivery rate)
    """
    buy_turnover = entry_px * qty
    stt = 0.001 * buy_turnover
    stamp_duty = 0.00015 * buy_turnover
    exchange_charges = 0.000035046 * buy_turnover
    sebi_charges = (10.0 / 10000000.0) * 1.18 * buy_turnover
    brokerage = 0.0
    return stt + stamp_duty + exchange_charges + sebi_charges + brokerage


def calculate_exit_charges(exit_px: float, qty: int) -> float:
    """
    Sell-side statutory charges for NSE Cash Equity Delivery:
    - STT (Securities Transaction Tax): 0.1% on Sell turnover
    - Stamp Duty: ₹0 (charged on Buy turnover only in India)
    - Exchange Turnover Charges: 0.00297% on Sell turnover + 18% GST (0.0035046% total)
    - SEBI Turnover Charges: ₹10 / crore + 18% GST on Sell turnover (0.000118% total)
    - Brokerage: ₹0 (standard discount broker delivery rate)
    """
    sell_turnover = exit_px * qty
    stt = 0.001 * sell_turnover
    exchange_charges = 0.000035046 * sell_turnover
    sebi_charges = (10.0 / 10000000.0) * 1.18 * sell_turnover
    brokerage = 0.0
    return stt + exchange_charges + sebi_charges + brokerage


def calculate_statutory_costs_equity(entry_px: float, exit_px: float, qty: int) -> float:
    """
    Calculates exact round-trip statutory taxes & regulatory fees for NSE Cash Equity Delivery:
    Combines buy-side entry charges and sell-side exit charges.
    """
    return calculate_entry_charges(entry_px, qty) + calculate_exit_charges(exit_px, qty)


def calculate_max_affordable_qty(available_cash: float, entry_price: float) -> int:
    """
    Calculates maximum affordable quantity such that:
        entry_value + entry_charges <= available_cash
    where:
        entry_value = entry_price * qty
        entry_charges = STT (0.1%) + Stamp Duty (0.015%) + Exchange Charges + GST (0.0035046%) + SEBI + GST (0.000118%)
    """
    if available_cash <= 0 or entry_price <= 0:
        return 0
    # Statutory buy-side charge rate per rupee of turnover
    buy_charge_rate = 0.001 + 0.00015 + 0.000035046 + (10.0 / 10000000.0) * 1.18
    effective_price = entry_price * (1.0 + buy_charge_rate)
    qty = int(available_cash / effective_price)

    # Strictly verify constraint: (qty * entry_price) + calculate_entry_charges(entry_price, qty) <= available_cash
    while qty > 0 and ((qty * entry_price) + calculate_entry_charges(entry_price, qty)) > available_cash:
        qty -= 1

    return max(0, qty)

# ==============================================================================
# 3. HISTORICAL DATA INGESTION & MONTHLY RESAMPLING
# ==============================================================================
# ==============================================================================
# 3. DHANHQ MARKET DATA FETCHER & PREHISTORY WARMUP PIPELINE
# ==============================================================================
def compute_ema_with_sma_seed(series: pd.Series, period: int) -> pd.Series:
    """
    Computes an Exponential Moving Average (EMA) seeded by the SMA of the initial 'period' bars,
    eliminating artificial initial-seed dominance and ensuring institutional numerical convergence.
    """
    ema = pd.Series(index=series.index, dtype=float)
    if len(series) < period:
        return series.ewm(span=period, adjust=False).mean()

    # Seed the first period with the SMA
    sma_seed = series.iloc[:period].mean()
    ema.iloc[period - 1] = sma_seed
    alpha = 2.0 / (period + 1.0)

    for i in range(period, len(series)):
        ema.iloc[i] = (series.iloc[i] * alpha) + (ema.iloc[i - 1] * (1.0 - alpha))

    # Pre-period warmup values populated with expanding SMA for display continuity
    ema.iloc[:period - 1] = series.iloc[:period - 1].expanding().mean()
    return ema

def fetch_dhan_daily_data(
    dhan: dhanhq,
    sec_info: Dict[str, str],
    backtest_years: int = ST07_BACKTEST_YEARS,
    prehistory_years: int = ST07_PREHISTORY_YEARS
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches long historical daily data spanning both the evaluation period and extended prehistory:
    - total_span = backtest_years + prehistory_years (e.g. 10 + 10 = 20 years).
    - At least 89+ monthly prehistory bars ensure the 89 EMA is 100% stabilized before evaluation.
    - Excludes incomplete current month from monthly indicator/signal generation.
    Returns:
    1. df_daily: Clean daily candles for day-by-day stop loss and gap execution.
    2. complete_monthly_df: Resampled completed monthly bars for Heikin Ashi and EMA signals.
    """
    end_date = dt.date.today()
    total_years = backtest_years + prehistory_years
    fetch_start_date = end_date - relativedelta(years=total_years)

    symbol = sec_info["symbol"]
    sec_id = sec_info["security_id"]
    segment = sec_info["exchange_segment"]
    inst_type = sec_info["instrument_type"]

    print(f"Fetching {total_years}-year daily data for {symbol} ({sec_id}) from {fetch_start_date} to {end_date} via DhanHQ (Includes {prehistory_years}y Prehistory Warmup)...")
    time.sleep(1.0)
    resp = dhan.historical_daily_data(
        security_id=sec_id,
        exchange_segment=segment,
        instrument_type=inst_type,
        from_date=fetch_start_date.strftime("%Y-%m-%d"),
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
    df_daily = df_daily[(df_daily["high"] >= df_daily["low"]) & (df_daily["open"] > 0) & (df_daily["close"] > 0)].reset_index(drop=True)
    df_daily["ym"] = pd.to_datetime(df_daily["date"]).dt.to_period("M")

    print(f"  ✓ Fetched {len(df_daily)} daily candles for {symbol} ({df_daily['date'].iloc[0]} to {df_daily['date'].iloc[-1]})")

    # Resample daily data to Monthly OHLCV
    df_daily_indexed = df_daily.set_index(pd.to_datetime(df_daily["date"]))

    # Monthly aggregation
    monthly_df = df_daily_indexed.resample("ME").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "date": "last"
    }).dropna().reset_index(drop=True)
    monthly_df["ym"] = pd.to_datetime(monthly_df["date"]).dt.to_period("M")

    # Exclude incomplete current month from monthly indicator & signal generation
    current_calendar_ym = pd.Timestamp.now().to_period("M")
    def is_month_complete(row):
        ym = row["ym"]
        if ym >= current_calendar_ym:
            return False
        cal_month_end = ym.end_time.date()
        return (cal_month_end - row["date"]).days <= 4

    monthly_df["is_complete"] = monthly_df.apply(is_month_complete, axis=1)
    complete_monthly_df = monthly_df[monthly_df["is_complete"]].drop(columns=["is_complete"]).reset_index(drop=True)

    print(f"  ✓ Resampled into {len(complete_monthly_df)} completed monthly candles for {symbol} (excluded incomplete {current_calendar_ym})")
    return df_daily, complete_monthly_df

# ==============================================================================
# 4. INDICATOR ENGINE (HEIKIN ASHI, SMA-SEEDED 89 EMA & 21 EMA)
# ==============================================================================
def compute_indicators(
    df: pd.DataFrame,
    ema_trend_period: int = ST07_EMA_TREND,
    ema_exit_period: int = ST07_EMA_EXIT,
    eval_start_date: Optional[dt.date] = None,
    warmup_bars: int = ST07_WARMUP_BARS
) -> pd.DataFrame:
    """
    Computes Heikin Ashi candles and 89 / 21 EMAs with SMA seeding on monthly closes.
    All bars prior to eval_start_date are marked as warmup to ensure full EMA stabilization.
    """
    df = df.copy()

    # 1. Recursive Heikin Ashi Calculation across full prehistory + evaluation series
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

    # 2. EMAs on Monthly Closes Seeded by 89-period SMA (Institutional Stability)
    df["ema89"] = compute_ema_with_sma_seed(df["close"], ema_trend_period)
    df["ema21"] = compute_ema_with_sma_seed(df["close"], ema_exit_period)

    # 3. 89 EMA Slope Filter (Uptrending)
    df["ema89_up"] = df["ema89"] > df["ema89"].shift(1)

    # 4. Setup Trigger: HA candle opens below 89 EMA and closes above 89 EMA
    df["ha_cross_89"] = (df["ha_open"] < df["ema89"]) & (df["ha_close"] > df["ema89"])
    df["setup_valid"] = df["ha_cross_89"] & df["ema89_up"]

    # 5. Warm-up Period: All prehistory prior to eval_start_date (or minimum warmup_bars)
    if eval_start_date is not None:
        df["is_warmup"] = df["date"] < eval_start_date
    else:
        df["is_warmup"] = np.arange(len(df)) < warmup_bars

    return df

# ==============================================================================
# 5. ST07 INTRADAY / POSITIONAL BACKTEST ENGINE (DUAL-TIMEFRAME)
# ==============================================================================
def run_st07_backtest(
    df_daily: pd.DataFrame,
    df_monthly: pd.DataFrame,
    symbol: str = "RELIANCE",
    initial_capital: float = INITIAL_CAPITAL,
    slippage_pct: float = ST07_SLIPPAGE_PCT
) -> Tuple[pd.DataFrame, pd.Series, List[Dict[str, Any]]]:
    """
    Executes ST07 Positional Strategy Backtest with Dual-Timeframe Execution:
    - Monthly signals: Monthly HA crossover with rising 89 EMA confirms entry setups and 21 EMA exits.
    - Excludes incomplete current month from signal generation; completed months only for indicators.
    - Daily execution: Stop-loss monitoring and MTM equity step through underlying daily candles.
    - Daily Gap-Down protection: If price gaps below SL on any day, fills at daily open (min(open, SL)).
    - Causal Realism: Exits at next-month open for 21 EMA, no same-month re-entry lookahead.
    - Realism: Fills, slippage, and P&L strictly calculated on actual exchange prices.
    - Active Open Holdings: Not artificially closed as END_OF_DATA; marked to market separately.
    """
    trades = []
    position = None
    current_capital = initial_capital
    slippage_mult = slippage_pct / 100.0
    mtm_equity = {}

    daily_by_ym = {ym: group.sort_values("date").reset_index(drop=True) for ym, group in df_daily.groupby("ym")}

    # 1. Iterate through completed months
    for m_idx in range(len(df_monthly)):
        curr_month = df_monthly.iloc[m_idx]
        ym = curr_month["ym"]
        if ym not in daily_by_ym:
            continue
        month_daily = daily_by_ym[ym]
        prev_month = df_monthly.iloc[m_idx - 1] if m_idx > 0 else None

        # Record whether a position was already open at the start of this month.
        # This guarantees causal execution: if an existing position exits during this month,
        # the engine CANNOT retroactively enter a new trade at this same month's open.
        position_at_month_open = position is not None
        first_day = month_daily.iloc[0]

        # ----------------------------------------------------------------------
        # A. PROCESS PENDING 21 EMA EXIT AT FIRST DAY OPEN
        # ----------------------------------------------------------------------
        if position is not None and position.get("pending_exit_21ema", False):
            exit_price = first_day["open"] * (1.0 - slippage_mult)
            exit_charges = calculate_exit_charges(exit_price, position["qty"])
            stat_cost = position.get("entry_charges", 0.0) + exit_charges
            pnl_inr_gross = (exit_price - position["entry"]) * position["qty"]
            pnl_inr_net = pnl_inr_gross - stat_cost
            current_capital += (exit_price * position["qty"]) - exit_charges

            # Execution-aware MAE: adverse excursion up to the execution fill at open
            final_mae_pts = max(position.get("mae_pts", 0.0), max(0.0, position["entry"] - exit_price))
            trades.append({
                "symbol": symbol,
                "date": first_day["date"],
                "entry_date": position["entry_date"],
                "exit_date": first_day["date"],
                "side": "LONG",
                "entry": position["entry"],
                "exit": exit_price,
                "sl": position["sl"],
                "qty": position["qty"],
                "pnl_inr": pnl_inr_net,
                "pnl_pts": pnl_inr_net / position["qty"],
                "exit_reason": "EXIT_21EMA",
                "stat_cost": stat_cost,
                "mae_inr": final_mae_pts * position["qty"],
            })
            position = None

        # ----------------------------------------------------------------------
        # B. TRIGGER NEW ENTRY (Only eligible if NO position was held at month open)
        # ----------------------------------------------------------------------
        if not position_at_month_open and prev_month is not None and not prev_month["is_warmup"]:
            if prev_month["setup_valid"]:
                entry_price = first_day["open"] * (1.0 + slippage_mult)
                sl_price = prev_month["low"]  # Real market low of crossover setup candle
                risk_pts = entry_price - sl_price

                if risk_pts > 0:
                    qty = calculate_max_affordable_qty(current_capital, entry_price)
                    entry_charges = calculate_entry_charges(entry_price, qty)
                    entry_total_cost = (entry_price * qty) + entry_charges
                    if qty > 0 and current_capital >= entry_total_cost:
                        current_capital -= entry_total_cost
                        position = {
                            "symbol": symbol,
                            "entry_date": first_day["date"],
                            "entry": entry_price,
                            "sl": sl_price,
                            "risk_pts": risk_pts,
                            "qty": qty,
                            "entry_charges": entry_charges,
                            "mae_pts": 0.0,
                            "pending_exit_21ema": False,
                        }

        # ----------------------------------------------------------------------
        # C. DAILY BAR LOOP (Intra-month Stop Monitoring & Daily MTM Equity)
        # ----------------------------------------------------------------------
        for _, day_row in month_daily.iterrows():
            d = day_row["date"]
            if position is not None:
                # Check stop-loss breach on the daily candle (captures true gap-downs)
                if day_row["low"] <= position["sl"]:
                    sl_fill = min(day_row["open"], position["sl"])
                    exit_price = sl_fill * (1.0 - slippage_mult)
                    exit_charges = calculate_exit_charges(exit_price, position["qty"])
                    stat_cost = position.get("entry_charges", 0.0) + exit_charges
                    pnl_inr_gross = (exit_price - position["entry"]) * position["qty"]
                    pnl_inr_net = pnl_inr_gross - stat_cost
                    current_capital += (exit_price * position["qty"]) - exit_charges

                    # Execution-aware MAE: trade stopped out at exit_price; ignore subsequent drop after exit
                    final_mae_pts = max(position.get("mae_pts", 0.0), max(0.0, position["entry"] - exit_price))
                    trades.append({
                        "symbol": symbol,
                        "date": d,
                        "entry_date": position["entry_date"],
                        "exit_date": d,
                        "side": "LONG",
                        "entry": position["entry"],
                        "exit": exit_price,
                        "sl": position["sl"],
                        "qty": position["qty"],
                        "pnl_inr": pnl_inr_net,
                        "pnl_pts": pnl_inr_net / position["qty"],
                        "exit_reason": "SL_ENTRY_CANDLE",
                        "stat_cost": stat_cost,
                        "mae_inr": final_mae_pts * position["qty"],
                    })
                    position = None
                else:
                    adverse = max(0.0, position["entry"] - day_row["low"])
                    position["mae_pts"] = max(position.get("mae_pts", 0.0), adverse)

            open_val = position["qty"] * day_row["close"] if position is not None else 0.0
            mtm_equity[d] = current_capital + open_val

        # ----------------------------------------------------------------------
        # D. MONTH-END 21 EMA CHECK
        # ----------------------------------------------------------------------
        if position is not None:
            if curr_month["ha_close"] < curr_month["ema21"]:
                position["pending_exit_21ema"] = True

    # 2. Process incomplete current month daily bars (e.g. current in-progress month)
    completed_yms = set(df_monthly["ym"])
    for ym, month_daily in daily_by_ym.items():
        if ym not in completed_yms:
            first_day = month_daily.iloc[0]
            position_at_month_open = position is not None

            # Pending 21 EMA exit confirmed from the last completed month
            if position is not None and position.get("pending_exit_21ema", False):
                exit_price = first_day["open"] * (1.0 - slippage_mult)
                exit_charges = calculate_exit_charges(exit_price, position["qty"])
                stat_cost = position.get("entry_charges", 0.0) + exit_charges
                pnl_inr_gross = (exit_price - position["entry"]) * position["qty"]
                pnl_inr_net = pnl_inr_gross - stat_cost
                current_capital += (exit_price * position["qty"]) - exit_charges

                # Execution-aware MAE: adverse excursion up to the execution fill at open
                final_mae_pts = max(position.get("mae_pts", 0.0), max(0.0, position["entry"] - exit_price))
                trades.append({
                    "symbol": symbol,
                    "date": first_day["date"],
                    "entry_date": position["entry_date"],
                    "exit_date": first_day["date"],
                    "side": "LONG",
                    "entry": position["entry"],
                    "exit": exit_price,
                    "sl": position["sl"],
                    "qty": position["qty"],
                    "pnl_inr": pnl_inr_net,
                    "pnl_pts": pnl_inr_net / position["qty"],
                    "exit_reason": "EXIT_21EMA",
                    "stat_cost": stat_cost,
                    "mae_inr": final_mae_pts * position["qty"],
                })
                position = None

            # New entry from the last completed month's setup
            if len(df_monthly) > 0:
                last_m = df_monthly.iloc[-1]
                if not position_at_month_open and not last_m["is_warmup"] and last_m["setup_valid"]:
                    entry_price = first_day["open"] * (1.0 + slippage_mult)
                    sl_price = last_m["low"]
                    risk_pts = entry_price - sl_price
                    if risk_pts > 0:
                        qty = calculate_max_affordable_qty(current_capital, entry_price)
                        entry_charges = calculate_entry_charges(entry_price, qty)
                        entry_total_cost = (entry_price * qty) + entry_charges
                        if qty > 0 and current_capital >= entry_total_cost:
                            current_capital -= entry_total_cost
                            position = {
                                "symbol": symbol,
                                "entry_date": first_day["date"],
                                "entry": entry_price,
                                "sl": sl_price,
                                "risk_pts": risk_pts,
                                "qty": qty,
                                "entry_charges": entry_charges,
                                "mae_pts": 0.0,
                                "pending_exit_21ema": False,
                            }

            # Daily bars inside incomplete month
            for _, day_row in month_daily.iterrows():
                d = day_row["date"]
                if position is not None:
                    if day_row["low"] <= position["sl"]:
                        sl_fill = min(day_row["open"], position["sl"])
                        exit_price = sl_fill * (1.0 - slippage_mult)
                        exit_charges = calculate_exit_charges(exit_price, position["qty"])
                        stat_cost = position.get("entry_charges", 0.0) + exit_charges
                        pnl_inr_gross = (exit_price - position["entry"]) * position["qty"]
                        pnl_inr_net = pnl_inr_gross - stat_cost
                        current_capital += (exit_price * position["qty"]) - exit_charges

                        # Execution-aware MAE: trade stopped out at exit_price; ignore subsequent drop after exit
                        final_mae_pts = max(position.get("mae_pts", 0.0), max(0.0, position["entry"] - exit_price))
                        trades.append({
                            "symbol": symbol,
                            "date": d,
                            "entry_date": position["entry_date"],
                            "exit_date": d,
                            "side": "LONG",
                            "entry": position["entry"],
                            "exit": exit_price,
                            "sl": position["sl"],
                            "qty": position["qty"],
                            "pnl_inr": pnl_inr_net,
                            "pnl_pts": pnl_inr_net / position["qty"],
                            "exit_reason": "SL_ENTRY_CANDLE",
                            "stat_cost": stat_cost,
                            "mae_inr": final_mae_pts * position["qty"],
                        })
                        position = None
                    else:
                        adverse = max(0.0, position["entry"] - day_row["low"])
                        position["mae_pts"] = max(position.get("mae_pts", 0.0), adverse)

                open_val = position["qty"] * day_row["close"] if position is not None else 0.0
                mtm_equity[d] = current_capital + open_val

    # 3. Active Open Holdings (Marked-to-Market, NOT forcibly closed as END_OF_DATA)
    open_holdings = []
    if position is not None:
        last_close = df_daily.iloc[-1]["close"]
        unrealized_pnl = (last_close - position["entry"]) * position["qty"] - position.get("entry_charges", 0.0)
        unrealized_return_pct = ((last_close - position["entry"]) / position["entry"]) * 100.0
        open_holdings.append({
            "symbol": symbol,
            "entry_date": position["entry_date"],
            "entry": position["entry"],
            "current_price": last_close,
            "qty": position["qty"],
            "sl": position["sl"],
            "entry_charges": position.get("entry_charges", 0.0),
            "unrealized_pnl": unrealized_pnl,
            "unrealized_return_pct": unrealized_return_pct,
        })

    return pd.DataFrame(trades), pd.Series(mtm_equity), open_holdings

# ==============================================================================
# 6. CHRONOLOGICAL SHARED-CASH PORTFOLIO SIMULATION ENGINE (DAILY EXECUTION)
# ==============================================================================
def run_st07_portfolio_backtest(
    daily_by_sym: Dict[str, pd.DataFrame],
    monthly_by_sym: Dict[str, pd.DataFrame],
    initial_capital: float = INITIAL_CAPITAL,
    max_positions: int = ST07_MAX_POSITIONS,
    slippage_pct: float = ST07_SLIPPAGE_PCT
) -> Tuple[pd.DataFrame, pd.Series, List[Dict[str, Any]]]:
    """
    Executes ST07 Chronological Multi-Asset Portfolio Simulation with Daily Execution:
    - Shared Cash Pool: Starts with 'initial_capital' shared across all universe symbols.
    - Allocation Policy: Each position gets up to (initial_capital / max_positions) allocation.
    - Daily Chronological Step: Steps day-by-day across all symbols simultaneously.
    - Intra-month Stops: Checked on every daily candle, capturing overnight gaps below SL.
    - Excludes incomplete month from signal generation; completed months only for 21 EMA exits.
    - Active Open Holdings: Retained open and marked to market separately without forced END_OF_DATA exit.
    """
    slippage_mult = slippage_pct / 100.0
    shared_cash = initial_capital
    positions = {}
    trades = []
    mtm_equity = {}

    # Build symbol lookup helpers
    sym_daily_indexed = {}
    first_day_of_month = {}
    last_day_of_month = {}
    date_to_ym = {}
    monthly_by_ym = {}
    prev_monthly_by_ym = {}

    symbols = list(daily_by_sym.keys())

    for sym in symbols:
        df_d = daily_by_sym[sym]
        df_m = monthly_by_sym[sym]
        sym_daily_indexed[sym] = df_d.set_index("date")

        first_day_of_month[sym] = {}
        last_day_of_month[sym] = {}
        date_to_ym[sym] = {}

        for ym, group in df_d.groupby("ym"):
            sorted_dates = group.sort_values("date")["date"].tolist()
            if sorted_dates:
                first_day_of_month[sym][ym] = sorted_dates[0]
                last_day_of_month[sym][ym] = sorted_dates[-1]
                for d in sorted_dates:
                    date_to_ym[sym][d] = ym

        monthly_by_ym[sym] = {}
        prev_monthly_by_ym[sym] = {}
        for m_idx in range(len(df_m)):
            row_m = df_m.iloc[m_idx]
            ym_val = row_m["ym"]
            monthly_by_ym[sym][ym_val] = row_m
            prev_monthly_by_ym[sym][ym_val] = df_m.iloc[m_idx - 1] if m_idx > 0 else None

    # Sorted union of all trading dates across all symbols
    all_dates = sorted(list(set.union(*[set(df["date"]) for df in daily_by_sym.values()])))

    for d in all_dates:
        # ----------------------------------------------------------------------
        # 1. PROCESS PENDING 21 EMA EXITS AT MONTH OPEN
        # ----------------------------------------------------------------------
        just_exited_at_open = set()
        for sym in list(positions.keys()):
            pos = positions[sym]
            if pos.get("pending_exit_21ema", False):
                if sym in sym_daily_indexed and d in sym_daily_indexed[sym].index:
                    ym = date_to_ym[sym].get(d)
                    if ym and first_day_of_month[sym].get(ym) == d:
                        row = sym_daily_indexed[sym].loc[d]
                        exit_price = row["open"] * (1.0 - slippage_mult)
                        exit_charges = calculate_exit_charges(exit_price, pos["qty"])
                        stat_cost = pos.get("entry_charges", 0.0) + exit_charges
                        pnl_net = (exit_price - pos["entry"]) * pos["qty"] - stat_cost
                        shared_cash += (exit_price * pos["qty"]) - exit_charges

                        # Execution-aware MAE: adverse excursion up to the execution fill at open
                        final_mae_pts = max(pos.get("mae_pts", 0.0), max(0.0, pos["entry"] - exit_price))
                        trades.append({
                            "symbol": sym,
                            "date": d,
                            "entry_date": pos["entry_date"],
                            "exit_date": d,
                            "side": "LONG",
                            "entry": pos["entry"],
                            "exit": exit_price,
                            "sl": pos["sl"],
                            "qty": pos["qty"],
                            "pnl_inr": pnl_net,
                            "pnl_pts": pnl_net / pos["qty"],
                            "exit_reason": "EXIT_21EMA",
                            "stat_cost": stat_cost,
                            "mae_inr": final_mae_pts * pos["qty"],
                        })
                        del positions[sym]
                        just_exited_at_open.add(sym)

        # ----------------------------------------------------------------------
        # 2. TRIGGER CANDIDATE ENTRIES (FIRST TRADING DAY OF MONTH)
        # ----------------------------------------------------------------------
        for sym in symbols:
            if sym in positions or sym in just_exited_at_open:
                continue
            if len(positions) >= max_positions:
                break
            if sym in sym_daily_indexed and d in sym_daily_indexed[sym].index:
                ym = date_to_ym[sym].get(d)
                if ym and first_day_of_month[sym].get(ym) == d:
                    prev_month = prev_monthly_by_ym[sym].get(ym)
                    if prev_month is not None and prev_month["setup_valid"] and not prev_month["is_warmup"]:
                        row = sym_daily_indexed[sym].loc[d]
                        entry_price = row["open"] * (1.0 + slippage_mult)
                        sl_price = prev_month["low"]
                        risk_pts = entry_price - sl_price

                        if risk_pts > 0:
                            alloc_cap = min(shared_cash, initial_capital / max_positions)
                            qty = calculate_max_affordable_qty(alloc_cap, entry_price)
                            entry_charges = calculate_entry_charges(entry_price, qty)
                            entry_total_cost = (entry_price * qty) + entry_charges
                            if qty > 0 and shared_cash >= entry_total_cost:
                                shared_cash -= entry_total_cost
                                positions[sym] = {
                                    "symbol": sym,
                                    "entry_date": d,
                                    "entry": entry_price,
                                    "sl": sl_price,
                                    "qty": qty,
                                    "entry_charges": entry_charges,
                                    "pending_exit_21ema": False,
                                    "mae_pts": 0.0,
                                }

        # ----------------------------------------------------------------------
        # 3. DAILY STOP-LOSS & MAE UPDATE FOR ALL OPEN POSITIONS
        # ----------------------------------------------------------------------
        for sym in list(positions.keys()):
            if sym in sym_daily_indexed and d in sym_daily_indexed[sym].index:
                row = sym_daily_indexed[sym].loc[d]
                pos = positions[sym]
                if row["low"] <= pos["sl"]:
                    sl_fill = min(row["open"], pos["sl"])
                    exit_price = sl_fill * (1.0 - slippage_mult)
                    exit_charges = calculate_exit_charges(exit_price, pos["qty"])
                    stat_cost = pos.get("entry_charges", 0.0) + exit_charges
                    pnl_net = (exit_price - pos["entry"]) * pos["qty"] - stat_cost
                    shared_cash += (exit_price * pos["qty"]) - exit_charges

                    # Execution-aware MAE: trade stopped out at exit_price; ignore subsequent drop after exit
                    final_mae_pts = max(pos.get("mae_pts", 0.0), max(0.0, pos["entry"] - exit_price))
                    trades.append({
                        "symbol": sym,
                        "date": d,
                        "entry_date": pos["entry_date"],
                        "exit_date": d,
                        "side": "LONG",
                        "entry": pos["entry"],
                        "exit": exit_price,
                        "sl": pos["sl"],
                        "qty": pos["qty"],
                        "pnl_inr": pnl_net,
                        "pnl_pts": pnl_net / pos["qty"],
                        "exit_reason": "SL_ENTRY_CANDLE",
                        "stat_cost": stat_cost,
                        "mae_inr": final_mae_pts * pos["qty"],
                    })
                    del positions[sym]
                else:
                    adverse = max(0.0, pos["entry"] - row["low"])
                    pos["mae_pts"] = max(pos.get("mae_pts", 0.0), adverse)

        # ----------------------------------------------------------------------
        # 4. MONTH-END 21 EMA CHECK (LAST TRADING DAY OF COMPLETED MONTHS ONLY)
        # ----------------------------------------------------------------------
        for sym in list(positions.keys()):
            if sym in sym_daily_indexed and d in sym_daily_indexed[sym].index:
                ym = date_to_ym[sym].get(d)
                # Only evaluate month-end break on completed months
                if ym and last_day_of_month[sym].get(ym) == d and ym in monthly_by_ym[sym]:
                    curr_month = monthly_by_ym[sym].get(ym)
                    if curr_month is not None and curr_month["ha_close"] < curr_month["ema21"]:
                        positions[sym]["pending_exit_21ema"] = True

        # ----------------------------------------------------------------------
        # 5. MARK-TO-MARKET PORTFOLIO EQUITY
        # ----------------------------------------------------------------------
        open_val = 0.0
        for sym, pos in positions.items():
            if sym in sym_daily_indexed and d in sym_daily_indexed[sym].index:
                open_val += pos["qty"] * sym_daily_indexed[sym].loc[d, "close"]
            else:
                open_val += pos["qty"] * pos["entry"]
        mtm_equity[d] = shared_cash + open_val

    # --------------------------------------------------------------------------
    # 6. ACTIVE OPEN HOLDINGS (Marked-to-Market, NOT forcibly closed)
    # --------------------------------------------------------------------------
    open_holdings = []
    for sym, pos in positions.items():
        last_close = daily_by_sym[sym].iloc[-1]["close"]
        unrealized_pnl = (last_close - pos["entry"]) * pos["qty"] - pos.get("entry_charges", 0.0)
        unrealized_return_pct = ((last_close - pos["entry"]) / pos["entry"]) * 100.0
        open_holdings.append({
            "symbol": sym,
            "entry_date": pos["entry_date"],
            "entry": pos["entry"],
            "current_price": last_close,
            "qty": pos["qty"],
            "sl": pos["sl"],
            "entry_charges": pos.get("entry_charges", 0.0),
            "unrealized_pnl": unrealized_pnl,
            "unrealized_return_pct": unrealized_return_pct,
        })

    return pd.DataFrame(trades), pd.Series(mtm_equity), open_holdings

# ==============================================================================
# 7. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    if DHAN_CLIENT_ID == "YOUR_CLIENT_ID" or DHAN_ACCESS_TOKEN == "YOUR_ACCESS_TOKEN":
        print("Warning: Please set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables in .env.")

    dhan_context = DhanContext(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
    dhan_client = dhanhq(dhan_context)

    universe = parse_universe(ST07_UNIVERSE_RAW)
    print(f"\nInitialized ST07 Backtest Engine for universe: {[x['symbol'] for x in universe]}")
    print(f"Strategy: Monthly Heikin Ashi + 89 EMA Crossover (Exit: 21 EMA / Entry Low)")
    print(f"Evaluation Window: {ST07_BACKTEST_YEARS} Years | Prehistory Warmup: {ST07_PREHISTORY_YEARS} Years ({ST07_PREHISTORY_YEARS * 12}+ months)")
    print(f"Initial Capital: ₹{INITIAL_CAPITAL:,.0f} | Max Positions: {ST07_MAX_POSITIONS}\n")

    end_date = dt.date.today()
    eval_start_date = end_date - relativedelta(years=ST07_BACKTEST_YEARS)

    daily_by_sym = {}
    monthly_by_sym = {}
    all_dates_set = set()

    for sec_info in universe:
        sym = sec_info["symbol"]
        try:
            df_daily, monthly_df = fetch_dhan_daily_data(
                dhan_client, sec_info,
                backtest_years=ST07_BACKTEST_YEARS,
                prehistory_years=ST07_PREHISTORY_YEARS
            )
            ind_df = compute_indicators(
                monthly_df,
                ema_trend_period=ST07_EMA_TREND,
                ema_exit_period=ST07_EMA_EXIT,
                eval_start_date=eval_start_date,
                warmup_bars=ST07_WARMUP_BARS
            )
            df_daily_eval = df_daily[df_daily["date"] >= eval_start_date].reset_index(drop=True)
            daily_by_sym[sym] = df_daily_eval
            monthly_by_sym[sym] = ind_df
            all_dates_set.update(df_daily_eval["date"])

            # 1. Standalone Single-Instrument Backtest (for per-stock assessment)
            trades_df, mtm_series, open_holdings = run_st07_backtest(df_daily_eval, ind_df, symbol=sym)
            title = f"ST07 STANDALONE — {sym} ({ST07_BACKTEST_YEARS}Y EVALUATION | {ST07_PREHISTORY_YEARS}Y WARMUP)"
            generate_performance_summary(
                trades_df=trades_df,
                all_dates=df_daily_eval["date"],
                initial_capital=INITIAL_CAPITAL,
                strategy_title=title,
                equity_curve=mtm_series,
                open_positions=open_holdings,
            )

            if trades_df.empty and not open_holdings:
                print(f"[DIAGNOSTIC] No trades or positions generated for {sym}:")
                eval_ind = ind_df[ind_df["date"] >= eval_start_date]
                min_ha_ratio = (eval_ind["ha_open"] / eval_ind["ema89"]).min() if not eval_ind.empty else 0.0
                crossovers = eval_ind["ha_cross_89"].sum() if not eval_ind.empty else 0
                uptrends = eval_ind["ema89_up"].sum() if not eval_ind.empty else 0
                pre_count = len(ind_df) - len(eval_ind)
                print(f"  • Monthly Bars in Eval Window: {len(eval_ind)} | Prehistory Warmup: {pre_count} bars")
                print(f"  • Crossovers (HA Open < 89 EMA & HA Close > 89 EMA): {crossovers}")
                print(f"  • 89 EMA Uptrending Bars: {uptrends} / {len(eval_ind)}")
                print(f"  • Min HA Open / 89 EMA Ratio: {min_ha_ratio:.4f}")
                if crossovers == 0:
                    print(f"  • Insight: {sym} traded strictly above its 89-month EMA throughout the period (secular bull run).")
                print()

        except Exception as e:
            print(f"Error backtesting symbol {sym}: {e}")

    # 2. Chronological Multi-Asset Shared-Cash Portfolio Simulation
    if len(daily_by_sym) > 1:
        portfolio_trades, portfolio_mtm, portfolio_open_holdings = run_st07_portfolio_backtest(
            daily_by_sym=daily_by_sym,
            monthly_by_sym=monthly_by_sym,
            initial_capital=INITIAL_CAPITAL,
            max_positions=ST07_MAX_POSITIONS,
            slippage_pct=ST07_SLIPPAGE_PCT
        )
        title = f"ST07 SHARED PORTFOLIO ({ST07_BACKTEST_YEARS}Y EVALUATION | {ST07_PREHISTORY_YEARS}Y WARMUP | CAPITAL: ₹{INITIAL_CAPITAL:,.0f} | MAX POSITIONS: {ST07_MAX_POSITIONS})"
        generate_performance_summary(
            trades_df=portfolio_trades,
            all_dates=pd.Series(sorted(list(all_dates_set))),
            initial_capital=INITIAL_CAPITAL,
            strategy_title=title,
            equity_curve=portfolio_mtm,
            open_positions=portfolio_open_holdings,
        )

if __name__ == "__main__":
    main()
