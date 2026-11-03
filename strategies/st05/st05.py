import os
import sys
import time
import datetime as dt
from pathlib import Path
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from tabulate import tabulate
from dhanhq import DhanContext, dhanhq

# Add project root to sys.path to import shared modules
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
# 1. CONFIGURATION & STRATEGY PARAMETERS (ST05 PLAYBOOK)
# ==============================================================================
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "YOUR_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")

BACKTEST_MONTHS = int(os.getenv("BACKTEST_MONTHS", "60"))
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000000.0"))
NIFTY_LOT_SIZE = int(os.getenv("NIFTY_LOT_SIZE", "75"))
NUM_LOTS = int(os.getenv("NUM_LOTS", "2"))
QUANTITY = NIFTY_LOT_SIZE * NUM_LOTS
EXECUTION_SLIPPAGE_PTS = float(os.getenv("EXECUTION_SLIPPAGE_PTS", "2.0"))

# Exact Statutory Cost Calculator for NSE Nifty Futures Turnover
def calculate_statutory_costs(entry_px: float, exit_px: float, qty: int, side: str = "LONG") -> float:
    """
    Calculates exact statutory taxes & regulatory fees for Nifty Futures turnover:
    - STT: 0.02% on Sell turnover leg (SEBI revised Oct 1, 2024)
    - Exchange Turnover Charge: 0.00173% on both legs + 18% GST (0.0020414% total)
    - Brokerage: ₹20 / executed order flat (₹40 roundtrip) + 18% GST (₹47.20)
    - Stamp Duty: 0.002% on Buy turnover leg
    - SEBI Turnover Charges: ₹10 / crore + 18% GST on both legs
    """
    buy_px = entry_px if side == "LONG" else exit_px
    sell_px = exit_px if side == "LONG" else entry_px
    
    buy_turnover = buy_px * qty
    sell_turnover = sell_px * qty
    total_turnover = buy_turnover + sell_turnover
    
    stt = 0.0002 * sell_turnover
    brokerage = 47.20
    exchange_charges = 0.000020414 * total_turnover
    stamp_duty = 0.00002 * buy_turnover
    sebi_charges = (10.0 / 10000000.0) * total_turnover * 1.18
    
    return stt + brokerage + exchange_charges + stamp_duty + sebi_charges

# Enhancement 4: Exact Regulatory 5-Tier Historical Lot Size Calendar
USE_HISTORICAL_LOT_SIZES = os.getenv("USE_HISTORICAL_LOT_SIZES", "True").lower() in ("true", "1", "yes")

def get_nifty_lot_size(trade_date: dt.date) -> int:
    """
    Returns exact NSE regulatory lot size for Nifty derivatives by time period:
    - Up to 30-Jun-2021: 75
    - 01-Jul-2021 to 25-Apr-2024: 50
    - 26-Apr-2024 to 19-Nov-2024: 25
    - 20-Nov-2024 to 05-Jan-2026: 75 (NSE Circular FAOP64625)
    - 06-Jan-2026 onwards: 65 (NSE Circular FAOP70616)
    """
    if not USE_HISTORICAL_LOT_SIZES:
        return NIFTY_LOT_SIZE
    if trade_date < dt.date(2021, 7, 1):
        return 75
    elif trade_date < dt.date(2024, 4, 26):
        return 50
    elif trade_date < dt.date(2024, 11, 20):
        return 25
    elif trade_date < dt.date(2026, 1, 6):
        return 75
    else:
        return 65

# Enhancement 1: Dynamic Position Sizing (Fixed Fractional Risk)
ENABLE_DYNAMIC_SIZING = os.getenv("ENABLE_DYNAMIC_SIZING", "True").lower() in ("true", "1", "yes")
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
MIN_LOTS = int(os.getenv("MIN_LOTS", "1"))
MAX_LOTS = int(os.getenv("MAX_LOTS", "5"))

# Enhancement 2: Daily Trade Limits & Circuit Breaker
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "2"))
MAX_LOSSES_PER_DAY = int(os.getenv("MAX_LOSSES_PER_DAY", "2"))

# Enhancement 3: Breakeven Stop-Loss Trailing
ENABLE_BREAKEVEN = os.getenv("ENABLE_BREAKEVEN", "True").lower() in ("true", "1", "yes")
BREAKEVEN_R_MULTIPLE = float(os.getenv("BREAKEVEN_R_MULTIPLE", "1.5"))

# Quant Fix 1: Enforce Fresh Crossover / First Pullback Only (Cuts churn and fee drag)
ENFORCE_FRESH_CROSSOVER_ONLY = os.getenv("ENFORCE_FRESH_CROSSOVER_ONLY", "True").lower() in ("true", "1", "yes")

# Quant Fix 2: ATR Setup Candle Filter & Volatility-Buffered SL
ENABLE_ATR_SETUP_FILTER = os.getenv("ENABLE_ATR_SETUP_FILTER", "True").lower() in ("true", "1", "yes")
ATR_SETUP_MIN_MULT = float(os.getenv("ATR_SETUP_MIN_MULT", "0.5"))
ATR_SETUP_MAX_MULT = float(os.getenv("ATR_SETUP_MAX_MULT", "1.3"))
ENTRY_TRIGGER_BUFFER_PTS = float(os.getenv("ENTRY_TRIGGER_BUFFER_PTS", "2.0"))
SL_ATR_BUFFER_MULT = float(os.getenv("SL_ATR_BUFFER_MULT", "0.25"))

# Quant Fix 3: Hard Time-of-Day Window (09:30 AM to 13:15 PM IST)
ENABLE_TIME_WINDOW_FILTER = os.getenv("ENABLE_TIME_WINDOW_FILTER", "True").lower() in ("true", "1", "yes")
ENTRY_WINDOW_START_TIME = dt.time(9, 30)
ENTRY_WINDOW_END_TIME = dt.time(13, 15)

# Quant Fix 4: ADX / EMA-Spread Choppiness Filter
ENABLE_CHOPPINESS_FILTER = os.getenv("ENABLE_CHOPPINESS_FILTER", "True").lower() in ("true", "1", "yes")
ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))
ADX_MIN_THRESHOLD = float(os.getenv("ADX_MIN_THRESHOLD", "20.0"))
EMA_SPREAD_MIN_PCT = float(os.getenv("EMA_SPREAD_MIN_PCT", "0.08"))  # 0.08% of close

# Quant Fix 5: Dynamic Trailing Stop (1.5R Breakeven + 4 pts, 2.0R Trail EMA 20)
ENABLE_DYNAMIC_TRAILING_STOP = os.getenv("ENABLE_DYNAMIC_TRAILING_STOP", "True").lower() in ("true", "1", "yes")
TRAILING_R1_MULT = float(os.getenv("TRAILING_R1_MULT", "1.5"))
TRAILING_R1_OFFSET_PTS = float(os.getenv("TRAILING_R1_OFFSET_PTS", "4.0"))
TRAILING_R2_MULT = float(os.getenv("TRAILING_R2_MULT", "2.0"))

EMA_FAST = 10
EMA_SLOW = 20
ST_PERIOD = 10               # Standard SuperTrend ATR Period
ST_MULTIPLIER = 3.0          # Standard SuperTrend Multiplier
RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "3.0"))      # 1:3.0 Target calibrated for intraday execution

# ==============================================================================
# 2. FETCH HISTORICAL INTRADAY NIFTY DATA VIA DHANHQ API (CHUNKED)
# ==============================================================================
def fetch_dhan_intraday(dhan: dhanhq, months: int = BACKTEST_MONTHS) -> pd.DataFrame:
    """
    Fetches intraday historical 15-minute data from DhanHQ in 85-day chunks
    (Dhan API limits intraday historical requests to ~90 days per call; supports up to 5 years).
    Uses Nifty 50 Index (security_id='13', IDX_I) for continuous history.
    """
    end_date = dt.date.today()
    start_date = end_date - relativedelta(months=months)
    
    all_chunks = []
    curr_start = start_date
    
    print(f"Fetching 15-min Nifty data for {months} months ({start_date} to {end_date}) via DhanHQ...")
    while curr_start < end_date:
        curr_end = min(curr_start + dt.timedelta(days=85), end_date)
        max_retries = 4
        chunk_success = False

        for attempt in range(max_retries):
            # Throttle between calls to respect DhanHQ rate limits
            time.sleep(1.2)
            resp = dhan.intraday_minute_data(
                security_id="13",             # Nifty 50 continuous index
                exchange_segment="IDX_I",     # Use "NSE_FNO" + active FUTIDX security_id for single expiry
                instrument_type="INDEX",
                from_date=curr_start.strftime("%Y-%m-%d"),
                to_date=curr_end.strftime("%Y-%m-%d"),
                interval=15                   # 15-Minute timeframe per ST05 rules
            )

            if isinstance(resp, dict) and resp.get("status") == "success" and resp.get("data"):
                data = resp["data"]
                # Review 1: Proper timezone localization via utc=True and Asia/Kolkata
                timestamps = (
                    pd.to_datetime(data["timestamp"], unit="s", utc=True)
                    .tz_convert("Asia/Kolkata")
                    .tz_localize(None)
                )
                df_chunk = pd.DataFrame({
                    "timestamp": timestamps,
                    "open": data["open"],
                    "high": data["high"],
                    "low": data["low"],
                    "close": data["close"],
                    "volume": data.get("volume", [0]*len(data["close"]))
                })
                all_chunks.append(df_chunk)
                print(f"  ✓ Fetched {curr_start} to {curr_end} ({len(df_chunk)} candles)")
                chunk_success = True
                break
            else:
                # Review 2: Safe error parsing for non-dict response payloads
                if isinstance(resp, dict):
                    err_msg = resp.get("error_message") or resp.get("remarks") or str(resp)
                else:
                    err_msg = str(resp)

                if "Rate_Limit" in str(resp) or "DH-904" in str(resp):
                    wait_sec = (attempt + 1) * 2.5
                    print(f"  ! Rate limit for {curr_start} -> {curr_end}. Pausing {wait_sec}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_sec)
                else:
                    print(f"  ! Attempt {attempt+1}/{max_retries} failed for {curr_start} -> {curr_end}: {err_msg}")
                    time.sleep(1.5)

        if not chunk_success:
            raise RuntimeError(
                f"Data Completeness Error: Failed to fetch chunk {curr_start} to {curr_end} after {max_retries} retries. "
                f"Aborting backtest to ensure metrics are not corrupted by missing periods."
            )

        curr_start = curr_end + dt.timedelta(days=1)

    if not all_chunks:
        raise ValueError("No data retrieved from DhanHQ. Check Client ID / Access Token.")

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Review 3: Data Integrity, OHLC sanity & Incomplete Live Candle filtering
    # A. Valid OHLC values
    df = df[(df["high"] >= df["low"]) & (df["open"] > 0) & (df["close"] > 0)]

    # B. Valid market hours (09:15 to 15:30)
    df["time"] = df["timestamp"].dt.time
    df = df[(df["time"] >= dt.time(9, 15)) & (df["time"] <= dt.time(15, 30))]

    # C. Exclude in-progress/incomplete current-day candles if running during market hours
    cutoff_time = dt.datetime.now() - dt.timedelta(minutes=15)
    df = df[df["timestamp"] <= cutoff_time]

    df["date"] = df["timestamp"].dt.date
    df = df.reset_index(drop=True)
    return df

# Backward-compatibility alias
fetch_dhan_24m_intraday = fetch_dhan_intraday

# ==============================================================================
# 3. INDICATOR CALCULATIONS (10 EMA, 20 EMA, SUPERTREND)
# ==============================================================================
WARMUP_CANDLES = 50 # Review 4: Exclude early initialization-dependent signals

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Review 4: Mark warm-up candles
    df["is_warmup"] = np.arange(len(df)) < WARMUP_CANDLES

    # 10 & 20 EMA
    df["ema10"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # EMA Slopes (both trending up or both trending down)
    df["ema10_up"] = df["ema10"] > df["ema10"].shift(1)
    df["ema20_up"] = df["ema20"] > df["ema20"].shift(1)
    df["ema10_down"] = df["ema10"] < df["ema10"].shift(1)
    df["ema20_down"] = df["ema20"] < df["ema20"].shift(1)

    # SuperTrend (10, 3)
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/ST_PERIOD, adjust=False).mean()

    hl2 = (high + low) / 2.0
    basic_ub = hl2 + (ST_MULTIPLIER * atr)
    basic_lb = hl2 - (ST_MULTIPLIER * atr)

    # Review 5: Robust SuperTrend Initialization (no zero-initialized bands, no forced +1 bias)
    final_ub = np.zeros(len(df))
    final_lb = np.zeros(len(df))
    st_dir = np.zeros(len(df), dtype=int)

    final_ub[0] = basic_ub.iloc[0]
    final_lb[0] = basic_lb.iloc[0]
    st_dir[0] = 1 if close.iloc[0] >= hl2.iloc[0] else -1

    for i in range(1, len(df)):
        final_ub[i] = basic_ub.iloc[i] if (basic_ub.iloc[i] < final_ub[i-1] or close.iloc[i-1] > final_ub[i-1]) else final_ub[i-1]
        final_lb[i] = basic_lb.iloc[i] if (basic_lb.iloc[i] > final_lb[i-1] or close.iloc[i-1] < final_lb[i-1]) else final_lb[i-1]

        if st_dir[i-1] == -1 and close.iloc[i] > final_ub[i]:
            st_dir[i] = 1
        elif st_dir[i-1] == 1 and close.iloc[i] < final_lb[i]:
            st_dir[i] = -1
        else:
            st_dir[i] = st_dir[i-1]

    df["st_dir"] = st_dir

    # Quant Fix 2: 14-period ATR for Setup Candle Size Filtering & SL Volatility Buffering
    df["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # Quant Fix 4: 14-period ADX & EMA Spread % Choppiness Filter
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    pos_dm_s = pd.Series(pos_dm, index=df.index).ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    neg_dm_s = pd.Series(neg_dm, index=df.index).ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    pos_di = 100.0 * (pos_dm_s / (df["atr14"] + 1e-9))
    neg_di = 100.0 * (neg_dm_s / (df["atr14"] + 1e-9))
    dx = 100.0 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-9)
    df["adx14"] = dx.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    df["adx14_rising"] = df["adx14"] > df["adx14"].shift(1)
    df["ema_spread_pct"] = (df["ema10"] - df["ema20"]).abs() / df["close"] * 100.0

    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    return df

# ==============================================================================
# 4. ST05 INTRADAY BACKTEST ENGINE
# ==============================================================================
def run_st05_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes ST05 Rules with Complete Institutional Backtesting Realism:
    - Long Setup: Close > EMA10 & EMA20, EMA10 > EMA20, both EMAs sloping up, SuperTrend == Green (1), Green candle
    - Short Setup: Close < EMA10 & EMA20, EMA10 < EMA20, both EMAs sloping down, SuperTrend == Red (-1), Red candle
    - Entry: Break of setup candle High (Long) or Low (Short)
    - SL: Low of setup candle (Long) or High of setup candle (Short)
    - Target: 1:3 Risk-Reward (or 15:15 IST intraday auto-square-off)
    - Fix 1 & 2: Immediate Entry-Candle Evaluation & Conservative Ambiguity Resolution (SL wins if both hit)
    - Fix 3: Zero-Lookahead Breakeven (Armed strictly for future bars)
    - Fix 4: Real Historical Lot Sizing & Statutory Derivatives Friction
    - Fix 5: Adverse Gap-Through-Stop Fills (filled at open if opened beyond stop)
    - Fix 6: 15:15 Square-off filled at 15:15:00 open price
    - Fix 7: EOD Fail-safe (prevents positions from disappearing on truncated/missing days)
    - Fix 8: True 1% risk position sizing (skips trades when account cannot afford 1 lot within 1% risk)
    """
    trades = []
    current_capital = INITIAL_CAPITAL
    
    for trade_date, day_df in df.groupby("date"):
        day_df = day_df.reset_index(drop=True)
        position = None      # None, "LONG", or "SHORT"
        pending_setup = None # Stores setup candle trigger levels
        day_trades_count = 0
        day_losses_count = 0
        lot_size = get_nifty_lot_size(trade_date)
        
        # Quant Fix 1: State tracking for First Pullback / Fresh Crossover Only
        long_wave_active = False
        short_wave_active = False
        prev_ema_cross = day_df.iloc[0]["ema10"] > day_df.iloc[0]["ema20"]
        prev_st_dir = day_df.iloc[0]["st_dir"]
        
        for i in range(1, len(day_df)):
            row = day_df.iloc[i]
            
            # Review 4: Skip warm-up candles to avoid initialization-dependent signals
            if row.get("is_warmup", False):
                continue
            
            # Quant Fix 1: Detect Crossover & Pullback Resets
            curr_ema_cross = row["ema10"] > row["ema20"]
            if curr_ema_cross != prev_ema_cross:
                long_wave_active = False
                short_wave_active = False
                prev_ema_cross = curr_ema_cross
                
            if row["st_dir"] != prev_st_dir:
                long_wave_active = False
                short_wave_active = False
                prev_st_dir = row["st_dir"]
                
            # If price pulls back to touch 20 EMA, a fresh pullback entry is armed
            if curr_ema_cross and row["low"] <= row["ema20"]:
                long_wave_active = False
            if not curr_ema_cross and row["high"] >= row["ema20"]:
                short_wave_active = False
            
            # ==================================================================
            # 1. MANAGE EXISTING OPEN POSITION (From prior candles)
            # ==================================================================
            if position is not None:
                # Update intratrade Maximum Adverse Excursion (MAE)
                if position["side"] == "LONG":
                    adverse = max(0.0, position["entry"] - row["low"])
                else:
                    adverse = max(0.0, row["high"] - position["entry"])
                position["mae_pts"] = max(position.get("mae_pts", 0.0), adverse)

                # Point 6: End of Day Intraday Square-off at 15:15 IST sharp (Fill at open of 15:15 bar)
                if row["time"] >= dt.time(15, 15):
                    exit_price = row["open"]
                    stat_cost = calculate_statutory_costs(position["entry"], exit_price, position["qty"], position["side"])
                    pnl_pts_gross = (exit_price - position["entry"]) if position["side"] == "LONG" else (position["entry"] - exit_price)
                    pnl_pts_gross -= EXECUTION_SLIPPAGE_PTS
                    pnl_inr_net = (pnl_pts_gross * position["qty"]) - stat_cost
                    pnl_pts_net = pnl_inr_net / position["qty"]
                    current_capital += pnl_inr_net
                    trades.append({**position, "exit_time": row["timestamp"], "exit": exit_price,
                                   "pnl_pts": pnl_pts_net, "pnl_inr": pnl_inr_net, "exit_reason": "EOD_1515",
                                   "stat_cost": stat_cost, "mae_inr": position["mae_pts"] * position["qty"]})
                    day_trades_count += 1
                    if pnl_inr_net <= 0:
                        day_losses_count += 1
                    position = None
                    break
                
                # B. Check SL and Target with Conservative Ambiguity Resolution & Gap Fills
                if position["side"] == "LONG":
                    hit_sl = row["low"] <= position["sl"]
                    hit_target = row["high"] >= position["target"]
                    
                    if hit_sl and hit_target:
                        sl_fill = min(row["open"], position["sl"])
                        stat_cost = calculate_statutory_costs(position["entry"], sl_fill, position["qty"], "LONG")
                        pnl_pts_gross = (sl_fill - position["entry"]) - EXECUTION_SLIPPAGE_PTS
                        pnl_inr_net = (pnl_pts_gross * position["qty"]) - stat_cost
                        pnl_pts_net = pnl_inr_net / position["qty"]
                        current_capital += pnl_inr_net
                        exit_reason = "TRAIL_EMA20" if position.get("trail_ema_active", False) else (
                            "BREAKEVEN_PLUS" if position.get("be_triggered", False) and sl_fill >= position["entry"] else "SL_AMBIGUOUS"
                        )
                        trades.append({**position, "exit_time": row["timestamp"], "exit": sl_fill,
                                       "pnl_pts": pnl_pts_net, "pnl_inr": pnl_inr_net, "exit_reason": exit_reason,
                                       "stat_cost": stat_cost, "mae_inr": position["mae_pts"] * position["qty"]})
                        day_trades_count += 1
                        if pnl_inr_net <= 0:
                            day_losses_count += 1
                        position = None
                    elif hit_sl:
                        sl_fill = min(row["open"], position["sl"])
                        stat_cost = calculate_statutory_costs(position["entry"], sl_fill, position["qty"], "LONG")
                        pnl_pts_gross = (sl_fill - position["entry"]) - EXECUTION_SLIPPAGE_PTS
                        pnl_inr_net = (pnl_pts_gross * position["qty"]) - stat_cost
                        pnl_pts_net = pnl_inr_net / position["qty"]
                        current_capital += pnl_inr_net
                        exit_reason = "TRAIL_EMA20" if position.get("trail_ema_active", False) else (
                            "BREAKEVEN_PLUS" if position.get("be_triggered", False) and sl_fill >= position["entry"] else "SL"
                        )
                        trades.append({**position, "exit_time": row["timestamp"], "exit": sl_fill,
                                       "pnl_pts": pnl_pts_net, "pnl_inr": pnl_inr_net, "exit_reason": exit_reason,
                                       "stat_cost": stat_cost, "mae_inr": position["mae_pts"] * position["qty"]})
                        day_trades_count += 1
                        if pnl_inr_net <= 0:
                            day_losses_count += 1
                        position = None
                    elif hit_target:
                        stat_cost = calculate_statutory_costs(position["entry"], position["target"], position["qty"], "LONG")
                        pnl_pts_gross = (position["target"] - position["entry"]) - EXECUTION_SLIPPAGE_PTS
                        pnl_inr_net = (pnl_pts_gross * position["qty"]) - stat_cost
                        pnl_pts_net = pnl_inr_net / position["qty"]
                        current_capital += pnl_inr_net
                        trades.append({**position, "exit_time": row["timestamp"], "exit": position["target"],
                                       "pnl_pts": pnl_pts_net, "pnl_inr": pnl_inr_net, "exit_reason": "TARGET",
                                       "stat_cost": stat_cost, "mae_inr": position["mae_pts"] * position["qty"]})
                        day_trades_count += 1
                        position = None
                    else:
                        if ENABLE_DYNAMIC_TRAILING_STOP:
                            if row["high"] >= position["entry"] + (TRAILING_R2_MULT * position["risk_pts"]):
                                position["trail_ema_active"] = True
                            if row["high"] >= position["entry"] + (TRAILING_R1_MULT * position["risk_pts"]):
                                position["be_triggered"] = True
                                position["sl"] = max(position["sl"], position["entry"] + TRAILING_R1_OFFSET_PTS)
                            if position.get("trail_ema_active", False):
                                position["sl"] = max(position["sl"], row["ema20"])
                        elif ENABLE_BREAKEVEN and not position["be_triggered"]:
                            if row["high"] >= position["entry"] + (BREAKEVEN_R_MULTIPLE * position["risk_pts"]):
                                position["sl"] = max(position["sl"], position["entry"])
                                position["be_triggered"] = True

                elif position["side"] == "SHORT":
                    hit_sl = row["high"] >= position["sl"]
                    hit_target = row["low"] <= position["target"]
                    
                    if hit_sl and hit_target:
                        sl_fill = max(row["open"], position["sl"])
                        stat_cost = calculate_statutory_costs(position["entry"], sl_fill, position["qty"], "SHORT")
                        pnl_pts_gross = (position["entry"] - sl_fill) - EXECUTION_SLIPPAGE_PTS
                        pnl_inr_net = (pnl_pts_gross * position["qty"]) - stat_cost
                        pnl_pts_net = pnl_inr_net / position["qty"]
                        current_capital += pnl_inr_net
                        exit_reason = "TRAIL_EMA20" if position.get("trail_ema_active", False) else (
                            "BREAKEVEN_PLUS" if position.get("be_triggered", False) and sl_fill <= position["entry"] else "SL_AMBIGUOUS"
                        )
                        trades.append({**position, "exit_time": row["timestamp"], "exit": sl_fill,
                                       "pnl_pts": pnl_pts_net, "pnl_inr": pnl_inr_net, "exit_reason": exit_reason,
                                       "stat_cost": stat_cost, "mae_inr": position["mae_pts"] * position["qty"]})
                        day_trades_count += 1
                        if pnl_inr_net <= 0:
                            day_losses_count += 1
                        position = None
                    elif hit_sl:
                        sl_fill = max(row["open"], position["sl"])
                        stat_cost = calculate_statutory_costs(position["entry"], sl_fill, position["qty"], "SHORT")
                        pnl_pts_gross = (position["entry"] - sl_fill) - EXECUTION_SLIPPAGE_PTS
                        pnl_inr_net = (pnl_pts_gross * position["qty"]) - stat_cost
                        pnl_pts_net = pnl_inr_net / position["qty"]
                        current_capital += pnl_inr_net
                        exit_reason = "TRAIL_EMA20" if position.get("trail_ema_active", False) else (
                            "BREAKEVEN_PLUS" if position.get("be_triggered", False) and sl_fill <= position["entry"] else "SL"
                        )
                        trades.append({**position, "exit_time": row["timestamp"], "exit": sl_fill,
                                       "pnl_pts": pnl_pts_net, "pnl_inr": pnl_inr_net, "exit_reason": exit_reason,
                                       "stat_cost": stat_cost, "mae_inr": position["mae_pts"] * position["qty"]})
                        day_trades_count += 1
                        if pnl_inr_net <= 0:
                            day_losses_count += 1
                        position = None
                    elif hit_target:
                        stat_cost = calculate_statutory_costs(position["entry"], position["target"], position["qty"], "SHORT")
                        pnl_pts_gross = (position["entry"] - position["target"]) - EXECUTION_SLIPPAGE_PTS
                        pnl_inr_net = (pnl_pts_gross * position["qty"]) - stat_cost
                        pnl_pts_net = pnl_inr_net / position["qty"]
                        current_capital += pnl_inr_net
                        trades.append({**position, "exit_time": row["timestamp"], "exit": position["target"],
                                       "pnl_pts": pnl_pts_net, "pnl_inr": pnl_inr_net, "exit_reason": "TARGET",
                                       "stat_cost": stat_cost, "mae_inr": position["mae_pts"] * position["qty"]})
                        day_trades_count += 1
                        position = None
                    else:
                        if ENABLE_DYNAMIC_TRAILING_STOP:
                            if row["low"] <= position["entry"] - (TRAILING_R2_MULT * position["risk_pts"]):
                                position["trail_ema_active"] = True
                            if row["low"] <= position["entry"] - (TRAILING_R1_MULT * position["risk_pts"]):
                                position["be_triggered"] = True
                                position["sl"] = min(position["sl"], position["entry"] - TRAILING_R1_OFFSET_PTS)
                            if position.get("trail_ema_active", False):
                                position["sl"] = min(position["sl"], row["ema20"])
                        elif ENABLE_BREAKEVEN and not position["be_triggered"]:
                            if row["low"] <= position["entry"] - (BREAKEVEN_R_MULTIPLE * position["risk_pts"]):
                                position["sl"] = min(position["sl"], position["entry"])
                                position["be_triggered"] = True

            # Daily circuit breaker & trade caps
            can_trade_today = (day_trades_count < MAX_TRADES_PER_DAY) and (day_losses_count < MAX_LOSSES_PER_DAY)

            # ==================================================================
            # 2. TRIGGER PENDING SETUP ON CURRENT BAR (Friction-Inclusive 1% Sizing & Sequencing Fix)
            # ==================================================================
            # Quant Fix 3: Hard Time-of-Day Window (09:30 AM to 13:15 PM IST)
            within_entry_window = (
                (row["time"] >= ENTRY_WINDOW_START_TIME and row["time"] <= ENTRY_WINDOW_END_TIME)
                if ENABLE_TIME_WINDOW_FILTER else (row["time"] <= dt.time(14, 30))
            )

            # Expire pending setup if past allowable entry window
            cutoff_entry_time = ENTRY_WINDOW_END_TIME if ENABLE_TIME_WINDOW_FILTER else dt.time(14, 30)
            if pending_setup is not None and row["time"] > cutoff_entry_time:
                pending_setup = None

            if position is None and pending_setup is not None and within_entry_window and can_trade_today:
                if pending_setup["side"] == "LONG" and row["high"] > pending_setup["trigger"]:
                    entry_px = max(row["open"], pending_setup["trigger"])
                    risk = entry_px - pending_setup["sl"]
                    if risk > 5:
                        if ENABLE_DYNAMIC_SIZING:
                            risk_budget = current_capital * (RISK_PER_TRADE_PCT / 100.0)
                            # Include execution slippage and estimated roundtrip statutory friction (~7 pts) in risk denominator
                            total_risk_per_share = risk + EXECUTION_SLIPPAGE_PTS + 7.0
                            risk_per_lot = total_risk_per_share * lot_size
                            affordable_lots = int(risk_budget / risk_per_lot) if risk_per_lot > 0 else 0
                            if affordable_lots < 1:
                                pending_setup = None
                                continue
                            lots = min(affordable_lots, MAX_LOTS)
                        else:
                            lots = NUM_LOTS
                        qty = lots * lot_size
                        
                        pos = {
                            "date": trade_date, "side": "LONG", "entry_time": row["timestamp"],
                            "entry": entry_px, "sl": pending_setup["sl"], "risk_pts": risk,
                            "target": entry_px + (RISK_REWARD_RATIO * risk),
                            "lots": lots, "qty": qty, "be_triggered": False, "trail_ema_active": False,
                            "mae_pts": max(0.0, entry_px - row["low"])
                        }
                        
                        # Entry Candle Sequencing:
                        # If opened >= trigger (gap up): low occurred post-entry -> genuine stop.
                        # If opened < trigger: candle rallied through trigger; only stop out if candle closed <= SL.
                        is_gap_entry = row["open"] >= pending_setup["trigger"]
                        closed_below_sl = row["close"] <= pos["sl"]
                        hit_sl_entry = (is_gap_entry and row["low"] <= pos["sl"]) or closed_below_sl
                        hit_tgt_entry = row["high"] >= pos["target"]
                        
                        if hit_sl_entry:
                            stat_cost = calculate_statutory_costs(pos["entry"], pos["sl"], pos["qty"], "LONG")
                            pnl_pts_gross = (pos["sl"] - pos["entry"]) - EXECUTION_SLIPPAGE_PTS
                            pnl_inr_net = (pnl_pts_gross * pos["qty"]) - stat_cost
                            pnl_pts_net = pnl_inr_net / pos["qty"]
                            current_capital += pnl_inr_net
                            trades.append({**pos, "exit_time": row["timestamp"], "exit": pos["sl"],
                                           "pnl_pts": pnl_pts_net, "pnl_inr": pnl_inr_net, "exit_reason": "SL_ENTRY_BAR",
                                           "stat_cost": stat_cost, "mae_inr": pos["mae_pts"] * pos["qty"]})
                            day_trades_count += 1
                            if pnl_inr_net <= 0:
                                day_losses_count += 1
                            position = None
                        elif hit_tgt_entry:
                            stat_cost = calculate_statutory_costs(pos["entry"], pos["target"], pos["qty"], "LONG")
                            pnl_pts_gross = (pos["target"] - pos["entry"]) - EXECUTION_SLIPPAGE_PTS
                            pnl_inr_net = (pnl_pts_gross * pos["qty"]) - stat_cost
                            pnl_pts_net = pnl_inr_net / pos["qty"]
                            current_capital += pnl_inr_net
                            trades.append({**pos, "exit_time": row["timestamp"], "exit": pos["target"],
                                           "pnl_pts": pnl_pts_net, "pnl_inr": pnl_inr_net, "exit_reason": "TARGET_ENTRY_BAR",
                                           "stat_cost": stat_cost, "mae_inr": pos["mae_pts"] * pos["qty"]})
                            day_trades_count += 1
                            position = None
                        else:
                            if ENABLE_DYNAMIC_TRAILING_STOP:
                                if row["high"] >= pos["entry"] + (TRAILING_R2_MULT * pos["risk_pts"]):
                                    pos["trail_ema_active"] = True
                                if row["high"] >= pos["entry"] + (TRAILING_R1_MULT * pos["risk_pts"]):
                                    pos["be_triggered"] = True
                                    pos["sl"] = max(pos["sl"], pos["entry"] + TRAILING_R1_OFFSET_PTS)
                                if pos.get("trail_ema_active", False):
                                    pos["sl"] = max(pos["sl"], row["ema20"])
                            elif ENABLE_BREAKEVEN and row["high"] >= pos["entry"] + (BREAKEVEN_R_MULTIPLE * pos["risk_pts"]):
                                pos["sl"] = max(pos["sl"], pos["entry"])
                                pos["be_triggered"] = True
                            position = pos
                            
                    long_wave_active = True
                    pending_setup = None

                elif pending_setup["side"] == "SHORT" and row["low"] < pending_setup["trigger"]:
                    entry_px = min(row["open"], pending_setup["trigger"])
                    risk = pending_setup["sl"] - entry_px
                    if risk > 5:
                        if ENABLE_DYNAMIC_SIZING:
                            risk_budget = current_capital * (RISK_PER_TRADE_PCT / 100.0)
                            total_risk_per_share = risk + EXECUTION_SLIPPAGE_PTS + 7.0
                            risk_per_lot = total_risk_per_share * lot_size
                            affordable_lots = int(risk_budget / risk_per_lot) if risk_per_lot > 0 else 0
                            if affordable_lots < 1:
                                pending_setup = None
                                continue
                            lots = min(affordable_lots, MAX_LOTS)
                        else:
                            lots = NUM_LOTS
                        qty = lots * lot_size
                        
                        pos = {
                            "date": trade_date, "side": "SHORT", "entry_time": row["timestamp"],
                            "entry": entry_px, "sl": pending_setup["sl"], "risk_pts": risk,
                            "target": entry_px - (RISK_REWARD_RATIO * risk),
                            "lots": lots, "qty": qty, "be_triggered": False, "trail_ema_active": False,
                            "mae_pts": max(0.0, row["high"] - entry_px)
                        }
                        
                        is_gap_entry = row["open"] <= pending_setup["trigger"]
                        closed_above_sl = row["close"] >= pos["sl"]
                        hit_sl_entry = (is_gap_entry and row["high"] >= pos["sl"]) or closed_above_sl
                        hit_tgt_entry = row["low"] <= pos["target"]
                        
                        if hit_sl_entry:
                            stat_cost = calculate_statutory_costs(pos["entry"], pos["sl"], pos["qty"], "SHORT")
                            pnl_pts_gross = (pos["entry"] - pos["sl"]) - EXECUTION_SLIPPAGE_PTS
                            pnl_inr_net = (pnl_pts_gross * pos["qty"]) - stat_cost
                            pnl_pts_net = pnl_inr_net / pos["qty"]
                            current_capital += pnl_inr_net
                            trades.append({**pos, "exit_time": row["timestamp"], "exit": pos["sl"],
                                           "pnl_pts": pnl_pts_net, "pnl_inr": pnl_inr_net, "exit_reason": "SL_ENTRY_BAR",
                                           "stat_cost": stat_cost, "mae_inr": pos["mae_pts"] * pos["qty"]})
                            day_trades_count += 1
                            if pnl_inr_net <= 0:
                                day_losses_count += 1
                            position = None
                        elif hit_tgt_entry:
                            stat_cost = calculate_statutory_costs(pos["entry"], pos["target"], pos["qty"], "SHORT")
                            pnl_pts_gross = (pos["entry"] - pos["target"]) - EXECUTION_SLIPPAGE_PTS
                            pnl_inr_net = (pnl_pts_gross * pos["qty"]) - stat_cost
                            pnl_pts_net = pnl_inr_net / pos["qty"]
                            current_capital += pnl_inr_net
                            trades.append({**pos, "exit_time": row["timestamp"], "exit": pos["target"],
                                           "pnl_pts": pnl_pts_net, "pnl_inr": pnl_inr_net, "exit_reason": "TARGET_ENTRY_BAR",
                                           "stat_cost": stat_cost, "mae_inr": pos["mae_pts"] * pos["qty"]})
                            day_trades_count += 1
                            position = None
                        else:
                            if ENABLE_DYNAMIC_TRAILING_STOP:
                                if row["low"] <= pos["entry"] - (TRAILING_R2_MULT * pos["risk_pts"]):
                                    pos["trail_ema_active"] = True
                                if row["low"] <= pos["entry"] - (TRAILING_R1_MULT * pos["risk_pts"]):
                                    pos["be_triggered"] = True
                                    pos["sl"] = min(pos["sl"], pos["entry"] - TRAILING_R1_OFFSET_PTS)
                                if pos.get("trail_ema_active", False):
                                    pos["sl"] = min(pos["sl"], row["ema20"])
                            elif ENABLE_BREAKEVEN and row["low"] <= pos["entry"] - (BREAKEVEN_R_MULTIPLE * pos["risk_pts"]):
                                pos["sl"] = min(pos["sl"], pos["entry"])
                                pos["be_triggered"] = True
                            position = pos
                            
                    short_wave_active = True
                    pending_setup = None
            elif not can_trade_today:
                pending_setup = None

            # ==================================================================
            # 3. IDENTIFY NEW SETUP CANDLE (ST05 Alignment + Quant Filters)
            # ==================================================================
            # Quant Fix 3: Skip 09:15-09:30 morning opening bar; cutoff setup formation at 13:00 (so trigger is <= 13:15)
            within_setup_window = (
                (row["time"] >= ENTRY_WINDOW_START_TIME and row["time"] < ENTRY_WINDOW_END_TIME)
                if ENABLE_TIME_WINDOW_FILTER else (row["time"] < dt.time(14, 15))
            )

            if position is None and within_setup_window and can_trade_today:
                is_green = row["close"] > row["open"]
                is_red = row["close"] < row["open"]
                
                long_aligned = (
                    row["close"] > row["ema10"] and
                    row["close"] > row["ema20"] and
                    row["ema10"] > row["ema20"] and
                    row["ema10_up"] and row["ema20_up"] and
                    row["st_dir"] == 1 and is_green
                )
                
                short_aligned = (
                    row["close"] < row["ema10"] and
                    row["close"] < row["ema20"] and
                    row["ema10"] < row["ema20"] and
                    row["ema10_down"] and row["ema20_down"] and
                    row["st_dir"] == -1 and is_red
                )
                
                # Quant Fix 4: ADX / EMA-Spread Choppiness Filter (Must not be pinched flat or non-trending)
                if ENABLE_CHOPPINESS_FILTER:
                    adx_rising = row.get("adx14_rising", False)
                    adx_trending = (row["adx14"] > ADX_MIN_THRESHOLD) and adx_rising
                    ema_spread_trending = row["ema_spread_pct"] > EMA_SPREAD_MIN_PCT
                    regime_trending = adx_trending or ema_spread_trending
                else:
                    regime_trending = True

                # Quant Fix 2: Setup Candle Size Filter (0.5 * ATR14 <= range <= 1.3 * ATR14)
                candle_range = row["high"] - row["low"]
                if ENABLE_ATR_SETUP_FILTER:
                    valid_candle_size = (ATR_SETUP_MIN_MULT * row["atr14"] <= candle_range <= ATR_SETUP_MAX_MULT * row["atr14"])
                else:
                    valid_candle_size = True

                # Quant Fix 1: Filter out repetitive entries in the same ongoing wave without a pullback
                can_take_long = (not long_wave_active) if ENFORCE_FRESH_CROSSOVER_ONLY else True
                can_take_short = (not short_wave_active) if ENFORCE_FRESH_CROSSOVER_ONLY else True

                if long_aligned and regime_trending and valid_candle_size and can_take_long:
                    # Quant Fix 2: Buffer Trigger (+2 pts) and Stop-Loss (-0.25*ATR14 or EMA20)
                    if ENABLE_ATR_SETUP_FILTER:
                        trigger_px = row["high"] + ENTRY_TRIGGER_BUFFER_PTS
                        sl_px = min(row["low"] - (SL_ATR_BUFFER_MULT * row["atr14"]), row["ema20"])
                    else:
                        trigger_px = row["high"]
                        sl_px = row["low"]
                    pending_setup = {"side": "LONG", "trigger": trigger_px, "sl": sl_px}
                    
                elif short_aligned and regime_trending and valid_candle_size and can_take_short:
                    # Quant Fix 2: Buffer Trigger (-2 pts) and Stop-Loss (+0.25*ATR14 or EMA20)
                    if ENABLE_ATR_SETUP_FILTER:
                        trigger_px = row["low"] - ENTRY_TRIGGER_BUFFER_PTS
                        sl_px = max(row["high"] + (SL_ATR_BUFFER_MULT * row["atr14"]), row["ema20"])
                    else:
                        trigger_px = row["low"]
                        sl_px = row["high"]
                    pending_setup = {"side": "SHORT", "trigger": trigger_px, "sl": sl_px}

        # End of Day Fail-Safe (Ensures positions NEVER silently disappear & flags truncated days)
        if position is not None:
            last_row = day_df.iloc[-1]
            if last_row["time"] < dt.time(15, 0):
                # Truncated session: mark unresolved rather than fabricating a fill
                trades.append({**position, "exit_time": last_row["timestamp"], "exit": last_row["close"],
                               "pnl_pts": 0.0, "pnl_inr": 0.0, "exit_reason": "UNRESOLVED_TRUNCATED_SESSION",
                               "stat_cost": 0.0, "mae_inr": position.get("mae_pts", 0.0) * position["qty"]})
            else:
                exit_price = last_row["close"]
                stat_cost = calculate_statutory_costs(position["entry"], exit_price, position["qty"], position["side"])
                pnl_pts_gross = (exit_price - position["entry"]) if position["side"] == "LONG" else (position["entry"] - exit_price)
                pnl_pts_gross -= EXECUTION_SLIPPAGE_PTS
                pnl_inr_net = (pnl_pts_gross * position["qty"]) - stat_cost
                pnl_pts_net = pnl_inr_net / position["qty"]
                current_capital += pnl_inr_net
                trades.append({**position, "exit_time": last_row["timestamp"], "exit": exit_price,
                               "pnl_pts": pnl_pts_net, "pnl_inr": pnl_inr_net, "exit_reason": "EOD_FORCED",
                               "stat_cost": stat_cost, "mae_inr": position.get("mae_pts", 0.0) * position["qty"]})
                day_trades_count += 1
                if pnl_inr_net <= 0:
                    day_losses_count += 1
            position = None

    return pd.DataFrame(trades)

if __name__ == "__main__":
    if DHAN_CLIENT_ID == "YOUR_CLIENT_ID" or DHAN_ACCESS_TOKEN == "YOUR_ACCESS_TOKEN":
        print("Warning: Please set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables or update the configuration.")
    dhan_context = DhanContext(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
    dhan_client = dhanhq(dhan_context)
    raw_df = fetch_dhan_intraday(dhan_client, months=BACKTEST_MONTHS)
    ind_df = compute_indicators(raw_df)
    trades = run_st05_backtest(ind_df)
    generate_performance_summary(
        trades,
        ind_df["date"],
        initial_capital=INITIAL_CAPITAL,
        strategy_title="NIFTY 15M ST05 STRATEGY"
    )
