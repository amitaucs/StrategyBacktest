import os
import time
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from tabulate import tabulate
from dhanhq import DhanContext, dhanhq

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
# Point 12: Realistic Derivative Friction (STT 5.0 pts + Exchange/GST 1.0 pt + Brokerage 0.6 pt + Stamp 0.4 pt + Slippage 1.5 pts = ~8.5 pts)
SLIPPAGE_AND_COSTS_PTS = float(os.getenv("SLIPPAGE_AND_COSTS_PTS", "8.0"))

# Enhancement 4: Historical Lot Size & Derivative Friction
USE_HISTORICAL_LOT_SIZES = os.getenv("USE_HISTORICAL_LOT_SIZES", "True").lower() in ("true", "1", "yes")

def get_nifty_lot_size(trade_date: dt.date) -> int:
    """
    Returns exact NSE regulatory lot size for Nifty derivatives by time period:
    - Prior to July 2021: 75
    - July 2021 to April 2024: 50
    - April 2024 to Nov 19, 2024: 25
    - Nov 20, 2024 onwards: 75 (or NIFTY_LOT_SIZE from .env)
    """
    if not USE_HISTORICAL_LOT_SIZES:
        return NIFTY_LOT_SIZE
    if trade_date < dt.date(2021, 7, 1):
        return 75
    elif trade_date < dt.date(2024, 4, 26):
        return 50
    elif trade_date < dt.date(2024, 11, 20):
        return 25
    else:
        return NIFTY_LOT_SIZE

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

# CAGR Performance Rating Thresholds (in %) from env
CAGR_BAD_BELOW = float(os.getenv("CAGR_BAD_BELOW", "12.0"))
CAGR_ACCEPTABLE_MAX = float(os.getenv("CAGR_ACCEPTABLE_MAX", "16.0"))
CAGR_GOOD_MIN = float(os.getenv("CAGR_GOOD_MIN", "20.0"))
CAGR_EXCELLENT_MIN = float(os.getenv("CAGR_EXCELLENT_MIN", "30.0"))

def classify_cagr(cagr: float) -> str:
    """Classifies CAGR according to configured percentage thresholds."""
    if cagr >= CAGR_EXCELLENT_MIN:
        return "EXCELLENT"
    elif cagr >= CAGR_GOOD_MIN:
        return "GOOD"
    elif cagr > CAGR_ACCEPTABLE_MAX:
        return "MODERATE"
    elif cagr >= CAGR_BAD_BELOW:
        return "ACCEPTABLE"
    else:
        return "BAD"

# Maximum Drawdown Rating Thresholds (in %) from env
MAX_DD_EXCELLENT_BELOW = float(os.getenv("MAX_DD_EXCELLENT_BELOW", "10.0"))
MAX_DD_GOOD_MAX = float(os.getenv("MAX_DD_GOOD_MAX", "15.0"))
MAX_DD_ACCEPTABLE_MAX = float(os.getenv("MAX_DD_ACCEPTABLE_MAX", "25.0"))
MAX_DD_RISKY_ABOVE = float(os.getenv("MAX_DD_RISKY_ABOVE", "25.0"))

def classify_max_dd(max_dd_pct: float) -> str:
    """Classifies Maximum Drawdown according to configured percentage thresholds."""
    if max_dd_pct < MAX_DD_EXCELLENT_BELOW:
        return "EXCELLENT"
    elif max_dd_pct <= MAX_DD_GOOD_MAX:
        return "GOOD"
    elif max_dd_pct <= MAX_DD_ACCEPTABLE_MAX:
        return "ACCEPTABLE"
    else:
        return "RISKY"

# Calmar Ratio Rating Thresholds from env
CALMAR_WEAK_BELOW = float(os.getenv("CALMAR_WEAK_BELOW", "1.0"))
CALMAR_ACCEPTABLE_MAX = float(os.getenv("CALMAR_ACCEPTABLE_MAX", "2.0"))
CALMAR_GOOD_MAX = float(os.getenv("CALMAR_GOOD_MAX", "3.0"))
CALMAR_EXCELLENT_MIN = float(os.getenv("CALMAR_EXCELLENT_MIN", "3.0"))

def classify_calmar(calmar: float) -> str:
    """Classifies Calmar Ratio according to configured thresholds."""
    if np.isnan(calmar) or calmar < CALMAR_WEAK_BELOW:
        return "WEAK"
    elif calmar < CALMAR_ACCEPTABLE_MAX:
        return "ACCEPTABLE"
    elif calmar < CALMAR_EXCELLENT_MIN:
        return "GOOD"
    else:
        return "EXCELLENT"

# Sharpe Ratio Rating Thresholds from env
SHARPE_WEAK_BELOW = float(os.getenv("SHARPE_WEAK_BELOW", "1.0"))
SHARPE_ACCEPTABLE_MAX = float(os.getenv("SHARPE_ACCEPTABLE_MAX", "1.5"))
SHARPE_GOOD_MAX = float(os.getenv("SHARPE_GOOD_MAX", "2.0"))
SHARPE_EXCELLENT_MIN = float(os.getenv("SHARPE_EXCELLENT_MIN", "2.0"))

def classify_sharpe(sharpe: float) -> str:
    """Classifies Sharpe Ratio according to configured thresholds."""
    if np.isnan(sharpe) or sharpe < SHARPE_WEAK_BELOW:
        return "WEAK"
    elif sharpe < SHARPE_ACCEPTABLE_MAX:
        return "ACCEPTABLE"
    elif sharpe < SHARPE_EXCELLENT_MIN:
        return "GOOD"
    else:
        return "EXCELLENT"

# Sortino Ratio Rating Thresholds from env
SORTINO_WEAK_BELOW = float(os.getenv("SORTINO_WEAK_BELOW", "1.5"))
SORTINO_ACCEPTABLE_MAX = float(os.getenv("SORTINO_ACCEPTABLE_MAX", "2.0"))
SORTINO_GOOD_MAX = float(os.getenv("SORTINO_GOOD_MAX", "3.0"))
SORTINO_EXCELLENT_MIN = float(os.getenv("SORTINO_EXCELLENT_MIN", "3.0"))

def classify_sortino(sortino: float) -> str:
    """Classifies Sortino Ratio according to configured thresholds."""
    if np.isnan(sortino) or sortino < SORTINO_WEAK_BELOW:
        return "WEAK"
    elif sortino < SORTINO_ACCEPTABLE_MAX:
        return "ACCEPTABLE"
    elif sortino < SORTINO_EXCELLENT_MIN:
        return "GOOD"
    else:
        return "EXCELLENT"

# Profit Factor Rating Thresholds from env
PROFIT_FACTOR_LOSS_BELOW = float(os.getenv("PROFIT_FACTOR_LOSS_BELOW", "1.0"))
PROFIT_FACTOR_WEAK_MAX = float(os.getenv("PROFIT_FACTOR_WEAK_MAX", "1.3"))
PROFIT_FACTOR_ACCEPTABLE_MAX = float(os.getenv("PROFIT_FACTOR_ACCEPTABLE_MAX", "1.5"))
PROFIT_FACTOR_GOOD_MAX = float(os.getenv("PROFIT_FACTOR_GOOD_MAX", "2.0"))
PROFIT_FACTOR_EXCELLENT_MIN = float(os.getenv("PROFIT_FACTOR_EXCELLENT_MIN", "2.0"))

def classify_profit_factor(pf: float) -> str:
    """Classifies Profit Factor according to configured thresholds."""
    if np.isnan(pf) or pf < PROFIT_FACTOR_LOSS_BELOW:
        return "LOSS MAKING"
    elif pf < PROFIT_FACTOR_WEAK_MAX:
        return "WEAK"
    elif pf < PROFIT_FACTOR_ACCEPTABLE_MAX:
        return "ACCEPTABLE"
    elif pf < PROFIT_FACTOR_EXCELLENT_MIN:
        return "GOOD"
    else:
        return "EXCELLENT"

# Expectancy Rating Thresholds (Per Trade in ₹) from env
EXPECTANCY_REJECT_BELOW = float(os.getenv("EXPECTANCY_REJECT_BELOW", "0.0"))
EXPECTANCY_WEAK_MAX = float(os.getenv("EXPECTANCY_WEAK_MAX", "200.0"))
EXPECTANCY_GOOD_MIN = float(os.getenv("EXPECTANCY_GOOD_MIN", "200.0"))
EXPECTANCY_GOOD_MAX = float(os.getenv("EXPECTANCY_GOOD_MAX", "1000.0"))
EXPECTANCY_EXCELLENT_MIN = float(os.getenv("EXPECTANCY_EXCELLENT_MIN", "1000.0"))

def classify_expectancy(exp: float) -> str:
    """Classifies Expectancy according to configured thresholds."""
    if np.isnan(exp) or exp < EXPECTANCY_REJECT_BELOW:
        return "REJECT"
    elif exp <= EXPECTANCY_WEAK_MAX:
        return "WEAK"
    elif exp < EXPECTANCY_EXCELLENT_MIN:
        return "GOOD"
    else:
        return "EXCELLENT"

# Recovery Factor Rating Thresholds from env
RECOVERY_FACTOR_WEAK_BELOW = float(os.getenv("RECOVERY_FACTOR_WEAK_BELOW", "2.0"))
RECOVERY_FACTOR_ACCEPTABLE_MAX = float(os.getenv("RECOVERY_FACTOR_ACCEPTABLE_MAX", "4.0"))
RECOVERY_FACTOR_GOOD_MIN = float(os.getenv("RECOVERY_FACTOR_GOOD_MIN", "4.0"))
RECOVERY_FACTOR_GOOD_MAX = float(os.getenv("RECOVERY_FACTOR_GOOD_MAX", "6.0"))
RECOVERY_FACTOR_EXCELLENT_MIN = float(os.getenv("RECOVERY_FACTOR_EXCELLENT_MIN", "6.0"))

def classify_recovery_factor(rf: float) -> str:
    """Classifies Recovery Factor according to configured thresholds."""
    if np.isnan(rf) or rf < RECOVERY_FACTOR_WEAK_BELOW:
        return "WEAK"
    elif rf < RECOVERY_FACTOR_ACCEPTABLE_MAX:
        return "ACCEPTABLE"
    elif rf < RECOVERY_FACTOR_EXCELLENT_MIN:
        return "GOOD"
    else:
        return "EXCELLENT"

# Consecutive Losing Trades Rating Thresholds from env
CONSEC_LOSSES_EXCELLENT_BELOW = int(os.getenv("CONSEC_LOSSES_EXCELLENT_BELOW", "5"))
CONSEC_LOSSES_GOOD_MAX = int(os.getenv("CONSEC_LOSSES_GOOD_MAX", "10"))
CONSEC_LOSSES_ACCEPTABLE_MAX = int(os.getenv("CONSEC_LOSSES_ACCEPTABLE_MAX", "15"))
CONSEC_LOSSES_UNACCEPTABLE_ABOVE = int(os.getenv("CONSEC_LOSSES_UNACCEPTABLE_ABOVE", "15"))

def classify_consec_losses(losses: int) -> str:
    """Classifies Consecutive Losing Trades according to configured thresholds."""
    if losses < CONSEC_LOSSES_EXCELLENT_BELOW:
        return "EXCELLENT"
    elif losses <= CONSEC_LOSSES_GOOD_MAX:
        return "GOOD"
    elif losses <= CONSEC_LOSSES_ACCEPTABLE_MAX:
        return "ACCEPTABLE"
    else:
        return "UNACCEPTABLE"

# Drawdown Duration Rating Thresholds (in Months) from env
DD_DURATION_EXCELLENT_BELOW = float(os.getenv("DD_DURATION_EXCELLENT_BELOW", "3.0"))
DD_DURATION_GOOD_MAX = float(os.getenv("DD_DURATION_GOOD_MAX", "6.0"))
DD_DURATION_ACCEPTABLE_MAX = float(os.getenv("DD_DURATION_ACCEPTABLE_MAX", "12.0"))
DD_DURATION_WEAK_ABOVE = float(os.getenv("DD_DURATION_WEAK_ABOVE", "12.0"))

def classify_dd_duration(months: float) -> str:
    """Classifies Drawdown Duration according to configured thresholds."""
    if np.isnan(months) or months < DD_DURATION_EXCELLENT_BELOW:
        return "EXCELLENT"
    elif months <= DD_DURATION_GOOD_MAX:
        return "GOOD"
    elif months <= DD_DURATION_ACCEPTABLE_MAX:
        return "ACCEPTABLE"
    else:
        return "WEAK"

# Win Rate Rating Thresholds (in %) from env
WIN_RATE_WEAK_BELOW = float(os.getenv("WIN_RATE_WEAK_BELOW", "30.0"))
WIN_RATE_ACCEPTABLE_MAX = float(os.getenv("WIN_RATE_ACCEPTABLE_MAX", "40.0"))
WIN_RATE_GOOD_MIN = float(os.getenv("WIN_RATE_GOOD_MIN", "40.0"))
WIN_RATE_GOOD_MAX = float(os.getenv("WIN_RATE_GOOD_MAX", "50.0"))
WIN_RATE_EXCELLENT_MIN = float(os.getenv("WIN_RATE_EXCELLENT_MIN", "50.0"))

def classify_win_rate(wr: float) -> str:
    """Classifies Win Rate according to configured thresholds."""
    if np.isnan(wr) or wr < WIN_RATE_WEAK_BELOW:
        return "WEAK"
    elif wr < WIN_RATE_GOOD_MIN:
        return "ACCEPTABLE"
    elif wr < WIN_RATE_EXCELLENT_MIN:
        return "GOOD"
    else:
        return "EXCELLENT"

# Avg Win / Avg Loss Ratio Rating Thresholds from env
WIN_LOSS_RATIO_WEAK_BELOW = float(os.getenv("WIN_LOSS_RATIO_WEAK_BELOW", "1.0"))
WIN_LOSS_RATIO_ACCEPTABLE_MAX = float(os.getenv("WIN_LOSS_RATIO_ACCEPTABLE_MAX", "1.5"))
WIN_LOSS_RATIO_GOOD_MIN = float(os.getenv("WIN_LOSS_RATIO_GOOD_MIN", "1.5"))
WIN_LOSS_RATIO_GOOD_MAX = float(os.getenv("WIN_LOSS_RATIO_GOOD_MAX", "2.5"))
WIN_LOSS_RATIO_EXCELLENT_MIN = float(os.getenv("WIN_LOSS_RATIO_EXCELLENT_MIN", "2.5"))

def classify_win_loss_ratio(ratio: float) -> str:
    """Classifies Avg Win / Avg Loss Ratio according to configured thresholds."""
    if np.isnan(ratio) or ratio < WIN_LOSS_RATIO_WEAK_BELOW:
        return "WEAK"
    elif ratio < WIN_LOSS_RATIO_GOOD_MIN:
        return "ACCEPTABLE"
    elif ratio < WIN_LOSS_RATIO_EXCELLENT_MIN:
        return "GOOD"
    else:
        return "EXCELLENT"

EMA_FAST = 10
EMA_SLOW = 20
ST_PERIOD = 10               # Standard SuperTrend ATR Period
ST_MULTIPLIER = 3.0          # Standard SuperTrend Multiplier
RISK_REWARD_RATIO = 3.0      # 1:3 Target (can change to 4.0 for 1:4)

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
            print(f"  ✗ Warning: Could not fetch chunk {curr_start} -> {curr_end} after {max_retries} retries.")

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
        
        for i in range(1, len(day_df)):
            row = day_df.iloc[i]
            
            # Review 4: Skip warm-up candles to avoid initialization-dependent signals
            if row.get("is_warmup", False):
                continue
            
            # ==================================================================
            # 1. MANAGE EXISTING OPEN POSITION (From prior candles)
            # ==================================================================
            if position is not None:
                # Point 6: End of Day Intraday Square-off at 15:15 IST sharp (Fill at open of 15:15 bar)
                if row["time"] >= dt.time(15, 15):
                    exit_price = row["open"]
                    pnl_pts = (exit_price - position["entry"]) if position["side"] == "LONG" else (position["entry"] - exit_price)
                    pnl_pts -= SLIPPAGE_AND_COSTS_PTS
                    pnl_inr = pnl_pts * position["qty"]
                    current_capital += pnl_inr
                    trades.append({**position, "exit_time": row["timestamp"], "exit": exit_price,
                                   "pnl_pts": pnl_pts, "pnl_inr": pnl_inr, "exit_reason": "EOD_1515"})
                    day_trades_count += 1
                    if pnl_pts <= 0:
                        day_losses_count += 1
                    position = None
                    break
                
                # B. Check SL and Target with Conservative Ambiguity Resolution (Fix 2) & Gap Fills (Fix 5)
                if position["side"] == "LONG":
                    hit_sl = row["low"] <= position["sl"]
                    hit_target = row["high"] >= position["target"]
                    
                    if hit_sl and hit_target:
                        # Ambiguous bar: Conservative assumption - Stop Loss was hit first!
                        # Point 5: Gap-through-stop fill at adverse open price if opened gap down
                        sl_fill = min(row["open"], position["sl"])
                        pnl_pts = (sl_fill - position["entry"]) - SLIPPAGE_AND_COSTS_PTS
                        pnl_inr = pnl_pts * position["qty"]
                        current_capital += pnl_inr
                        exit_reason = "BREAKEVEN" if position["be_triggered"] and sl_fill >= position["entry"] else "SL_AMBIGUOUS"
                        trades.append({**position, "exit_time": row["timestamp"], "exit": sl_fill,
                                       "pnl_pts": pnl_pts, "pnl_inr": pnl_inr, "exit_reason": exit_reason})
                        day_trades_count += 1
                        if pnl_pts <= 0:
                            day_losses_count += 1
                        position = None
                    elif hit_sl:
                        # Point 5: Gap fill
                        sl_fill = min(row["open"], position["sl"])
                        pnl_pts = (sl_fill - position["entry"]) - SLIPPAGE_AND_COSTS_PTS
                        pnl_inr = pnl_pts * position["qty"]
                        current_capital += pnl_inr
                        exit_reason = "BREAKEVEN" if position["be_triggered"] and sl_fill >= position["entry"] else "SL"
                        trades.append({**position, "exit_time": row["timestamp"], "exit": sl_fill,
                                       "pnl_pts": pnl_pts, "pnl_inr": pnl_inr, "exit_reason": exit_reason})
                        day_trades_count += 1
                        if pnl_pts <= 0:
                            day_losses_count += 1
                        position = None
                    elif hit_target:
                        pnl_pts = (position["target"] - position["entry"]) - SLIPPAGE_AND_COSTS_PTS
                        pnl_inr = pnl_pts * position["qty"]
                        current_capital += pnl_inr
                        trades.append({**position, "exit_time": row["timestamp"], "exit": position["target"],
                                       "pnl_pts": pnl_pts, "pnl_inr": pnl_inr, "exit_reason": "TARGET"})
                        day_trades_count += 1
                        position = None
                    else:
                        # Fix 3: Zero-Lookahead Breakeven (Armed strictly for future bars)
                        if ENABLE_BREAKEVEN and not position["be_triggered"]:
                            if row["high"] >= position["entry"] + (BREAKEVEN_R_MULTIPLE * position["risk_pts"]):
                                position["sl"] = max(position["sl"], position["entry"])
                                position["be_triggered"] = True

                elif position["side"] == "SHORT":
                    hit_sl = row["high"] >= position["sl"]
                    hit_target = row["low"] <= position["target"]
                    
                    if hit_sl and hit_target:
                        # Point 5: Gap-through-stop fill at adverse open price if opened gap up
                        sl_fill = max(row["open"], position["sl"])
                        pnl_pts = (position["entry"] - sl_fill) - SLIPPAGE_AND_COSTS_PTS
                        pnl_inr = pnl_pts * position["qty"]
                        current_capital += pnl_inr
                        exit_reason = "BREAKEVEN" if position["be_triggered"] and sl_fill <= position["entry"] else "SL_AMBIGUOUS"
                        trades.append({**position, "exit_time": row["timestamp"], "exit": sl_fill,
                                       "pnl_pts": pnl_pts, "pnl_inr": pnl_inr, "exit_reason": exit_reason})
                        day_trades_count += 1
                        if pnl_pts <= 0:
                            day_losses_count += 1
                        position = None
                    elif hit_sl:
                        # Point 5: Gap fill
                        sl_fill = max(row["open"], position["sl"])
                        pnl_pts = (position["entry"] - sl_fill) - SLIPPAGE_AND_COSTS_PTS
                        pnl_inr = pnl_pts * position["qty"]
                        current_capital += pnl_inr
                        exit_reason = "BREAKEVEN" if position["be_triggered"] and sl_fill <= position["entry"] else "SL"
                        trades.append({**position, "exit_time": row["timestamp"], "exit": sl_fill,
                                       "pnl_pts": pnl_pts, "pnl_inr": pnl_inr, "exit_reason": exit_reason})
                        day_trades_count += 1
                        if pnl_pts <= 0:
                            day_losses_count += 1
                        position = None
                    elif hit_target:
                        pnl_pts = (position["entry"] - position["target"]) - SLIPPAGE_AND_COSTS_PTS
                        pnl_inr = pnl_pts * position["qty"]
                        current_capital += pnl_inr
                        trades.append({**position, "exit_time": row["timestamp"], "exit": position["target"],
                                       "pnl_pts": pnl_pts, "pnl_inr": pnl_inr, "exit_reason": "TARGET"})
                        day_trades_count += 1
                        position = None
                    else:
                        if ENABLE_BREAKEVEN and not position["be_triggered"]:
                            if row["low"] <= position["entry"] - (BREAKEVEN_R_MULTIPLE * position["risk_pts"]):
                                position["sl"] = min(position["sl"], position["entry"])
                                position["be_triggered"] = True

            # Daily circuit breaker & trade caps
            can_trade_today = (day_trades_count < MAX_TRADES_PER_DAY) and (day_losses_count < MAX_LOSSES_PER_DAY)

            # ==================================================================
            # 2. TRIGGER PENDING SETUP ON CURRENT BAR (Fix 1: Immediate Bar Testing & Fix 8: True 1% Sizing)
            # ==================================================================
            if position is None and pending_setup is not None and row["time"] <= dt.time(14, 30) and can_trade_today:
                if pending_setup["side"] == "LONG" and row["high"] > pending_setup["trigger"]:
                    entry_px = max(row["open"], pending_setup["trigger"])
                    risk = entry_px - pending_setup["sl"]
                    if risk > 5:
                        if ENABLE_DYNAMIC_SIZING:
                            risk_budget = current_capital * (RISK_PER_TRADE_PCT / 100.0)
                            risk_per_lot = risk * lot_size
                            affordable_lots = int(risk_budget / risk_per_lot) if risk_per_lot > 0 else 0
                            # Point 8: Skip trade if account cannot afford 1 lot within 1% risk
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
                            "lots": lots, "qty": qty, "be_triggered": False
                        }
                        
                        # Fix 1: Immediately evaluate the entry bar
                        hit_sl_entry = row["low"] <= pos["sl"]
                        hit_tgt_entry = row["high"] >= pos["target"]
                        
                        if hit_sl_entry: # If both or SL hit, SL wins (Conservative)
                            pnl_pts = (pos["sl"] - pos["entry"]) - SLIPPAGE_AND_COSTS_PTS
                            pnl_inr = pnl_pts * pos["qty"]
                            current_capital += pnl_inr
                            trades.append({**pos, "exit_time": row["timestamp"], "exit": pos["sl"],
                                           "pnl_pts": pnl_pts, "pnl_inr": pnl_inr, "exit_reason": "SL_ENTRY_BAR"})
                            day_trades_count += 1
                            if pnl_pts <= 0:
                                day_losses_count += 1
                            position = None
                        elif hit_tgt_entry:
                            pnl_pts = (pos["target"] - pos["entry"]) - SLIPPAGE_AND_COSTS_PTS
                            pnl_inr = pnl_pts * pos["qty"]
                            current_capital += pnl_inr
                            trades.append({**pos, "exit_time": row["timestamp"], "exit": pos["target"],
                                           "pnl_pts": pnl_pts, "pnl_inr": pnl_inr, "exit_reason": "TARGET_ENTRY_BAR"})
                            day_trades_count += 1
                            position = None
                        else:
                            if ENABLE_BREAKEVEN and row["high"] >= pos["entry"] + (BREAKEVEN_R_MULTIPLE * pos["risk_pts"]):
                                pos["sl"] = max(pos["sl"], pos["entry"])
                                pos["be_triggered"] = True
                            position = pos
                            
                    pending_setup = None

                elif pending_setup["side"] == "SHORT" and row["low"] < pending_setup["trigger"]:
                    entry_px = min(row["open"], pending_setup["trigger"])
                    risk = pending_setup["sl"] - entry_px
                    if risk > 5:
                        if ENABLE_DYNAMIC_SIZING:
                            risk_budget = current_capital * (RISK_PER_TRADE_PCT / 100.0)
                            risk_per_lot = risk * lot_size
                            affordable_lots = int(risk_budget / risk_per_lot) if risk_per_lot > 0 else 0
                            # Point 8: Skip trade if account cannot afford 1 lot within 1% risk
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
                            "lots": lots, "qty": qty, "be_triggered": False
                        }
                        
                        # Fix 1: Immediately evaluate the entry bar
                        hit_sl_entry = row["high"] >= pos["sl"]
                        hit_tgt_entry = row["low"] <= pos["target"]
                        
                        if hit_sl_entry:
                            pnl_pts = (pos["entry"] - pos["sl"]) - SLIPPAGE_AND_COSTS_PTS
                            pnl_inr = pnl_pts * pos["qty"]
                            current_capital += pnl_inr
                            trades.append({**pos, "exit_time": row["timestamp"], "exit": pos["sl"],
                                           "pnl_pts": pnl_pts, "pnl_inr": pnl_inr, "exit_reason": "SL_ENTRY_BAR"})
                            day_trades_count += 1
                            if pnl_pts <= 0:
                                day_losses_count += 1
                            position = None
                        elif hit_tgt_entry:
                            pnl_pts = (pos["entry"] - pos["target"]) - SLIPPAGE_AND_COSTS_PTS
                            pnl_inr = pnl_pts * pos["qty"]
                            current_capital += pnl_inr
                            trades.append({**pos, "exit_time": row["timestamp"], "exit": pos["target"],
                                           "pnl_pts": pnl_pts, "pnl_inr": pnl_inr, "exit_reason": "TARGET_ENTRY_BAR"})
                            day_trades_count += 1
                            position = None
                        else:
                            if ENABLE_BREAKEVEN and row["low"] <= pos["entry"] - (BREAKEVEN_R_MULTIPLE * pos["risk_pts"]):
                                pos["sl"] = min(pos["sl"], pos["entry"])
                                pos["be_triggered"] = True
                            position = pos
                            
                    pending_setup = None
            elif not can_trade_today:
                pending_setup = None

            # ==================================================================
            # 3. IDENTIFY NEW SETUP CANDLE (ST05 Alignment)
            # ==================================================================
            if position is None and row["time"] < dt.time(14, 15) and can_trade_today:
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
                
                if long_aligned:
                    pending_setup = {"side": "LONG", "trigger": row["high"], "sl": row["low"]}
                elif short_aligned:
                    pending_setup = {"side": "SHORT", "trigger": row["low"], "sl": row["high"]}

        # Point 7: End of Day Fail-Safe (Ensures positions NEVER silently disappear)
        if position is not None:
            last_row = day_df.iloc[-1]
            exit_price = last_row["close"]
            pnl_pts = (exit_price - position["entry"]) if position["side"] == "LONG" else (position["entry"] - exit_price)
            pnl_pts -= SLIPPAGE_AND_COSTS_PTS
            pnl_inr = pnl_pts * position["qty"]
            current_capital += pnl_inr
            trades.append({**position, "exit_time": last_row["timestamp"], "exit": exit_price,
                           "pnl_pts": pnl_pts, "pnl_inr": pnl_inr, "exit_reason": "EOD_FORCED"})
            day_trades_count += 1
            if pnl_pts <= 0:
                day_losses_count += 1
            position = None

    return pd.DataFrame(trades)

# ==============================================================================
# 5. CALCULATE ALL 12 PERFORMANCE SUMMARY RATIOS (MATCHING SCREENSHOT)
# ==============================================================================
def generate_performance_summary(trades_df: pd.DataFrame, all_dates: pd.Series):
    if trades_df.empty:
        print("No trades generated during the backtest period.")
        return

    start_date = pd.to_datetime(all_dates.min())
    end_date = pd.to_datetime(all_dates.max())
    total_days = all_dates.nunique()
    total_trades = len(trades_df)

    # Point 9: Add Day 0 (t0) baseline before the first trading day with INITIAL_CAPITAL
    sorted_dates = sorted(all_dates.unique())
    daily_pnl = trades_df.groupby("date")["pnl_inr"].sum().reindex(sorted_dates, fill_value=0.0)
    t0_date = pd.to_datetime(sorted_dates[0]) - pd.Timedelta(days=1)
    
    equity_curve = pd.concat([pd.Series([INITIAL_CAPITAL], index=[t0_date]), INITIAL_CAPITAL + daily_pnl.cumsum()])
    daily_returns = equity_curve.pct_change().dropna()

    # 1. CAGR (Annual Return)
    years = max((end_date - start_date).days / 365.25, 0.01)
    ending_capital = equity_curve.iloc[-1]
    cagr = ((ending_capital / INITIAL_CAPITAL) ** (1 / years) - 1) * 100

    # 2. Maximum Drawdown (% and ₹) - Points 9 & 10 (Daily EOD & Trade-by-trade realized)
    running_peak = equity_curve.cummax()
    dd_inr_series = running_peak - equity_curve
    dd_pct_series = (dd_inr_series / running_peak) * 100
    daily_max_dd_pct = dd_pct_series.max()
    daily_max_dd_inr = dd_inr_series.max()

    # Point 10: Trade-by-trade cumulative drawdown
    trade_equity = pd.concat([pd.Series([INITIAL_CAPITAL]), INITIAL_CAPITAL + trades_df["pnl_inr"].cumsum()])
    trade_peak = trade_equity.cummax()
    trade_max_dd_pct = ((trade_peak - trade_equity) / trade_peak).max() * 100
    trade_max_dd_inr = (trade_peak - trade_equity).max()

    # Take the conservative (maximum) of daily and trade-by-trade drawdown
    max_dd_pct = max(daily_max_dd_pct, trade_max_dd_pct)
    max_dd_inr = max(daily_max_dd_inr, trade_max_dd_inr)

    # 3. Calmar Ratio (CAGR / Max Drawdown)
    calmar_ratio = cagr / max_dd_pct if max_dd_pct > 0 else np.nan

    # 4. Sharpe Ratio (Risk Adjusted Return using Std Dev, Rf = 6.5% p.a.)
    rf_daily = 0.065 / 252
    excess_returns = daily_returns - rf_daily
    sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0

    # 5. Sortino Ratio (Risk Adjusted Return using Downside Deviation)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

    # 6. Profit Factor (Gross Profit / Gross Loss)
    winners = trades_df[trades_df["pnl_inr"] > 0]["pnl_inr"]
    losers = trades_df[trades_df["pnl_inr"] <= 0]["pnl_inr"]
    gross_profit = winners.sum()
    gross_loss = abs(losers.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # 7. Expectancy (Per Trade)
    combined_total_pnl = trades_df["pnl_inr"].sum()
    expectancy_per_trade = combined_total_pnl / total_trades

    # 8. Recovery Factor (Net Profit / Max Drawdown ₹)
    recovery_factor = combined_total_pnl / max_dd_inr if max_dd_inr > 0 else np.inf

    # 9. Win Rate (Winning Trades / Total Trades)
    win_rate = (len(winners) / total_trades) * 100

    # 10. Avg Win / Avg Loss Ratio
    avg_win = winners.mean() if len(winners) > 0 else 0.0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 1.0
    avg_win_loss_ratio = avg_win / avg_loss

    # 11. Consecutive Losing Trades (Max Losing Streak)
    is_loss = (trades_df["pnl_inr"] <= 0).astype(int)
    loss_streaks = is_loss.groupby((is_loss != is_loss.shift()).cumsum()).sum()
    max_consecutive_losses = int(loss_streaks.max()) if not loss_streaks.empty else 0

    # 12. Drawdown Duration (Point 11: Reports Max Duration Underwater, with Average alongside)
    in_dd = equity_curve < running_peak
    dd_groups = (~in_dd).cumsum()[in_dd]
    if not dd_groups.empty:
        max_dd_days = dd_groups.value_counts().max()
        avg_dd_days = dd_groups.value_counts().mean()
        max_dd_duration_months = max_dd_days / 21.0 # ~21 trading days per month
        avg_dd_duration_months = avg_dd_days / 21.0
        dd_duration_months = max_dd_duration_months # Primary metric is MAX
    else:
        max_dd_duration_months = 0.0
        avg_dd_duration_months = 0.0
        dd_duration_months = 0.0

    # Footer Aggregations (Best Year, Best Month, Worst Month)
    trades_df["year"] = pd.to_datetime(trades_df["date"]).dt.year
    trades_df["ym"] = pd.to_datetime(trades_df["date"]).dt.strftime("%b %Y")
    yearly_pnl = trades_df.groupby("year")["pnl_inr"].sum()
    monthly_pnl = trades_df.groupby("ym")["pnl_inr"].sum()

    best_year = f"{yearly_pnl.idxmax()} (₹{yearly_pnl.max():,.2f})"
    best_month = f"{monthly_pnl.idxmax()} (₹{monthly_pnl.max():,.2f})"
    worst_month = f"{monthly_pnl.idxmin()} (₹{monthly_pnl.min():,.2f})"
    cagr_rating = classify_cagr(cagr)
    max_dd_rating = classify_max_dd(max_dd_pct)
    calmar_rating = classify_calmar(calmar_ratio)
    sharpe_rating = classify_sharpe(sharpe_ratio)
    sortino_rating = classify_sortino(sortino_ratio)
    pf_rating = classify_profit_factor(profit_factor)
    expectancy_rating = classify_expectancy(expectancy_per_trade)
    rf_rating = classify_recovery_factor(recovery_factor)
    consec_losses_rating = classify_consec_losses(max_consecutive_losses)
    dd_duration_rating = classify_dd_duration(dd_duration_months)
    win_rate_rating = classify_win_rate(win_rate)
    win_loss_rating = classify_win_loss_ratio(avg_win_loss_ratio)

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

    # Print Dashboard Report
    print("\n" + "="*85)
    print(f" BACKTEST PERFORMANCE SUMMARY — NIFTY 15M ST05 STRATEGY")
    print(f" PERIOD: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} | "
          f"TOTAL TRADES: {total_trades} | TOTAL DAYS: {total_days} | CAPITAL: ₹{INITIAL_CAPITAL:,.0f}")
    print("="*85)

    summary_table = [
        ["1. CAGR (Annual Return)", f"{cagr:.2f}% ({cagr_rating})", "2. Maximum Drawdown", f"{max_dd_pct:.2f}% ({max_dd_rating})"],
        ["3. Calmar Ratio", f"{calmar_ratio:.2f} ({calmar_rating})", "4. Sharpe Ratio", f"{sharpe_ratio:.2f} ({sharpe_rating})"],
        ["5. Sortino Ratio", f"{sortino_ratio:.2f} ({sortino_rating})", "6. Profit Factor", f"{profit_factor:.2f} ({pf_rating})"],
        ["7. Expectancy (Per Trade)", f"₹{expectancy_per_trade:,.2f} ({expectancy_rating})", "8. Recovery Factor", f"{recovery_factor:.2f} ({rf_rating})"],
        ["9. Win Rate", f"{win_rate:.2f}% ({win_rate_rating})", "10. Avg Win / Avg Loss", f"{avg_win_loss_ratio:.2f} ({win_loss_rating})"],
        ["11. Consecutive Losing Trades", f"{max_consecutive_losses} ({consec_losses_rating})", "12. Max DD Duration", f"{max_dd_duration_months:.1f}M [Avg: {avg_dd_duration_months:.1f}M] ({dd_duration_rating})"]
    ]
    print(tabulate(summary_table, tablefmt="fancy_grid"))

    exit_counts = trades_df["exit_reason"].value_counts().to_dict() if "exit_reason" in trades_df else {}
    exit_breakdown_str = " | ".join([f"{k}: {v}" for k, v in exit_counts.items()])
    print(f" TRADE EXITS        : {exit_breakdown_str}")
    print(f" COMBINED TOTAL PNL : ₹{combined_total_pnl:,.2f}")
    print(f" BEST YEAR          : {best_year}")
    print(f" BEST MONTH         : {best_month}")
    print(f" WORST MONTH        : {worst_month}")
    print(f" CAGR RATING        : {cagr_rating} ({cagr:.2f}% vs thresholds: <{CAGR_BAD_BELOW:.0f}% Bad, {CAGR_BAD_BELOW:.0f}-{CAGR_ACCEPTABLE_MAX:.0f}% Acceptable, {CAGR_GOOD_MIN:.0f}-{CAGR_EXCELLENT_MIN:.0f}% Good, >{CAGR_EXCELLENT_MIN:.0f}% Excellent)")
    print(f" MAX DD RATING      : {max_dd_rating} ({max_dd_pct:.2f}% vs thresholds: <{MAX_DD_EXCELLENT_BELOW:.0f}% Excellent, {MAX_DD_EXCELLENT_BELOW:.0f}-{MAX_DD_GOOD_MAX:.0f}% Good, {MAX_DD_GOOD_MAX:.0f}-{MAX_DD_ACCEPTABLE_MAX:.0f}% Acceptable, >{MAX_DD_ACCEPTABLE_MAX:.0f}% Risky)")
    print(f" CALMAR RATING      : {calmar_rating} ({calmar_ratio:.2f} vs thresholds: <{CALMAR_WEAK_BELOW:.1f} Weak, {CALMAR_WEAK_BELOW:.1f}-{CALMAR_ACCEPTABLE_MAX:.1f} Acceptable, {CALMAR_ACCEPTABLE_MAX:.1f}-{CALMAR_GOOD_MAX:.1f} Good, >{CALMAR_GOOD_MAX:.1f} Excellent)")
    print(f" SHARPE RATING      : {sharpe_rating} ({sharpe_ratio:.2f} vs thresholds: <{SHARPE_WEAK_BELOW:.1f} Weak, {SHARPE_WEAK_BELOW:.1f}-{SHARPE_ACCEPTABLE_MAX:.1f} Acceptable, {SHARPE_ACCEPTABLE_MAX:.1f}-{SHARPE_EXCELLENT_MIN:.1f} Good, >{SHARPE_EXCELLENT_MIN:.1f} Excellent)")
    print(f" SORTINO RATING     : {sortino_rating} ({sortino_ratio:.2f} vs thresholds: <{SORTINO_WEAK_BELOW:.1f} Weak, {SORTINO_WEAK_BELOW:.1f}-{SORTINO_ACCEPTABLE_MAX:.1f} Acceptable, {SORTINO_ACCEPTABLE_MAX:.1f}-{SORTINO_EXCELLENT_MIN:.1f} Good, >{SORTINO_EXCELLENT_MIN:.1f} Excellent)")
    print(f" PROFIT FACTOR RATING: {pf_rating} ({profit_factor:.2f} vs thresholds: <{PROFIT_FACTOR_LOSS_BELOW:.1f} Loss Making, {PROFIT_FACTOR_LOSS_BELOW:.1f}-{PROFIT_FACTOR_WEAK_MAX:.1f} Weak, {PROFIT_FACTOR_WEAK_MAX:.1f}-{PROFIT_FACTOR_ACCEPTABLE_MAX:.1f} Acceptable, {PROFIT_FACTOR_ACCEPTABLE_MAX:.1f}-{PROFIT_FACTOR_EXCELLENT_MIN:.1f} Good, >{PROFIT_FACTOR_EXCELLENT_MIN:.1f} Excellent)")
    print(f" EXPECTANCY RATING  : {expectancy_rating} (₹{expectancy_per_trade:,.2f} vs thresholds: <₹{EXPECTANCY_REJECT_BELOW:.0f} Reject, ₹{EXPECTANCY_REJECT_BELOW:.0f}-₹{EXPECTANCY_WEAK_MAX:.0f} Weak, ₹{EXPECTANCY_WEAK_MAX:.0f}-₹{EXPECTANCY_EXCELLENT_MIN:.0f} Good, >₹{EXPECTANCY_EXCELLENT_MIN:.0f} Excellent)")
    print(f" RECOVERY FACTOR RATING: {rf_rating} ({recovery_factor:.2f} vs thresholds: <{RECOVERY_FACTOR_WEAK_BELOW:.1f} Weak, {RECOVERY_FACTOR_WEAK_BELOW:.1f}-{RECOVERY_FACTOR_ACCEPTABLE_MAX:.1f} Acceptable, {RECOVERY_FACTOR_GOOD_MIN:.1f}-{RECOVERY_FACTOR_EXCELLENT_MIN:.1f} Good, >{RECOVERY_FACTOR_EXCELLENT_MIN:.1f} Excellent)")
    print(f" CONSEC LOSSES RATING: {consec_losses_rating} ({max_consecutive_losses} vs thresholds: <{CONSEC_LOSSES_EXCELLENT_BELOW} Excellent, {CONSEC_LOSSES_EXCELLENT_BELOW}-{CONSEC_LOSSES_GOOD_MAX} Good, {CONSEC_LOSSES_GOOD_MAX}-{CONSEC_LOSSES_ACCEPTABLE_MAX} Acceptable, >{CONSEC_LOSSES_UNACCEPTABLE_ABOVE} Unacceptable)")
    print(f" DD DURATION RATING : {dd_duration_rating} ({dd_duration_months:.1f}M vs thresholds: <{DD_DURATION_EXCELLENT_BELOW:.0f}M Excellent, {DD_DURATION_EXCELLENT_BELOW:.0f}-{DD_DURATION_GOOD_MAX:.0f}M Good, {DD_DURATION_GOOD_MAX:.0f}-{DD_DURATION_ACCEPTABLE_MAX:.0f}M Acceptable, >{DD_DURATION_WEAK_ABOVE:.0f}M Weak)")
    print(f" WIN RATE RATING    : {win_rate_rating} ({win_rate:.2f}% vs thresholds: <{WIN_RATE_WEAK_BELOW:.0f}% Weak, {WIN_RATE_WEAK_BELOW:.0f}-{WIN_RATE_ACCEPTABLE_MAX:.0f}% Acceptable, {WIN_RATE_GOOD_MIN:.0f}-{WIN_RATE_EXCELLENT_MIN:.0f}% Good, >{WIN_RATE_EXCELLENT_MIN:.0f}% Excellent)")
    print(f" WIN/LOSS RATING    : {win_loss_rating} ({avg_win_loss_ratio:.2f} vs thresholds: <{WIN_LOSS_RATIO_WEAK_BELOW:.1f} Weak, {WIN_LOSS_RATIO_WEAK_BELOW:.1f}-{WIN_LOSS_RATIO_ACCEPTABLE_MAX:.1f} Acceptable, {WIN_LOSS_RATIO_GOOD_MIN:.1f}-{WIN_LOSS_RATIO_EXCELLENT_MIN:.1f} Good, >{WIN_LOSS_RATIO_EXCELLENT_MIN:.1f} Excellent)")
    print("-" * 85)
    print(f" ★ OVERALL TEST OUTCOME : {overall_outcome}")
    print(f" SCORECARD SUMMARY     : {all_ratings.count('EXCELLENT')} Excellent | {all_ratings.count('GOOD')} Good | {all_ratings.count('ACCEPTABLE') + all_ratings.count('MODERATE')} Acceptable | {all_ratings.count('WEAK')} Weak | {all_ratings.count('RISKY') + all_ratings.count('UNACCEPTABLE') + all_ratings.count('LOSS MAKING') + all_ratings.count('REJECT')} Critical")
    print("="*85 + "\n")

if __name__ == "__main__":
    if DHAN_CLIENT_ID == "YOUR_CLIENT_ID" or DHAN_ACCESS_TOKEN == "YOUR_ACCESS_TOKEN":
        print("Warning: Please set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables or update the configuration.")
    dhan_context = DhanContext(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
    dhan_client = dhanhq(dhan_context)
    raw_df = fetch_dhan_intraday(dhan_client, months=BACKTEST_MONTHS)
    ind_df = compute_indicators(raw_df)
    trades = run_st05_backtest(ind_df)
    generate_performance_summary(trades, ind_df["date"])
