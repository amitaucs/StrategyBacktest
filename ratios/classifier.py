"""
Performance metric classification module.

Provides standardized thresholds and ratings for all 12 institutional trading ratios,
dynamically loaded from environment variables (.env) with production-grade defaults.
"""

import os
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# Threshold Configurations
# ==============================================================================

# 1. CAGR Rating Thresholds (in %)
CAGR_BAD_BELOW = float(os.getenv("CAGR_BAD_BELOW", "12.0"))
CAGR_ACCEPTABLE_MAX = float(os.getenv("CAGR_ACCEPTABLE_MAX", "16.0"))
CAGR_GOOD_MIN = float(os.getenv("CAGR_GOOD_MIN", "20.0"))
CAGR_EXCELLENT_MIN = float(os.getenv("CAGR_EXCELLENT_MIN", "30.0"))

# 2. Maximum Drawdown Rating Thresholds (in %)
MAX_DD_EXCELLENT_BELOW = float(os.getenv("MAX_DD_EXCELLENT_BELOW", "10.0"))
MAX_DD_GOOD_MAX = float(os.getenv("MAX_DD_GOOD_MAX", "15.0"))
MAX_DD_ACCEPTABLE_MAX = float(os.getenv("MAX_DD_ACCEPTABLE_MAX", "25.0"))
MAX_DD_RISKY_ABOVE = float(os.getenv("MAX_DD_RISKY_ABOVE", "25.0"))

# 3. Calmar Ratio Rating Thresholds
CALMAR_WEAK_BELOW = float(os.getenv("CALMAR_WEAK_BELOW", "1.0"))
CALMAR_ACCEPTABLE_MAX = float(os.getenv("CALMAR_ACCEPTABLE_MAX", "2.0"))
CALMAR_GOOD_MAX = float(os.getenv("CALMAR_GOOD_MAX", "3.0"))
CALMAR_EXCELLENT_MIN = float(os.getenv("CALMAR_EXCELLENT_MIN", "3.0"))

# 4. Sharpe Ratio Rating Thresholds
SHARPE_WEAK_BELOW = float(os.getenv("SHARPE_WEAK_BELOW", "1.0"))
SHARPE_ACCEPTABLE_MAX = float(os.getenv("SHARPE_ACCEPTABLE_MAX", "1.5"))
SHARPE_GOOD_MAX = float(os.getenv("SHARPE_GOOD_MAX", "2.0"))
SHARPE_EXCELLENT_MIN = float(os.getenv("SHARPE_EXCELLENT_MIN", "2.0"))

# 5. Sortino Ratio Rating Thresholds
SORTINO_WEAK_BELOW = float(os.getenv("SORTINO_WEAK_BELOW", "1.5"))
SORTINO_ACCEPTABLE_MAX = float(os.getenv("SORTINO_ACCEPTABLE_MAX", "2.0"))
SORTINO_GOOD_MAX = float(os.getenv("SORTINO_GOOD_MAX", "3.0"))
SORTINO_EXCELLENT_MIN = float(os.getenv("SORTINO_EXCELLENT_MIN", "3.0"))

# 6. Profit Factor Rating Thresholds
PROFIT_FACTOR_LOSS_BELOW = float(os.getenv("PROFIT_FACTOR_LOSS_BELOW", "1.0"))
PROFIT_FACTOR_WEAK_MAX = float(os.getenv("PROFIT_FACTOR_WEAK_MAX", "1.3"))
PROFIT_FACTOR_ACCEPTABLE_MAX = float(os.getenv("PROFIT_FACTOR_ACCEPTABLE_MAX", "1.5"))
PROFIT_FACTOR_GOOD_MAX = float(os.getenv("PROFIT_FACTOR_GOOD_MAX", "2.0"))
PROFIT_FACTOR_EXCELLENT_MIN = float(os.getenv("PROFIT_FACTOR_EXCELLENT_MIN", "2.0"))

# 7. Expectancy Rating Thresholds (Per Trade in ₹)
EXPECTANCY_REJECT_BELOW = float(os.getenv("EXPECTANCY_REJECT_BELOW", "0.0"))
EXPECTANCY_WEAK_MAX = float(os.getenv("EXPECTANCY_WEAK_MAX", "200.0"))
EXPECTANCY_GOOD_MIN = float(os.getenv("EXPECTANCY_GOOD_MIN", "200.0"))
EXPECTANCY_GOOD_MAX = float(os.getenv("EXPECTANCY_GOOD_MAX", "1000.0"))
EXPECTANCY_EXCELLENT_MIN = float(os.getenv("EXPECTANCY_EXCELLENT_MIN", "1000.0"))

# 8. Recovery Factor Rating Thresholds
RECOVERY_FACTOR_WEAK_BELOW = float(os.getenv("RECOVERY_FACTOR_WEAK_BELOW", "2.0"))
RECOVERY_FACTOR_ACCEPTABLE_MAX = float(os.getenv("RECOVERY_FACTOR_ACCEPTABLE_MAX", "4.0"))
RECOVERY_FACTOR_GOOD_MIN = float(os.getenv("RECOVERY_FACTOR_GOOD_MIN", "4.0"))
RECOVERY_FACTOR_GOOD_MAX = float(os.getenv("RECOVERY_FACTOR_GOOD_MAX", "6.0"))
RECOVERY_FACTOR_EXCELLENT_MIN = float(os.getenv("RECOVERY_FACTOR_EXCELLENT_MIN", "6.0"))

# 9. Consecutive Losing Trades Rating Thresholds
CONSEC_LOSSES_EXCELLENT_BELOW = int(os.getenv("CONSEC_LOSSES_EXCELLENT_BELOW", "5"))
CONSEC_LOSSES_GOOD_MAX = int(os.getenv("CONSEC_LOSSES_GOOD_MAX", "10"))
CONSEC_LOSSES_ACCEPTABLE_MAX = int(os.getenv("CONSEC_LOSSES_ACCEPTABLE_MAX", "15"))
CONSEC_LOSSES_UNACCEPTABLE_ABOVE = int(os.getenv("CONSEC_LOSSES_UNACCEPTABLE_ABOVE", "15"))

# 10. Drawdown Duration Rating Thresholds (in Months)
DD_DURATION_EXCELLENT_BELOW = float(os.getenv("DD_DURATION_EXCELLENT_BELOW", "3.0"))
DD_DURATION_GOOD_MAX = float(os.getenv("DD_DURATION_GOOD_MAX", "6.0"))
DD_DURATION_ACCEPTABLE_MAX = float(os.getenv("DD_DURATION_ACCEPTABLE_MAX", "12.0"))
DD_DURATION_WEAK_ABOVE = float(os.getenv("DD_DURATION_WEAK_ABOVE", "12.0"))

# 11. Win Rate Rating Thresholds (in %)
WIN_RATE_WEAK_BELOW = float(os.getenv("WIN_RATE_WEAK_BELOW", "30.0"))
WIN_RATE_ACCEPTABLE_MAX = float(os.getenv("WIN_RATE_ACCEPTABLE_MAX", "40.0"))
WIN_RATE_GOOD_MIN = float(os.getenv("WIN_RATE_GOOD_MIN", "40.0"))
WIN_RATE_GOOD_MAX = float(os.getenv("WIN_RATE_GOOD_MAX", "50.0"))
WIN_RATE_EXCELLENT_MIN = float(os.getenv("WIN_RATE_EXCELLENT_MIN", "50.0"))

# 12. Avg Win / Avg Loss Ratio Rating Thresholds
WIN_LOSS_RATIO_WEAK_BELOW = float(os.getenv("WIN_LOSS_RATIO_WEAK_BELOW", "1.0"))
WIN_LOSS_RATIO_ACCEPTABLE_MAX = float(os.getenv("WIN_LOSS_RATIO_ACCEPTABLE_MAX", "1.5"))
WIN_LOSS_RATIO_GOOD_MIN = float(os.getenv("WIN_LOSS_RATIO_GOOD_MIN", "1.5"))
WIN_LOSS_RATIO_GOOD_MAX = float(os.getenv("WIN_LOSS_RATIO_GOOD_MAX", "2.5"))
WIN_LOSS_RATIO_EXCELLENT_MIN = float(os.getenv("WIN_LOSS_RATIO_EXCELLENT_MIN", "2.5"))


# ==============================================================================
# Classifier Functions
# ==============================================================================

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


__all__ = [
    "classify_cagr",
    "classify_max_dd",
    "classify_calmar",
    "classify_sharpe",
    "classify_sortino",
    "classify_profit_factor",
    "classify_expectancy",
    "classify_recovery_factor",
    "classify_consec_losses",
    "classify_dd_duration",
    "classify_win_rate",
    "classify_win_loss_ratio",
    # Threshold Constants (for display or reference)
    "CAGR_BAD_BELOW",
    "CAGR_ACCEPTABLE_MAX",
    "CAGR_GOOD_MIN",
    "CAGR_EXCELLENT_MIN",
    "MAX_DD_EXCELLENT_BELOW",
    "MAX_DD_GOOD_MAX",
    "MAX_DD_ACCEPTABLE_MAX",
    "MAX_DD_RISKY_ABOVE",
    "CALMAR_WEAK_BELOW",
    "CALMAR_ACCEPTABLE_MAX",
    "CALMAR_GOOD_MAX",
    "CALMAR_EXCELLENT_MIN",
    "SHARPE_WEAK_BELOW",
    "SHARPE_ACCEPTABLE_MAX",
    "SHARPE_GOOD_MAX",
    "SHARPE_EXCELLENT_MIN",
    "SORTINO_WEAK_BELOW",
    "SORTINO_ACCEPTABLE_MAX",
    "SORTINO_GOOD_MAX",
    "SORTINO_EXCELLENT_MIN",
    "PROFIT_FACTOR_LOSS_BELOW",
    "PROFIT_FACTOR_WEAK_MAX",
    "PROFIT_FACTOR_ACCEPTABLE_MAX",
    "PROFIT_FACTOR_GOOD_MAX",
    "PROFIT_FACTOR_EXCELLENT_MIN",
    "EXPECTANCY_REJECT_BELOW",
    "EXPECTANCY_WEAK_MAX",
    "EXPECTANCY_GOOD_MIN",
    "EXPECTANCY_GOOD_MAX",
    "EXPECTANCY_EXCELLENT_MIN",
    "RECOVERY_FACTOR_WEAK_BELOW",
    "RECOVERY_FACTOR_ACCEPTABLE_MAX",
    "RECOVERY_FACTOR_GOOD_MIN",
    "RECOVERY_FACTOR_GOOD_MAX",
    "RECOVERY_FACTOR_EXCELLENT_MIN",
    "CONSEC_LOSSES_EXCELLENT_BELOW",
    "CONSEC_LOSSES_GOOD_MAX",
    "CONSEC_LOSSES_ACCEPTABLE_MAX",
    "CONSEC_LOSSES_UNACCEPTABLE_ABOVE",
    "DD_DURATION_EXCELLENT_BELOW",
    "DD_DURATION_GOOD_MAX",
    "DD_DURATION_ACCEPTABLE_MAX",
    "DD_DURATION_WEAK_ABOVE",
    "WIN_RATE_WEAK_BELOW",
    "WIN_RATE_ACCEPTABLE_MAX",
    "WIN_RATE_GOOD_MIN",
    "WIN_RATE_GOOD_MAX",
    "WIN_RATE_EXCELLENT_MIN",
    "WIN_LOSS_RATIO_WEAK_BELOW",
    "WIN_LOSS_RATIO_ACCEPTABLE_MAX",
    "WIN_LOSS_RATIO_GOOD_MIN",
    "WIN_LOSS_RATIO_GOOD_MAX",
    "WIN_LOSS_RATIO_EXCELLENT_MIN",
]

