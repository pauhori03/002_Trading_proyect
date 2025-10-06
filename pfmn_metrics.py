import numpy as np
import pandas as pd

# Métricas de performance: Max Drawdown, CAGR, Sharpe, Sortino, Calmar, Win Rate
def max_drawdown(equity: pd.Series):
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    return dd.min(), dd

def cagr(equity: pd.Series, bars_per_year: int):
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 2:
        return np.nan
    years = len(eq) / bars_per_year
    if years <= 0:
        return np.nan
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    return (1.0 + total_ret) ** (1.0 / years) - 1.0

def sharpe_ratio(returns: pd.Series, bars_per_year: int, rf: float = 0.0):
    r = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.std(ddof=0) == 0 or len(r) == 0:
        return np.nan
    # Convertimos rf anual a por-barra
    rf_bar = rf / bars_per_year
    excess = r - rf_bar
    return np.sqrt(bars_per_year) * excess.mean() / excess.std(ddof=0)

def sortino_ratio(returns: pd.Series, bars_per_year: int, rf: float = 0.0):
    r = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return np.nan
    rf_bar = rf / bars_per_year
    excess = r - rf_bar
    downside = excess[excess < 0]
    if downside.std(ddof=0) == 0 or len(downside) == 0:
        return np.nan
    return np.sqrt(bars_per_year) * excess.mean() / downside.std(ddof=0)

def calmar_ratio(equity: pd.Series, bars_per_year: int):
    cg = cagr(equity, bars_per_year)
    mdd, _ = max_drawdown(equity)
    denom = abs(mdd)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return cg / denom

def win_rate(trade_col: pd.Series):
    t = trade_col.astype(str).fillna("")
    wins = t.str.contains("CLOSE_.*_TP").sum()
    loss = t.str.contains("CLOSE_.*_SL").sum()
    total = wins + loss
    return (wins / total) if total > 0 else np.nan
