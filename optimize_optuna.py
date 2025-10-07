import numpy as np
import pandas as pd
import optuna

from backtesting import run_backtest  # o: from backtest import run_backtest

# ===================== MÉTRICAS =====================
def max_drawdown(equity: pd.Series) -> float:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 2: return np.nan
    roll = eq.cummax()
    dd = eq/roll - 1.0
    return float(dd.min())

def cagr(equity: pd.Series, bars_per_year: int) -> float:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 2: return np.nan
    years = len(eq)/bars_per_year
    if years <= 0: return np.nan
    total = eq.iloc[-1]/eq.iloc[0] - 1.0
    return (1.0 + total)**(1.0/years) - 1.0

def calmar_ratio(equity: pd.Series, bars_per_year: int) -> float:
    cg = cagr(equity, bars_per_year)
    mdd = max_drawdown(equity)
    den = abs(mdd)
    if np.isnan(cg) or np.isnan(den) or den == 0: return -np.inf
    return cg/den

# ===================== INDICADORES & SEÑALES =====================
def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean().replace(0, np.nan)
    rs = avg_gain/avg_loss
    return 100 - (100/(1+rs))

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _bollinger(series: pd.Series, period: int, std_mult: float):
    mid = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    up = mid + std_mult*std
    lo = mid - std_mult*std
    return mid, up, lo

def build_signals(df: pd.DataFrame,
                  rsi_period: int, rsi_buy: int, rsi_sell: int,
                  ema_period: int, bb_period: int, bb_std: float) -> pd.DataFrame:
    """Señales 2-de-3 (RSI/EMA/Bollinger). Devuelve df con buy_signal/sell_signal."""
    data = df.copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["close"]).reset_index(drop=True)

    rsi_vals = _rsi(data["close"], rsi_period)
    ema_vals = _ema(data["close"], ema_period)
    _, bb_up, bb_low = _bollinger(data["close"], bb_period, bb_std)

    rsi_buy_sig  = (rsi_vals <= rsi_buy)
    rsi_sell_sig = (rsi_vals >= rsi_sell)
    ema_buy_sig  = (data["close"] > ema_vals)
    ema_sell_sig = (data["close"] < ema_vals)
    bb_buy_sig   = (data["close"] <= bb_low)
    bb_sell_sig  = (data["close"] >= bb_up)

    buys  = rsi_buy_sig.fillna(False).astype(int) + ema_buy_sig.fillna(False).astype(int) + bb_buy_sig.fillna(False).astype(int)
    sells = rsi_sell_sig.fillna(False).astype(int) + ema_sell_sig.fillna(False).astype(int) + bb_sell_sig.fillna(False).astype(int)

    data["buy_signal"]  = buys  >= 2
    data["sell_signal"] = sells >= 2

    warm = max(rsi_period, ema_period, bb_period)
    if warm > 0 and len(data) > warm:
        data = data.iloc[warm:].reset_index(drop=True)
    return data

# ===================== SPLIT 60 / 20 / 20 =====================
def split_60_20_20(df: pd.DataFrame):
    n = len(df)
    i1 = int(0.60*n)
    i2 = int(0.80*n)
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()

# ===================== OBJETIVO (TRAIN; TEST para pruning) =====================
def make_objective_60_20_20(train_df, test_df, bars_per_year=8760):
    def objective(trial):
        # --- rangos algo más amplios ---
        sl  = trial.suggest_float("sl", 0.01, 0.06)
        tp  = trial.suggest_float("tp", 0.03, 0.20)
        n_shares = trial.suggest_float("n_shares", 0.5, 2.0)

        rsi_period = trial.suggest_int("rsi_period", 8, 30)
        rsi_buy    = trial.suggest_int("rsi_buy",   20, 45)
        rsi_sell   = trial.suggest_int("rsi_sell",  60, 90)
        if rsi_sell <= rsi_buy: rsi_sell = rsi_buy + 5
        ema_period = trial.suggest_int("ema_period", 8, 55)
        bb_period  = trial.suggest_int("bb_period",  14, 30)
        bb_std     = trial.suggest_float("bb_std",   1.5, 3.0)

        # --- TRAIN ---
        sig_tr = build_signals(train_df, rsi_period, rsi_buy, rsi_sell, ema_period, bb_period, bb_std)
        if len(sig_tr) < 10:
            return -1e9  # penaliza, no prune inmediato

        bt_tr = run_backtest(sig_tr, n_shares=n_shares, com=0.00125, tp=tp, sl=sl)
        score_tr = calmar_ratio(bt_tr["equity"], bars_per_year)
        if not np.isfinite(score_tr):
            return -1e9

        # --- TEST ---
        sig_te = build_signals(test_df, rsi_period, rsi_buy, rsi_sell, ema_period, bb_period, bb_std)
        if len(sig_te) >= 10:
            bt_te = run_backtest(sig_te, n_shares=n_shares, com=0.00125, tp=tp, sl=sl)
            score_te = calmar_ratio(bt_te["equity"], bars_per_year)
            if not np.isfinite(score_te):
                score_te = -1e9
            trial.report(score_te, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(score_tr)
    return objective