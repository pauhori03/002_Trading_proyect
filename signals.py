import pandas as pd
import numpy as np

# --------------------------
# INDICADORES
# --------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder con warm-up controlado
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    # Evita división por cero
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals

def ema(series: pd.Series, span: int = 20) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    up = mid + std_mult * std
    low = mid - std_mult * std
    return mid, up, low

# --------------------------
# SEÑALES
# --------------------------
def make_signals(df: pd.DataFrame) -> pd.DataFrame:
   
    data = df.copy()

    # 0) Validación y limpieza mínima
    if "close" not in data.columns:
        raise KeyError("Se requiere columna 'close'.")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")

    # 1) Indicadores (internos, no se devuelven)
    rsi_vals = rsi(data["close"], period=14)
    ema_vals = ema(data["close"], span=50)
    _, bb_up, bb_low = bollinger(data["close"], period=20, std_mult=2.0)

    # 2) Condiciones por indicador (booleanas)
    rsi_buy  = (rsi_vals <= 30)
    rsi_sell = (rsi_vals >= 70)

    ema_buy  = (data["close"] > ema_vals)
    ema_sell = (data["close"] < ema_vals)

    bb_buy   = (data["close"] <= bb_low)
    bb_sell  = (data["close"] >= bb_up)

    # Confirmación 2 de 3 indicadores
    buy_votes  = (
        rsi_buy.fillna(False).astype(int) +
        ema_buy.fillna(False).astype(int) +
        bb_buy.fillna(False).astype(int)
    )
    sell_votes = (
        rsi_sell.fillna(False).astype(int) +
        ema_sell.fillna(False).astype(int) +
        bb_sell.fillna(False).astype(int)
    )

    buy_signal  = (buy_votes  >= 2)
    sell_signal = (sell_votes >= 2)

    # Resultado: dataset original + señales (booleanas limpias)
    result = data.copy()
    result["buy_signal"]  = buy_signal.astype(bool)
    result["sell_signal"] = sell_signal.astype(bool)

    return result
