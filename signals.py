import pandas as pd
import numpy as np

# FUNCIONES DE INDICADORES

# Calcula RSI (Relative Strength Index)
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calcula media móvil exponencial (EMA)
def ema(series, span=50):
    return series.ewm(span=span, adjust=False).mean()

# Calcula bandas de Bollinger
def bollinger(series, period=20, std_mult=2):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return mid, upper, lower


# FUNCIÓN PRINCIPAL

def make_signals(df):
    """Devuelve el dataset original + columnas buy_signal y sell_signal."""
    data = df.copy()

    # Calcular los tres indicadores
    rsi_values = rsi(data['close'])
    ema_values = ema(data['close'])
    bb_up, bb_low = bollinger(data['close'])

    # Señales de compra por indicador
    rsi_buy = rsi_values < 30
    ema_buy = data['close'] > ema_values
    bb_buy = data['close'] < bb_low

    # Señales de venta por indicador
    rsi_sell = rsi_values > 70
    ema_sell = data['close'] < ema_values
    bb_sell = data['close'] > bb_up

    # Confirmación 2 de 3
    data['buy_signal'] = (rsi_buy + ema_buy + bb_buy) >= 2
    data['sell_signal'] = (rsi_sell + ema_sell + bb_sell) >= 2

    # Devuelve solo las dos columnas de señales
    return data[['buy_signal', 'sell_signal']]

