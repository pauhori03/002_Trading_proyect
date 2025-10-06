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
    """Recibe DataFrame con columna 'close' y devuelve señales de compra/venta"""
    data = df.copy()

    # Calcular indicadores
    data['RSI'] = rsi(data['close'])
    data['EMA'] = ema(data['close'])
    data['BB_mid'], data['BB_up'], data['BB_low'] = bollinger(data['close'])

    # Condiciones de compra (BUY) por indicador
    rsi_buy = data['RSI'] < 30
    ema_buy = data['close'] > data['EMA']
    bb_buy = data['close'] < data['BB_low']

    # Condiciones de venta (SELL) por indicador
    rsi_sell = data['RSI'] > 70
    ema_sell = data['close'] < data['EMA']
    bb_sell = data['close'] > data['BB_up']

    # Regla 2 de 3 → si 2 o más indicadores coinciden
    data['buy_signal'] = (rsi_buy + ema_buy + bb_buy) >= 2
    data['sell_signal'] = (rsi_sell + ema_sell + bb_sell) >= 2

    # Señal compacta (+1 compra, -1 venta, 0 nada)
    data['signal'] = 0
    data.loc[data['buy_signal'], 'signal'] = 1
    data.loc[data['sell_signal'], 'signal'] = -1

    return data


