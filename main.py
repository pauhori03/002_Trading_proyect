import pandas as pd
from signals import make_signals
from backtesting import run_backtest

# Import dataset
df = pd.read_csv("Data/Binance_BTCUSDT_1h.csv")
df = df.iloc[::-1].reset_index(drop=True)
df.columns = df.columns.str.strip().str.lower()

# Generate buy/sell signals
df_signals = make_signals(df)
print(df_signals.head(20))

# Backtesting
result = run_backtest(
    df_signals,
    n_shares=2,   # cantidad de unidades
    sl=0.01,      # Stop Loss 1%
    tp=0.02       # Take Profit 2%
)

print(result[["close", "buy_signal", "sell_signal", "equity"]].head(20))

final_value = result["equity"].iloc[-1]
print(f"\nValor final del portafolio: {final_value:,.2f}")