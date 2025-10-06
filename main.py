import pandas as pd
from signals import make_signals

# Import dataset
df = pd.read_csv("Data/Binance_BTCUSDT_1h.csv")
df.columns = df.columns.str.strip().str.lower()

# Generate buy/sell signals
df_signals = make_signals(df)
print(df_signals.head(20))
