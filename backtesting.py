import pandas as pd

def run_backtest(
    data: pd.DataFrame,
    n_shares: int = 1,
    com: float = 0.00125,
    tp: float = 0.03,   # Take Profit 3%
    sl: float = 0.02    # Stop Loss 2%
):

    df = data.copy()

    # Validaciones
    required = {"close", "buy_signal", "sell_signal"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas necesarias: {required - set(df.columns)}")

    # Asegurar tipos
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["buy_signal"] = df["buy_signal"].astype(bool)
    df["sell_signal"] = df["sell_signal"].astype(bool)
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    # Estado inicial
    cash = 1000000       # capital inicial
    position = 0         # 0 = sin posición, 1 = long, -1 = short
    entry_price = None

    # Historial
    cash_hist, pos_hist, eq_hist, trade_hist = [], [], [], []

    for i, row in df.iterrows():
        price = row["close"]
        trade = ""

        # Si hay posición abierta, revisa TP/SL
        if position != 0 and entry_price is not None:
            # LONG
            if position == 1:
                # Take Profit
                if price >= entry_price * (1 + tp):
                    cash += n_shares * price * (1 - com)
                    position = 0
                    entry_price = None
                    trade = "CLOSE_LONG_TP"
                # Stop Loss
                elif price <= entry_price * (1 - sl):
                    cash += n_shares * price * (1 - com)
                    position = 0
                    entry_price = None
                    trade = "CLOSE_LONG_SL"

            # SHORT
            elif position == -1:
                # Take Profit
                if price <= entry_price * (1 - tp):
                    cash -= n_shares * price * (1 + com)
                    position = 0
                    entry_price = None
                    trade = "CLOSE_SHORT_TP"
                # Stop Loss
                elif price >= entry_price * (1 + sl):
                    cash -= n_shares * price * (1 + com)
                    position = 0
                    entry_price = None
                    trade = "CLOSE_SHORT_SL"

        # Señales contrarias: cerrar y revertir
        if position == 1 and row["sell_signal"]:
            cash += n_shares * price * (1 - com)
            position = 0
            entry_price = None
            trade = "CLOSE_LONG_SIGNAL"

        elif position == -1 and row["buy_signal"]:
            cash -= n_shares * price * (1 + com)
            position = 0
            entry_price = None
            trade = "CLOSE_SHORT_SIGNAL"

        # Abrir nuevas posiciones
        if position == 0:
            if row["buy_signal"]:
                cash -= n_shares * price * (1 + com)
                position = 1
                entry_price = price
                trade = "BUY"

            elif row["sell_signal"]:
                cash += n_shares * price * (1 - com)
                position = -1
                entry_price = price
                trade = "SELL"

        # Calcular equity (valor total del portafolio)
        if position == 1:
            equity = cash + n_shares * price
        elif position == -1:
            equity = cash - n_shares * price
        else:
            equity = cash

        cash_hist.append(cash)
        pos_hist.append(position)
        eq_hist.append(equity)
        trade_hist.append(trade)

    # Agregar resultados al DataFrame
    df["cash"] = cash_hist
    df["position"] = pos_hist
    df["equity"] = eq_hist
    df["trade"] = trade_hist
    df["returns"] = df["equity"].pct_change().fillna(0)

    return df
