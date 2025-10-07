import pandas as pd
import numpy as np
from signals import make_signals
from backtesting import run_backtest
import optuna
from optimize_optuna import build_signals, split_60_20_20, make_objective_60_20_20
from pfmn_metrics import (
    max_drawdown, cagr, sharpe_ratio, sortino_ratio, calmar_ratio, win_rate
    )
from plots import equity_curve, save_returns_tables

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
    n_shares=1,   
    sl=0.02,      # Stop Loss 1%
    tp=0.03       # Take Profit 2%
)

print(result[["close", "buy_signal", "sell_signal", "equity"]].head(20))

final_value = result["equity"].iloc[-1]
print(f"\nValor final del portafolio: {final_value:,.2f}")

# Performance metrics
BPY = 8760  # barras por año para timeframe 1H
rf_annual = 0.0  # tasa libre de riesgo anual (déjala en 0 si no quieres ajustar)

mdd, dd_series = max_drawdown(result["equity"])
sr   = sharpe_ratio(result["returns"], BPY, rf=rf_annual)
so   = sortino_ratio(result["returns"], BPY, rf=rf_annual)
cg   = cagr(result["equity"], BPY)
calm = calmar_ratio(result["equity"], BPY)
wr   = win_rate(result["trade"])

print("\n=== MÉTRICAS ===")
print(f"Sharpe Ratio:    {sr:.3f}" if not np.isnan(sr) else "Sharpe Ratio:    n/a")
print(f"Sortino Ratio:   {so:.3f}" if not np.isnan(so) else "Sortino Ratio:   n/a")
print(f"Calmar Ratio:    {calm:.3f}" if not np.isnan(calm) else "Calmar Ratio:    n/a")
print(f"Maximum DD:      {mdd:.2%}")
print(f"CAGR (aprox):    {cg:.2%}" if not np.isnan(cg) else "CAGR (aprox):    n/a")
print(f"Win Rate (TP/SL):{wr:.2%}" if not np.isnan(wr) else "Win Rate (TP/SL): n/a")

# Optimization with Optuna
train_df, test_df, val_df = split_60_20_20(df)

# 3) Optuna (>= 50 trials)
objective = make_objective_60_20_20(train_df, test_df, bars_per_year=8760)
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    study_name="btc_1h_calmar_60_20_20"
    )
study.optimize(objective, n_trials=100, n_jobs=1)

# 4) Best Trial (el “best study” que quieres ver)
print("\n=== BEST TRIAL (Train Calmar) ===")
print("Value:", study.best_value)
print("Params:")

for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# 5) Evaluación rápida en Test y Validation con best_params
best = study.best_params

def _eval_on(block: pd.DataFrame, name: str):
        sig = build_signals(
            block,
            rsi_period=int(best["rsi_period"]),
            rsi_buy=int(best["rsi_buy"]),
            rsi_sell=int(best["rsi_sell"]) if int(best["rsi_sell"]) > int(best["rsi_buy"]) else int(best["rsi_buy"]) + 5,
            ema_period=int(best["ema_period"]),
            bb_period=int(best["bb_period"]),
            bb_std=float(best["bb_std"])
        )
        bt = run_backtest(
            sig,
            n_shares=float(best.get("n_shares", 1)),
            com=0.00125,
            tp=float(best.get("tp", 0.03)),
            sl=float(best.get("sl", 0.02))
        )
        eq = bt["equity"]
        calm = calmar_ratio(eq, 8760)
        print(f"\n[{name}]")
        print(f"  Final equity: {eq.iloc[-1]:,.2f}")
        print(f"  Calmar:       {calm:.3f}")
        return bt

_eval_on(test_df, "TEST (20%)")
_eval_on(val_df,  "VALIDATION (20%)")



# Backtest final con los mejores params
print("\n=== EVALUACIÓN FINAL CON PARÁMETROS ÓPTIMOS ===")
best = study.best_params

# Generar señales con parámetros óptimos
optimized_signals = build_signals(
    df,
    rsi_period=int(best["rsi_period"]),
    rsi_buy=int(best["rsi_buy"]),
    rsi_sell=int(best["rsi_sell"]) if int(best["rsi_sell"]) > int(best["rsi_buy"]) else int(best["rsi_buy"]) + 5,
    ema_period=int(best["ema_period"]),
    bb_period=int(best["bb_period"]),
    bb_std=float(best["bb_std"])
)

# Ejecutar backtest con parámetros óptimos
optimized_result = run_backtest(
    optimized_signals,
    n_shares=float(best.get("n_shares", 1)),
    com=0.00125,
    tp=float(best.get("tp", 0.03)),
    sl=float(best.get("sl", 0.02))
)

print(f"Valor final del portafolio: {eq.iloc[-1]:,.2f}")
