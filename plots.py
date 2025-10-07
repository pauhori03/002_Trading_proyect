import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from signals import make_signals
from backtesting import run_backtest


# === FUNCIONES AUXILIARES ===

def equity_curve(curves, title="Equity Curve", save_path="reports/equity_curve.png", show=False):
    """Grafica curvas de equity y guarda la imagen."""
    def _ensure_series(eq):
        if isinstance(eq, pd.DataFrame):
            eq = eq.iloc[:, 0]
        s = eq.dropna().copy()
        if isinstance(s.index, pd.DatetimeIndex) and not s.index.is_monotonic_increasing:
            s = s.sort_index()
        return s.astype(float)

    plt.figure(figsize=(11, 5))
    if isinstance(curves, pd.Series):
        s = _ensure_series(curves)
        plt.plot(s.index, s.values, label="Equity")
    else:
        for name, series in curves.items():
            s = _ensure_series(series)
            plt.plot(s.index, s.values, label=name)

    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Valor del Portafolio")
    plt.legend()
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def _period_returns(eq: pd.Series, freq: str = "M") -> pd.Series:
    """Calcula retornos porcentuales por periodo."""
    eq = eq.copy()
    if not isinstance(eq.index, pd.DatetimeIndex):
        raise ValueError("La Serie de equity debe tener índice de fechas (DatetimeIndex).")
    px = eq.resample(freq).last().dropna()
    return px.pct_change().dropna()


def save_returns_tables(eq: pd.Series, out_dir="reports/"):
    """Genera y guarda tablas de retornos mensuales, trimestrales y anuales."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Mensual
    r_m = _period_returns(eq, "M")
    m = r_m.to_frame("ret")
    m["Year"] = m.index.year
    m["Month"] = m.index.month
    monthly = m.pivot(index="Year", columns="Month", values="ret").sort_index()
    monthly.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(monthly.columns)]
    monthly.to_csv(Path(out_dir) / "returns_monthly.csv")

    # Trimestral
    r_q = _period_returns(eq, "Q")
    q = r_q.to_frame("ret")
    q["Year"] = q.index.year
    q["Quarter"] = q.index.quarter
    quarterly = q.pivot(index="Year", columns="Quarter", values="ret").sort_index()
    quarterly.columns = [f"Q{c}" for c in quarterly.columns]
    quarterly.to_csv(Path(out_dir) / "returns_quarterly.csv")

    # Anual
    annual = _period_returns(eq, "Y").to_frame("ret")
    annual.index.name = "YearEnd"
    annual.to_csv(Path(out_dir) / "returns_annual.csv")

    return {"monthly": monthly, "quarterly": quarterly, "annual": annual}