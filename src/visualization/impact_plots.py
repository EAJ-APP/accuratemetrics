# src/visualization/impact_plots.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Iterable


# --- Mapeo de columnas esperadas ---
COLUMN_MAPPING: Dict[str, Iterable] = {
    "actual": ["actual", "response", "y", 0],
    "predicted": ["predicted", "preds", "point_pred", "point_prediction", 1],
    "predicted_lower": ["predicted_lower", "preds_lower", "point_pred_lower", 2],
    "predicted_upper": ["predicted_upper", "preds_upper", "point_pred_upper", 3],
    "point_effect": ["point_effect", "effect", "point_effects"],
    "cumulative_effect": ["cumulative_effect", "cum_effect", "post_cum_effect"],
}


def _get_column(df: pd.DataFrame, key: str, required: bool = True) -> Optional[pd.Series]:
    """
    Devuelve la serie para la clave 'key' según COLUMN_MAPPING.
    - Si 'required' es True y no se encuentra, lanza ValueError.
    - Si 'required' es False y no se encuentra, devuelve None.
    """
    candidates = COLUMN_MAPPING.get(key, [])
    for col_name in candidates:
        if isinstance(col_name, int):
            if col_name < len(df.columns):
                return df.iloc[:, col_name]
        elif col_name in df.columns:
            return df[col_name]

    if required:
        raise ValueError(
            f"No se encontró la columna para '{key}'. "
            f"Columnas disponibles: {df.columns.tolist()}. "
            f"Añade un alias válido en COLUMN_MAPPING o normaliza antes."
        )
    return None


def _padded_limits(*series: Iterable[pd.Series], pad_ratio: float = 0.05) -> Optional[Tuple[float, float]]:
    """
    Calcula límites ymin, ymax con padding para ajustar el eje Y a los datos.
    Si no hay datos válidos, devuelve None y deja que Matplotlib autoescale.
    """
    vals_list = []
    for s in series:
        if s is None:
            continue
        s = pd.Series(s).dropna()
        if len(s):
            vals_list.append(s)

    if not vals_list:
        return None

    vals = pd.concat(vals_list, axis=0)
    vmin, vmax = float(vals.min()), float(vals.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None

    if vmin == vmax:
        delta = 1.0 if vmax == 0 else abs(vmax) * 0.1
        return (vmin - delta, vmax + delta)

    pad = (vmax - vmin) * pad_ratio
    return (vmin - pad, vmax + pad)


def plot_observed_vs_predicted(df: pd.DataFrame, intervention_date: Optional[pd.Timestamp] = None) -> plt.Figure:
    """Gráfico Observado vs Predicho con bandas opcionales."""
    actual = _get_column(df, "actual", required=True)
    predicted = _get_column(df, "predicted", required=True)
    lower = _get_column(df, "predicted_lower", required=False)
    upper = _get_column(df, "predicted_upper", required=False)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(df.index, actual, label="Observado", linewidth=2)
    ax.plot(df.index, predicted, label="Predicho", linewidth=2, linestyle="--")

    if lower is not None and upper is not None:
        ax.fill_between(df.index, lower, upper, alpha=0.15, label="IC Predicho")

    if intervention_date is not None:
        ax.axvline(pd.to_datetime(intervention_date), linestyle=":", linewidth=1.5, label="Intervención")

    limits = _padded_limits(actual, predicted, lower, upper)
    if limits:
        ax.set_ylim(*limits)
    ax.margins(x=0.02)
    ax.set_title("Observado vs Predicho")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_point_effect(df: pd.DataFrame, intervention_date: Optional[pd.Timestamp] = None) -> plt.Figure:
    """Gráfico del Efecto Puntual."""
    point_eff = _get_column(df, "point_effect", required=True)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(df.index, point_eff, label="Efecto puntual", linewidth=2)

    if intervention_date is not None:
        ax.axvline(pd.to_datetime(intervention_date), linestyle=":", linewidth=1.5, label="Intervención")

    limits = _padded_limits(point_eff)
    if limits:
        ax.set_ylim(*limits)
    ax.margins(x=0.02)
    ax.set_title("Efecto puntual")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Δ (observado - predicho)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cumulative_effect(df: pd.DataFrame, intervention_date: Optional[pd.Timestamp] = None) -> plt.Figure:
    """Gráfico del Efecto Acumulado."""
    cum_eff = _get_column(df, "cumulative_effect", required=True)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(df.index, cum_eff, label="Efecto acumulado", linewidth=2)

    if intervention_date is not None:
        ax.axvline(pd.to_datetime(intervention_date), linestyle=":", linewidth=1.5, label="Intervención")

    limits = _padded_limits(cum_eff)
    if limits:
        ax.set_ylim(*limits)
    ax.margins(x=0.02)
    ax.set_title("Efecto acumulado")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Σ Δ")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_impact_dashboard(df: pd.DataFrame, intervention_date: Optional[pd.Timestamp] = None) -> Dict[str, plt.Figure]:
    """
    Devuelve las 3 figuras principales para el dashboard.
    """
    figs = {
        "observado_vs_predicho": plot_observed_vs_predicted(df, intervention_date),
        "efecto_puntual": plot_point_effect(df, intervention_date),
        "efecto_acumulado": plot_cumulative_effect(df, intervention_date),
    }
    return figs
