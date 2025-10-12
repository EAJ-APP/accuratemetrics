# src/visualization/__init__.py
"""
Módulo de visualización para AccurateMetrics.
Incluye herramientas para graficar resultados de análisis y una clase de compatibilidad
`ImpactVisualizer` para mantener código histórico funcionando.
"""

from __future__ import annotations

from typing import Optional, Dict
import pandas as pd

from .impact_plots import (
    plot_observed_vs_predicted,
    plot_point_effect,
    plot_cumulative_effect,
    plot_impact_dashboard,
)

__all__ = [
    "plot_observed_vs_predicted",
    "plot_point_effect",
    "plot_cumulative_effect",
    "plot_impact_dashboard",
    "ImpactVisualizer",
]


class ImpactVisualizer:
    """
    Clase *de compatibilidad* para no romper código que importaba
    `ImpactVisualizer` desde `src.visualization`.

    Uso típico:
        vis = ImpactVisualizer(df, intervention_date)
        fig1 = vis.observed_vs_predicted()
        fig2 = vis.point_effect()
        fig3 = vis.cumulative_effect()
        figs = vis.dashboard()
    """

    def __init__(self, df: pd.DataFrame, intervention_date: Optional[pd.Timestamp] = None):
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                raise TypeError(f"Los datos deben ser un DataFrame o convertible a uno: {e}")
        self.df = df
        self.intervention_date = pd.to_datetime(intervention_date) if intervention_date is not None else None

    def observed_vs_predicted(self):
        return plot_observed_vs_predicted(self.df, self.intervention_date)

    def point_effect(self):
        return plot_point_effect(self.df, self.intervention_date)

    def cumulative_effect(self):
        return plot_cumulative_effect(self.df, self.intervention_date)

    def dashboard(self) -> Dict[str, "matplotlib.figure.Figure"]:
        return plot_impact_dashboard(self.df, self.intervention_date)
