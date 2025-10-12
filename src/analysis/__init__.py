# src/analysis/__init__.py
"""
Módulo de análisis para AccurateMetrics
Incluye herramientas para análisis de impacto causal.
Expone funciones de estandarización y resumen, y una clase de compatibilidad
con el nombre histórico `CausalImpactAnalyzer`.
"""

from __future__ import annotations

from typing import Any, Dict
import pandas as pd

# Importa las funciones reales del módulo causal_impact
from .causal_impact import (
    standardize_causal_columns,
    compute_effect_summary,
)

__all__ = [
    "standardize_causal_columns",
    "compute_effect_summary",
    "CausalImpactAnalyzer",
]


class CausalImpactAnalyzer:
    """
    Clase *de compatibilidad* para no romper código existente que esperaba
    `CausalImpactAnalyzer` en `src.analysis`.

    Uso típico:
        cia = CausalImpactAnalyzer(df)
        df_std = cia.standardize()
        summary = cia.summary(intervention_date, alpha=0.05)
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                raise TypeError(f"Los datos deben ser un DataFrame o convertible a uno: {e}")
        self.df_raw = df

    def standardize(self) -> pd.DataFrame:
        """Devuelve el DataFrame estandarizado con las columnas canónicas."""
        return standardize_causal_columns(self.df_raw)

    def summary(self, intervention_date: Any, alpha: float = 0.05) -> Dict[str, float]:
        """
        Calcula el resumen de efectos SOLO en el periodo POST a la fecha de intervención.
        Devuelve: dict con avg_effect, cum_effect, pct_change_mean e is_significant.
        """
        df_std = self.standardize()
        return compute_effect_summary(df_std, pd.to_datetime(intervention_date), alpha=alpha)
