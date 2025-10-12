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
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                raise TypeError(f"Los datos deben ser un DataFrame o convertible a uno: {e}")
        self.df_raw = df

    def standardize(self) -> pd.DataFrame:
        return standardize_causal_columns(self.df_raw)

    def summary(self, intervention_date: Any, alpha: float = 0.05) -> Dict[str, float]:
        df_std = self.standardize()
        return compute_effect_summary(df_std, pd.to_datetime(intervention_date), alpha=alpha)
