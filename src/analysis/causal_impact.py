# src/analysis/causal_impact.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional


COLUMN_MAPPING: Dict[str, Iterable] = {
    "actual": ["actual", "response", "y", 0],
    "predicted": ["predicted", "preds", "point_pred", "point_prediction", 1],
    "predicted_lower": ["predicted_lower", "preds_lower", "point_pred_lower", 2],
    "predicted_upper": ["predicted_upper", "preds_upper", "point_pred_upper", 3],
    "point_effect": ["point_effect", "effect", "point_effects"],
    "cumulative_effect": ["cumulative_effect", "cum_effect", "post_cum_effect"],
    "p_value": ["p_value", "p", "pval"],
}


def _pick_column(df: pd.DataFrame, key: str, required: bool = True) -> Optional[pd.Series]:
    candidates = COLUMN_MAPPING.get(key, [])
    for col_name in candidates:
        if isinstance(col_name, int):
            if col_name < len(df.columns):
                return df.iloc[:, col_name]
        elif col_name in df.columns:
            return df[col_name]

    if required:
        raise ValueError(
            f"No se encontró la columna para '{key}'. Columnas disponibles: {df.columns.tolist()}."
        )
    return None


def standardize_causal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza nombres de columnas a:
    actual, predicted, predicted_lower, predicted_upper, point_effect, cumulative_effect, p_value (opcional)
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"]))
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError("El índice debe ser DatetimeIndex o incluir 'date'.")

    actual = _pick_column(df, "actual", required=False)
    predicted = _pick_column(df, "predicted", required=False)
    lower = _pick_column(df, "predicted_lower", required=False)
    upper = _pick_column(df, "predicted_upper", required=False)
    point_eff = _pick_column(df, "point_effect", required=False)
    cum_eff = _pick_column(df, "cumulative_effect", required=False)
    p_value = _pick_column(df, "p_value", required=False)

    if actual is None:
        raise ValueError("Falta 'actual' (observado).")
    if predicted is None:
        raise ValueError("Falta 'predicted' (predicho).")

    if point_eff is None:
        point_eff = actual - predicted
    if cum_eff is None:
        cum_eff = point_eff.cumsum()

    out = pd.DataFrame(
        {
            "actual": pd.to_numeric(actual, errors="coerce"),
            "predicted": pd.to_numeric(predicted, errors="coerce"),
            "point_effect": pd.to_numeric(point_eff, errors="coerce"),
            "cumulative_effect": pd.to_numeric(cum_eff, errors="coerce"),
        },
        index=df.index,
    )

    if lower is not None:
        out["predicted_lower"] = pd.to_numeric(lower, errors="coerce")
    if upper is not None:
        out["predicted_upper"] = pd.to_numeric(upper, errors="coerce")
    if p_value is not None:
        out["p_value"] = pd.to_numeric(p_value, errors="coerce")

    out = out.sort_index()
    return out


def compute_effect_summary(df_std: pd.DataFrame, intervention_date: pd.Timestamp, alpha: float = 0.05):
    """
    Calcula métricas de resumen SOLO en POST:
    - avg_effect (media de point_effect en POST)
    - cum_effect (acumulado final en POST)
    - pct_change_mean (media de (point_effect/predicted)*100 en POST)
    - is_significant (p<alpha si hay p; si no, heurística con IC si existen; si no, 0)
    """
    if not isinstance(intervention_date, pd.Timestamp):
        intervention_date = pd.to_datetime(intervention_date)

    post = df_std.loc[df_std.index >= intervention_date].copy()
    if post.empty:
        raise ValueError("No hay observaciones en el periodo POST.")

    avg_effect = float(post["point_effect"].mean())
    cum_effect = float(post["cumulative_effect"].iloc[-1])

    denom = post["predicted"].replace(0, np.nan).abs()
    pct_change = (post["point_effect"] / denom) * 100.0
    pct_change_mean = float(pct_change.dropna().mean()) if pct_change.notna().any() else float("nan")

    is_significant = 0
    if "p_value" in post.columns and post["p_value"].notna().any():
        p = float(post["p_value"].dropna().iloc[-1])
        is_significant = int(p < alpha)
    elif {"predicted_lower", "predicted_upper"}.issubset(post.columns):
        eff_lo = post["actual"] - post["predicted_upper"]
        eff_hi = post["actual"] - post["predicted_lower"]
        crosses_zero = (eff_lo <= 0) & (eff_hi >= 0)
        frac_no_cross = 1.0 - (crosses_zero.sum() / len(crosses_zero))
        is_significant = int(frac_no_cross >= 0.8)
    else:
        is_significant = 0

    return {
        "avg_effect": avg_effect,
        "cum_effect": cum_effect,
        "pct_change_mean": pct_change_mean,
        "is_significant": is_significant,
    }
