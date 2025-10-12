# pages/2_📈_Causal_Impact.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from src.analysis.causal_impact import standardize_causal_columns, compute_effect_summary
from src.visualization.impact_plots import (
    plot_observed_vs_predicted,
    plot_point_effect,
    plot_cumulative_effect,
)

st.set_page_config(page_title="Causal Impact", page_icon="📈", layout="wide")
st.title("📈 Causal Impact")

st.markdown(
    """
    Flujo: **login → elegir propiedad GA4 → Extraer datos → venir aquí → Ejecutar análisis**.
    Esta página **recoge automáticamente** los datos que hayas extraído (de `st.session_state`), 
    aunque no estén guardados bajo `ci_data`.
    """
)

# ----------------------------------------------------------------------
# 1) RECUPERACIÓN ROBUSTA: toma los datos tal como los dejó la extracción
# ----------------------------------------------------------------------
PREFERRED_KEYS = (
    # claves típicas tras extracción GA4 (varias alternativas por si tu app usa otra)
    "ci_data", "ga4_data", "ga4_df", "df_ga4", "ga4_results", "extract_data",
    "data", "dataset", "df", "timeseries", "series", "report", "table"
)

def _try_to_dataframe(x):
    if isinstance(x, pd.DataFrame):
        return x
    try:
        df = pd.DataFrame(x)
        # descarta falsos positivos (e.g., escalares)
        if df.empty and not isinstance(x, (list, dict)):
            return None
        return df
    except Exception:
        return None

def _is_timeseries(df: pd.DataFrame) -> bool:
    if isinstance(df.index, pd.DatetimeIndex):
        return True
    cols_lower = [c.lower() for c in df.columns]
    return any(c in cols_lower for c in ("date", "fecha", "ds"))

def _has_numeric(df: pd.DataFrame) -> bool:
    return any(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)

def recover_extracted_df() -> pd.DataFrame | None:
    # 1) preferidas explícitas
    for k in PREFERRED_KEYS:
        if k in st.session_state:
            cand = _try_to_dataframe(st.session_state[k])
            if cand is not None and _is_timeseries(cand) and _has_numeric(cand):
                st.info(f"Usando datos de extracción: `st.session_state['{k}']`.")
                return cand

    # 2) si no, busca el mejor candidato entre TODO el session_state
    candidates = []
    for k, v in st.session_state.items():
        cand = _try_to_dataframe(v)
        if cand is None:
            continue
        score = 0
        if _is_timeseries(cand): score += 1
        if _has_numeric(cand):   score += 1
        # ligera preferencia si la clave "suena" a datos GA4
        for hint in ("ga4", "extract", "data", "df", "report", "table", "timeseries", "series"):
            if hint in k.lower():
                score += 1
                break
        if score >= 2:
            candidates.append((k, cand, score))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[2], reverse=True)
    best_key, best_df, _ = candidates[0]
    st.info(f"Usando datos de `st.session_state['{best_key}']` (mejor candidato).")
    return best_df

df_input = recover_extracted_df()
if df_input is None:
    st.warning(
        "No encuentro datos de la extracción en `st.session_state`. "
        "Vuelve a la página principal, pulsa **Extraer**, y regresa."
    )
    st.stop()

# --------------------------------------------------------
# 2) NORMALIZA FECHAS Y ASEGURA 'actual' (sin exigir ci_data)
# --------------------------------------------------------
try:
    if not isinstance(df_input.index, pd.DatetimeIndex):
        # intenta columna de fecha común
        for cand in ("date", "Date", "fecha", "ds"):
            if cand in df_input.columns:
                df_input[cand] = pd.to_datetime(df_input[cand])
                df_input = df_input.set_index(cand)
                break
        if not isinstance(df_input.index, pd.DatetimeIndex):
            # como último recurso: parsea índice
            df_input.index = pd.to_datetime(df_input.index)

    # quita tz si viniera con zona horaria (evita desajustes)
    if isinstance(df_input.index, pd.DatetimeIndex) and df_input.index.tz is not None:
        df_input.index = df_input.index.tz_localize(None)

    df_input = df_input.sort_index()
except Exception:
    st.error("No pude interpretar el índice/columna de fechas. Asegúrate de que haya fecha (columna o índice).")
    st.stop()

# Asegura columna objetivo 'actual'
if "actual" not in df_input.columns:
    # intenta convenciones típicas de GA4
    for cand in ("sessions", "totalUsers", "activeUsers", "screenPageViews", "eventCount", "value"):
        if cand in df_input.columns:
            df_input = df_input.rename(columns={cand: "actual"})
            break
    else:
        # toma la primera numérica
        num_cols = [c for c in df_input.columns if pd.api.types.is_numeric_dtype(df_input[c])]
        if num_cols:
            df_input = df_input.rename(columns={num_cols[0]: "actual"})
        else:
            st.error("No encuentro una columna numérica para usar como 'actual'.")
            st.stop()

# --------------------------------------------------------
# 3) UI: FECHA DE INTERVENCIÓN + ALPHA
# --------------------------------------------------------
min_date = df_input.index.min()
max_date = df_input.index.max()

col1, col2 = st.columns(2)
with col1:
    intervention_date = st.date_input(
        "Fecha de intervención",
        value=min_date if pd.notna(min_date) else None,
        min_value=min_date if pd.notna(min_date) else None,
        max_value=max_date if pd.notna(max_date) else None,
    )
with col2:
    alpha = st.number_input("Nivel de significación (alpha)", value=0.05, min_value=0.001, max_value=0.5, step=0.005)

run = st.button("Ejecutar análisis")
if not run:
    st.info("Selecciona la fecha y pulsa **Ejecutar análisis**.")
    st.stop()

# --------------------------------------------------------
# 4) SI NO HAY 'predicted', CALCÚLALO AHORA CON CAUSALIMPACT
#    (periodos PRE/POST con Timestamps DEL ÍNDICE, no .date())
# --------------------------------------------------------
if "predicted" not in df_input.columns:
    try:
        from causalimpact import CausalImpact
    except Exception as e:
        st.error(
            "Falta la dependencia 'causalimpact' o no se puede importar. "
            "Añádela en requirements.txt. "
            f"Detalle: {e}"
        )
        st.stop()

    y = df_input["actual"]

    # Regresores potenciales: todo lo que no sea columnas reservadas
    reserved = {"actual", "predicted", "predicted_lower", "predicted_upper", "point_effect", "cumulative_effect", "p", "p_value"}
    X_cols = [c for c in df_input.columns if c not in reserved and pd.api.types.is_numeric_dtype(df_input[c])]
    X = df_input[X_cols] if X_cols else None

    data = pd.concat([y] + ([X] if X is not None else []), axis=1)
    data = data.sort_index()

    idx = data.index
    # Si el índice viene con tz, quítala para evitar desajustes
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_localize(None)
        data.index = idx

    # 'intervention_date' viene de st.date_input (date); conviértelo a Timestamp
    intervention_ts = pd.Timestamp(intervention_date)

    # Posición donde empieza el POST (primer índice >= intervention_ts)
    post_start_pos = idx.searchsorted(intervention_ts, side="left")
    pre_end_pos = post_start_pos - 1

    if pre_end_pos < 0:
        st.error("No hay datos PRE anteriores a la fecha de intervención seleccionada.")
        st.stop()
    if post_start_pos >= len(idx):
        st.error("No hay datos POST en/tras la fecha de intervención seleccionada.")
        st.stop()

    # Usar exactamente los elementos del índice (Timestamps) que EXISTEN
    pre_period = [idx[0], idx[pre_end_pos]]
    post_period = [idx[post_start_pos], idx[-1]]

    # Construir y ejecutar CausalImpact con periodos válidos
    ci = CausalImpact(data, pre_period, post_period)
    infer = ci.inferences

    # Trae columnas relevantes si existen
    cols_ci = [c for c in ["point_pred", "point_pred_lower", "point_pred_upper", "point_effect", "cumulative_effect", "p"] if c in infer.columns]
    df_input = df_input.join(infer[cols_ci], how="left").rename(
        columns={
            "point_pred": "predicted",
            "point_pred_lower": "predicted_lower",
            "point_pred_upper": "predicted_upper",
            "p": "p_value",
        }
    )

# --------------------------------------------------------
# 5) ESTANDARIZA + RESUMEN (solo POST) + GRÁFICOS (autoejes)
# --------------------------------------------------------
with st.spinner("Normalizando columnas y calculando métricas..."):
    try:
        df_std = standardize_causal_columns(df_input)
    except Exception as e:
        st.error(f"Error al estandarizar columnas: {e}")
        st.write("Columnas disponibles:", list(df_input.columns))
        st.stop()

    with st.expander("Ver DataFrame estandarizado (debug)"):
        st.write(df_std.head())
        st.write(df_std.dtypes)

    try:
        summary = compute_effect_summary(df_std, pd.to_datetime(intervention_date), alpha=float(alpha))
    except Exception as e:
        st.error(f"Error al calcular el resumen POST: {e}")
        st.stop()

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Efecto promedio (POST)", f"{summary['avg_effect']:.2f}")
k2.metric("Efecto acumulado (POST)", f"{summary['cum_effect']:.2f}")
pct_display = "-" if (np.isnan(summary["pct_change_mean"]) or pd.isna(summary["pct_change_mean"])) else f"{summary['pct_change_mean']:.2f}%"
k3.metric("% cambio medio (POST)", pct_display)
k4.metric("Significativo", "Sí ✅" if summary["is_significant"] else "No ❌")
st.caption("Las métricas se calculan **solo sobre el periodo POST** desde la fecha seleccionada.")

# Gráficos (usar width='stretch' como recomienda Streamlit)
g1, g2 = st.columns((2, 1))
with g1:
    fig1 = plot_observed_vs_predicted(df_std, pd.to_datetime(intervention_date))
    st.pyplot(fig1, width="stretch")

with g2:
    fig2 = plot_point_effect(df_std, pd.to_datetime(intervention_date))
    st.pyplot(fig2, width="stretch")

fig3 = plot_cumulative_effect(df_std, pd.to_datetime(intervention_date))
st.pyplot(fig3, width="stretch")

st.success("Análisis completado.")
