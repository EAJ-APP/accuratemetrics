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
    Esta página ejecuta y **visualiza** el análisis de impacto causal.
    - Asegúrate de traer un `DataFrame` con **observado** y (si no existe `predicted`) generaremos el **predicho** con *CausalImpact* usando el periodo PRE.
    - Los datos deben estar en `st.session_state['ci_data']` (ver página 🔍 Diagnostic).
    """
)

# =====================
# Entrada de datos
# =====================
df_input = st.session_state.get("ci_data")
if df_input is None:
    st.warning(
        "No encontré `st.session_state['ci_data']`. "
        "Ve a la página 🔍 Diagnostic, ejecuta la extracción y vuelve."
    )
    st.stop()

# Asegurar índice temporal ordenado
try:
    if not isinstance(df_input.index, pd.DatetimeIndex):
        if "date" in df_input.columns:
            df_input = df_input.set_index(pd.to_datetime(df_input["date"]))
        else:
            df_input.index = pd.to_datetime(df_input.index)
    df_input = df_input.sort_index()
except Exception:
    st.error("No pude interpretar el índice/columna de fechas. Asegúrate de incluir una columna 'date' o un índice temporal.")
    st.stop()

# Inferir 'actual' si no existe todavía
if "actual" not in df_input.columns:
    if "sessions" in df_input.columns:
        df_input = df_input.rename(columns={"sessions": "actual"})
    elif "value" in df_input.columns:
        df_input = df_input.rename(columns={"value": "actual"})

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

# =====================
# Si no hay 'predicted', ejecuta CausalImpact para estimarlo
# =====================
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

    if "actual" not in df_input.columns:
        st.error("No encuentro columna 'actual' para ejecutar CausalImpact.")
        st.stop()

    y = df_input["actual"]

    exclude_cols = {"actual", "predicted", "predicted_lower", "predicted_upper", "point_effect", "cumulative_effect"}
    candidate_cols = [c for c in df_input.columns if c not in exclude_cols]
    X = df_input[candidate_cols] if candidate_cols else None

    data = pd.concat([y] + ([X] if X is not None else []), axis=1)

    intervention_ts = pd.to_datetime(intervention_date)
    pre_period = [data.index.min().date(), (intervention_ts - pd.Timedelta(days=1)).date()]
    post_period = [intervention_ts.date(), data.index.max().date()]

    if pre_period[1] < pre_period[0] or post_period[1] < post_period[0]:
        st.error("El periodo PRE o POST es inválido. Revisa la fecha de intervención y el rango temporal.")
        st.stop()

    ci = CausalImpact(data, pre_period, post_period)
    infer = ci.inferences

    cols_ci = [c for c in ["point_pred", "point_pred_lower", "point_pred_upper", "point_effect", "cumulative_effect", "p"] if c in infer.columns]
    df_input = df_input.join(infer[cols_ci], how="left").rename(
        columns={
            "point_pred": "predicted",
            "point_pred_lower": "predicted_lower",
            "point_pred_upper": "predicted_upper",
            "p": "p_value",
        }
    )

# =====================
# Normalización + Cálculo (usa funciones del módulo)
# =====================
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

# =====================
# KPIs de resumen
# =====================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Efecto promedio (POST)", f"{summary['avg_effect']:.2f}")
k2.metric("Efecto acumulado (POST)", f"{summary['cum_effect']:.2f}")
pct_display = "-" if (np.isnan(summary["pct_change_mean"]) or pd.isna(summary["pct_change_mean"])) else f"{summary['pct_change_mean']:.2f}%"
k3.metric("% cambio medio (POST)", pct_display)
k4.metric("Significativo", "Sí ✅" if summary["is_significant"] else "No ❌")
st.caption("Las métricas se calculan **solo sobre el periodo POST** desde la fecha seleccionada.")

# =====================
# Gráficos (autoajuste de ejes)
# =====================
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
