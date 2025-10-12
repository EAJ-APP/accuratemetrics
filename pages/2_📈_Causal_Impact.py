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
    - Asegúrate de traer un `DataFrame` con **observado** y **predicho** (y opcionalmente intervalos) desde la página de diagnóstico o tu pipeline.
    - El DataFrame debe estar disponible en `st.session_state['ci_data']` (puedes adaptar el nombre si ya usas otro).
    """
)

# =====================
# Entrada de datos
# =====================
df_input = st.session_state.get("ci_data")  # <-- ADAPTA este nombre a tu flujo si usas otro
if df_input is None:
    st.warning(
        "No encontré `st.session_state['ci_data']`. "
        "Por favor, guarda ahí tu DataFrame antes de venir a esta página."
    )
    st.stop()

# Si es dict/list o viene con 'data', intenta convertir
if not isinstance(df_input, pd.DataFrame):
    try:
        df_input = pd.DataFrame(df_input)
    except Exception as e:
        st.error(f"No pude convertir los datos a DataFrame: {e}")
        st.stop()

# Rango de fechas para intervención
try:
    # Intentar obtener límites del índice/columna de fecha
    if not isinstance(df_input.index, pd.DatetimeIndex):
        if "date" in df_input.columns:
            df_input = df_input.set_index(pd.to_datetime(df_input["date"]))
        else:
            df_input.index = pd.to_datetime(df_input.index)
except Exception:
    st.error("No pude interpretar el índice/columna de fechas. Asegúrate de incluir una columna 'date' o un índice temporal.")
    st.stop()

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
# Normalización + Cálculo
# =====================
with st.spinner("Normalizando columnas y calculando métricas..."):
    try:
        df_std = standardize_causal_columns(df_input)
    except Exception as e:
        st.error(f"Error al estandarizar columnas: {e}")
        st.write("Columnas disponibles:", list(df_input.columns))
        st.stop()

    # Debug opcional
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
# % cambio medio puede ser NaN si predicho fue 0
pct_display = "-" if (np.isnan(summary["pct_change_mean"]) or pd.isna(summary["pct_change_mean"])) else f"{summary['pct_change_mean']:.2f}%"
k3.metric("% cambio medio (POST)", pct_display)
k4.metric("Significativo", "Sí ✅" if summary["is_significant"] else "No ❌")

st.caption("Las métricas se calculan **solo sobre el periodo POST** desde la fecha de intervención seleccionada.")

# =====================
# Gráficos (autoajuste de ejes)
# =====================
g1, g2 = st.columns((2, 1))
with g1:
    fig1 = plot_observed_vs_predicted(df_std, pd.to_datetime(intervention_date))
    st.pyplot(fig1, use_container_width=True)

with g2:
    fig2 = plot_point_effect(df_std, pd.to_datetime(intervention_date))
    st.pyplot(fig2, use_container_width=True)

fig3 = plot_cumulative_effect(df_std, pd.to_datetime(intervention_date))
st.pyplot(fig3, use_container_width=True)

st.success("Análisis completado.")
