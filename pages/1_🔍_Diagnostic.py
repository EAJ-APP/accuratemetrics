# pages/1_🔍_Diagnostic.py
from __future__ import annotations

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Diagnostic", page_icon="🔍", layout="wide")
st.title("🔍 Diagnostic")

st.markdown(
    """
    Usa esta página para **obtener/preparar** tus datos y enviarlos a la página 📈 Causal Impact.
    Opciones:
    1) **Sube un CSV** con columnas `date` y tu métrica (p. ej., `sessions`).
    2) (Opcional) Si tienes un conector GA4 en tu repo, úsalo y luego guarda el DataFrame aquí.
    """
)

with st.expander("1) Subir CSV (rápido)"):
    file = st.file_uploader("Sube un CSV con 'date' y, por ejemplo, 'sessions' (o 'value')", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
            st.write("Vista previa CSV:", df.head())
        except Exception as e:
            st.error(f"No pude leer el CSV: {e}")
            st.stop()

        def _prepare_for_causal_impact(df_in: pd.DataFrame) -> pd.DataFrame:
            df_out = df_in.copy()

            # Índice temporal
            if not isinstance(df_out.index, pd.DatetimeIndex):
                if 'date' in df_out.columns:
                    df_out['date'] = pd.to_datetime(df_out['date'])
                    df_out = df_out.set_index('date')
                else:
                    # Si no hay 'date', intenta parsear el índice actual
                    try:
                        df_out.index = pd.to_datetime(df_out.index)
                    except Exception:
                        raise ValueError("Falta columna 'date' y el índice no es temporal.")

            df_out = df_out.sort_index()

            # Asegurar 'actual'
            if 'actual' not in df_out.columns:
                if 'sessions' in df_out.columns:
                    df_out = df_out.rename(columns={'sessions': 'actual'})
                elif 'value' in df_out.columns:
                    df_out = df_out.rename(columns={'value': 'actual'})
                else:
                    # Como mínimo, intenta elegir la primera numérica si existe
                    numeric_cols = [c for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c])]
                    if numeric_cols:
                        df_out = df_out.rename(columns={numeric_cols[0]: 'actual'})
                    else:
                        raise ValueError("No encuentro una columna numérica para 'actual'.")

            return df_out

        try:
            df_ci = _prepare_for_causal_impact(df)
        except Exception as e:
            st.error(f"Error preparando datos: {e}")
            st.stop()

        st.session_state['ci_data'] = df_ci
        st.success(f"Datos listos para 📈 Causal Impact: {df_ci.shape[0]} filas, columnas: {list(df_ci.columns)}")

        try:
            st.page_link("pages/2_📈_Causal_Impact.py", label="Ir a 📈 Causal Impact", icon="📈")
        except Exception:
            st.info("Navega a la página 📈 Causal Impact desde la barra lateral.")

with st.expander("2) (Opcional) Usar tu conector GA4"):
    st.info("Si tu conector GA4 devuelve un DataFrame, asígnalo a `df_ga4` y pulsa el botón para guardarlo.")
    code = st.text_area(
        "Pega aquí tu snippet Python que genera `df_ga4`",
        value=(
            "# Ejemplo (adáptalo a tu conector):\n"
            "# from src.data.ga4_connector import tu_funcion\n"
            "# df_ga4 = tu_funcion(...)\n"
            "# df_ga4 debe tener 'date' y tu métrica (p. ej. 'sessions')\n"
        ),
        height=160,
    )
    run = st.button("Ejecutar snippet y preparar datos (avanzado)")
    if run:
        try:
            # Contexto aislado para ejecutar el snippet
            local_ctx = {}
            exec(code, {}, local_ctx)
            if "df_ga4" not in local_ctx:
                st.error("Tu snippet debe definir un DataFrame llamado `df_ga4`.")
            else:
                df_ga4 = local_ctx["df_ga4"]
                if not isinstance(df_ga4, pd.DataFrame):
                    st.error("`df_ga4` no es un DataFrame.")
                else:
                    # Reutiliza la función de preparación
                    def _prepare_for_causal_impact(df_in: pd.DataFrame) -> pd.DataFrame:
                        df_out = df_in.copy()
                        if not isinstance(df_out.index, pd.DatetimeIndex):
                            if 'date' in df_out.columns:
                                df_out['date'] = pd.to_datetime(df_out['date'])
                                df_out = df_out.set_index('date')
                            else:
                                df_out.index = pd.to_datetime(df_out.index)
                        df_out = df_out.sort_index()
                        if 'actual' not in df_out.columns:
                            if 'sessions' in df_out.columns:
                                df_out = df_out.rename(columns={'sessions': 'actual'})
                            elif 'value' in df_out.columns:
                                df_out = df_out.rename(columns={'value': 'actual'})
                        return df_out

                    df_ci = _prepare_for_causal_impact(df_ga4)
                    st.session_state['ci_data'] = df_ci
                    st.success(f"Datos listos para 📈 Causal Impact: {df_ci.shape[0]} filas, columnas: {list(df_ci.columns)}")
                    try:
                        st.page_link("pages/2_📈_Causal_Impact.py", label="Ir a 📈 Causal Impact", icon="📈")
                    except Exception:
                        st.info("Navega a la página 📈 Causal Impact desde la barra lateral.")
        except Exception as e:
            st.error(f"Falló la ejecución del snippet: {e}")
