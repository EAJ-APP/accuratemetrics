"""
Página de Análisis de Causal Impact Avanzado
AccurateMetrics - Módulo Avanzado

Funcionalidades:
- Extracción de múltiples métricas de GA4
- Matriz de correlación para selección de variables de control
- Análisis de hasta 2 intervenciones
- Comparación de intervenciones
- Gráficos detallados con matplotlib/seaborn
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Añadir directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Causal Impact Avanzado - AccurateMetrics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# VERIFICAR DEPENDENCIAS
# ============================================================================
IMPORTS_OK = True
MISSING_DEPS = []

try:
    from src.data.ga4_advanced_extractor import GA4AdvancedExtractor, generate_sample_data
except ImportError as e:
    IMPORTS_OK = False
    MISSING_DEPS.append(f"ga4_advanced_extractor: {e}")

try:
    from src.analysis.causal_impact_advanced import CausalImpactAdvancedAnalyzer
except ImportError as e:
    IMPORTS_OK = False
    MISSING_DEPS.append(f"causal_impact_advanced: {e}")

try:
    from src.visualization.matplotlib_plots import (
        plot_exploratory_analysis,
        plot_correlation_heatmap,
        plot_causal_impact_custom,
        plot_intervention_comparison,
        plot_recommended_variables,
        fig_to_bytes
    )
except ImportError as e:
    IMPORTS_OK = False
    MISSING_DEPS.append(f"matplotlib_plots: {e}")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    IMPORTS_OK = False
    MISSING_DEPS.append(f"matplotlib/seaborn: {e}")

if not IMPORTS_OK:
    st.error("Error importando módulos necesarios")
    for dep in MISSING_DEPS:
        st.error(f"  - {dep}")
    st.info("Por favor, verifica que todos los módulos estén correctamente instalados.")
    st.stop()

# ============================================================================
# INICIALIZAR SESSION STATE
# ============================================================================
if 'advanced_ga4_data' not in st.session_state:
    st.session_state.advanced_ga4_data = None
if 'ci_result_1' not in st.session_state:
    st.session_state.ci_result_1 = None
if 'ci_result_2' not in st.session_state:
    st.session_state.ci_result_2 = None
if 'comparison_df' not in st.session_state:
    st.session_state.comparison_df = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None

# ============================================================================
# TÍTULO Y DESCRIPCIÓN
# ============================================================================
st.title("📊 Análisis de Causal Impact Avanzado")
st.markdown("""
Módulo avanzado para análisis de impacto causal con:
- **Múltiples métricas** de Google Analytics 4
- **Matriz de correlación** para seleccionar variables de control
- **Hasta 2 intervenciones** para comparar
- **Gráficos detallados** con matplotlib/seaborn
""")

st.markdown("---")

# ============================================================================
# SIDEBAR - INFORMACIÓN DE SESIÓN
# ============================================================================
with st.sidebar:
    st.header("🔐 Estado de Sesión")

    if 'authenticated' in st.session_state and st.session_state.authenticated:
        st.success("Sesión activa")
        if 'user_info' in st.session_state and st.session_state.user_info:
            st.write(f"**Usuario:** {st.session_state.user_info.get('email', 'N/A')}")

        if 'property_id' in st.session_state and st.session_state.property_id:
            st.write(f"**Propiedad:** {st.session_state.property_id}")
    else:
        st.warning("No autenticado")
        st.info("Ve a la página principal para autenticarte con Google.")

    st.markdown("---")

    # Opción de datos de ejemplo
    st.header("🧪 Modo Demo")
    use_demo_data = st.checkbox(
        "Usar datos de ejemplo",
        value=False,
        help="Genera datos simulados para probar la funcionalidad"
    )

    if use_demo_data:
        st.info("Los datos de ejemplo incluyen intervenciones simuladas.")

# ============================================================================
# VERIFICAR AUTENTICACIÓN O MODO DEMO
# ============================================================================
is_authenticated = 'authenticated' in st.session_state and st.session_state.authenticated
can_proceed = is_authenticated or use_demo_data

if not can_proceed:
    st.warning("⚠️ No hay sesión activa")
    st.info("""
    Para usar este módulo necesitas:
    1. **Autenticarte** con Google en la página principal, o
    2. **Activar el modo demo** en el sidebar para usar datos de ejemplo
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Ir a la página principal", use_container_width=True):
            st.switch_page("app.py")
    with col2:
        st.info("O activa 'Usar datos de ejemplo' en el sidebar →")

    st.stop()

# ============================================================================
# PASO 1: EXTRACCIÓN DE DATOS
# ============================================================================
st.header("📥 Paso 1: Extracción de Datos")

if use_demo_data:
    # Generar datos de ejemplo
    with st.expander("⚙️ Configuración de datos de ejemplo", expanded=True):
        demo_days = st.slider("Días de datos:", 90, 365, 180)

        if st.button("🎲 Generar Datos de Ejemplo", type="primary", use_container_width=True):
            with st.spinner("Generando datos simulados..."):
                demo_data = generate_sample_data(days=demo_days)

                # Simular intervenciones
                n_days = len(demo_data)
                int1_idx = n_days // 3
                int2_idx = 2 * n_days // 3

                # Aplicar efectos
                demo_data.loc[demo_data.index[int1_idx:int1_idx+30], 'conversiones'] *= 1.15
                demo_data.loc[demo_data.index[int2_idx:int2_idx+30], 'conversiones'] *= 1.25

                st.session_state.advanced_ga4_data = demo_data
                st.session_state.demo_intervention_dates = [
                    demo_data.index[int1_idx].strftime('%Y-%m-%d'),
                    demo_data.index[int2_idx].strftime('%Y-%m-%d')
                ]

                st.success(f"Datos generados: {len(demo_data)} días")
                st.info(f"""
                **Intervenciones simuladas:**
                - Intervención 1: {st.session_state.demo_intervention_dates[0]} (+15%)
                - Intervención 2: {st.session_state.demo_intervention_dates[1]} (+25%)
                """)
                st.rerun()

else:
    # Extracción real de GA4
    with st.expander("⚙️ Configuración de extracción", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📅 Rango de Fechas")
            default_start = datetime.now() - timedelta(days=120)
            start_date = st.date_input(
                "Fecha de inicio:",
                value=default_start,
                max_value=datetime.now() - timedelta(days=1),
                help="Mínimo recomendado: 90 días (60 pre + 30 post)"
            )

            end_date = st.date_input(
                "Fecha de fin:",
                value=datetime.now() - timedelta(days=1),
                max_value=datetime.now() - timedelta(days=1)
            )

            days_selected = (end_date - start_date).days + 1
            if days_selected < 60:
                st.warning(f"⚠️ Solo {days_selected} días. Se recomiendan al menos 90.")
            else:
                st.success(f"✓ {days_selected} días seleccionados")

        with col2:
            st.subheader("🔍 Filtros (Opcional)")

            # Los filtros se cargarán dinámicamente si hay conexión
            channel_filter = st.selectbox(
                "Canal:",
                options=['Todos', 'Organic Search', 'Direct', 'Paid Search', 'Display', 'Social'],
                help="Filtrar por canal de adquisición"
            )

            device_filter = st.selectbox(
                "Dispositivo:",
                options=['Todos', 'desktop', 'mobile', 'tablet'],
                help="Filtrar por tipo de dispositivo"
            )

            country_filter = st.text_input(
                "País (opcional):",
                placeholder="Ej: Spain",
                help="Dejar vacío para todos los países"
            )

        # Botón de extracción
        if st.button("📥 Extraer Datos de GA4", type="primary", use_container_width=True):
            if 'credentials' not in st.session_state or not st.session_state.credentials:
                st.error("No hay credenciales disponibles. Autentícate primero.")
            elif 'property_id' not in st.session_state or not st.session_state.property_id:
                st.error("No hay propiedad seleccionada. Ve a la página principal.")
            else:
                with st.spinner("Extrayendo datos de GA4..."):
                    try:
                        extractor = GA4AdvancedExtractor(st.session_state.credentials)

                        df = extractor.get_advanced_metrics(
                            property_id=st.session_state.property_id,
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d'),
                            channel_filter=channel_filter if channel_filter != 'Todos' else None,
                            device_filter=device_filter if device_filter != 'Todos' else None,
                            country_filter=country_filter if country_filter else None,
                            include_channel_breakdown=True
                        )

                        if df.empty:
                            st.error("No se encontraron datos para los filtros seleccionados")
                        else:
                            st.session_state.advanced_ga4_data = df
                            st.success(f"Datos extraídos: {len(df)} días, {len(df.columns)} métricas")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Error extrayendo datos: {str(e)}")
                        with st.expander("Ver detalles del error"):
                            import traceback
                            st.code(traceback.format_exc())

# ============================================================================
# MOSTRAR DATOS SI ESTÁN DISPONIBLES
# ============================================================================
if st.session_state.advanced_ga4_data is not None:
    df = st.session_state.advanced_ga4_data

    st.success(f"✓ Datos cargados: {len(df)} días, {len(df.columns)} columnas")

    # Métricas resumen
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'sesiones_totales' in df.columns:
            st.metric("Total Sesiones", f"{df['sesiones_totales'].sum():,.0f}")
    with col2:
        if 'usuarios_unicos' in df.columns:
            st.metric("Usuarios Únicos", f"{df['usuarios_unicos'].sum():,.0f}")
    with col3:
        if 'conversiones' in df.columns:
            st.metric("Conversiones", f"{df['conversiones'].sum():,.0f}")
    with col4:
        st.metric("Días de Datos", len(df))

    # Vista previa de datos
    with st.expander("📋 Vista previa de datos"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # ==========================================================================
    # PASO 2: ANÁLISIS EXPLORATORIO
    # ==========================================================================
    st.header("🔬 Paso 2: Análisis Exploratorio")

    tab_exp1, tab_exp2, tab_exp3 = st.tabs([
        "📈 Series Temporales",
        "🔥 Matriz de Correlación",
        "💡 Variables Recomendadas"
    ])

    with tab_exp1:
        st.subheader("Series Temporales")

        # Obtener fechas de intervención si están en modo demo
        intervention_dates = None
        if use_demo_data and 'demo_intervention_dates' in st.session_state:
            intervention_dates = st.session_state.demo_intervention_dates

        # Seleccionar variable respuesta para el gráfico
        response_var = st.selectbox(
            "Variable respuesta:",
            options=['conversiones', 'sesiones_totales', 'usuarios_unicos'],
            index=0,
            key='exp_response_var'
        )

        try:
            fig_exp = plot_exploratory_analysis(
                data=df,
                intervention_dates=intervention_dates,
                response_variable=response_var
            )
            st.pyplot(fig_exp, use_container_width=True)
            plt.close(fig_exp)
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")

    with tab_exp2:
        st.subheader("Matriz de Correlación")

        try:
            fig_corr = plot_correlation_heatmap(df)
            st.pyplot(fig_corr, use_container_width=True)
            plt.close(fig_corr)
        except Exception as e:
            st.error(f"Error generando matriz: {e}")

    with tab_exp3:
        st.subheader("Variables Recomendadas como Control")

        response_var_rec = st.selectbox(
            "Variable respuesta para correlaciones:",
            options=['conversiones', 'sesiones_totales', 'usuarios_unicos'],
            index=0,
            key='rec_response_var'
        )

        threshold = st.slider(
            "Umbral de correlación:",
            min_value=0.3,
            max_value=0.8,
            value=0.5,
            step=0.1,
            help="Variables con correlación mayor a este umbral serán recomendadas"
        )

        # Calcular correlaciones
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if response_var_rec in numeric_cols:
            corr_series = df[numeric_cols].corr()[response_var_rec].drop(response_var_rec)
            corr_series = corr_series.sort_values(ascending=False)

            # Mostrar recomendaciones
            recommended = corr_series[abs(corr_series) >= threshold]

            if not recommended.empty:
                st.success(f"**Variables recomendadas** (correlación >= {threshold}):")
                for var, corr in recommended.items():
                    emoji = "✅" if corr > 0 else "⚠️"
                    st.write(f"  {emoji} **{var.replace('_', ' ').title()}**: {corr:.2f}")
            else:
                st.info(f"No hay variables con correlación >= {threshold}")

            # Gráfico
            try:
                fig_rec = plot_recommended_variables(corr_series, threshold=threshold)
                st.pyplot(fig_rec, use_container_width=True)
                plt.close(fig_rec)
            except Exception as e:
                st.error(f"Error generando gráfico: {e}")

    st.markdown("---")

    # ==========================================================================
    # PASO 3: CONFIGURACIÓN DEL ANÁLISIS
    # ==========================================================================
    st.header("⚙️ Paso 3: Configuración del Análisis")

    col_config1, col_config2 = st.columns(2)

    with col_config1:
        st.subheader("📊 Variables")

        # Variable respuesta
        response_variable = st.selectbox(
            "Variable respuesta:",
            options=['conversiones', 'sesiones_totales', 'usuarios_unicos'],
            index=0,
            help="La métrica que quieres analizar"
        )

        # Variables de control
        available_controls = [c for c in df.select_dtypes(include=[np.number]).columns
                             if c not in ['date', response_variable]]

        # Preseleccionar recomendadas
        default_controls = []
        if response_variable in df.columns:
            corr = df.select_dtypes(include=[np.number]).corr()[response_variable]
            default_controls = [c for c in available_controls if abs(corr.get(c, 0)) >= 0.5][:3]

        control_variables = st.multiselect(
            "Variables de control:",
            options=available_controls,
            default=default_controls,
            help="Variables que no fueron afectadas por la intervención"
        )

        if not control_variables:
            st.warning("⚠️ Se recomienda seleccionar al menos una variable de control")

    with col_config2:
        st.subheader("📅 Intervenciones")

        # Rango de fechas disponible
        if 'date' in df.columns:
            min_date = df['date'].min()
            max_date = df['date'].max()
        else:
            min_date = df.index.min()
            max_date = df.index.max()

        # Intervención 1 (obligatoria)
        st.markdown("**Intervención 1** (obligatoria)")

        if use_demo_data and 'demo_intervention_dates' in st.session_state:
            default_int1 = pd.to_datetime(st.session_state.demo_intervention_dates[0])
        else:
            default_int1 = min_date + (max_date - min_date) / 2

        intervention_1 = st.date_input(
            "Fecha de intervención 1:",
            value=default_int1,
            min_value=min_date + timedelta(days=14),
            max_value=max_date - timedelta(days=7),
            key='int1_date'
        )

        int1_name = st.text_input(
            "Nombre intervención 1:",
            value="Intervención 1",
            key='int1_name'
        )

        st.markdown("---")

        # Intervención 2 (opcional)
        use_intervention_2 = st.checkbox(
            "Añadir segunda intervención",
            value=use_demo_data,
            help="Analizar una segunda intervención para comparar"
        )

        intervention_2 = None
        int2_name = None

        if use_intervention_2:
            st.markdown("**Intervención 2** (opcional)")

            if use_demo_data and 'demo_intervention_dates' in st.session_state:
                default_int2 = pd.to_datetime(st.session_state.demo_intervention_dates[1])
            else:
                default_int2 = intervention_1 + timedelta(days=30)

            intervention_2 = st.date_input(
                "Fecha de intervención 2:",
                value=default_int2,
                min_value=intervention_1 + timedelta(days=7),
                max_value=max_date - timedelta(days=7),
                key='int2_date'
            )

            int2_name = st.text_input(
                "Nombre intervención 2:",
                value="Intervención 2",
                key='int2_name'
            )

    # Validaciones
    st.markdown("---")

    pre_days_1 = (intervention_1 - min_date.date()).days if hasattr(min_date, 'date') else (intervention_1 - min_date.date()).days
    post_days_1 = (max_date.date() - intervention_1).days if hasattr(max_date, 'date') else (max_date.date() - intervention_1).days

    col_val1, col_val2 = st.columns(2)

    with col_val1:
        st.info(f"""
        **Intervención 1:**
        - Pre-período: {pre_days_1} días
        - Post-período: {post_days_1} días
        """)

        if pre_days_1 < 14:
            st.error("⚠️ Se necesitan al menos 14 días pre-intervención")
        if post_days_1 < 7:
            st.error("⚠️ Se necesitan al menos 7 días post-intervención")

    if use_intervention_2 and intervention_2:
        with col_val2:
            pre_days_2 = (intervention_2 - min_date.date()).days if hasattr(min_date, 'date') else (intervention_2 - min_date.date()).days
            post_days_2 = (max_date.date() - intervention_2).days if hasattr(max_date, 'date') else (max_date.date() - intervention_2).days

            st.info(f"""
            **Intervención 2:**
            - Pre-período: {pre_days_2} días
            - Post-período: {post_days_2} días
            """)

    st.markdown("---")

    # ==========================================================================
    # PASO 4: EJECUTAR ANÁLISIS
    # ==========================================================================
    st.header("🚀 Paso 4: Ejecutar Análisis")

    can_run = True
    if pre_days_1 < 14:
        can_run = False
        st.error("No hay suficientes días pre-intervención para el análisis")
    if post_days_1 < 7:
        can_run = False
        st.error("No hay suficientes días post-intervención para el análisis")

    if can_run:
        if st.button("🚀 Ejecutar Análisis de Causal Impact", type="primary", use_container_width=True):

            # Preparar datos
            with st.spinner("Preparando datos..."):
                try:
                    # Crear DataFrame para CausalImpact
                    ci_data = df.copy()

                    # Establecer índice de fecha
                    if 'date' in ci_data.columns:
                        ci_data.set_index('date', inplace=True)

                    # Renombrar variable respuesta a 'y'
                    ci_data['y'] = ci_data[response_variable]

                    # Seleccionar columnas
                    cols_to_use = ['y'] + [c for c in control_variables if c in ci_data.columns]
                    ci_data = ci_data[cols_to_use]

                    # Asegurar frecuencia diaria
                    ci_data = ci_data.asfreq('D')
                    ci_data = ci_data.fillna(method='ffill').fillna(method='bfill').fillna(0)

                    st.success("Datos preparados correctamente")

                except Exception as e:
                    st.error(f"Error preparando datos: {e}")
                    st.stop()

            # Crear analizador
            with st.spinner("Inicializando analizador..."):
                try:
                    analyzer = CausalImpactAdvancedAnalyzer(
                        data=ci_data,
                        control_variables=control_variables if control_variables else None
                    )

                    # Validar datos
                    is_valid, validation_msg = analyzer.validate_data()
                    if not is_valid:
                        st.error(f"Validación fallida: {validation_msg}")
                        st.stop()

                    st.success("Analizador inicializado")

                except Exception as e:
                    st.error(f"Error inicializando analizador: {e}")
                    st.stop()

            # Analizar Intervención 1
            progress_bar = st.progress(0, text="Analizando intervención 1...")

            try:
                result_1 = analyzer.analyze_intervention(
                    intervention_date=intervention_1.strftime('%Y-%m-%d'),
                    intervention_name=int1_name
                )

                st.session_state.ci_result_1 = result_1
                progress_bar.progress(50, text="Intervención 1 completada")

            except Exception as e:
                st.error(f"Error analizando intervención 1: {e}")
                with st.expander("Ver detalles"):
                    import traceback
                    st.code(traceback.format_exc())
                st.stop()

            # Analizar Intervención 2 (si existe)
            if use_intervention_2 and intervention_2:
                progress_bar.progress(50, text="Analizando intervención 2...")

                try:
                    result_2 = analyzer.analyze_intervention(
                        intervention_date=intervention_2.strftime('%Y-%m-%d'),
                        intervention_name=int2_name
                    )

                    st.session_state.ci_result_2 = result_2

                except Exception as e:
                    st.error(f"Error analizando intervención 2: {e}")

            # Guardar analizador y comparación
            st.session_state.analyzer = analyzer
            st.session_state.comparison_df = analyzer.compare_interventions()

            progress_bar.progress(100, text="✓ Análisis completado")
            st.success("🎉 Análisis completado exitosamente")
            st.balloons()

    st.markdown("---")

    # ==========================================================================
    # PASO 5: RESULTADOS
    # ==========================================================================
    st.header("📊 Paso 5: Resultados")

    if st.session_state.ci_result_1 is None:
        st.info("👆 Ejecuta el análisis para ver los resultados")
    else:
        # Tabs de resultados
        result_tabs = st.tabs([
            f"🎯 {st.session_state.ci_result_1['nombre']}",
            f"🎯 {st.session_state.ci_result_2['nombre']}" if st.session_state.ci_result_2 else "📊 Sin 2da Intervención",
            "📊 Comparación"
        ])

        # ====== TAB RESULTADO 1 ======
        with result_tabs[0]:
            result_1 = st.session_state.ci_result_1

            st.subheader(f"Resultados: {result_1['nombre']}")

            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Efecto Diario",
                    f"{result_1['metricas']['efecto_diario']:+.1f}",
                    delta=f"{result_1['metricas']['cambio_porcentual']:+.1f}%"
                )

            with col2:
                st.metric(
                    "Efecto Total",
                    f"{result_1['metricas']['efecto_total']:+,.0f}"
                )

            with col3:
                sig_text = "Sí ✅" if result_1['estadisticas']['es_significativo'] else "No ❌"
                st.metric(
                    "Significativo",
                    sig_text,
                    delta=f"p={result_1['estadisticas']['p_value']:.4f}"
                )

            with col4:
                st.metric(
                    "Cambio %",
                    f"{result_1['metricas']['cambio_porcentual']:+.1f}%"
                )

            # Interpretación
            st.markdown("---")

            interp = result_1['interpretacion']
            if result_1['estadisticas']['es_significativo']:
                if result_1['metricas']['efecto_diario'] > 0:
                    st.success(f"**{interp['significancia']}**")
                else:
                    st.warning(f"**{interp['significancia']}**")
            else:
                st.info(f"**{interp['significancia']}**")

            st.markdown(f"**Conclusión:** {interp['conclusion']}")

            # Gráfico
            st.markdown("---")
            st.subheader("📈 Gráficos de Causal Impact")

            try:
                analyzer = st.session_state.analyzer
                plot_data_1 = analyzer.get_plot_data(result_1['nombre'])

                fig_ci_1 = plot_causal_impact_custom(
                    ci_result=analyzer.impact_objects[result_1['nombre']],
                    data=analyzer.data,
                    intervention_date=result_1['fecha'],
                    title=f"Análisis Causal Impact - {result_1['nombre']}"
                )

                st.pyplot(fig_ci_1, use_container_width=True)
                plt.close(fig_ci_1)

                # Botón de descarga
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    png_bytes = fig_to_bytes(fig_ci_1, format='png')
                    st.download_button(
                        label="📥 Descargar Gráfico (PNG)",
                        data=png_bytes,
                        file_name=f"causal_impact_{result_1['nombre'].replace(' ', '_')}.png",
                        mime="image/png"
                    )

            except Exception as e:
                st.error(f"Error generando gráfico: {e}")

        # ====== TAB RESULTADO 2 ======
        with result_tabs[1]:
            if st.session_state.ci_result_2:
                result_2 = st.session_state.ci_result_2

                st.subheader(f"Resultados: {result_2['nombre']}")

                # Métricas principales
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Efecto Diario",
                        f"{result_2['metricas']['efecto_diario']:+.1f}",
                        delta=f"{result_2['metricas']['cambio_porcentual']:+.1f}%"
                    )

                with col2:
                    st.metric(
                        "Efecto Total",
                        f"{result_2['metricas']['efecto_total']:+,.0f}"
                    )

                with col3:
                    sig_text = "Sí ✅" if result_2['estadisticas']['es_significativo'] else "No ❌"
                    st.metric(
                        "Significativo",
                        sig_text,
                        delta=f"p={result_2['estadisticas']['p_value']:.4f}"
                    )

                with col4:
                    st.metric(
                        "Cambio %",
                        f"{result_2['metricas']['cambio_porcentual']:+.1f}%"
                    )

                # Interpretación
                st.markdown("---")

                interp = result_2['interpretacion']
                if result_2['estadisticas']['es_significativo']:
                    if result_2['metricas']['efecto_diario'] > 0:
                        st.success(f"**{interp['significancia']}**")
                    else:
                        st.warning(f"**{interp['significancia']}**")
                else:
                    st.info(f"**{interp['significancia']}**")

                st.markdown(f"**Conclusión:** {interp['conclusion']}")

                # Gráfico
                st.markdown("---")
                st.subheader("📈 Gráficos de Causal Impact")

                try:
                    analyzer = st.session_state.analyzer

                    fig_ci_2 = plot_causal_impact_custom(
                        ci_result=analyzer.impact_objects[result_2['nombre']],
                        data=analyzer.data,
                        intervention_date=result_2['fecha'],
                        title=f"Análisis Causal Impact - {result_2['nombre']}"
                    )

                    st.pyplot(fig_ci_2, use_container_width=True)
                    plt.close(fig_ci_2)

                except Exception as e:
                    st.error(f"Error generando gráfico: {e}")
            else:
                st.info("No se configuró una segunda intervención")

        # ====== TAB COMPARACIÓN ======
        with result_tabs[2]:
            st.subheader("📊 Comparación de Intervenciones")

            if st.session_state.comparison_df is not None and len(st.session_state.comparison_df) > 1:
                comparison_df = st.session_state.comparison_df

                # Tabla comparativa
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                # Gráfico de comparación
                st.markdown("---")

                try:
                    fig_comp = plot_intervention_comparison(comparison_df)
                    st.pyplot(fig_comp, use_container_width=True)
                    plt.close(fig_comp)
                except Exception as e:
                    st.error(f"Error generando gráfico de comparación: {e}")

                # Ganador
                st.markdown("---")
                winner = st.session_state.analyzer.get_winner()

                if winner and winner['hay_ganador']:
                    st.success(f"""
                    ## 🏆 {winner['mensaje']}

                    **Detalles:**
                    - Efecto diario: {winner['efecto_diario']:+.1f} unidades
                    - Efecto total: {winner['efecto_total']:+,.0f} unidades
                    - Cambio porcentual: {winner['cambio_porcentual']:+.1f}%
                    """)
                elif winner:
                    st.info(f"ℹ️ {winner['mensaje']}")

            else:
                st.info("Se necesitan al menos 2 intervenciones para comparar")

        # ==========================================================================
        # EXPORTAR RESULTADOS
        # ==========================================================================
        st.markdown("---")
        st.header("💾 Exportar Resultados")

        col_exp1, col_exp2, col_exp3 = st.columns(3)

        with col_exp1:
            # Exportar datos preparados
            if st.session_state.analyzer:
                csv_data = st.session_state.analyzer.data.to_csv()
                st.download_button(
                    label="📥 Descargar Datos (CSV)",
                    data=csv_data,
                    file_name=f"datos_causal_impact_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col_exp2:
            # Exportar comparación
            if st.session_state.comparison_df is not None:
                comp_csv = st.session_state.comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Comparación (CSV)",
                    data=comp_csv,
                    file_name=f"comparacion_intervenciones_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col_exp3:
            # Exportar reporte
            if st.session_state.ci_result_1:
                report = f"""
REPORTE DE ANÁLISIS DE CAUSAL IMPACT AVANZADO
=============================================
Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{'='*50}
INTERVENCIÓN 1: {st.session_state.ci_result_1['nombre']}
{'='*50}

Fecha: {st.session_state.ci_result_1['fecha']}

MÉTRICAS:
- Efecto diario: {st.session_state.ci_result_1['metricas']['efecto_diario']:+.2f}
- Efecto total: {st.session_state.ci_result_1['metricas']['efecto_total']:+.0f}
- Cambio porcentual: {st.session_state.ci_result_1['metricas']['cambio_porcentual']:+.2f}%
- P-value: {st.session_state.ci_result_1['estadisticas']['p_value']:.4f}
- Significativo: {'Sí' if st.session_state.ci_result_1['estadisticas']['es_significativo'] else 'No'}

CONCLUSIÓN:
{st.session_state.ci_result_1['interpretacion']['conclusion']}
"""

                if st.session_state.ci_result_2:
                    report += f"""

{'='*50}
INTERVENCIÓN 2: {st.session_state.ci_result_2['nombre']}
{'='*50}

Fecha: {st.session_state.ci_result_2['fecha']}

MÉTRICAS:
- Efecto diario: {st.session_state.ci_result_2['metricas']['efecto_diario']:+.2f}
- Efecto total: {st.session_state.ci_result_2['metricas']['efecto_total']:+.0f}
- Cambio porcentual: {st.session_state.ci_result_2['metricas']['cambio_porcentual']:+.2f}%
- P-value: {st.session_state.ci_result_2['estadisticas']['p_value']:.4f}
- Significativo: {'Sí' if st.session_state.ci_result_2['estadisticas']['es_significativo'] else 'No'}

CONCLUSIÓN:
{st.session_state.ci_result_2['interpretacion']['conclusion']}
"""

                st.download_button(
                    label="📄 Descargar Reporte (TXT)",
                    data=report,
                    file_name=f"reporte_causal_impact_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("AccurateMetrics v0.3 - Módulo de Causal Impact Avanzado | Powered by pycausalimpact, matplotlib y seaborn")
