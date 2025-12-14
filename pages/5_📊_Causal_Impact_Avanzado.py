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
    from src.data.ga4_properties import GA4PropertyManager
except ImportError as e:
    IMPORTS_OK = False
    MISSING_DEPS.append(f"ga4_properties: {e}")

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
        plot_dual_intervention_timeline,
        plot_monetary_impact,
        plot_monetary_comparison,
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
# Session state para propiedades GA4 avanzado
if 'adv_ga4_properties' not in st.session_state:
    st.session_state.adv_ga4_properties = None
if 'adv_properties_loaded' not in st.session_state:
    st.session_state.adv_properties_loaded = False
if 'adv_selected_property_id' not in st.session_state:
    st.session_state.adv_selected_property_id = None
if 'adv_available_filters' not in st.session_state:
    st.session_state.adv_available_filters = None

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

        # Mostrar propiedad seleccionada en este módulo
        if st.session_state.adv_selected_property_id:
            st.write(f"**Propiedad:** `{st.session_state.adv_selected_property_id}`")
        elif 'property_id' in st.session_state and st.session_state.property_id:
            st.write(f"**Propiedad (principal):** `{st.session_state.property_id}`")
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

    # Información de datos cargados
    if st.session_state.advanced_ga4_data is not None:
        st.markdown("---")
        st.header("📊 Datos Cargados")
        df_info = st.session_state.advanced_ga4_data
        st.write(f"**Días:** {len(df_info)}")
        st.write(f"**Métricas:** {len(df_info.columns)}")
        if 'date' in df_info.columns:
            st.write(f"**Desde:** {df_info['date'].min().strftime('%Y-%m-%d')}")
            st.write(f"**Hasta:** {df_info['date'].max().strftime('%Y-%m-%d')}")

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

    # ==========================================
    # SELECTOR DE PROPIEDAD GA4
    # ==========================================
    with st.expander("🎯 Selección de Propiedad GA4", expanded=True):

        # Verificar credenciales
        if 'credentials' not in st.session_state or not st.session_state.credentials:
            st.error("No hay credenciales disponibles. Por favor, autentícate primero en la página principal.")
            st.stop()

        # Cargar propiedades si no están cargadas
        if not st.session_state.adv_properties_loaded:
            with st.spinner("🔄 Cargando tus propiedades de Google Analytics..."):
                try:
                    property_manager = GA4PropertyManager(st.session_state.credentials)
                    properties_dict = property_manager.get_properties_dict()

                    if properties_dict:
                        st.session_state.adv_ga4_properties = properties_dict
                        st.session_state.adv_properties_loaded = True
                        st.rerun()
                    else:
                        st.warning("⚠️ No se encontraron propiedades de GA4 en tu cuenta")
                        st.info("Verifica que tengas acceso a al menos una propiedad de Google Analytics 4")

                except Exception as e:
                    st.error(f"Error al cargar propiedades: {str(e)}")
                    st.info("💡 Intenta cerrar sesión y volver a autenticarte")

        # Mostrar selector de propiedades
        if st.session_state.adv_properties_loaded and st.session_state.adv_ga4_properties:
            col_prop1, col_prop2 = st.columns([3, 1])

            with col_prop1:
                # Selector de propiedades
                selected_property_name = st.selectbox(
                    "📊 Selecciona una Propiedad de GA4:",
                    options=list(st.session_state.adv_ga4_properties.keys()),
                    help="Selecciona la propiedad que quieres analizar",
                    key="adv_property_selector"
                )

                # Obtener el ID de la propiedad seleccionada
                selected_property_id = st.session_state.adv_ga4_properties[selected_property_name]
                st.session_state.adv_selected_property_id = selected_property_id

                # Mostrar el ID para referencia
                st.caption(f"**Property ID:** `{selected_property_id}`")

            with col_prop2:
                st.markdown("####")
                if st.button("🔄 Recargar", use_container_width=True, help="Recargar lista de propiedades"):
                    st.session_state.adv_properties_loaded = False
                    st.session_state.adv_ga4_properties = None
                    st.session_state.adv_available_filters = None
                    st.rerun()

            # Mostrar información de la propiedad seleccionada
            st.success(f"✓ Propiedad seleccionada: **{selected_property_name}**")

        else:
            # Fallback: Input manual si no se pudieron cargar propiedades
            st.warning("⚠️ No se pudieron cargar las propiedades automáticamente")

            col_manual1, col_manual2 = st.columns([3, 1])

            with col_manual1:
                manual_property_id = st.text_input(
                    "Property ID de GA4 (manual):",
                    value=st.session_state.adv_selected_property_id or "",
                    placeholder="123456789",
                    help="Encuentra tu Property ID en Admin > Property Settings de Google Analytics"
                )

                if manual_property_id:
                    st.session_state.adv_selected_property_id = manual_property_id

            with col_manual2:
                st.markdown("####")
                if st.button("🔄 Reintentar", use_container_width=True):
                    st.session_state.adv_properties_loaded = False
                    st.rerun()

    # ==========================================
    # CONFIGURACIÓN DE EXTRACCIÓN
    # ==========================================
    with st.expander("⚙️ Configuración de extracción", expanded=True):

        # Verificar que hay propiedad seleccionada
        if not st.session_state.adv_selected_property_id:
            st.warning("⚠️ Selecciona una propiedad GA4 primero")
            st.stop()

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

            # Botón para recargar filtros
            col_filter_reload = st.columns([3, 1])
            with col_filter_reload[1]:
                if st.button("🔄", help="Recargar filtros de GA4"):
                    st.session_state.adv_available_filters = None
                    st.rerun()

            # Cargar filtros disponibles dinámicamente si no están cargados
            if st.session_state.adv_available_filters is None and st.session_state.adv_selected_property_id:
                try:
                    with st.spinner("Cargando filtros disponibles..."):
                        extractor = GA4AdvancedExtractor(st.session_state.credentials)
                        filters = extractor.get_available_filters(
                            property_id=st.session_state.adv_selected_property_id,
                            start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                            end_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                        )
                        st.session_state.adv_available_filters = filters

                        # Mostrar resumen de filtros cargados
                        n_canales = len(filters.get('canales', [])) - 1  # -1 por "Todos"
                        n_dispositivos = len(filters.get('dispositivos', [])) - 1
                        n_paises = len(filters.get('paises', [])) - 1
                        if n_canales > 0 or n_dispositivos > 0 or n_paises > 0:
                            st.success(f"✓ Filtros cargados: {n_canales} canales, {n_dispositivos} dispositivos, {n_paises} países")
                        else:
                            st.warning("⚠️ No se cargaron valores de filtro de GA4")

                except Exception as e:
                    st.error(f"Error cargando filtros: {e}")
                    st.session_state.adv_available_filters = {
                        'canales': ['Todos', 'Organic Search', 'Direct', 'Paid Search', 'Display', 'Social', 'Referral', 'Email'],
                        'dispositivos': ['Todos', 'desktop', 'mobile', 'tablet'],
                        'paises': ['Todos'],
                        'ciudades': ['Todos']
                    }
                    st.info("Usando filtros predeterminados")

            # Usar filtros dinámicos o defaults
            filters = st.session_state.adv_available_filters or {
                'canales': ['Todos', 'Organic Search', 'Direct', 'Paid Search', 'Display', 'Social', 'Referral', 'Email'],
                'dispositivos': ['Todos', 'desktop', 'mobile', 'tablet'],
                'paises': ['Todos'],
                'ciudades': ['Todos']
            }

            # Debug: mostrar opciones disponibles
            with st.expander("Ver opciones de filtro disponibles"):
                st.write(f"**Canales:** {filters.get('canales', [])}")
                st.write(f"**Dispositivos:** {filters.get('dispositivos', [])}")
                st.write(f"**Países:** {len(filters.get('paises', []))} opciones")
                st.write(f"**Ciudades:** {len(filters.get('ciudades', []))} opciones")

            channel_filter = st.selectbox(
                "Canal:",
                options=filters.get('canales', ['Todos']),
                help="Filtrar por canal de adquisición"
            )

            device_filter = st.selectbox(
                "Dispositivo:",
                options=filters.get('dispositivos', ['Todos']),
                help="Filtrar por tipo de dispositivo"
            )

            country_filter = st.selectbox(
                "País:",
                options=filters.get('paises', ['Todos']),
                help="Filtrar por país"
            )

            city_filter = st.selectbox(
                "Ciudad:",
                options=filters.get('ciudades', ['Todos']),
                help="Filtrar por ciudad"
            )

        # Mostrar filtros activos
        active_filters = []
        if channel_filter != 'Todos':
            active_filters.append(f"Canal: {channel_filter}")
        if device_filter != 'Todos':
            active_filters.append(f"Dispositivo: {device_filter}")
        if country_filter != 'Todos':
            active_filters.append(f"País: {country_filter}")
        if city_filter != 'Todos':
            active_filters.append(f"Ciudad: {city_filter}")

        if active_filters:
            st.info(f"🔍 **Filtros activos:** {' | '.join(active_filters)}")

        # Debug mode
        debug_mode = st.checkbox("🐛 Modo depuración", value=False, help="Mostrar información detallada de la extracción")

        # Botón de extracción
        if st.button("📥 Extraer Datos de GA4", type="primary", use_container_width=True):
            if 'credentials' not in st.session_state or not st.session_state.credentials:
                st.error("No hay credenciales disponibles. Autentícate primero.")
            elif not st.session_state.adv_selected_property_id:
                st.error("No hay propiedad seleccionada. Selecciona una propiedad arriba.")
            else:
                with st.spinner("Extrayendo datos de GA4..."):
                    try:
                        extractor = GA4AdvancedExtractor(st.session_state.credentials)

                        # Mostrar info de debug antes de extraer
                        if debug_mode:
                            st.write("### 🐛 Debug Info")
                            st.write(f"**Property ID:** {st.session_state.adv_selected_property_id}")
                            st.write(f"**Fechas:** {start_date} a {end_date}")
                            st.write(f"**Channel filter (raw):** `{channel_filter}`")
                            st.write(f"**Channel filter (passed):** `{channel_filter if channel_filter != 'Todos' else None}`")
                            st.write(f"**Device filter:** `{device_filter if device_filter != 'Todos' else None}`")
                            st.write(f"**Country filter:** `{country_filter if country_filter != 'Todos' else None}`")
                            st.write(f"**City filter:** `{city_filter if city_filter != 'Todos' else None}`")

                        df = extractor.get_advanced_metrics(
                            property_id=st.session_state.adv_selected_property_id,
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d'),
                            channel_filter=channel_filter if channel_filter != 'Todos' else None,
                            device_filter=device_filter if device_filter != 'Todos' else None,
                            country_filter=country_filter if country_filter != 'Todos' else None,
                            city_filter=city_filter if city_filter != 'Todos' else None,
                            include_channel_breakdown=True,
                            debug=debug_mode
                        )

                        if debug_mode:
                            st.write(f"**Filas retornadas:** {len(df)}")
                            st.write(f"**Columnas:** {list(df.columns)}")
                            if 'sesiones_totales' in df.columns:
                                st.write(f"**Total sesiones:** {df['sesiones_totales'].sum():,.0f}")

                        if df.empty:
                            st.error("No se encontraron datos para los filtros seleccionados")
                            if debug_mode:
                                st.warning("⚠️ El DataFrame está vacío. Posibles causas:")
                                st.write("- El filtro no coincide con ningún dato")
                                st.write("- El nombre del canal/dispositivo/país tiene diferente formato")
                                st.write("- No hay datos en el rango de fechas seleccionado")
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

        # Variable respuesta (por defecto: Sesiones)
        response_variable = st.selectbox(
            "Variable respuesta:",
            options=['sesiones_totales', 'conversiones', 'usuarios_unicos'],
            index=0,
            help="La métrica que quieres analizar"
        )

        # Variables de control
        available_controls = [c for c in df.select_dtypes(include=[np.number]).columns
                             if c not in ['date', response_variable]]

        # Variables de control predeterminadas (en orden de preferencia)
        preferred_controls = [
            'sesiones_totales',
            'conversiones',
            'usuarios_unicos',
            'trafico_directo',
            'trafico_pago',
            'trafico_organico'
        ]
        # Filtrar las que existen y no son la variable respuesta
        default_controls = [c for c in preferred_controls
                          if c in available_controls and c != response_variable]

        # Si no hay ninguna preferida disponible, usar las primeras 3 disponibles
        if not default_controls and available_controls:
            default_controls = available_controls[:3]

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

        # Tipo de intervención 1
        int1_type = st.radio(
            "Tipo de intervención:",
            options=['puntual', 'prolongada'],
            index=0,
            horizontal=True,
            key='int1_type',
            help="**Puntual:** Acción de un día con efecto permanente. **Prolongada:** Campaña con duración definida."
        )

        # Fecha fin para intervención prolongada
        int1_end_date = None
        int1_analyze_residual = False

        if int1_type == 'prolongada':
            default_end_1 = intervention_1 + timedelta(days=30)
            if default_end_1 > max_date.date() if hasattr(max_date, 'date') else max_date:
                default_end_1 = max_date.date() if hasattr(max_date, 'date') else max_date - timedelta(days=1)

            int1_end_date = st.date_input(
                "Fecha fin de campaña:",
                value=default_end_1,
                min_value=intervention_1 + timedelta(days=1),
                max_value=max_date - timedelta(days=1) if hasattr(max_date, 'date') else max_date - timedelta(days=1),
                key='int1_end_date',
                help="Fecha en que termina la campaña/intervención"
            )

            int1_analyze_residual = st.checkbox(
                "Analizar efecto residual (post-campaña)",
                value=True,
                key='int1_residual',
                help="Analiza si el efecto persiste después de que termine la campaña"
            )

            # Mostrar duración de campaña
            campaign_days = (int1_end_date - intervention_1).days
            st.caption(f"📅 Duración de campaña: **{campaign_days} días**")

        st.markdown("---")

        # Intervención 2 (opcional)
        use_intervention_2 = st.checkbox(
            "Añadir segunda intervención",
            value=use_demo_data,
            help="Analizar una segunda intervención para comparar"
        )

        intervention_2 = None
        int2_name = None
        int2_type = 'puntual'
        int2_end_date = None
        int2_analyze_residual = False

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

            # Tipo de intervención 2
            int2_type = st.radio(
                "Tipo de intervención:",
                options=['puntual', 'prolongada'],
                index=0,
                horizontal=True,
                key='int2_type',
                help="**Puntual:** Acción de un día con efecto permanente. **Prolongada:** Campaña con duración definida."
            )

            if int2_type == 'prolongada':
                default_end_2 = intervention_2 + timedelta(days=30)
                if default_end_2 > max_date.date() if hasattr(max_date, 'date') else max_date:
                    default_end_2 = max_date.date() if hasattr(max_date, 'date') else max_date - timedelta(days=1)

                int2_end_date = st.date_input(
                    "Fecha fin de campaña:",
                    value=default_end_2,
                    min_value=intervention_2 + timedelta(days=1),
                    max_value=max_date - timedelta(days=1) if hasattr(max_date, 'date') else max_date - timedelta(days=1),
                    key='int2_end_date',
                    help="Fecha en que termina la campaña/intervención"
                )

                int2_analyze_residual = st.checkbox(
                    "Analizar efecto residual (post-campaña)",
                    value=True,
                    key='int2_residual',
                    help="Analiza si el efecto persiste después de que termine la campaña"
                )

                campaign_days_2 = (int2_end_date - intervention_2).days
                st.caption(f"📅 Duración de campaña: **{campaign_days_2} días**")

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
                # Preparar parámetros según tipo de intervención
                int1_params = {
                    'intervention_date': intervention_1.strftime('%Y-%m-%d'),
                    'intervention_name': int1_name,
                    'intervention_type': int1_type
                }

                # Añadir parámetros para intervención prolongada
                if int1_type == 'prolongada' and int1_end_date:
                    int1_params['intervention_end_date'] = int1_end_date.strftime('%Y-%m-%d')
                    int1_params['analyze_residual'] = int1_analyze_residual

                result_1 = analyzer.analyze_intervention(**int1_params)

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
                    # Preparar parámetros según tipo de intervención
                    int2_params = {
                        'intervention_date': intervention_2.strftime('%Y-%m-%d'),
                        'intervention_name': int2_name,
                        'intervention_type': int2_type
                    }

                    # Añadir parámetros para intervención prolongada
                    if int2_type == 'prolongada' and int2_end_date:
                        int2_params['intervention_end_date'] = int2_end_date.strftime('%Y-%m-%d')
                        int2_params['analyze_residual'] = int2_analyze_residual

                    result_2 = analyzer.analyze_intervention(**int2_params)

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

            # Mostrar tipo de intervención
            tipo_interv = result_1.get('tipo_intervencion', 'puntual')
            if tipo_interv == 'prolongada':
                campana = result_1.get('campana', {})
                st.info(f"""
                **📅 Intervención Prolongada (Campaña)**
                - Inicio: {campana.get('fecha_inicio', 'N/A')}
                - Fin: {campana.get('fecha_fin', 'N/A')}
                - Duración: {campana.get('duracion_dias', 'N/A')} días
                """)
            else:
                st.info(f"**📍 Intervención Puntual** - Fecha: {result_1['fecha']}")

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

            # ====== EFECTO RESIDUAL (solo para intervenciones prolongadas) ======
            if tipo_interv == 'prolongada' and 'efecto_residual' in result_1:
                residual = result_1['efecto_residual']
                st.markdown("---")
                st.subheader("🔄 Efecto Residual (Post-Campaña)")

                if residual.get('disponible', False):
                    st.info(f"""
                    **Período analizado:** {residual.get('periodo_inicio', 'N/A')} a {residual.get('periodo_fin', 'N/A')} ({residual.get('dias', 0)} días después de la campaña)
                    """)

                    res_data = residual.get('resultados', {})
                    res_metricas = res_data.get('metricas', {})
                    res_stats = res_data.get('estadisticas', {})

                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

                    with col_r1:
                        st.metric(
                            "Efecto Diario Residual",
                            f"{res_metricas.get('efecto_diario', 0):+.1f}",
                            delta=f"{res_metricas.get('cambio_porcentual', 0):+.1f}%"
                        )

                    with col_r2:
                        st.metric(
                            "Efecto Total Residual",
                            f"{res_metricas.get('efecto_total', 0):+,.0f}"
                        )

                    with col_r3:
                        sig_residual = "Sí ✅" if res_stats.get('es_significativo', False) else "No ❌"
                        st.metric(
                            "Significativo",
                            sig_residual,
                            delta=f"p={res_stats.get('p_value', 1):.4f}"
                        )

                    with col_r4:
                        st.metric(
                            "Cambio % Residual",
                            f"{res_metricas.get('cambio_porcentual', 0):+.1f}%"
                        )

                    # Interpretación del efecto residual
                    if res_stats.get('es_significativo', False):
                        if res_metricas.get('efecto_diario', 0) > 0:
                            st.success("✅ **El efecto persiste** después de la campaña")
                        else:
                            st.warning("⚠️ **Efecto negativo residual** detectado")
                    else:
                        st.info("ℹ️ **No hay efecto residual significativo** - El efecto desapareció al terminar la campaña")

                else:
                    st.warning(f"⚠️ {residual.get('mensaje', 'No se pudo analizar el efecto residual')}")

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

            # ====== IMPACTO MONETARIO ======
            st.markdown("---")
            st.subheader("💰 Impacto Monetario")

            try:
                # Obtener datos de ingresos
                if 'ingresos' in df.columns and 'compras' in df.columns:
                    ingresos_totales = df['ingresos'].sum()
                    compras_totales = df['compras'].sum()
                else:
                    # Estimar si no hay datos reales
                    ingresos_totales = df.get('ingresos', pd.Series([0])).sum()
                    compras_totales = df.get('compras', pd.Series([1])).sum()
                    if compras_totales == 0:
                        compras_totales = 1

                # Datos del efecto
                efecto_total = result_1['metricas']['efecto_total']

                # Calcular conversiones con y sin impacto
                # Efecto total = real - predicho, entonces predicho = real - efecto
                summary_data = result_1.get('summary_data', {})
                if 'cumulative' in summary_data:
                    conversiones_con_impacto = summary_data['cumulative'].get('actual', efecto_total * 2)
                    conversiones_sin_impacto = conversiones_con_impacto - efecto_total
                else:
                    # Estimar basado en efecto total
                    conversiones_sin_impacto = max(100, abs(efecto_total) * 5)
                    conversiones_con_impacto = conversiones_sin_impacto + efecto_total

                fig_monetary = plot_monetary_impact(
                    efecto_conversiones=efecto_total,
                    conversiones_sin_impacto=conversiones_sin_impacto,
                    conversiones_con_impacto=conversiones_con_impacto,
                    ingresos_totales=ingresos_totales,
                    compras_totales=compras_totales,
                    nombre_intervencion=result_1['nombre']
                )
                st.pyplot(fig_monetary, use_container_width=True)
                plt.close(fig_monetary)

            except Exception as e:
                st.warning(f"No se pudo generar el gráfico de impacto monetario: {e}")
                st.info("Asegúrate de que los datos incluyen métricas de ingresos y compras")

        # ====== TAB RESULTADO 2 ======
        with result_tabs[1]:
            if st.session_state.ci_result_2:
                result_2 = st.session_state.ci_result_2

                st.subheader(f"Resultados: {result_2['nombre']}")

                # Mostrar tipo de intervención
                tipo_interv_2 = result_2.get('tipo_intervencion', 'puntual')
                if tipo_interv_2 == 'prolongada':
                    campana_2 = result_2.get('campana', {})
                    st.info(f"""
                    **📅 Intervención Prolongada (Campaña)**
                    - Inicio: {campana_2.get('fecha_inicio', 'N/A')}
                    - Fin: {campana_2.get('fecha_fin', 'N/A')}
                    - Duración: {campana_2.get('duracion_dias', 'N/A')} días
                    """)
                else:
                    st.info(f"**📍 Intervención Puntual** - Fecha: {result_2['fecha']}")

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

                # ====== EFECTO RESIDUAL (solo para intervenciones prolongadas) ======
                if tipo_interv_2 == 'prolongada' and 'efecto_residual' in result_2:
                    residual_2 = result_2['efecto_residual']
                    st.markdown("---")
                    st.subheader("🔄 Efecto Residual (Post-Campaña)")

                    if residual_2.get('disponible', False):
                        st.info(f"""
                        **Período analizado:** {residual_2.get('periodo_inicio', 'N/A')} a {residual_2.get('periodo_fin', 'N/A')} ({residual_2.get('dias', 0)} días después de la campaña)
                        """)

                        res_data_2 = residual_2.get('resultados', {})
                        res_metricas_2 = res_data_2.get('metricas', {})
                        res_stats_2 = res_data_2.get('estadisticas', {})

                        col_r1, col_r2, col_r3, col_r4 = st.columns(4)

                        with col_r1:
                            st.metric(
                                "Efecto Diario Residual",
                                f"{res_metricas_2.get('efecto_diario', 0):+.1f}",
                                delta=f"{res_metricas_2.get('cambio_porcentual', 0):+.1f}%"
                            )

                        with col_r2:
                            st.metric(
                                "Efecto Total Residual",
                                f"{res_metricas_2.get('efecto_total', 0):+,.0f}"
                            )

                        with col_r3:
                            sig_residual_2 = "Sí ✅" if res_stats_2.get('es_significativo', False) else "No ❌"
                            st.metric(
                                "Significativo",
                                sig_residual_2,
                                delta=f"p={res_stats_2.get('p_value', 1):.4f}"
                            )

                        with col_r4:
                            st.metric(
                                "Cambio % Residual",
                                f"{res_metricas_2.get('cambio_porcentual', 0):+.1f}%"
                            )

                        # Interpretación del efecto residual
                        if res_stats_2.get('es_significativo', False):
                            if res_metricas_2.get('efecto_diario', 0) > 0:
                                st.success("✅ **El efecto persiste** después de la campaña")
                            else:
                                st.warning("⚠️ **Efecto negativo residual** detectado")
                        else:
                            st.info("ℹ️ **No hay efecto residual significativo** - El efecto desapareció al terminar la campaña")

                    else:
                        st.warning(f"⚠️ {residual_2.get('mensaje', 'No se pudo analizar el efecto residual')}")

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

                # ====== IMPACTO MONETARIO ======
                st.markdown("---")
                st.subheader("💰 Impacto Monetario")

                try:
                    # Obtener datos de ingresos
                    if 'ingresos' in df.columns and 'compras' in df.columns:
                        ingresos_totales = df['ingresos'].sum()
                        compras_totales = df['compras'].sum()
                    else:
                        ingresos_totales = df.get('ingresos', pd.Series([0])).sum()
                        compras_totales = df.get('compras', pd.Series([1])).sum()
                        if compras_totales == 0:
                            compras_totales = 1

                    efecto_total_2 = result_2['metricas']['efecto_total']

                    summary_data_2 = result_2.get('summary_data', {})
                    if 'cumulative' in summary_data_2:
                        conversiones_con_impacto_2 = summary_data_2['cumulative'].get('actual', efecto_total_2 * 2)
                        conversiones_sin_impacto_2 = conversiones_con_impacto_2 - efecto_total_2
                    else:
                        conversiones_sin_impacto_2 = max(100, abs(efecto_total_2) * 5)
                        conversiones_con_impacto_2 = conversiones_sin_impacto_2 + efecto_total_2

                    fig_monetary_2 = plot_monetary_impact(
                        efecto_conversiones=efecto_total_2,
                        conversiones_sin_impacto=conversiones_sin_impacto_2,
                        conversiones_con_impacto=conversiones_con_impacto_2,
                        ingresos_totales=ingresos_totales,
                        compras_totales=compras_totales,
                        nombre_intervencion=result_2['nombre']
                    )
                    st.pyplot(fig_monetary_2, use_container_width=True)
                    plt.close(fig_monetary_2)

                except Exception as e:
                    st.warning(f"No se pudo generar el gráfico de impacto monetario: {e}")

            else:
                st.info("No se configuró una segunda intervención")

        # ====== TAB COMPARACIÓN ======
        with result_tabs[2]:
            st.subheader("📊 Comparación de Intervenciones")

            if st.session_state.comparison_df is not None and len(st.session_state.comparison_df) > 1:
                comparison_df = st.session_state.comparison_df

                # Tabla comparativa
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                # Gráfico timeline con ambas intervenciones
                st.markdown("---")
                st.subheader("📈 Serie Temporal con Intervenciones")

                try:
                    result_1 = st.session_state.ci_result_1
                    result_2 = st.session_state.ci_result_2

                    # Preparar datos de intervenciones
                    int1_data = {
                        'fecha': result_1['fecha'],
                        'nombre': result_1['nombre'],
                        'fecha_fin': result_1.get('campana', {}).get('fecha_fin', result_1['fecha'])
                    }

                    int2_data = None
                    if result_2 is not None and isinstance(result_2, dict) and 'fecha' in result_2:
                        int2_data = {
                            'fecha': result_2['fecha'],
                            'nombre': result_2['nombre'],
                            'fecha_fin': result_2.get('campana', {}).get('fecha_fin', result_2['fecha'])
                        }
                        st.caption(f"📍 Intervención 1: {int1_data['nombre']} | 📍 Intervención 2: {int2_data['nombre']}")
                    else:
                        st.caption(f"📍 Intervención: {int1_data['nombre']}")

                    fig_timeline = plot_dual_intervention_timeline(
                        data=df,
                        intervention_1=int1_data,
                        intervention_2=int2_data,
                        response_variable=response_variable
                    )
                    st.pyplot(fig_timeline, use_container_width=True)
                    plt.close(fig_timeline)

                except Exception as e:
                    st.error(f"Error generando gráfico timeline: {e}")
                    import traceback
                    with st.expander("Ver detalles del error"):
                        st.code(traceback.format_exc())

                # Gráfico de comparación barras
                st.markdown("---")
                st.subheader("📊 Comparación de Efectos")

                try:
                    fig_comp = plot_intervention_comparison(comparison_df)
                    st.pyplot(fig_comp, use_container_width=True)
                    plt.close(fig_comp)
                except Exception as e:
                    st.error(f"Error generando gráfico de comparación: {e}")

                # Gráfico de comparación monetaria
                st.markdown("---")
                st.subheader("💰 Comparación de Impacto Monetario")

                try:
                    # Obtener datos de ingresos
                    if 'ingresos' in df.columns and 'compras' in df.columns:
                        ingresos_totales = df['ingresos'].sum()
                        compras_totales = df['compras'].sum()
                    else:
                        ingresos_totales = df.get('ingresos', pd.Series([0])).sum()
                        compras_totales = df.get('compras', pd.Series([1])).sum()
                        if compras_totales == 0:
                            compras_totales = 1

                    # Preparar datos de intervenciones
                    intervenciones_data = []
                    result_1 = st.session_state.ci_result_1
                    result_2 = st.session_state.ci_result_2

                    if result_1:
                        intervenciones_data.append({
                            'nombre': result_1['nombre'],
                            'efecto_total': result_1['metricas']['efecto_total'],
                            'significativo': result_1['estadisticas']['es_significativo']
                        })

                    if result_2 is not None and isinstance(result_2, dict):
                        intervenciones_data.append({
                            'nombre': result_2['nombre'],
                            'efecto_total': result_2['metricas']['efecto_total'],
                            'significativo': result_2['estadisticas']['es_significativo']
                        })

                    if len(intervenciones_data) >= 2:
                        fig_monetary_comp = plot_monetary_comparison(
                            intervenciones=intervenciones_data,
                            ingresos_totales=ingresos_totales,
                            compras_totales=compras_totales
                        )
                        st.pyplot(fig_monetary_comp, use_container_width=True)
                        plt.close(fig_monetary_comp)
                    else:
                        st.info("Se necesitan 2 intervenciones para comparar el impacto monetario")

                except Exception as e:
                    st.warning(f"No se pudo generar el gráfico de comparación monetaria: {e}")

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
