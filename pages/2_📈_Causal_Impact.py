"""
Página de Análisis de Impacto Causal
AccurateMetrics - Fase 2
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Causal Impact - AccurateMetrics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# VERIFICAR DEPENDENCIAS
# ============================================================================
try:
    from src.analysis.causal_impact import CausalImpactAnalyzer
    from src.visualization.impact_plots import ImpactVisualizer
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    st.error(f"❌ Error importando módulos: {e}")
    st.info("""
    Por favor, asegúrate de:
    1. Instalar las dependencias: `pip install -r requirements.txt`
    2. Verificar que pycausalimpact esté instalado: `pip install pycausalimpact`
    """)
    st.stop()

# ============================================================================
# TÍTULO Y DESCRIPCIÓN
# ============================================================================
st.title("📈 Análisis de Impacto Causal")
st.markdown("""
Analiza el impacto de intervenciones en tus métricas de Google Analytics usando 
la metodología **Causal Impact** de Google.
""")

# ============================================================================
# VERIFICAR DATOS
# ============================================================================
if 'ga4_data' not in st.session_state or st.session_state.ga4_data is None:
    st.warning("⚠️ No hay datos de GA4 cargados")
    st.info("""
    👉 Por favor, ve a la página principal y:
    1. Autentícate con Google
    2. Selecciona una propiedad de GA4
    3. Extrae los datos
    """)
    
    if st.button("🏠 Ir a la página principal"):
        st.switch_page("app.py")
    st.stop()

# ============================================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuración del Análisis")
    
    # Selección de métrica
    st.subheader("📊 Métrica a Analizar")
    metric_column = st.selectbox(
        "Selecciona la métrica:",
        options=['sessions', 'conversions'],
        format_func=lambda x: 'Sesiones' if x == 'sessions' else 'Conversiones',
        help="Elige qué métrica quieres analizar"
    )
    
    st.markdown("---")
    
    # Fecha de intervención
    st.subheader("📅 Fecha de Intervención")
    
    # Obtener rango de fechas disponibles
    df = st.session_state.ga4_data
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    # Calcular fecha por defecto (mitad del período)
    days_range = (max_date - min_date).days
    default_intervention = min_date + timedelta(days=days_range // 2)
    
    intervention_date = st.date_input(
        "¿Cuándo ocurrió la intervención?",
        value=default_intervention,
        min_value=min_date + timedelta(days=14),  # Necesitamos al menos 2 semanas pre
        max_value=max_date - timedelta(days=7),   # Necesitamos al menos 1 semana post
        help="La fecha cuando implementaste el cambio que quieres medir"
    )
    
    st.markdown("---")
    
    # Configuración de períodos
    st.subheader("⏱️ Períodos de Análisis")
    
    use_custom_periods = st.checkbox(
        "Personalizar períodos",
        value=False,
        help="Por defecto se usa todo el período disponible"
    )
    
    if use_custom_periods:
        # Período pre-intervención
        max_pre_days = (intervention_date - min_date.date()).days
        pre_period_days = st.slider(
            "Días pre-intervención:",
            min_value=14,
            max_value=max_pre_days,
            value=min(30, max_pre_days),
            help="Período de entrenamiento antes de la intervención"
        )
        
        # Período post-intervención
        max_post_days = (max_date.date() - intervention_date).days
        post_period_days = st.slider(
            "Días post-intervención:",
            min_value=7,
            max_value=max_post_days,
            value=min(14, max_post_days),
            help="Período de evaluación después de la intervención"
        )
    else:
        pre_period_days = None
        post_period_days = None
    
    st.markdown("---")
    
    # Información sobre los períodos
    st.info(f"""
    **📊 Resumen de datos:**
    - Total de días: {len(df)}
    - Desde: {min_date.strftime('%d/%m/%Y')}
    - Hasta: {max_date.strftime('%d/%m/%Y')}
    """)

# ============================================================================
# ÁREA PRINCIPAL
# ============================================================================

# Tabs para organizar el contenido
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Análisis", 
    "📈 Visualizaciones", 
    "📋 Resumen", 
    "❓ Ayuda"
])

# ============================================================================
# TAB 1: ANÁLISIS
# ============================================================================
with tab1:
    st.header("🔬 Ejecutar Análisis de Impacto Causal")
    
    # Mostrar configuración actual
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Métrica", metric_column.title())
    with col2:
        st.metric("Fecha Intervención", intervention_date.strftime('%d/%m/%Y'))
    with col3:
        period_text = "Personalizado" if use_custom_periods else "Completo"
        st.metric("Período", period_text)
    
    st.markdown("---")
    
    # Botón de análisis
    if st.button("🚀 Ejecutar Análisis", type="primary", use_container_width=True):
        
        with st.spinner("🔄 Preparando datos..."):
            try:
                # Crear analizador
                analyzer = CausalImpactAnalyzer(
                    data=st.session_state.ga4_data,
                    metric_column=metric_column
                )
                
                # Validar datos
                is_valid, validation_msg = analyzer.validate_data_requirements()
                
                if not is_valid:
                    st.error(f"❌ {validation_msg}")
                    st.stop()
                
                st.success("✅ Datos validados correctamente")
            
            except Exception as e:
                st.error(f"❌ Error preparando datos: {e}")
                st.stop()
        
        with st.spinner("📈 Ejecutando análisis de Causal Impact..."):
            try:
                # Ejecutar análisis
                results = analyzer.analyze_single_intervention(
                    intervention_date=intervention_date.strftime('%Y-%m-%d'),
                    pre_period_days=pre_period_days,
                    post_period_days=post_period_days
                )
                
                # Guardar resultados en session_state
                st.session_state.causal_results = results
                st.session_state.causal_analyzer = analyzer
                st.session_state.causal_plot_data = analyzer.get_plot_data()
                st.session_state.causal_intervention_date = pd.to_datetime(intervention_date)
                
                st.success("✅ Análisis completado exitosamente")
                
            except Exception as e:
                st.error(f"❌ Error en el análisis: {e}")
                st.info("""
                Posibles causas:
                - Período pre-intervención muy corto (mínimo 14 días)
                - Datos insuficientes o con poca variabilidad
                - Fecha de intervención muy cercana a los extremos
                """)
                st.stop()
        
        # Mostrar resumen inmediato
        st.markdown("---")
        st.subheader("📊 Resultados Principales")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            efecto_promedio = results['average']['rel_effect'] * 100
            delta_color = "normal" if abs(efecto_promedio) < 5 else ("inverse" if efecto_promedio < 0 else "off")
            st.metric(
                "Efecto Promedio",
                f"{efecto_promedio:.1f}%",
                delta=f"{efecto_promedio:.1f}%",
                delta_color=delta_color
            )
        
        with col2:
            efecto_acumulado = results['cumulative']['rel_effect'] * 100
            st.metric(
                "Efecto Acumulado",
                f"{efecto_acumulado:.1f}%"
            )
        
        with col3:
            p_value = results['p_value']
            is_significant = results['is_significant']
            sig_text = "Sí ✅" if is_significant else "No ⚠️"
            st.metric(
                "Significativo",
                sig_text,
                delta=f"p={p_value:.3f}"
            )
        
        with col4:
            impacto_absoluto = results['cumulative']['abs_effect']
            st.metric(
                f"{metric_column.title()} Impacto",
                f"{impacto_absoluto:,.0f}"
            )
        
        # Resumen narrativo
        st.markdown("---")
        summary_text = analyzer.get_summary_text()
        st.markdown(summary_text)

# ============================================================================
# TAB 2: VISUALIZACIONES
# ============================================================================
with tab2:
    st.header("📈 Visualizaciones del Impacto")
    
    if 'causal_results' not in st.session_state:
        st.info("👆 Ejecuta primero el análisis en la pestaña 'Análisis'")
    else:
        visualizer = ImpactVisualizer()
        
        # Debug: Ver qué datos tenemos
        with st.expander("🔍 Debug: Ver estructura de datos", expanded=False):
            if 'causal_plot_data' in st.session_state:
                plot_data = st.session_state.causal_plot_data
                st.write(f"Tipo de plot_data: {type(plot_data)}")
                st.write(f"Shape: {plot_data.shape if hasattr(plot_data, 'shape') else 'N/A'}")
                if not plot_data.empty:
                    st.write(f"Columnas: {plot_data.columns.tolist()}")
                    st.write("Primeras 5 filas:")
                    st.dataframe(plot_data.head())
                else:
                    st.write("DataFrame está vacío")
            else:
                st.write("No hay datos de plot en session_state")
        
        # Gráfico principal
        try:
            st.subheader("1. Serie Temporal Completa")
            
            # Verificar que tenemos datos antes de graficar
            if 'causal_plot_data' in st.session_state and not st.session_state.causal_plot_data.empty:
                fig_main = visualizer.plot_impact_analysis(
                    plot_data=st.session_state.causal_plot_data,
                    intervention_date=st.session_state.causal_intervention_date,
                    metric_name=metric_column.title()
                )
                st.plotly_chart(fig_main, use_container_width=True)
            else:
                st.warning("⚠️ No hay datos suficientes para crear el gráfico principal")
        except Exception as e:
            st.error(f"Error creando gráfico principal: {str(e)}")
            with st.expander("Ver detalles del error"):
                import traceback
                st.code(traceback.format_exc())
        
        # Gráficos secundarios
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                st.subheader("2. Resumen de Efectos")
                fig_summary = visualizer.plot_summary_metrics(
                    st.session_state.causal_results
                )
                st.plotly_chart(fig_summary, use_container_width=True)
            except Exception as e:
                st.error(f"Error en gráfico de resumen: {str(e)[:100]}")
        
        with col2:
            try:
                st.subheader("3. Comparación Pre vs Post")
                fig_comparison = visualizer.plot_period_comparison(
                    data=st.session_state.ga4_data,
                    intervention_date=st.session_state.causal_intervention_date,
                    metric_column=metric_column
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            except Exception as e:
                st.error(f"Error en comparación: {str(e)[:100]}")

# ============================================================================
# TAB 3: RESUMEN
# ============================================================================
with tab3:
    st.header("📋 Resumen Detallado")
    
    if 'causal_results' not in st.session_state:
        st.info("👆 Ejecuta primero el análisis en la pestaña 'Análisis'")
    else:
        results = st.session_state.causal_results
        
        # Información de períodos
        st.subheader("📅 Períodos Analizados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Pre-intervención:**")
            st.write(f"Desde: {results['periods']['pre_period']['start']}")
            st.write(f"Hasta: {results['periods']['pre_period']['end']}")
            st.write(f"Días: {results['periods']['pre_period']['days']}")
        
        with col2:
            st.markdown("**Intervención:**")
            st.write(f"Fecha: {results['periods']['intervention_date']}")
        
        with col3:
            st.markdown("**Post-intervención:**")
            st.write(f"Desde: {results['periods']['post_period']['start']}")
            st.write(f"Hasta: {results['periods']['post_period']['end']}")
            st.write(f"Días: {results['periods']['post_period']['days']}")
        
        st.markdown("---")
        
        # Tabla de resultados detallados
        st.subheader("📊 Métricas Detalladas")
        
        # Crear DataFrame con resultados
        metrics_data = []
        
        # Métricas promedio
        avg = results['average']
        metrics_data.append({
            'Tipo': 'Promedio Diario',
            'Real': f"{avg['actual']:,.1f}",
            'Predicho': f"{avg['predicted']:,.1f}",
            'Efecto Absoluto': f"{avg['abs_effect']:,.1f}",
            'Efecto Relativo': f"{avg['rel_effect']*100:.1f}%",
            'IC Inferior': f"{avg['rel_effect_lower']*100:.1f}%",
            'IC Superior': f"{avg['rel_effect_upper']*100:.1f}%"
        })
        
        # Métricas acumuladas
        cum = results['cumulative']
        metrics_data.append({
            'Tipo': 'Acumulado Total',
            'Real': f"{cum['actual']:,.0f}",
            'Predicho': f"{cum['predicted']:,.0f}",
            'Efecto Absoluto': f"{cum['abs_effect']:,.0f}",
            'Efecto Relativo': f"{cum['rel_effect']*100:.1f}%",
            'IC Inferior': f"{cum['rel_effect_lower']*100:.1f}%",
            'IC Superior': f"{cum['rel_effect_upper']*100:.1f}%"
        })
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Interpretación
        st.subheader("🎯 Interpretación de Resultados")
        
        if results['is_significant']:
            if results['average']['rel_effect'] > 0:
                st.success("""
                ✅ **Impacto Positivo Significativo**
                
                La intervención tuvo un efecto positivo estadísticamente significativo en las {}.
                Puedes confiar en que el cambio observado no se debe al azar.
                """.format(metric_column))
            else:
                st.warning("""
                ⚠️ **Impacto Negativo Significativo**
                
                La intervención tuvo un efecto negativo estadísticamente significativo en las {}.
                El cambio observado indica una disminución real en la métrica.
                """.format(metric_column))
        else:
            st.info("""
            ℹ️ **Impacto No Significativo**
            
            No hay evidencia estadística suficiente para afirmar que la intervención 
            tuvo un efecto real en las {}. El cambio observado podría deberse al azar 
            o a la variabilidad natural de los datos.
            """.format(metric_column))
        
        # Exportar resultados
        st.markdown("---")
        st.subheader("💾 Exportar Resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Preparar datos para exportar
            export_data = st.session_state.causal_plot_data.copy()
            export_data['intervention'] = export_data.index >= st.session_state.causal_intervention_date
            
            csv = export_data.to_csv()
            st.download_button(
                label="📥 Descargar Datos (CSV)",
                data=csv,
                file_name=f"causal_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Crear resumen en texto
            report_text = f"""
REPORTE DE ANÁLISIS DE IMPACTO CAUSAL
=====================================
Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M')}

CONFIGURACIÓN
-------------
Métrica analizada: {metric_column}
Fecha de intervención: {results['periods']['intervention_date']}
Período pre-intervención: {results['periods']['pre_period']['days']} días
Período post-intervención: {results['periods']['post_period']['days']} días

RESULTADOS PRINCIPALES
----------------------
Efecto promedio: {results['average']['rel_effect']*100:.1f}%
Intervalo de confianza: [{results['average']['rel_effect_lower']*100:.1f}%, {results['average']['rel_effect_upper']*100:.1f}%]

Efecto acumulado: {results['cumulative']['abs_effect']:,.0f} {metric_column}
Efecto relativo acumulado: {results['cumulative']['rel_effect']*100:.1f}%

P-value: {results['p_value']:.4f}
Significancia estadística: {'Sí' if results['is_significant'] else 'No'}

INTERPRETACIÓN
--------------
{st.session_state.causal_analyzer.get_summary_text()}
            """
            
            st.download_button(
                label="📄 Descargar Reporte (TXT)",
                data=report_text,
                file_name=f"reporte_causal_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# ============================================================================
# TAB 4: AYUDA
# ============================================================================
with tab4:
    st.header("❓ Ayuda y Documentación")
    
    st.markdown("""
    ### 📚 ¿Qué es Causal Impact?
    
    **Causal Impact** es una metodología desarrollada por Google para estimar el efecto causal 
    de una intervención en una serie temporal. Utiliza un enfoque bayesiano para crear un 
    modelo contrafactual (qué hubiera pasado sin la intervención) y lo compara con los 
    datos observados.
    
    ### 🎯 ¿Cuándo usar este análisis?
    
    Este análisis es útil cuando quieres medir el impacto de:
    - 🚀 Lanzamiento de campañas de marketing
    - 🔄 Cambios en el sitio web
    - 📱 Actualizaciones de producto
    - 💰 Cambios de precio
    - 📰 Eventos de PR o noticias
    
    ### 📊 Requisitos de datos
    
    Para obtener resultados confiables necesitas:
    - **Mínimo 21 días de datos** (idealmente más de 30)
    - **Al menos 14 días pre-intervención** para entrenar el modelo
    - **Al menos 7 días post-intervención** para evaluar el impacto
    - **Datos con variabilidad** (no todos los valores iguales)
    
    ### 📈 Interpretación de resultados
    
    **Efecto Promedio:**
    - Cambio porcentual promedio diario en la métrica
    - Positivo = mejora, Negativo = empeoramiento
    
    **Efecto Acumulado:**
    - Suma total del impacto durante todo el período post-intervención
    - Útil para entender el impacto total en términos absolutos
    
    **P-value y Significancia:**
    - P-value < 0.05 = Resultado estadísticamente significativo
    - P-value ≥ 0.05 = No hay evidencia suficiente de impacto
    
    **Intervalos de Confianza:**
    - Rango donde probablemente está el efecto real (95% de confianza)
    - Si incluye el 0, el efecto podría no ser real
    
    ### 🔧 Consejos para mejores resultados
    
    1. **Más datos = Mejor modelo**: Intenta tener al menos 30-60 días pre-intervención
    2. **Evita múltiples cambios**: Analiza una intervención a la vez
    3. **Considera la estacionalidad**: El modelo intenta detectarla automáticamente
    4. **Valida los supuestos**: Revisa que no haya otros eventos importantes en el período
    
    ### 📖 Referencias
    
    - [Paper original de Google](https://google.github.io/CausalImpact/CausalImpact.html)
    - [Documentación de pycausalimpact](https://github.com/dafiti/causalimpact)
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("AccurateMetrics v0.2 - Análisis de Impacto Causal | Powered by Google's CausalImpact")
