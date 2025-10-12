"""
Página de diagnóstico para CausalImpact
Guardar como: pages/3_🧪_Test_CausalImpact.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Test CausalImpact",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Diagnóstico de CausalImpact")
st.markdown("Herramienta para diagnosticar problemas con el análisis de Causal Impact")

# ============================================================================
# TEST 1: Verificar instalación
# ============================================================================
st.header("1️⃣ Verificar Instalación")

try:
    from causalimpact import CausalImpact
    st.success("✅ pycausalimpact está instalado")
    
    try:
        import pkg_resources
        version = pkg_resources.get_distribution("pycausalimpact").version
        st.info(f"Versión: {version}")
    except:
        st.warning("Versión: No se pudo determinar")
except ImportError:
    st.error("❌ pycausalimpact NO está instalado")
    st.code("pip install pycausalimpact")
    st.stop()

# ============================================================================
# TEST 2: Crear datos de prueba
# ============================================================================
st.header("2️⃣ Crear Datos de Prueba")

if st.button("🎲 Generar Datos de Prueba", type="primary"):
    np.random.seed(42)
    
    # Generar 90 días
    dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
    baseline = 1000
    
    # Pre: normal, Post: +20%
    pre_data = baseline + np.random.normal(0, 100, 60)
    post_data = baseline * 1.2 + np.random.normal(0, 100, 30)
    sessions = np.concatenate([pre_data, post_data])
    
    df = pd.DataFrame({
        'date': dates,
        'sessions': sessions
    })
    
    st.session_state['test_data'] = df
    st.session_state['intervention_date'] = pd.Timestamp('2024-03-01')
    
    st.success(f"✅ Generados {len(df)} días de datos")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Días totales", len(df))
    with col2:
        st.metric("Media Pre", f"{sessions[:60].mean():.0f}")
    with col3:
        st.metric("Media Post", f"{sessions[60:].mean():.0f}")
    
    st.info("📊 Cambio real simulado: **+20%**")

# ============================================================================
# TEST 3: Mostrar datos
# ============================================================================
if 'test_data' in st.session_state:
    st.header("3️⃣ Datos Generados")
    
    df = st.session_state['test_data']
    intervention = st.session_state['intervention_date']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Vista de Tabla")
        st.dataframe(df, height=300)
    
    with col2:
        st.subheader("📈 Gráfico")
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['sessions'],
            mode='lines',
            name='Sesiones'
        ))
        
        fig.add_vline(
            x=intervention,
            line_dash="dash",
            line_color="red",
            annotation_text="Intervención"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TEST 4: Ejecutar CausalImpact
# ============================================================================
if 'test_data' in st.session_state:
    st.header("4️⃣ Ejecutar CausalImpact")
    
    if st.button("🚀 Ejecutar Análisis", type="primary"):
        
        df = st.session_state['test_data']
        intervention = st.session_state['intervention_date']
        
        with st.spinner("Ejecutando CausalImpact..."):
            
            # Preparar datos
            df_analysis = df.set_index('date')[['sessions']]
            df_analysis.index = pd.DatetimeIndex(df_analysis.index, freq='D')
            
            # Definir períodos
            pre_start = df_analysis.index.min()
            pre_end = intervention - pd.Timedelta(days=1)
            post_start = intervention
            post_end = df_analysis.index.max()
            
            pre_period = [pre_start, pre_end]
            post_period = [post_start, post_end]
            
            st.info(f"""
            **Períodos:**
            - Pre: {pre_start.date()} a {pre_end.date()} ({(pre_end - pre_start).days + 1} días)
            - Post: {post_start.date()} a {post_end.date()} ({(post_end - post_start).days + 1} días)
            """)
            
            # Ejecutar
            try:
                ci = CausalImpact(
                    df_analysis,
                    pre_period,
                    post_period,
                    model_args={'nseasons': 7}
                )
                st.success("✅ CausalImpact ejecutado con nseasons=7")
            except TypeError:
                st.warning("⚠️ TypeError con nseasons, intentando sin ese parámetro...")
                ci = CausalImpact(
                    df_analysis,
                    pre_period,
                    post_period
                )
                st.success("✅ CausalImpact ejecutado (sin nseasons)")
            
            st.session_state['ci_result'] = ci

# ============================================================================
# TEST 5: Examinar resultado
# ============================================================================
if 'ci_result' in st.session_state:
    st.header("5️⃣ Examinar Resultado")
    
    ci = st.session_state['ci_result']
    
    # Atributos
    with st.expander("🔍 Ver Atributos del Objeto", expanded=False):
        attrs = [attr for attr in dir(ci) if not attr.startswith('_')]
        st.write("Atributos disponibles:")
        st.code(", ".join(attrs[:30]))
    
    # Summary
    st.subheader("📊 Summary")
    
    if hasattr(ci, 'summary_data'):
        st.success("✅ Encontrado: ci.summary_data")
        summary = ci.summary_data
        st.write(f"**Type:** {type(summary)}")
        
        if isinstance(summary, pd.DataFrame):
            st.write(f"**Shape:** {summary.shape}")
            st.write(f"**Index:** {summary.index.tolist()}")
            st.write(f"**Columns:** {summary.columns.tolist()}")
            st.dataframe(summary)
        else:
            st.write(summary)
    
    elif hasattr(ci, 'summary'):
        st.success("✅ Encontrado: ci.summary()")
        try:
            summary = ci.summary()
            st.write(f"**Type:** {type(summary)}")
            if isinstance(summary, pd.DataFrame):
                st.dataframe(summary)
            else:
                st.write(summary)
        except Exception as e:
            st.error(f"Error llamando summary(): {e}")
    else:
        st.error("❌ No se encontró ni summary_data ni summary()")
    
    # Inferences
    st.subheader("📈 Inferences")
    
    if hasattr(ci, 'inferences'):
        st.success("✅ Encontrado: ci.inferences")
        inferences = ci.inferences
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape:** {inferences.shape}")
            st.write(f"**Columns:** {inferences.columns.tolist()}")
        
        with col2:
            st.write("**Primeras filas:**")
            st.dataframe(inferences.head(3))
        
        st.markdown("---")
        st.write("**Últimas filas:**")
        st.dataframe(inferences.tail(3))
        
        # Calcular métricas manualmente
        st.subheader("🔢 Métricas Calculadas Manualmente")
        
        intervention = st.session_state['intervention_date']
        post_mask = inferences.index >= intervention
        
        actual_post = inferences.loc[post_mask, 'response']
        pred_post = inferences.loc[post_mask, 'preds']
        
        actual_mean = actual_post.mean()
        pred_mean = pred_post.mean()
        effect_mean = actual_mean - pred_mean
        rel_effect = (effect_mean / pred_mean) * 100 if pred_mean != 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Actual Promedio", f"{actual_mean:.2f}")
        with col2:
            st.metric("Predicho Promedio", f"{pred_mean:.2f}")
        with col3:
            st.metric("Efecto Absoluto", f"{effect_mean:.2f}")
        with col4:
            st.metric("Efecto Relativo", f"{rel_effect:.2f}%")
        
        st.markdown("---")
        
        actual_sum = actual_post.sum()
        pred_sum = pred_post.sum()
        effect_sum = actual_sum - pred_sum
        rel_effect_cum = (effect_sum / pred_sum) * 100 if pred_sum != 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Actual Acumulado", f"{actual_sum:.2f}")
        with col2:
            st.metric("Predicho Acumulado", f"{pred_sum:.2f}")
        with col3:
            st.metric("Efecto Acumulado", f"{effect_sum:.2f}")
        with col4:
            st.metric("Efecto Relativo Acum.", f"{rel_effect_cum:.2f}%")
        
        # Comparar con lo esperado
        st.markdown("---")
        expected_effect = 20.0  # Sabemos que simulamos +20%
        
        if abs(rel_effect - expected_effect) < 5:
            st.success(f"✅ El efecto calculado ({rel_effect:.1f}%) está cerca del esperado (+20%)")
        else:
            st.warning(f"⚠️ El efecto calculado ({rel_effect:.1f}%) difiere del esperado (+20%)")
    
    else:
        st.error("❌ No se encontró inferences")
    
    # P-value
    st.subheader("📊 P-value")
    if hasattr(ci, 'p_value'):
        p_val = ci.p_value
        st.metric("P-value", f"{p_val:.4f}")
        
        if p_val < 0.05:
            st.success("✅ Resultado estadísticamente significativo (p < 0.05)")
        else:
            st.warning("⚠️ Resultado NO significativo (p ≥ 0.05)")
    else:
        st.error("❌ No se encontró p_value")

# ============================================================================
# RESUMEN
# ============================================================================
if 'ci_result' in st.session_state:
    st.markdown("---")
    st.header("📋 Resumen del Diagnóstico")
    
    checks = []
    
    # Check instalación
    try:
        from causalimpact import CausalImpact
        checks.append(("pycausalimpact instalado", True))
    except:
        checks.append(("pycausalimpact instalado", False))
    
    # Check ejecución
    if 'ci_result' in st.session_state:
        checks.append(("CausalImpact ejecutado", True))
    else:
        checks.append(("CausalImpact ejecutado", False))
    
    # Check summary
    ci = st.session_state.get('ci_result')
    if ci and (hasattr(ci, 'summary_data') or hasattr(ci, 'summary')):
        checks.append(("Summary disponible", True))
    else:
        checks.append(("Summary disponible", False))
    
    # Check inferences
    if ci and hasattr(ci, 'inferences'):
        checks.append(("Inferences disponible", True))
    else:
        checks.append(("Inferences disponible", False))
    
    # Mostrar checks
    for name, status in checks:
        if status:
            st.success(f"✅ {name}")
        else:
            st.error(f"❌ {name}")
    
    st.markdown("---")
    
    # Recomendaciones
    st.info("""
    **💡 Información Clave para tu Código:**
    
    1. Usa `ci.inferences` para acceder a los datos
    2. Las columnas son: `response`, `preds`, `preds_lower`, `preds_upper`
    3. Si `summary_data` está vacío, calcula manualmente desde `inferences`
    4. Filtra el período post usando: `inferences.index >= intervention_date`
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("AccurateMetrics - Herramienta de Diagnóstico CausalImpact")
