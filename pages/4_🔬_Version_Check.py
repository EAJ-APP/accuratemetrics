"""
Página de verificación de versión de CausalImpact
Guardar como: pages/4_🔬_Version_Check.py
"""
import streamlit as st
import sys
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Version Check - CausalImpact",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Verificación de Versión y Estructura")
st.markdown("Herramienta para verificar qué versión de CausalImpact tienes y qué columnas genera")

st.markdown("---")

# ============================================================================
# 1. INFORMACIÓN DEL SISTEMA
# ============================================================================
st.header("1️⃣ Información del Sistema")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🐍 Python")
    st.code(f"Versión: {sys.version.split()[0]}")
    st.code(f"Completo: {sys.version}")

with col2:
    st.subheader("📦 Streamlit")
    st.code(f"Versión: {st.__version__}")

# ============================================================================
# 2. VERSIÓN DE PYCAUSALIMPACT
# ============================================================================
st.markdown("---")
st.header("2️⃣ Versión de pycausalimpact")

try:
    from causalimpact import CausalImpact
    st.success("✅ pycausalimpact está instalado")
    
    try:
        import pkg_resources
        version = pkg_resources.get_distribution("pycausalimpact").version
        st.info(f"**Versión detectada: `{version}`**")
    except:
        st.warning("⚠️ No se pudo determinar la versión exacta")
        st.caption("Esto es normal en algunos entornos")

except ImportError:
    st.error("❌ pycausalimpact NO está instalado")
    st.code("pip install pycausalimpact")
    st.stop()

# ============================================================================
# 3. TEST CON DATOS SINTÉTICOS
# ============================================================================
st.markdown("---")
st.header("3️⃣ Test con Datos Sintéticos")

st.info("Vamos a crear datos de prueba y ejecutar CausalImpact para ver qué columnas genera")

if st.button("🚀 Ejecutar Test", type="primary", use_container_width=True):
    
    with st.spinner("Generando datos de prueba..."):
        # Generar datos
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        
        # Pre: ~1000, Post: ~1200 (+20%)
        data = pd.DataFrame({
            'y': np.concatenate([
                np.random.normal(1000, 100, 60),
                np.random.normal(1200, 100, 30)
            ])
        }, index=dates)
        
        st.success(f"✅ Generados {len(data)} días de datos")
        
        # Mostrar datos
        with st.expander("📊 Ver datos generados"):
            st.dataframe(data.head(10))
            st.line_chart(data)
    
    with st.spinner("Ejecutando CausalImpact..."):
        try:
            ci = CausalImpact(
                data, 
                ['2024-01-01', '2024-02-29'], 
                ['2024-03-01', '2024-03-30']
            )
            st.success("✅ CausalImpact ejecutado correctamente")
            
            # Guardar en session_state
            st.session_state['test_ci'] = ci
            
        except Exception as e:
            st.error(f"❌ Error ejecutando CausalImpact: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

# ============================================================================
# 4. ANÁLISIS DE ESTRUCTURA
# ============================================================================
if 'test_ci' in st.session_state:
    st.markdown("---")
    st.header("4️⃣ Análisis de Estructura del Resultado")
    
    ci = st.session_state['test_ci']
    
    # Atributos disponibles
    st.subheader("📋 Atributos Disponibles")
    
    attrs = [attr for attr in dir(ci) if not attr.startswith('_')]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Atributos principales:**")
        main_attrs = ['inferences', 'summary', 'summary_data', 'p_value', 'plot']
        for attr in main_attrs:
            if hasattr(ci, attr):
                st.success(f"✅ `{attr}`")
            else:
                st.error(f"❌ `{attr}`")
    
    with col2:
        st.markdown("**Todos los atributos:**")
        with st.expander("Ver lista completa"):
            st.code(", ".join(attrs))
    
    # ========================================================================
    # INFERENCES
    # ========================================================================
    st.markdown("---")
    st.subheader("📊 Análisis de `inferences`")
    
    if hasattr(ci, 'inferences'):
        inferences = ci.inferences
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Filas", inferences.shape[0])
        with col2:
            st.metric("Columnas", inferences.shape[1])
        with col3:
            st.metric("Tipo", type(inferences).__name__)
        
        st.markdown("**Columnas disponibles:**")
        cols_df = pd.DataFrame({
            'Columna': inferences.columns.tolist(),
            'Tipo': [str(dtype) for dtype in inferences.dtypes],
            'Nulos': inferences.isnull().sum().tolist()
        })
        st.dataframe(cols_df, use_container_width=True)
        
        # Verificar columnas clave
        st.markdown("**Verificación de columnas clave:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Para predicciones:**")
            if 'point_pred' in inferences.columns:
                st.success("✅ `point_pred` (estilo Colab)")
            elif 'preds' in inferences.columns:
                st.success("✅ `preds` (estilo alternativo)")
            else:
                st.error("❌ No se encontró columna de predicciones")
        
        with col2:
            st.markdown("**Para intervalos de confianza:**")
            if 'point_pred_lower' in inferences.columns:
                st.success("✅ `point_pred_lower/upper`")
            elif 'preds_lower' in inferences.columns:
                st.success("✅ `preds_lower/upper`")
            else:
                st.warning("⚠️ No se encontraron ICs")
        
        # Mostrar primeras filas
        st.markdown("**Primeras 5 filas:**")
        st.dataframe(inferences.head(), use_container_width=True)
        
        # Mostrar últimas filas
        st.markdown("**Últimas 5 filas:**")
        st.dataframe(inferences.tail(), use_container_width=True)
        
        # Estadísticas de columnas clave
        st.markdown("**Estadísticas de columnas clave:**")
        
        # Detectar columna de predicción
        pred_col = None
        if 'point_pred' in inferences.columns:
            pred_col = 'point_pred'
        elif 'preds' in inferences.columns:
            pred_col = 'preds'
        
        if pred_col:
            stats_df = pd.DataFrame({
                'Métrica': ['Min', 'Max', 'Media', 'Std', 'Mediana'],
                pred_col: [
                    f"{inferences[pred_col].min():.2f}",
                    f"{inferences[pred_col].max():.2f}",
                    f"{inferences[pred_col].mean():.2f}",
                    f"{inferences[pred_col].std():.2f}",
                    f"{inferences[pred_col].median():.2f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
    
    else:
        st.error("❌ No se encontró atributo `inferences`")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    st.markdown("---")
    st.subheader("📈 Análisis de `summary` / `summary_data`")
    
    summary_found = False
    summary_df = None
    
    if hasattr(ci, 'summary_data'):
        st.success("✅ Usando `summary_data`")
        summary_df = ci.summary_data
        summary_found = True
    elif hasattr(ci, 'summary'):
        st.success("✅ Usando `summary()`")
        try:
            summary_df = ci.summary()
            summary_found = True
        except Exception as e:
            st.error(f"Error llamando summary(): {e}")
    
    if summary_found and summary_df is not None:
        st.markdown("**Estructura del Summary:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Filas", summary_df.shape[0])
            st.metric("Tipo", type(summary_df).__name__)
        with col2:
            st.metric("Columnas", summary_df.shape[1])
        
        st.markdown("**Index:**")
        st.code(summary_df.index.tolist())
        
        st.markdown("**Columns:**")
        st.code(summary_df.columns.tolist())
        
        st.markdown("**Contenido completo:**")
        st.dataframe(summary_df, use_container_width=True)
        
        # Verificar si tiene valores
        if (summary_df == 0).all().all():
            st.warning("⚠️ ADVERTENCIA: Todos los valores son 0")
            st.info("Esto significa que hay que calcular desde `inferences`")
        else:
            st.success("✅ El summary tiene valores válidos")
    
    else:
        st.error("❌ No se encontró summary ni summary_data")
    
    # ========================================================================
    # P-VALUE
    # ========================================================================
    st.markdown("---")
    st.subheader("📊 P-value")
    
    if hasattr(ci, 'p_value'):
        p_val = ci.p_value
        st.success(f"✅ P-value disponible: **{p_val:.4f}**")
        
        if p_val < 0.05:
            st.success("🎯 Resultado estadísticamente significativo (p < 0.05)")
        else:
            st.warning("⚠️ Resultado NO significativo (p ≥ 0.05)")
    else:
        st.error("❌ No se encontró p_value")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    st.markdown("---")
    st.header("5️⃣ Resumen de Compatibilidad")
    
    st.markdown("### ✅ Lo que tu versión tiene:")
    
    checks = []
    
    if hasattr(ci, 'inferences'):
        checks.append("✅ `inferences` - DataFrame con resultados diarios")
    
    if 'preds' in ci.inferences.columns:
        checks.append("✅ Columna `preds` para predicciones")
    elif 'point_pred' in ci.inferences.columns:
        checks.append("✅ Columna `point_pred` para predicciones")
    
    if 'preds_lower' in ci.inferences.columns:
        checks.append("✅ Columnas `preds_lower/upper` para ICs")
    elif 'point_pred_lower' in ci.inferences.columns:
        checks.append("✅ Columnas `point_pred_lower/upper` para ICs")
    
    if hasattr(ci, 'summary_data') or hasattr(ci, 'summary'):
        checks.append("✅ Método `summary` disponible")
    
    if hasattr(ci, 'p_value'):
        checks.append("✅ Atributo `p_value` disponible")
    
    for check in checks:
        st.markdown(check)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Configuración recomendada para tu código:")
    
    # Generar código recomendado
    if 'point_pred' in ci.inferences.columns:
        pred_col = 'point_pred'
        lower_col = 'point_pred_lower'
        upper_col = 'point_pred_upper'
    else:
        pred_col = 'preds'
        lower_col = 'preds_lower'
        upper_col = 'preds_upper'
    
    recommended_code = f"""
# Para tu versión de pycausalimpact:

# Obtener predicciones:
predicted = ci.inferences['{pred_col}']

# Obtener intervalos de confianza:
pred_lower = ci.inferences['{lower_col}']
pred_upper = ci.inferences['{upper_col}']

# Obtener valores reales (hay que añadirlos manualmente):
actual = data.loc[ci.inferences.index, 'tu_metrica']

# Calcular efecto:
effect = actual - predicted
"""
    
    st.code(recommended_code, language='python')
    
    # ========================================================================
    # EXPORTAR INFORMACIÓN
    # ========================================================================
    st.markdown("---")
    st.subheader("💾 Exportar Información")
    
    export_text = f"""
INFORMACIÓN DE VERSIÓN DE CAUSALIMPACT
{'='*60}

Python: {sys.version.split()[0]}
Streamlit: {st.__version__}

ATRIBUTOS DISPONIBLES:
{', '.join(attrs[:20])}

COLUMNAS EN INFERENCES:
{ci.inferences.columns.tolist()}

ESTRUCTURA DEL SUMMARY:
Index: {summary_df.index.tolist() if summary_df is not None else 'N/A'}
Columns: {summary_df.columns.tolist() if summary_df is not None else 'N/A'}

P-VALUE:
{ci.p_value if hasattr(ci, 'p_value') else 'N/A'}

COLUMNAS RECOMENDADAS:
- Predicción: {pred_col}
- IC Inferior: {lower_col}
- IC Superior: {upper_col}
"""
    
    st.download_button(
        label="📥 Descargar Informe de Versión",
        data=export_text,
        file_name="causalimpact_version_info.txt",
        mime="text/plain",
        use_container_width=True
    )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("AccurateMetrics - Verificación de Versión de CausalImpact")
