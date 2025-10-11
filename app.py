"""
AccurateMetrics - Análisis de Impacto Causal con GA4
Fase 1: Autenticación y extracción de datos
"""
import streamlit as st
import sys
import os

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="AccurateMetrics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PESTAÑA DE DEBUG
# ============================================================================
DEBUG_MODE = True

if DEBUG_MODE:
    with st.sidebar:
        with st.expander("🐛 DEBUG INFO", expanded=True):
            st.write("**Python Version:**", sys.version)
            st.write("**Streamlit Version:**", st.__version__)
            
            st.write("---")
            st.write("**📦 Verificando librerías...**")
            
            libs_status = {}
            
            try:
                import google.auth
                libs_status['google-auth'] = f"✅ {google.auth.__version__}"
            except ImportError as e:
                libs_status['google-auth'] = f"❌ {str(e)}"
            
            try:
                import google_auth_oauthlib
                libs_status['google-auth-oauthlib'] = "✅ OK"
            except ImportError as e:
                libs_status['google-auth-oauthlib'] = f"❌ {str(e)}"
            
            try:
                from google.analytics.data_v1beta import BetaAnalyticsDataClient
                libs_status['google-analytics-data'] = "✅ OK"
            except ImportError as e:
                libs_status['google-analytics-data'] = f"❌ {str(e)}"
            
            try:
                from googleapiclient.discovery import build
                libs_status['google-api-python-client'] = "✅ OK"
            except ImportError as e:
                libs_status['google-api-python-client'] = f"❌ {str(e)}"
            
            for lib, status in libs_status.items():
                if "✅" in status:
                    st.success(f"{lib}: {status}")
                else:
                    st.error(f"{lib}: {status}")
            
            st.write("---")
            st.write("**🔐 Secrets:**")
            try:
                if 'google_oauth' in st.secrets:
                    st.success("✅ google_oauth configurado")
                    st.code(f"client_id: {st.secrets['google_oauth']['client_id'][:30]}...")
                    st.code(f"redirect_uri: {st.secrets['google_oauth']['redirect_uri']}")
                else:
                    st.error("❌ google_oauth NO encontrado")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================================================================
# IMPORTS PRINCIPALES
# ============================================================================
try:
    from src.auth.google_oauth import GoogleAuthenticator
    from src.data.ga4_connector import GA4Connector
    import pandas as pd
    from datetime import datetime, timedelta
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    st.error(f"❌ Error al importar módulos: {e}")
    st.stop()

# ============================================================================
# INICIALIZAR SESSION STATE
# ============================================================================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'credentials' not in st.session_state:
    st.session_state.credentials = None
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'ga4_data' not in st.session_state:
    st.session_state.ga4_data = None
if 'property_id' not in st.session_state:
    st.session_state.property_id = None

# ============================================================================
# TÍTULO PRINCIPAL
# ============================================================================
st.title("📊 AccurateMetrics")
st.markdown("### Análisis de Impacto Causal con Google Analytics 4")
st.markdown("---")

# ============================================================================
# AUTENTICADOR
# ============================================================================
auth = GoogleAuthenticator()

# ============================================================================
# REEMPLAZAR SOLO LA SECCIÓN DE SIDEBAR - AUTENTICACIÓN
# En tu app.py, reemplaza desde "with st.sidebar:" hasta antes de "if st.session_state.authenticated:"
# ============================================================================

with st.sidebar:
    st.header("🔐 Autenticación")
    
    if not st.session_state.authenticated:
        # ✅ CORREGIDO: Verificar si hay credenciales guardadas
        saved_creds = auth.load_credentials()
        if saved_creds:
            st.session_state.credentials = saved_creds
            st.session_state.user_info = auth.get_user_info(saved_creds)
            st.session_state.authenticated = True
            st.rerun()
    
    if st.session_state.authenticated:
        st.success("✅ Autenticado")
        
        if st.session_state.user_info:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.session_state.user_info.get('picture'):
                    st.image(st.session_state.user_info['picture'], width=50)
            with col2:
                st.write(f"**{st.session_state.user_info.get('name', 'Usuario')}**")
                st.caption(st.session_state.user_info.get('email', ''))
        
        st.markdown("---")
        
        if st.button("🚪 Cerrar sesión", type="secondary", use_container_width=True):
            auth.logout()
            st.session_state.authenticated = False
            st.session_state.credentials = None
            st.session_state.user_info = None
            st.session_state.ga4_data = None
            st.session_state.property_id = None
            st.rerun()
    
    else:
        st.warning("⚠️ No autenticado")
        st.markdown("---")
        
        with st.expander("📖 ¿Cómo autenticarse?"):
            st.markdown("""
            1. Click en **"Iniciar sesión con Google"**
            2. Se abrirá una ventana de autorización
            3. Acepta los permisos solicitados
            4. Serás redirigido a una página con un código
            5. **Copia todo el código** de la URL (después de `code=`)
            6. Pégalo abajo y presiona Enter
            """)
        
        # ✅ BOTÓN MEJORADO: Genera URL al cargar, no al hacer click
        try:
            # Generar URL de autorización AL CARGAR
            auth_url = auth.get_authorization_url()
            
            # Botón con link directo
            st.markdown(
                f'<a href="{auth_url}" target="_blank">'
                '<button style="background-color:#4285f4;color:white;padding:10px 20px;'
                'border:none;border-radius:4px;cursor:pointer;width:100%;font-size:16px;">'
                '🔑 Iniciar sesión con Google'
                '</button></a>',
                unsafe_allow_html=True
            )
            
            st.info("👆 Click en el botón, autoriza la app, y copia el código de la URL")
            
        except FileNotFoundError as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Asegúrate de tener `credentials.json` en la raíz o configurar secrets")
        except Exception as e:
            st.error(f"❌ Error generando URL: {str(e)}")
            
            # Debug info
            with st.expander("🔍 Debug"):
                st.code(f"Error: {type(e).__name__}: {str(e)}")
                
                # Verificar secrets
                try:
                    if 'oauth' in st.secrets:
                        st.success("✅ Secrets 'oauth' encontrados")
                        st.code(f"client_id: {st.secrets['oauth']['client_id'][:30]}...")
                        st.code(f"redirect_uri: {st.secrets['oauth']['redirect_uri']}")
                    else:
                        st.error("❌ No se encontró 'oauth' en secrets")
                        st.info("💡 Debe ser [oauth], no [google_oauth]")
                except Exception as secret_err:
                    st.error(f"Error leyendo secrets: {secret_err}")
        
        st.markdown("---")
        
        # ✅ INPUT DE CÓDIGO CORREGIDO
        auth_code = st.text_input(
            "Pega el código aquí:",
            type="password",
            placeholder="4/0AfJ...",
            help="El código completo que aparece en la URL después de 'code='",
            key="oauth_code_input"
        )
        
        if auth_code:
            with st.spinner("Autenticando..."):
                try:
                    creds = auth.authenticate_with_code(auth_code)
                    st.session_state.credentials = creds
                    st.session_state.user_info = auth.get_user_info(creds)
                    st.session_state.authenticated = True
                    
                    st.success("✅ ¡Autenticación exitosa!")
                    st.balloons()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error en autenticación: {str(e)}")
                    st.info("💡 Asegúrate de copiar el código completo")
                    
                    # Debug
                    with st.expander("🔍 Ver error completo"):
                        import traceback
                        st.code(traceback.format_exc())

# ============================================================================
# FIN DEL REEMPLAZO - El resto del código app.py sigue igual
# ============================================================================

# ============================================================================
# CONTENIDO PRINCIPAL
# ============================================================================
if st.session_state.authenticated:
    st.success("🎉 ¡Bienvenido! Conecta con Google Analytics 4 para comenzar")
    
    st.header("⚙️ Configuración de GA4")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        property_id = st.text_input(
            "Property ID de GA4",
            value=st.session_state.property_id if st.session_state.property_id else "",
            placeholder="123456789",
            help="Encuentra tu Property ID en Admin > Property Settings de Google Analytics"
        )
    
    with col2:
        st.markdown("##### 📅 Rango de Fechas")
        
        default_start = datetime.now() - timedelta(days=90)
        default_end = datetime.now() - timedelta(days=1)
        
        date_range = st.date_input(
            "Selecciona el periodo:",
            value=(default_start, default_end),
            max_value=datetime.now() - timedelta(days=1),
            help="GA4 tiene un delay de ~24-48 horas en los datos"
        )
    
    date_range_valid = len(date_range) == 2
    
    extract_button = st.button(
        "📥 Extraer Datos de GA4",
        type="primary",
        disabled=not (property_id and date_range_valid),
        use_container_width=False
    )
    
    if extract_button:
        if not property_id:
            st.error("⚠️ Por favor ingresa un Property ID")
        elif not date_range_valid:
            st.error("⚠️ Por favor selecciona un rango de fechas válido")
        else:
            try:
                with st.spinner("🔄 Conectando con Google Analytics 4..."):
                    ga4 = GA4Connector(st.session_state.credentials)
                    
                    start_date = date_range[0].strftime('%Y-%m-%d')
                    end_date = date_range[1].strftime('%Y-%m-%d')
                    
                    df = ga4.get_sessions_and_conversions(
                        property_id=property_id,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if df.empty:
                        st.warning("⚠️ No se encontraron datos para el periodo seleccionado")
                        st.info("💡 Verifica que el Property ID sea correcto y que haya datos en ese periodo")
                    else:
                        st.session_state.ga4_data = df
                        st.session_state.property_id = property_id
                        
                        st.success(f"✅ Datos extraídos correctamente: **{len(df)}** registros")
            
            except Exception as e:
                st.error(f"❌ Error al extraer datos de GA4")
                
                with st.expander("🔍 Ver detalles del error"):
                    st.code(str(e))
                
                st.info("""
                **Posibles causas:**
                - Property ID incorrecto
                - No tienes permisos de lectura en esa propiedad
                - La API de GA4 no está habilitada
                - Problemas de conexión
                """)
    
    if st.session_state.ga4_data is not None and not st.session_state.ga4_data.empty:
        st.markdown("---")
        st.header("📊 Datos Extraídos")
        
        df = st.session_state.ga4_data
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_sessions = df['sessions'].sum()
        total_conversions = df['conversions'].sum()
        conversion_rate = (total_conversions / total_sessions * 100) if total_sessions > 0 else 0
        num_days = len(df)
        
        with col1:
            st.metric("📈 Total Sesiones", f"{total_sessions:,.0f}")
        with col2:
            st.metric("🎯 Total Conversiones", f"{total_conversions:,.0f}")
        with col3:
            st.metric("📊 Tasa de Conversión", f"{conversion_rate:.2f}%")
        with col4:
            st.metric("📅 Días de Datos", f"{num_days}")
        
        st.subheader("📈 Evolución Temporal")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Sesiones Diarias', 'Conversiones Diarias'),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['sessions'],
                mode='lines',
                name='Sesiones',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['conversions'],
                mode='lines',
                name='Conversiones',
                line=dict(color='#ff7f0e', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 127, 14, 0.1)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        fig.update_yaxes(title_text="Sesiones", row=1, col=1)
        fig.update_yaxes(title_text="Conversiones", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Estadísticas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sesiones**")
            stats_sessions = pd.DataFrame({
                'Métrica': ['Promedio diario', 'Máximo', 'Mínimo', 'Desviación estándar'],
                'Valor': [
                    f"{df['sessions'].mean():,.0f}",
                    f"{df['sessions'].max():,.0f}",
                    f"{df['sessions'].min():,.0f}",
                    f"{df['sessions'].std():,.0f}"
                ]
            })
            st.dataframe(stats_sessions, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**Conversiones**")
            stats_conversions = pd.DataFrame({
                'Métrica': ['Promedio diario', 'Máximo', 'Mínimo', 'Desviación estándar'],
                'Valor': [
                    f"{df['conversions'].mean():,.2f}",
                    f"{df['conversions'].max():,.0f}",
                    f"{df['conversions'].min():,.0f}",
                    f"{df['conversions'].std():,.2f}"
                ]
            })
            st.dataframe(stats_conversions, hide_index=True, use_container_width=True)
        
        st.subheader("📋 Tabla de Datos")
        
        df_display = df.copy()
        df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d')
        df_display['sessions'] = df_display['sessions'].apply(lambda x: f"{x:,.0f}")
        df_display['conversions'] = df_display['conversions'].apply(lambda x: f"{x:,.2f}")
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name=f"ga4_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='GA4 Data')
                
                st.download_button(
                    label="📥 Descargar Excel",
                    data=buffer.getvalue(),
                    file_name=f"ga4_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except ImportError:
                pass

else:
    st.info("👈 Por favor, inicia sesión con Google en el panel lateral para continuar")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 ¿Qué es AccurateMetrics?
        
        **AccurateMetrics** es una herramienta de análisis de impacto causal que te permite:
        
        - 🔐 **Conectar de forma segura** con Google Analytics 4
        - 📊 **Extraer datos** de sesiones y conversiones
        - 📈 **Analizar el impacto causal** de tus campañas de marketing usando modelos bayesianos
        - 🎯 **Segmentar resultados** por dispositivo, canal, ciudad, etc.
        - 📉 **Predecir contrafactuales** - qué habría pasado sin la intervención
        
        ### 📚 Fundamento Científico
        
        Basado en el paper académico:
        
        > *Brodersen et al. (2015) - Inferring causal impact using Bayesian structural time-series models*
        > 
        > The Annals of Applied Statistics
        
        ### 🚀 ¿Cómo empezar?
        
        1. **Inicia sesión** con tu cuenta de Google (panel lateral)
        2. **Ingresa tu Property ID** de GA4
        3. **Selecciona el periodo** que quieres analizar
        4. **Extrae los datos** y visualiza las métricas
        5. *(Próximamente)* Define tu intervención y analiza el impacto causal
        """)
    
    with col2:
        st.markdown("### 📋 Estado del Proyecto")
        
        st.success("✅ **FASE 1**: Autenticación y Datos")
        st.info("🔄 **FASE 2**: Modelo CausalImpact")
        st.info("⏳ **FASE 3**: Segmentación Avanzada")
        
        st.markdown("---")
        
        st.markdown("### 🔒 Seguridad")
        st.markdown("""
        - Autenticación OAuth 2.0
        - Permisos solo de lectura
        - Sin almacenamiento de credenciales
        - Conexión directa con Google
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.caption("AccurateMetrics v0.1 - FASE 1 | Powered by Streamlit & Google Analytics 4")

with col2:
    st.caption("📄 [Documentación](https://github.com/EAJ-APP/accuratemetrics)")

with col3:
    st.caption("🐛 [Reportar Bug](https://github.com/EAJ-APP/accuratemetrics/issues)")
