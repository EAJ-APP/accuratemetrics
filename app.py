"""
AccurateMetrics - Análisis de Impacto Causal con GA4
Fase 1: Autenticación y extracción de datos
"""
import streamlit as st
from src.auth.google_oauth import GoogleAuthenticator
from src.data.ga4_connector import GA4Connector
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# SIDEBAR - AUTENTICACIÓN
# ============================================================================
with st.sidebar:
    st.header("🔐 Autenticación")
    
    # Verificar si ya hay credenciales guardadas
    if not st.session_state.authenticated:
        saved_creds = auth.load_credentials()
        if saved_creds:
            st.session_state.credentials = saved_creds
            st.session_state.user_info = auth.get_user_info(saved_creds)
            st.session_state.authenticated = True
            st.rerun()
    
    # Mostrar estado de autenticación
    if st.session_state.authenticated:
        st.success("✅ Autenticado")
        
        # Mostrar info del usuario
        if st.session_state.user_info:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.session_state.user_info.get('picture'):
                    st.image(st.session_state.user_info['picture'], width=50)
            with col2:
                st.write(f"**{st.session_state.user_info.get('name', 'Usuario')}**")
                st.caption(st.session_state.user_info.get('email', ''))
        
        st.markdown("---")
        
        # Botón de logout
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
        
        # Instrucciones
        with st.expander("📖 ¿Cómo autenticarse?"):
            st.markdown("""
            1. Click en **"Iniciar sesión con Google"**
            2. Se abrirá una ventana de autorización
            3. Acepta los permisos solicitados
            4. Serás redirigido a una página con un código
            5. **Copia todo el código** de la URL (después de `code=`)
            6. Pégalo abajo y presiona Enter
            """)
        
        # Botón de login
        if st.button("🔑 Iniciar sesión con Google", type="primary", use_container_width=True):
            try:
                auth_url = auth.get_authorization_url()
                st.markdown(f"### [👉 Click aquí para autenticarte]({auth_url})")
                st.info("⬆️ Click en el enlace, autoriza la app, y copia el código")
            except FileNotFoundError as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Asegúrate de tener `credentials.json` en la raíz del proyecto")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        # Input para el código de autorización
        st.markdown("---")
        auth_code = st.text_input(
            "Pega el código aquí:",
            type="password",
            placeholder="4/0AfJ...",
            help="El código completo que aparece en la URL después de 'code='"
        )
        
        if auth_code:
            with st.spinner("Autenticando..."):
                try:
                    creds = auth.authenticate_with_code(auth_code)
                    st.session_state.credentials = creds
                    st.session_state.user_info = auth.get_user_info(creds)
                    st.session_state.authenticated = True
                    st.success("✅ ¡Autenticación exitosa!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error en autenticación: {str(e)}")
                    st.info("💡 Asegúrate de copiar el código completo")

# ============================================================================
# CONTENIDO PRINCIPAL
# ============================================================================
if st.session_state.authenticated:
    st.success("🎉 ¡Bienvenido! Conecta con Google Analytics 4 para comenzar")
    
    # ========================================================================
    # SECCIÓN: CONFIGURACIÓN DE GA4
    # ========================================================================
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
        
        # Fechas por defecto: últimos 90 días
        default_start = datetime.now() - timedelta(days=90)
        default_end = datetime.now() - timedelta(days=1)
        
        date_range = st.date_input(
            "Selecciona el periodo:",
            value=(default_start, default_end),
            max_value=datetime.now() - timedelta(days=1),
            help="GA4 tiene un delay de ~24-48 horas en los datos"
        )
    
    # Verificar que se seleccionó un rango válido
    date_range_valid = len(date_range) == 2
    
    # Botón para extraer datos
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
                    # Crear conector
                    ga4 = GA4Connector(st.session_state.credentials)
                    
                    # Formatear fechas
                    start_date = date_range[0].strftime('%Y-%m-%d')
                    end_date = date_range[1].strftime('%Y-%m-%d')
                    
                    # Obtener datos
                    df = ga4.get_sessions_and_conversions(
                        property_id=property_id,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Validar datos
                    if df.empty:
                        st.warning("⚠️ No se encontraron datos para el periodo seleccionado")
                        st.info("💡 Verifica que el Property ID sea correcto y que haya datos en ese periodo")
                    else:
                        # Guardar en session state
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
    
    # ========================================================================
    # SECCIÓN: VISUALIZACIÓN DE DATOS
    # ========================================================================
    if st.session_state.ga4_data is not None and not st.session_state.ga4_data.empty:
        st.markdown("---")
        st.header("📊 Datos Extraídos")
        
        df = st.session_state.ga4_data
        
        # Métricas resumen
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
        
        # Gráfico de series temporales
        st.subheader("📈 Evolución Temporal")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Sesiones Diarias', 'Conversiones Diarias'),
