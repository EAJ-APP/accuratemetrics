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
# PESTAÑA DE DEBUG PROFESIONAL
# ============================================================================
DEBUG_MODE = True

if DEBUG_MODE:
    with st.sidebar:
        st.markdown("---")
        with st.expander("🔧 Sistema", expanded=False):
            
            # Versiones
            st.markdown("**Versiones**")
            version_data = {
                "Python": sys.version.split()[0],
                "Streamlit": st.__version__
            }
            for key, value in version_data.items():
                st.text(f"{key}: {value}")
            
            st.markdown("---")
            
            # Estado de librerías
            st.markdown("**Librerías OAuth**")
            
            libs_to_check = [
                ('google.auth', 'google-auth'),
                ('google_auth_oauthlib', 'google-auth-oauthlib'),
                ('google.analytics.data_v1beta', 'google-analytics-data'),
                ('google.analytics.admin', 'google-analytics-admin'),
                ('googleapiclient.discovery', 'google-api-python-client')
            ]
            
            all_ok = True
            for module_name, display_name in libs_to_check:
                try:
                    __import__(module_name)
                    st.text(f"✅ {display_name}")
                except ImportError:
                    st.text(f"❌ {display_name}")
                    all_ok = False
            
            if all_ok:
                st.success("Todas las librerías cargadas")
            
            st.markdown("---")
            
            # Configuración OAuth
            st.markdown("**Configuración OAuth**")
            try:
                if 'oauth' in st.secrets:
                    st.text("✅ Secrets configurados")
                    client_id = st.secrets['oauth']['client_id']
                    st.text(f"Client ID: {client_id[:20]}...")
                elif 'google_oauth' in st.secrets:
                    st.text("⚠️ Usando legacy config")
                else:
                    st.text("❌ Secrets no encontrados")
            except Exception as e:
                st.text(f"❌ Error: {str(e)[:50]}")

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
# SIDEBAR - AUTENTICACIÓN PROFESIONAL
# ============================================================================

# PRIMERO: Manejar callback OAuth ANTES de mostrar UI
query_params = st.query_params
if 'code' in query_params:
    auth_code = query_params['code']
    
    with st.spinner("🔄 Procesando autenticación..."):
        try:
            creds = auth.authenticate_with_code(auth_code)
            st.session_state.credentials = creds
            st.session_state.user_info = auth.get_user_info(creds)
            st.session_state.authenticated = True
            
            st.query_params.clear()
            st.success("✅ Autenticación completada")
            st.balloons()
            st.rerun()
            
        except Exception as e:
            st.error("❌ Error durante la autenticación")
            
            with st.expander("📋 Detalles técnicos"):
                st.code(str(e))
            
            st.query_params.clear()

# SEGUNDO: Mostrar sidebar
with st.sidebar:
    st.header("🔐 Autenticación")
    
    # Verificar credenciales guardadas
    if not st.session_state.authenticated:
        saved_creds = auth.load_credentials()
        if saved_creds:
            st.session_state.credentials = saved_creds
            st.session_state.user_info = auth.get_user_info(saved_creds)
            st.session_state.authenticated = True
            st.rerun()
    
    # Usuario autenticado
    if st.session_state.authenticated:
        st.success("✅ Sesión activa")
        
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
    
    # Usuario NO autenticado
    else:
        st.warning("⚠️ Sesión no iniciada")
        st.markdown("---")
        
        try:
            # Generar URL de autorización
            auth_url = auth.get_authorization_url()
            
            # Botón de autenticación
            if st.button("🔑 Iniciar sesión con Google", type="primary", use_container_width=True):
                st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', unsafe_allow_html=True)
            
            st.caption("Se abrirá la página de Google para autorizar el acceso")
            
        except Exception as e:
            st.error("❌ Error al inicializar autenticación")
            
            with st.expander("📋 Detalles técnicos"):
                st.code(str(e))

# ============================================================================
# CONTENIDO PRINCIPAL - CON SELECTOR DE PROPIEDADES
# ============================================================================

if st.session_state.authenticated:
    st.success("🎉 ¡Bienvenido! Selecciona tu propiedad de Google Analytics 4")
    
    st.header("⚙️ Configuración de GA4")
    
    # ==========================================
    # SELECTOR DE PROPIEDADES GA4
    # ==========================================
    
    # Importar el gestor de propiedades
    from src.data.ga4_properties import GA4PropertyManager
    
    # Inicializar session_state para propiedades
    if 'ga4_properties' not in st.session_state:
        st.session_state.ga4_properties = None
    if 'properties_loaded' not in st.session_state:
        st.session_state.properties_loaded = False
    
    # Cargar propiedades si no están cargadas
    if not st.session_state.properties_loaded:
        with st.spinner("🔄 Cargando tus propiedades de Google Analytics..."):
            try:
                property_manager = GA4PropertyManager(st.session_state.credentials)
                properties_dict = property_manager.get_properties_dict()
                
                if properties_dict:
                    st.session_state.ga4_properties = properties_dict
                    st.session_state.properties_loaded = True
                else:
                    st.warning("⚠️ No se encontraron propiedades de GA4 en tu cuenta")
                    st.info("Verifica que tengas acceso a al menos una propiedad de Google Analytics 4")
            
            except Exception as e:
                st.error(f"❌ Error al cargar propiedades: {str(e)}")
                st.info("💡 Intenta cerrar sesión y volver a autenticarte")
    
    # Mostrar selector o input manual
    if st.session_state.properties_loaded and st.session_state.ga4_properties:
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### 🎯 Selecciona una Propiedad")
            
            # Selector de propiedades
            selected_property_name = st.selectbox(
                "Propiedad de GA4:",
                options=list(st.session_state.ga4_properties.keys()),
                help="Selecciona la propiedad que quieres analizar"
            )
            
            # Obtener el ID de la propiedad seleccionada
            property_id = st.session_state.ga4_properties[selected_property_name]
            
            # Guardar en session_state
            st.session_state.property_id = property_id
            
            # Mostrar el ID para referencia
            st.caption(f"Property ID: `{property_id}`")
        
        with col2:
            st.markdown("#### 🔄")
            if st.button("Recargar Propiedades", use_container_width=True):
                st.session_state.properties_loaded = False
                st.rerun()
    
    else:
        # Fallback: Input manual si no se pudieron cargar propiedades
        st.warning("⚠️ No se pudieron cargar las propiedades automáticamente")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            property_id = st.text_input(
                "Property ID de GA4",
                value=st.session_state.property_id if st.session_state.property_id else "",
                placeholder="123456789",
                help="Encuentra tu Property ID en Admin > Property Settings de Google Analytics"
            )
            
            if property_id:
                st.session_state.property_id = property_id
        
        with col2:
            st.markdown("#### 🔄")
            if st.button("Reintentar Carga", use_container_width=True):
                st.session_state.properties_loaded = False
                st.rerun()
    
    # ==========================================
    # RANGO DE FECHAS
    # ==========================================
    
    st.markdown("#### 📅 Rango de Fechas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_start = datetime.now() - timedelta(days=90)
        start_date = st.date_input(
            "Fecha de inicio:",
            value=default_start,
            max_value=datetime.now() - timedelta(days=1),
            help="GA4 tiene un delay de ~24-48 horas en los datos"
        )
    
    with col2:
        default_end = datetime.now() - timedelta(days=1)
        end_date = st.date_input(
            "Fecha de fin:",
            value=default_end,
            max_value=datetime.now() - timedelta(days=1)
        )
    
    # Validar rango de fechas
    date_range_valid = start_date <= end_date
    
    if not date_range_valid:
        st.error("⚠️ La fecha de inicio debe ser anterior a la fecha de fin")
    
    # Mostrar resumen del período
    if date_range_valid:
        days_diff = (end_date - start_date).days + 1
        st.info(f"📊 Período seleccionado: **{days_diff} días** ({start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')})")
    
    # ==========================================
    # BOTÓN DE EXTRACCIÓN
    # ==========================================
    
    st.markdown("---")
    
    # Verificar que todo esté listo
    can_extract = bool(st.session_state.property_id) and date_range_valid
    
    if can_extract:
        if st.button("📥 Extraer Datos de GA4", type="primary", use_container_width=True):
            try:
                with st.spinner("🔄 Conectando con Google Analytics 4..."):
                    ga4 = GA4Connector(st.session_state.credentials)
                    
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    end_date_str = end_date.strftime('%Y-%m-%d')
                    
                    df = ga4.get_sessions_and_conversions(
                        property_id=st.session_state.property_id,
                        start_date=start_date_str,
                        end_date=end_date_str
                    )
                    
                    if df.empty:
                        st.warning("⚠️ No se encontraron datos para el periodo seleccionado")
                        st.info("💡 Verifica que el Property ID sea correcto y que haya datos en ese periodo")
                    else:
                        st.session_state.ga4_data = df
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
    else:
        st.info("👆 Selecciona una propiedad y un rango de fechas válido para continuar")
    
    # ==========================================
    # VISUALIZACIÓN DE DATOS
    # ==========================================
    
    if st.session_state.ga4_data is not None and not st.session_state.ga4_data.empty:
        st.markdown("---")
        st.header("📊 Datos Extraídos")
        
        df = st.session_state.ga4_data
        
        # Métricas principales
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
        
        # Gráficos de evolución
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
        
        # Estadísticas
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
        
        # Tabla de datos
        st.subheader("📋 Tabla de Datos")
        
        df_display = df.copy()
        df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d')
        df_display['sessions'] = df_display['sessions'].apply(lambda x: f"{x:,.0f}")
        df_display['conversions'] = df_display['conversions'].apply(lambda x: f"{x:,.2f}")
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Botones de descarga
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

# ============================================================================
# CONTENIDO PARA USUARIO NO AUTENTICADO
# ============================================================================

else:
    # Usuario no autenticado - Vista profesional
    
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h2>🔐 Acceso Requerido</h2>
        <p style="font-size: 18px; color: #666; margin-top: 20px;">
            Inicia sesión con tu cuenta de Google para acceder a Google Analytics 4
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("""
        **Requisitos:**
        
        ✅ Cuenta de Google con acceso a Google Analytics 4  
        ✅ Permisos de lectura en al menos una propiedad GA4
        """)
        
        st.markdown("---")
        
        st.markdown("""
        **Funcionalidades:**
        
        📊 Extracción de datos de sesiones y conversiones  
        📈 Análisis de impacto causal  
        📉 Visualizaciones interactivas  
        📥 Exportación a CSV y Excel
        """)
    
    st.markdown("---")
    
    # Features en cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="padding: 20px; background: #f0f2f6; border-radius: 10px;">
            <h3>🔒 Seguro</h3>
            <p>Autenticación OAuth 2.0 de Google. Tus credenciales nunca se almacenan.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 20px; background: #f0f2f6; border-radius: 10px;">
            <h3>⚡ Rápido</h3>
            <p>Conexión directa con la API de Google Analytics 4 sin intermediarios.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="padding: 20px; background: #f0f2f6; border-radius: 10px;">
            <h3>📊 Completo</h3>
            <p>Análisis estadístico avanzado con metodología Causal Impact.</p>
        </div>
        """, unsafe_allow_html=True)

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
