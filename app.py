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
# 🐛 PESTAÑA DE DEBUG (temporal)
# ============================================================================
DEBUG_MODE = True  # Cambiar a False en producción

if DEBUG_MODE:
    with st.sidebar:
        with st.expander("🐛 DEBUG INFO", expanded=False):
            st.write("**Python Version:**", sys.version)
            st.write("**Streamlit Version:**", st.__version__)
            
            # Verificar instalación de librerías
            st.write("---")
            st.write("**📦 Librerías Instaladas:**")
            
            libs_to_check = [
                ('google.auth', 'google-auth'),
                ('google_auth_oauthlib', 'google-auth-oauthlib'),
                ('google.analytics.data_v1beta', 'google-analytics-data'),
                ('googleapiclient', 'google-api-python-client'),
                ('pandas', 'pandas'),
                ('plotly', 'plotly'),
            ]
            
            for module_name, package_name in libs_to_check:
                try:
                    module = __import__(module_name.split('.')[0])
                    if hasattr(module, '__version__'):
                        st.success(f"✅ {package_name}: {module.__version__}")
                    else:
                        st.success(f"✅ {package_name}: installed")
                except ImportError as e:
                    st.error(f"❌ {package_name}: NOT INSTALLED")
                    st.code(str(e))
            
            # Verificar secrets
            st.write("---")
            st.write("**🔐 Secrets:**")
            try:
                if 'google_oauth' in st.secrets:
                    st.success("✅ google_oauth configurado")
                    st.write("- client_id:", st.secrets['google_oauth']['client_id'][:20] + "...")
                    st.write("- redirect_uri:", st.secrets['google_oauth']['redirect_uri'])
                else:
                    st.warning("⚠️ google_oauth NO encontrado en secrets")
            except Exception as e:
                st.error(f"❌ Error al leer secrets: {e}")
            
            # Verificar archivos
            st.write("---")
            st.write("**📁 Archivos:**")
            st.write("- Working Directory:", os.getcwd())
            
            files_to_check = ['credentials.json', 'token.json']
            for file in files_to_check:
                if os.path.exists(file):
                    st.success(f"✅ {file} existe")
                else:
                    st.info(f"ℹ️ {file} no existe (normal en Cloud)")
            
            # Session state
            st.write("---")
            st.write("**💾 Session State:**")
            st.write("- authenticated:", st.session_state.get('authenticated', False))
            st.write("- has credentials:", st.session_state.get('credentials') is not None)
            st.write("- has user_info:", st.session_state.get('user_info') is not None)
            st.write("- has ga4_data:", st.session_state.get('ga4_data') is not None)


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
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        # Gráfico de sesiones
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
        
        # Gráfico de conversiones
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
        
        # Configurar layout
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
        
        # Estadísticas adicionales
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
        
        # Formatear DataFrame para mostrar
        df_display = df.copy()
        df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d')
        df_display['sessions'] = df_display['sessions'].apply(lambda x: f"{x:,.0f}")
        df_display['conversions'] = df_display['conversions'].apply(lambda x: f"{x:,.2f}")
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Botón de descarga
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
            # Preparar Excel (opcional, requiere openpyxl)
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
    # ========================================================================
    # PANTALLA DE BIENVENIDA (NO AUTENTICADO)
    # ========================================================================
    st.info("👈 Por favor, inicia sesión con Google en el panel lateral para continuar")
    
    # Información sobre la app
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
    st.caption("📄 [Documentación](https://github.com/tuusuario/accuratemetrics)")

with col3:
    st.caption("🐛 [Reportar Bug](https://github.com/tuusuario/accuratemetrics/issues)")
