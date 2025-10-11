"""
Herramienta de diagnóstico OAuth para AccurateMetrics
"""
import streamlit as st
import sys
from src.utils.config import get_redirect_uri, get_client_config

st.set_page_config(
    page_title="Diagnóstico - AccurateMetrics",
    page_icon="🔍",
    layout="wide"
)

# ============================================================================
# HEADER
# ============================================================================
st.title("🔍 Diagnóstico del Sistema")
st.markdown("Herramienta técnica para verificar la configuración OAuth y conexiones")
st.markdown("---")

# ============================================================================
# TABS PARA ORGANIZAR INFORMACIÓN
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumen", "🔐 OAuth", "🔧 Sistema", "🧪 Test"])

# ============================================================================
# TAB 1: RESUMEN
# ============================================================================
with tab1:
    st.header("Verificación Rápida")
    
    checks = []
    
    # Check 1: Secrets
    try:
        if 'oauth' in st.secrets or 'google_oauth' in st.secrets:
            checks.append(("Secrets OAuth", True, "Configurados correctamente"))
        else:
            checks.append(("Secrets OAuth", False, "No encontrados"))
    except Exception as e:
        checks.append(("Secrets OAuth", False, str(e)))
    
    # Check 2: Redirect URI
    try:
        uri = get_redirect_uri()
        if uri and uri.startswith('http'):
            checks.append(("Redirect URI", True, uri[:50] + "..."))
        else:
            checks.append(("Redirect URI", False, "URI inválida"))
    except Exception as e:
        checks.append(("Redirect URI", False, str(e)))
    
    # Check 3: Client Config
    try:
        config = get_client_config()
        if config and 'web' in config:
            checks.append(("Client Config", True, "Estructura correcta"))
        else:
            checks.append(("Client Config", False, "Estructura inválida"))
    except Exception as e:
        checks.append(("Client Config", False, str(e)))
    
    # Check 4: Librerías
    try:
        import google.auth
        import google_auth_oauthlib
        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.admin import AnalyticsAdminServiceClient
        checks.append(("Librerías Google", True, "Todas disponibles"))
    except ImportError as e:
        checks.append(("Librerías Google", False, f"Falta: {str(e)}"))
    
    # Mostrar checks
    col1, col2 = st.columns([1, 3])
    
    all_ok = all(check[1] for check in checks)
    
    with col1:
        if all_ok:
            st.success("### ✅ Todo OK")
        else:
            st.error("### ⚠️ Problemas detectados")
    
    with col2:
        for name, status, detail in checks:
            if status:
                st.success(f"**{name}**: {detail}")
            else:
                st.error(f"**{name}**: {detail}")
    
    if all_ok:
        st.markdown("---")
        st.info("✨ El sistema está correctamente configurado y listo para usar")

# ============================================================================
# TAB 2: OAUTH
# ============================================================================
with tab2:
    st.header("Configuración OAuth")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Secrets")
        
        try:
            secrets_key = None
            if 'oauth' in st.secrets:
                secrets_key = 'oauth'
                st.success("Usando configuración **[oauth]**")
            elif 'google_oauth' in st.secrets:
                secrets_key = 'google_oauth'
                st.warning("Usando configuración **[google_oauth]** (legacy)")
            else:
                st.error("No se encontraron secrets OAuth")
            
            if secrets_key:
                st.markdown("**Valores configurados:**")
                
                data = {
                    "client_id": st.secrets[secrets_key]['client_id'][:30] + "...",
                    "project_id": st.secrets[secrets_key]['project_id'],
                    "redirect_uri": st.secrets[secrets_key]['redirect_uri']
                }
                
                for key, value in data.items():
                    st.code(f"{key}:\n{value}")
        
        except Exception as e:
            st.error("Error leyendo secrets")
            st.code(str(e))
    
    with col2:
        st.subheader("🔗 Redirect URI")
        
        try:
            redirect_uri = get_redirect_uri()
            
            st.markdown("**URI configurada:**")
            st.code(redirect_uri)
            
            # Análisis
            metrics_data = {
                "Longitud": len(redirect_uri),
                "Protocolo": "HTTPS" if redirect_uri.startswith('https://') else "HTTP",
                "Termina en /": "Sí" if redirect_uri.endswith('/') else "No"
            }
            
            for key, value in metrics_data.items():
                st.metric(key, value)
        
        except Exception as e:
            st.error("Error obteniendo redirect URI")
            st.code(str(e))
    
    st.markdown("---")
    
    # URIs para Google Cloud Console
    st.subheader("☁️ Configuración en Google Cloud Console")
    
    st.info("Asegúrate de que estas URIs estén en tu OAuth Client ID")
    
    try:
        uri = get_redirect_uri()
        uri_sin_barra = uri.rstrip('/')
        uri_con_barra = uri_sin_barra + '/'
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.code(uri_sin_barra)
        with col2:
            st.code(uri_con_barra)
        with col3:
            st.code("http://localhost:8501")
        
        st.markdown(f"""
        **Pasos:**
        1. Ve a [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
        2. Edita tu OAuth 2.0 Client ID
        3. Añade las 3 URIs mostradas arriba
        4. Guarda y espera 5 minutos
        """)
    
    except Exception as e:
        st.error("No se pudo generar las URIs")

# ============================================================================
# TAB 3: SISTEMA
# ============================================================================
with tab3:
    st.header("Información del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🐍 Python")
        st.code(f"Versión: {sys.version}")
        
        st.subheader("📦 Streamlit")
        st.code(f"Versión: {st.__version__}")
    
    with col2:
        st.subheader("📚 Librerías Google")
        
        libs_to_check = {
            'google.auth': 'google-auth',
            'google_auth_oauthlib': 'google-auth-oauthlib',
            'google.analytics.data_v1beta': 'google-analytics-data',
            'google.analytics.admin': 'google-analytics-admin',
            'googleapiclient.discovery': 'google-api-python-client'
        }
        
        for module_name, display_name in libs_to_check.items():
            try:
                mod = __import__(module_name)
                version = getattr(mod, '__version__', 'OK')
                st.success(f"✅ {display_name}: {version}")
            except ImportError:
                st.error(f"❌ {display_name}: No instalado")
    
    st.markdown("---")
    
    # Client Config completo
    st.subheader("⚙️ Client Config")
    
    try:
        config = get_client_config()
        
        if config:
            # Ocultar client_secret
            safe_config = config.copy()
            safe_config['web'] = safe_config['web'].copy()
            safe_config['web']['client_secret'] = "***OCULTO***"
            
            st.json(safe_config)
        else:
            st.warning("No se pudo obtener la configuración")
    
    except Exception as e:
        st.error("Error obteniendo configuración")
        st.code(str(e))

# ============================================================================
# TAB 4: TEST
# ============================================================================
with tab4:
    st.header("Prueba de Autenticación")
    
    st.markdown("""
    Este test genera una URL de autenticación y verifica que todos los componentes 
    funcionen correctamente.
    """)
    
    if st.button("🧪 Ejecutar Test de OAuth", type="primary", use_container_width=True):
        try:
            from src.auth.google_oauth import GoogleAuthenticator
            
            with st.spinner("Generando URL de autenticación..."):
                auth = GoogleAuthenticator()
                auth_url = auth.get_authorization_url()
            
            st.success("✅ URL generada correctamente")
            
            # Análisis de la URL
            st.subheader("📋 Análisis de la URL")
            
            import urllib.parse
            parsed = urllib.parse.urlparse(auth_url)
            params = urllib.parse.parse_qs(parsed.query)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Parámetros OAuth:**")
                for key in ['client_id', 'redirect_uri', 'scope', 'response_type', 'access_type']:
                    if key in params:
                        value = params[key][0]
                        if key == 'client_id':
                            value = value[:30] + "..."
                        st.code(f"{key}:\n{value}")
            
            with col2:
                st.markdown("**Validación:**")
                
                validations = []
                
                # Validar redirect_uri
                if 'redirect_uri' in params:
                    redirect_in_url = params['redirect_uri'][0]
                    expected_uri = get_redirect_uri()
                    
                    if redirect_in_url == expected_uri:
                        validations.append(("Redirect URI", True, "Coincide con config"))
                    else:
                        validations.append(("Redirect URI", False, "NO coincide"))
                
                # Validar scopes
                if 'scope' in params:
                    scopes = params['scope'][0]
                    if 'analytics' in scopes:
                        validations.append(("Scopes Analytics", True, "Incluido"))
                    else:
                        validations.append(("Scopes Analytics", False, "No incluido"))
                
                # Validar access_type
                if 'access_type' in params:
                    if params['access_type'][0] == 'offline':
                        validations.append(("Access Type", True, "Offline (correcto)"))
                    else:
                        validations.append(("Access Type", False, "No es offline"))
                
                for name, status, detail in validations:
                    if status:
                        st.success(f"✅ {name}: {detail}")
                    else:
                        st.error(f"❌ {name}: {detail}")
            
            st.markdown("---")
            
            # Botón para probar autenticación
            st.subheader("🚀 Probar Autenticación")
            
            st.markdown("Haz click en el botón para abrir la página de Google:")
            
            st.markdown(
                f'<a href="{auth_url}" target="_blank">'
                '<button style="'
                'background: #4285f4;'
                'color: white;'
                'padding: 12px 24px;'
                'border: none;'
                'border-radius: 4px;'
                'cursor: pointer;'
                'font-size: 16px;'
                'font-weight: 500;'
                '">🔑 Autenticar con Google</button>'
                '</a>',
                unsafe_allow_html=True
            )
            
            st.markdown("---")
            
            # URL completa para referencia
            with st.expander("🔗 Ver URL completa"):
                st.code(auth_url)
        
        except Exception as e:
            st.error("❌ Error en el test")
            
            with st.expander("📋 Detalles del error"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("AccurateMetrics - Herramienta de Diagnóstico v2.0")
