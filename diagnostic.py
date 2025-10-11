"""
Script de diagnóstico OAuth
Ejecuta esto para ver exactamente qué está pasando
"""
import streamlit as st
from src.utils.config import get_redirect_uri, get_client_config

st.title("🔍 Diagnóstico OAuth")

st.markdown("---")

# 1. Verificar secrets
st.header("1️⃣ Verificación de Secrets")

if 'google_oauth' in st.secrets:
    st.success("✅ google_oauth encontrado en secrets")
    
    # Mostrar lo que hay en secrets
    st.subheader("Contenido de secrets:")
    
    secrets_data = {
        "client_id": st.secrets['google_oauth']['client_id'][:30] + "...",
        "project_id": st.secrets['google_oauth']['project_id'],
        "redirect_uri": st.secrets['google_oauth']['redirect_uri'],
    }
    
    for key, value in secrets_data.items():
        st.code(f"{key}: {value}")
    
    # Análisis de redirect_uri
    st.subheader("🔎 Análisis detallado de redirect_uri:")
    
    uri = st.secrets['google_oauth']['redirect_uri']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Longitud", len(uri))
        st.metric("Termina en /", "Sí" if uri.endswith('/') else "No")
    
    with col2:
        st.metric("Empieza con https://", "Sí" if uri.startswith('https://') else "No")
        st.code(f"Últimos 5 chars: {repr(uri[-5:])}")
    
    st.code(f"URI completa:\n{uri}")
    
else:
    st.error("❌ google_oauth NO encontrado en secrets")

st.markdown("---")

# 2. Verificar función get_redirect_uri()
st.header("2️⃣ Función get_redirect_uri()")

try:
    redirect_uri = get_redirect_uri()
    st.success(f"✅ URI obtenida: {redirect_uri}")
    
    # Comparar con el valor directo de secrets
    if 'google_oauth' in st.secrets:
        direct_uri = st.secrets['google_oauth']['redirect_uri']
        
        if redirect_uri == direct_uri:
            st.success("✅ La URI de la función coincide con secrets")
        else:
            st.error("❌ La URI de la función NO coincide con secrets")
            st.code(f"Secrets: {direct_uri}")
            st.code(f"Función: {redirect_uri}")
            st.code(f"Diferencia: {set(direct_uri) - set(redirect_uri)}")
    
except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    st.code(traceback.format_exc())

st.markdown("---")

# 3. Verificar client_config
st.header("3️⃣ Client Config")

try:
    config = get_client_config()
    
    if config:
        st.success("✅ Config obtenido correctamente")
        
        st.subheader("URIs en config:")
        st.code(config['web']['redirect_uris'])
        
        st.subheader("Config completo (sin secret):")
        safe_config = config.copy()
        safe_config['web'] = safe_config['web'].copy()
        safe_config['web']['client_secret'] = "***OCULTO***"
        st.json(safe_config)
        
    else:
        st.error("❌ No se pudo obtener config")
        
except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    st.code(traceback.format_exc())

st.markdown("---")

# 4. URIs que deberías tener en Google Cloud Console
st.header("4️⃣ URIs para Google Cloud Console")

st.info("""
**En Google Cloud Console, en tu OAuth Client ID, debes tener ESTAS URIs:**

(Copia y pega exactamente como aparecen abajo)
""")

if 'google_oauth' in st.secrets:
    uri = st.secrets['google_oauth']['redirect_uri']
    
    # Generar ambas versiones
    uri_sin_barra = uri.rstrip('/')
    uri_con_barra = uri_sin_barra + '/'
    
    st.code(uri_sin_barra)
    st.code(uri_con_barra)
    st.code("http://localhost:8501")  # Para desarrollo local
    
    st.warning("""
    ⚠️ IMPORTANTE:
    
    1. Ve a: https://console.cloud.google.com/apis/credentials
    2. Edita tu OAuth 2.0 Client ID
    3. En "URIs de redireccionamiento autorizados", añade las 3 URIs de arriba
    4. Guarda y espera 5 minutos
    5. Reinicia tu app en Streamlit Cloud
    """)

st.markdown("---")

# 5. Test de generación de auth_url
st.header("5️⃣ Test de Auth URL")

if st.button("🧪 Generar Auth URL de prueba"):
    try:
        from src.auth.google_oauth import GoogleAuthenticator
        
        auth = GoogleAuthenticator()
        auth_url = auth.get_authorization_url()
        
        st.success("✅ Auth URL generada correctamente")
        
        # Analizar la URL
        st.subheader("Análisis de la URL:")
        
        if 'redirect_uri=' in auth_url:
            # Extraer redirect_uri de la URL
            import urllib.parse
            parsed = urllib.parse.urlparse(auth_url)
            params = urllib.parse.parse_qs(parsed.query)
            
            if 'redirect_uri' in params:
                redirect_in_url = params['redirect_uri'][0]
                st.code(f"redirect_uri en la URL:\n{redirect_in_url}")
                
                # Comparar
                expected_uri = get_redirect_uri()
                
                if redirect_in_url == expected_uri:
                    st.success("✅ La redirect_uri en la URL coincide con la configuración")
                else:
                    st.error("❌ NO coinciden")
                    st.code(f"Esperado: {expected_uri}")
                    st.code(f"En URL:   {redirect_in_url}")
        
        st.markdown("### URL completa:")
        st.code(auth_url)
        
        st.info("Esta es la URL que se genera. La redirect_uri debe coincidir EXACTAMENTE con Google Cloud Console")
        
    except Exception as e:
        st.error(f"❌ Error generando auth URL: {e}")
        import traceback
        st.code(traceback.format_exc())
