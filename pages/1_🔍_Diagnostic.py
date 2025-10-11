"""
Script de diagnóstico OAuth
Página adicional de Streamlit para debugging
"""
import streamlit as st
from src.utils.config import get_redirect_uri, get_client_config

st.set_page_config(
    page_title="Diagnóstico OAuth",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Diagnóstico OAuth")
st.markdown("Herramienta de debugging para identificar problemas de autenticación")

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
    
    # Mostrar en HEX para detectar caracteres invisibles
    st.subheader("🔬 Análisis HEX (para detectar caracteres invisibles):")
    hex_repr = ' '.join([f"{ord(c):02x}" for c in uri[-10:]])
    st.code(f"Últimos 10 chars en HEX: {hex_repr}")
    
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
            
            col1, col2 = st.columns(2)
            with col1:
                st.code(f"Secrets:\n{direct_uri}")
            with col2:
                st.code(f"Función:\n{redirect_uri}")
            
            # Análisis de diferencias
            st.subheader("Diferencias detectadas:")
            if len(direct_uri) != len(redirect_uri):
                st.warning(f"Longitudes diferentes: {len(direct_uri)} vs {len(redirect_uri)}")
            
            # Mostrar diferencia carácter por carácter
            min_len = min(len(direct_uri), len(redirect_uri))
            for i in range(min_len):
                if direct_uri[i] != redirect_uri[i]:
                    st.error(f"Diferencia en posición {i}: '{direct_uri[i]}' vs '{redirect_uri[i]}'")
    
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
        for idx, uri in enumerate(config['web']['redirect_uris']):
            st.code(f"[{idx}] {uri}")
        
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

Ve a: https://console.cloud.google.com/apis/credentials

Edita tu OAuth 2.0 Client ID y añade las siguientes URIs en "URIs de redireccionamiento autorizados":
""")

if 'google_oauth' in st.secrets:
    uri = st.secrets['google_oauth']['redirect_uri']
    
    # Generar ambas versiones
    uri_sin_barra = uri.rstrip('/')
    uri_con_barra = uri_sin_barra + '/'
    
    st.code(uri_sin_barra)
    if st.button("📋 Copiar URI sin barra", key="copy1"):
        st.success("Copiado al portapapeles (usa Ctrl+C manualmente)")
    
    st.code(uri_con_barra)
    if st.button("📋 Copiar URI con barra", key="copy2"):
        st.success("Copiado al portapapeles (usa Ctrl+C manualmente)")
    
    st.code("http://localhost:8501")
    if st.button("📋 Copiar URI localhost", key="copy3"):
        st.success("Para desarrollo local")
    
    st.warning("""
    ⚠️ PASOS A SEGUIR:
    
    1. Ve a: https://console.cloud.google.com/apis/credentials
    2. Busca tu OAuth 2.0 Client ID: `107939987575-k47d8vvi6e40vcdg5uderh5jgt12j0e3`
    3. Haz click en el icono del lápiz ✏️ para editar
    4. Baja hasta "URIs de redireccionamiento autorizados"
    5. Añade las 3 URIs mostradas arriba
    6. Haz click en GUARDAR (abajo de la página)
    7. Espera 5 minutos para que los cambios se propaguen
    8. Vuelve a la página principal y prueba el login
    """)

st.markdown("---")

# 5. Test de generación de auth_url
st.header("5️⃣ Test de Auth URL")

if st.button("🧪 Generar Auth URL de prueba", type="primary"):
    try:
        from src.auth.google_oauth import GoogleAuthenticator
        
        with st.spinner("Generando URL de autenticación..."):
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
                
                st.markdown("**redirect_uri extraída de la URL:**")
                st.code(redirect_in_url)
                
                # Comparar
                expected_uri = get_redirect_uri()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("URI esperada (config)", expected_uri)
                with col2:
                    st.metric("URI en la URL generada", redirect_in_url)
                
                if redirect_in_url == expected_uri:
                    st.success("✅ La redirect_uri en la URL coincide con la configuración")
                else:
                    st.error("❌ NO coinciden - ESTE ES EL PROBLEMA")
                    st.code(f"Esperado: {expected_uri}")
                    st.code(f"En URL:   {redirect_in_url}")
            
            # Mostrar todos los parámetros
            st.subheader("Todos los parámetros OAuth:")
            for key, value in params.items():
                st.code(f"{key}: {value[0]}")
        
        st.markdown("### URL completa:")
        st.text_area("Auth URL", auth_url, height=150)
        
        st.info("""
        💡 **Cómo usar esta información:**
        
        La `redirect_uri` que aparece en esta URL debe estar EXACTAMENTE igual 
        en Google Cloud Console. Si no coincide, ese es el problema.
        
        Copia la redirect_uri de arriba y añádela en Google Cloud Console.
        """)
        
    except Exception as e:
        st.error(f"❌ Error generando auth URL: {e}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")

# 6. Información del entorno
with st.expander("📊 Información del entorno"):
    import sys
    
    st.code(f"Python: {sys.version}")
    st.code(f"Streamlit: {st.__version__}")
    
    try:
        import google.auth
        st.code(f"google-auth: {google.auth.__version__}")
    except:
        st.code("google-auth: No disponible")
    
    try:
        import google_auth_oauthlib
        st.code("google-auth-oauthlib: OK")
    except:
        st.code("google-auth-oauthlib: No disponible")

st.markdown("---")
st.caption("AccurateMetrics - Herramienta de Diagnóstico OAuth v1.0")
