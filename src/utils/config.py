"""
Configuración de la aplicación - CON DEBUG DE URI
"""
import streamlit as st

# Paths (solo para local)
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

# Scopes básicos (sin Analytics por ahora)
SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]

def get_redirect_uri():
    """Obtener redirect URI según el entorno - CON DEBUG"""
    try:
        if 'google_oauth' in st.secrets:
            uri = st.secrets['google_oauth']['redirect_uri']
            
            # ⚠️ IMPORTANTE: NO modificar la URI aquí
            # Debe coincidir EXACTAMENTE con Google Cloud Console
            
            # DEBUG: Mostrar en consola
            print("=" * 60)
            print("🔍 DEBUG REDIRECT URI")
            print("=" * 60)
            print(f"URI desde secrets: '{uri}'")
            print(f"Longitud: {len(uri)}")
            print(f"Termina en /: {uri.endswith('/')}")
            print(f"Caracteres finales: {repr(uri[-5:])}")
            print("=" * 60)
            
            return uri
    except Exception as e:
        print(f"⚠️ Error leyendo secrets: {e}")
    
    # Fallback a localhost
    return 'http://localhost:8501'

def get_client_config():
    """Obtener configuración OAuth desde secrets"""
    try:
        if 'google_oauth' in st.secrets:
            redirect_uri = get_redirect_uri()
            
            config = {
                "web": {
                    "client_id": st.secrets["google_oauth"]["client_id"],
                    "client_secret": st.secrets["google_oauth"]["client_secret"],
                    "project_id": st.secrets["google_oauth"]["project_id"],
                    "auth_uri": st.secrets["google_oauth"]["auth_uri"],
                    "token_uri": st.secrets["google_oauth"]["token_uri"],
                    "auth_provider_x509_cert_url": st.secrets["google_oauth"]["auth_provider_x509_cert_url"],
                    "redirect_uris": [redirect_uri]
                }
            }
            
            # DEBUG adicional
            print(f"✅ Client ID: {config['web']['client_id'][:30]}...")
            print(f"✅ Redirect URI en config: '{redirect_uri}'")
            
            return config
            
    except Exception as e:
        print(f"❌ Error cargando secrets: {e}")
        import traceback
        traceback.print_exc()
    
    return None
