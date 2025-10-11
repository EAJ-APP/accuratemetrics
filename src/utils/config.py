"""
Configuración de la aplicación
"""
import streamlit as st

# Paths (solo para local)
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

# OAuth Scopes
SCOPES = [
    'https://www.googleapis.com/auth/analytics.readonly',
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]

# OAuth Redirect URI
def get_redirect_uri():
    """Obtener redirect URI según el entorno"""
    try:
        if 'google_oauth' in st.secrets:
            uri = st.secrets['google_oauth']['redirect_uri']
            # IMPORTANTE: Quitar barra final si existe
            return uri.rstrip('/')
    except:
        pass
    
    # Fallback a localhost
    return 'http://localhost:8501'

# Google OAuth Client Config
def get_client_config():
    """Obtener configuración OAuth desde secrets o archivo"""
    try:
        if 'google_oauth' in st.secrets:
            redirect_uri = get_redirect_uri()  # Ya sin barra final
            
            return {
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
    except Exception as e:
        print(f"No se pudieron cargar secrets: {e}")
    
    # Si no hay secrets, se usará el archivo credentials.json
    return None
