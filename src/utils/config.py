"""
Configuración de la aplicación AccurateMetrics - CORREGIDA
Basada en el sistema que funciona en Modular
"""
import streamlit as st

# Paths (solo para local)
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

# ✅ SCOPES COMPLETOS - Incluye Analytics
SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/analytics.readonly'  # ← AGREGADO
]

def get_redirect_uri():
    """
    Obtener redirect URI según el entorno
    IMPORTANTE: Debe coincidir EXACTAMENTE con Google Cloud Console
    """
    try:
        # CAMBIO CRÍTICO: Usar 'oauth' en lugar de 'google_oauth'
        if 'oauth' in st.secrets:
            uri = st.secrets['oauth']['redirect_uri']
            
            # NO modificar la URI - debe ser exactamente como en Google Cloud
            return uri
    except Exception as e:
        print(f"⚠️ Error leyendo secrets: {e}")
    
    # Fallback a localhost
    return 'http://localhost:8501'

def get_client_config():
    """
    Obtener configuración OAuth desde secrets
    FORMATO CORREGIDO: compatible con google-auth-oauthlib
    """
    try:
        # CAMBIO CRÍTICO: Usar 'oauth' en lugar de 'google_oauth'
        if 'oauth' in st.secrets:
            redirect_uri = get_redirect_uri()
            
            # ✅ FORMATO CORRECTO - igual que Modular
            config = {
                "web": {
                    "client_id": st.secrets["oauth"]["client_id"],
                    "client_secret": st.secrets["oauth"]["client_secret"],
                    "project_id": st.secrets["oauth"]["project_id"],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": [redirect_uri]
                }
            }
            
            return config
            
    except Exception as e:
        print(f"❌ Error cargando secrets: {e}")
        import traceback
        traceback.print_exc()
    
    return None
