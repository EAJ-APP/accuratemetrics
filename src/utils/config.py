"""
Configuración de la aplicación AccurateMetrics - CORREGIDA
Basada en el sistema que funciona en Modular
"""
import streamlit as st

# Paths (solo para local)
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

# ✅ SCOPES COMPLETOS - Incluye Analytics Admin
SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/analytics.readonly',
    'https://www.googleapis.com/auth/analytics.edit'  # ← Para Admin API
]

def get_redirect_uri():
    """
    Obtener redirect URI según el entorno
    IMPORTANTE: Debe coincidir EXACTAMENTE con Google Cloud Console
    """
    try:
        # Prioridad 1: secrets.toml con [oauth]
        if 'oauth' in st.secrets:
            uri = st.secrets['oauth']['redirect_uri']
            # NO modificar - usar exactamente como está
            return uri
            
        # Prioridad 2: secrets.toml con [google_oauth] (legacy)
        if 'google_oauth' in st.secrets:
            uri = st.secrets['google_oauth']['redirect_uri']
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
        # Intentar primero con [oauth]
        secrets_key = None
        if 'oauth' in st.secrets:
            secrets_key = 'oauth'
        elif 'google_oauth' in st.secrets:
            secrets_key = 'google_oauth'
        
        if secrets_key:
            redirect_uri = get_redirect_uri()
            
            # ✅ FORMATO CORRECTO - igual que Modular
            config = {
                "web": {
                    "client_id": st.secrets[secrets_key]["client_id"],
                    "client_secret": st.secrets[secrets_key]["client_secret"],
                    "project_id": st.secrets[secrets_key]["project_id"],
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
