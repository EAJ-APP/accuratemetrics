"""
Configuración de la aplicación - VERSIÓN CORREGIDA
"""
import streamlit as st

# Paths (solo para local)
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

# ⚠️ IMPORTANTE: Para apps NO VERIFICADAS en Google, usar SOLO scopes básicos
# Si tu app NO está verificada, comenta los scopes de Analytics
SCOPES = [
    # 🔴 COMENTAR TEMPORALMENTE si la app no está verificada:
    # 'https://www.googleapis.com/auth/analytics.readonly',
    
    # ✅ Scopes BÁSICOS (siempre funcionan):
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]

# Si ya verificaste tu app en Google Cloud Console, descomenta esto:
# SCOPES = [
#     'https://www.googleapis.com/auth/analytics.readonly',
#     'openid',
#     'https://www.googleapis.com/auth/userinfo.email',
#     'https://www.googleapis.com/auth/userinfo.profile'
# ]

def get_redirect_uri():
    """Obtener redirect URI según el entorno"""
    try:
        if 'google_oauth' in st.secrets:
            uri = st.secrets['google_oauth']['redirect_uri']
            # IMPORTANTE: Quitar barra final si existe
            uri = uri.rstrip('/')
            
            # 🔍 DEBUG: Verificar que la URI sea correcta
            print(f"✅ Redirect URI configurado: {uri}")
            
            return uri
    except Exception as e:
        print(f"⚠️ Error leyendo secrets: {e}")
    
    # Fallback a localhost
    return 'http://localhost:8501'

def get_client_config():
    """Obtener configuración OAuth desde secrets o archivo"""
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
            
            print(f"✅ Client ID: {config['web']['client_id'][:20]}...")
            print(f"✅ Redirect URI en config: {redirect_uri}")
            
            return config
            
    except Exception as e:
        print(f"❌ Error cargando secrets: {e}")
    
    # Si no hay secrets, se usará el archivo credentials.json
    return None
