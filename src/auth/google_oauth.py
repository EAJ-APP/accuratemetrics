"""
Módulo de autenticación con Google OAuth 2.0 - CORREGIDO
Compatible con google-auth-oauthlib
"""
import os
import json
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from src.utils.config import CREDENTIALS_FILE, TOKEN_FILE, SCOPES, get_redirect_uri, get_client_config

class GoogleAuthenticator:
    """Gestor de autenticación con Google OAuth 2.0"""
    
    def __init__(self):
        self.credentials_file = CREDENTIALS_FILE
        self.token_file = TOKEN_FILE
        self.scopes = SCOPES
        self.flow = None
        
    def load_credentials(self):
        """Cargar credenciales guardadas si existen"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r') as token:
                    creds_data = json.load(token)
                    creds = Credentials.from_authorized_user_info(creds_data, self.scopes)
                    
                    # Refrescar si están expiradas
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                        self.save_credentials(creds)
                    
                    return creds
            except Exception as e:
                print(f"Error loading credentials: {e}")
                return None
        return None
    
    def save_credentials(self, creds):
        """Guardar credenciales en archivo"""
        with open(self.token_file, 'w') as token:
            token.write(creds.to_json())
    
    def get_authorization_url(self):
        """
        Obtener URL de autorización de Google
        CORREGIDO: Usa la configuración correcta de secrets
        """
        redirect_uri = get_redirect_uri()
        client_config = get_client_config()
        
        if client_config:
            # ✅ Usar configuración de secrets (producción)
            self.flow = Flow.from_client_config(
                client_config,
                scopes=self.scopes,
                redirect_uri=redirect_uri
            )
        else:
            # Usar archivo credentials.json (desarrollo local)
            if not os.path.exists(self.credentials_file):
                raise FileNotFoundError(
                    f"No se encontró {self.credentials_file}. "
                    "Descárgalo desde Google Cloud Console o configura secrets en Streamlit."
                )
            
            self.flow = Flow.from_client_secrets_file(
                self.credentials_file,
                scopes=self.scopes,
                redirect_uri=redirect_uri
            )
        
        # ✅ IMPORTANTE: Guardar flow en session_state para uso posterior
        st.session_state['oauth_flow'] = self.flow
        
        auth_url, state = self.flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        # Guardar state para verificación
        st.session_state['oauth_state'] = state
        
        return auth_url
    
    def authenticate_with_code(self, code):
        """
        Autenticar usando el código de autorización
        CORREGIDO: Recupera el flow del session_state
        """
        # ✅ Recuperar flow desde session_state
        if 'oauth_flow' in st.session_state:
            self.flow = st.session_state['oauth_flow']
        
        if not self.flow:
            # Si no existe flow, crear uno nuevo
            redirect_uri = get_redirect_uri()
            client_config = get_client_config()
            
            if client_config:
                self.flow = Flow.from_client_config(
                    client_config,
                    scopes=self.scopes,
                    redirect_uri=redirect_uri
                )
            else:
                raise Exception("No se pudo crear el flow de OAuth")
        
        # Obtener token
        self.flow.fetch_token(code=code)
        creds = self.flow.credentials
        self.save_credentials(creds)
        
        return creds
    
    def get_user_info(self, creds):
        """Obtener información del usuario autenticado"""
        try:
            service = build('oauth2', 'v2', credentials=creds)
            user_info = service.userinfo().get().execute()
            
            return {
                'email': user_info.get('email'),
                'name': user_info.get('name'),
                'picture': user_info.get('picture')
            }
        except Exception as e:
            print(f"Error getting user info: {e}")
            return {
                'email': 'Unknown',
                'name': 'Unknown',
                'picture': ''
            }
    
    def logout(self):
        """Cerrar sesión eliminando tokens"""
        if os.path.exists(self.token_file):
            os.remove(self.token_file)
        
        # Limpiar session_state
        if 'oauth_flow' in st.session_state:
            del st.session_state['oauth_flow']
        if 'oauth_state' in st.session_state:
            del st.session_state['oauth_state']
