"""
Configuración de la aplicación
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración general
APP_ENV = os.getenv('APP_ENV', 'development')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
APP_NAME = os.getenv('APP_NAME', 'AccurateMetrics')

# Paths
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

# OAuth
REDIRECT_URI = 'http://localhost:8501'  # Para local
REDIRECT_URI = 'accuratemetrics-taaxo3j532yujjhuceabjk.streamlit.app'  # Para producción
