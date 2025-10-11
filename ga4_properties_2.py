"""
Módulo para obtener propiedades de Google Analytics 4
Crear como: src/data/ga4_properties.py
"""

from google.analytics.admin import AnalyticsAdminServiceClient
from google.analytics.admin_v1alpha.types import ListAccountsRequest, ListPropertiesRequest


class GA4PropertyManager:
    """Gestor de propiedades de Google Analytics 4"""
    
    def __init__(self, credentials):
        """
        Inicializar con credenciales OAuth
        
        Args:
            credentials: Credenciales de Google OAuth
        """
        self.credentials = credentials
        self.admin_client = AnalyticsAdminServiceClient(credentials=credentials)
    
    def get_all_properties(self):
        """
        Obtener todas las propiedades GA4 accesibles
        
        Returns:
            list: Lista de diccionarios con información de propiedades
                  [{'id': '123456789', 'name': 'Mi Propiedad', 'account': 'Mi Cuenta'}, ...]
        """
        properties_list = []
        
        try:
            # 1. Obtener todas las cuentas
            accounts_request = ListAccountsRequest()
            accounts = self.admin_client.list_accounts(request=accounts_request)
            
            # 2. Para cada cuenta, obtener sus propiedades
            for account in accounts:
                account_name = account.display_name
                account_resource_name = account.name  # Format: accounts/123456
                
                try:
                    # Listar propiedades de esta cuenta
                    properties_request = ListPropertiesRequest(
                        filter=f"parent:{account_resource_name}"
                    )
                    properties = self.admin_client.list_properties(request=properties_request)
                    
                    for property in properties:
                        # Extraer el ID numérico de la propiedad
                        # Format: properties/123456789 -> 123456789
                        property_id = property.name.split('/')[-1]
                        
                        properties_list.append({
                            'id': property_id,
                            'name': property.display_name,
                            'account': account_name,
                            'full_name': f"{account_name} - {property.display_name}",
                            'resource_name': property.name
                        })
                
                except Exception as prop_error:
                    print(f"Error obteniendo propiedades de {account_name}: {prop_error}")
                    continue
        
        except Exception as e:
            print(f"Error obteniendo cuentas: {e}")
            return []
        
        return properties_list
    
    def get_properties_dict(self):
        """
        Obtener propiedades como diccionario para selectbox
        
        Returns:
            dict: {display_name: property_id}
        """
        properties = self.get_all_properties()
        
        return {
            prop['full_name']: prop['id'] 
            for prop in properties
        }
