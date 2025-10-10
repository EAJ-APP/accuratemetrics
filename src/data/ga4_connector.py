
"""
Conector para Google Analytics 4 Data API
"""
import pandas as pd

# Lazy imports - no importar aquí
# from google.analytics.data_v1beta import BetaAnalyticsDataClient
# from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
# from google.oauth2.credentials import Credentials

class GA4Connector:
    """Conector para extraer datos de Google Analytics 4"""
    
    def __init__(self, credentials):
        # Lazy import
        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        
        self.credentials = credentials
        self.client = BetaAnalyticsDataClient(credentials=credentials)
    
    def _format_property_id(self, property_id: str) -> str:
        """Asegurar formato correcto de property_id"""
        if not property_id.startswith('properties/'):
            return f'properties/{property_id}'
        return property_id
    
    def get_sessions_and_conversions(
        self,
        property_id: str,
        start_date: str,
        end_date: str,
        dimensions: list = None
    ) -> pd.DataFrame:
        """
        Obtener sesiones y conversiones de GA4
        
        Args:
            property_id: ID de la propiedad GA4 (formato: '123456789' o 'properties/123456789')
            start_date: Fecha inicio (formato: 'YYYY-MM-DD')
            end_date: Fecha fin (formato: 'YYYY-MM-DD')
            dimensions: Lista de dimensiones adicionales 
                       (ej: ['deviceCategory', 'sessionDefaultChannelGroup'])
        
        Returns:
            DataFrame con los datos
        """
        # Lazy import
        from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
        
        property_id = self._format_property_id(property_id)
        
        # Dimensiones base
        dims = [Dimension(name="date")]
        
        # Añadir dimensiones adicionales si se especifican
        if dimensions:
            for dim in dimensions:
                dims.append(Dimension(name=dim))
        
        # Métricas
        metrics = [
            Metric(name="sessions"),
            Metric(name="conversions"),
        ]
        
        # Configurar request
        request = RunReportRequest(
            property=property_id,
            dimensions=dims,
            metrics=metrics,
            date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        )
        
        # Ejecutar request
        response = self.client.run_report(request)
        
        # Convertir a DataFrame
        data = []
        for row in response.rows:
            row_data = {}
            
            # Dimensiones
            for i, dimension_value in enumerate(row.dimension_values):
                dim_name = response.dimension_headers[i].name
                row_data[dim_name] = dimension_value.value
            
            # Métricas
            for i, metric_value in enumerate(row.metric_values):
                metric_name = response.metric_headers[i].name
                row_data[metric_name] = float(metric_value.value)
            
            data.append(row_data)
        
        df = pd.DataFrame(data)
        
        # Convertir fecha a datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def get_data_with_dimensions(
        self,
        property_id: str,
        start_date: str,
        end_date: str,
        metrics: list = None,
        dimensions: list = None
    ) -> pd.DataFrame:
        """
        Método genérico para obtener datos con métricas y dimensiones personalizadas
        
        Args:
            property_id: ID de la propiedad GA4
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)
            metrics: Lista de nombres de métricas (ej: ['sessions', 'conversions'])
            dimensions: Lista de nombres de dimensiones (ej: ['date', 'deviceCategory'])
        
        Returns:
            DataFrame con los datos solicitados
        """
        # Lazy import
        from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
        
        property_id = self._format_property_id(property_id)
        
        # Métricas por defecto
        if metrics is None:
            metrics = ['sessions', 'conversions']
        
        # Dimensiones por defecto
        if dimensions is None:
            dimensions = ['date']
        
        # Construir request
        dims = [Dimension(name=dim) for dim in dimensions]
        mets = [Metric(name=met) for met in metrics]
        
        request = RunReportRequest(
            property=property_id,
            dimensions=dims,
            metrics=mets,
            date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        )
        
        response = self.client.run_report(request)
        
        # Convertir a DataFrame
        data = []
        for row in response.rows:
            row_data = {}
            for i, dimension_value in enumerate(row.dimension_values):
                dim_name = response.dimension_headers[i].name
                row_data[dim_name] = dimension_value.value
            
            for i, metric_value in enumerate(row.metric_values):
                metric_name = response.metric_headers[i].name
                row_data[metric_name] = float(metric_value.value)
            
            data.append(row_data)
        
        df = pd.DataFrame(data)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df = df.sort_values('date').reset_index(drop=True)
        
        return df