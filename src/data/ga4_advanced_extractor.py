"""
Extractor Avanzado de Datos de Google Analytics 4
AccurateMetrics - Módulo de Causal Impact Avanzado

Este módulo extiende las capacidades del GA4Connector para extraer
múltiples métricas con filtros y segmentación por canal.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class GA4AdvancedExtractor:
    """
    Extractor avanzado de datos de GA4 con soporte para:
    - Múltiples métricas simultáneas
    - Filtros por canal, dispositivo, país, ciudad
    - Segmentación de tráfico por tipo
    """

    # Mapeo de métricas GA4 a nombres en español
    METRICS_MAP = {
        'sessions': 'sesiones_totales',
        'activeUsers': 'usuarios_unicos',
        'conversions': 'conversiones',
        'bounceRate': 'bounce_rate',
        'screenPageViewsPerSession': 'paginas_por_sesion',
        'newUsers': 'usuarios_nuevos',
        'engagedSessions': 'sesiones_engagement',
        'engagementRate': 'tasa_engagement',
        'averageSessionDuration': 'duracion_sesion_promedio',
        'screenPageViews': 'paginas_vistas'
    }

    # Canales para clasificar tráfico
    ORGANIC_CHANNELS = ['Organic Search', 'Organic Social', 'Organic Video']
    DIRECT_CHANNELS = ['Direct']
    PAID_CHANNELS = ['Paid Search', 'Paid Social', 'Display', 'Paid Video', 'Paid Shopping']

    def __init__(self, credentials):
        """
        Inicializar el extractor con credenciales OAuth

        Args:
            credentials: Credenciales de Google OAuth
        """
        # Lazy import para evitar errores si no está instalado
        from google.analytics.data_v1beta import BetaAnalyticsDataClient

        self.credentials = credentials
        self.client = BetaAnalyticsDataClient(credentials=credentials)

    def _format_property_id(self, property_id: str) -> str:
        """Asegurar formato correcto de property_id"""
        if not property_id.startswith('properties/'):
            return f'properties/{property_id}'
        return property_id

    def get_advanced_metrics(
        self,
        property_id: str,
        start_date: str,
        end_date: str,
        channel_filter: Optional[str] = None,
        device_filter: Optional[str] = None,
        country_filter: Optional[str] = None,
        city_filter: Optional[str] = None,
        include_channel_breakdown: bool = True
    ) -> pd.DataFrame:
        """
        Extraer métricas avanzadas de GA4 con filtros opcionales

        Args:
            property_id: ID de la propiedad GA4
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)
            channel_filter: Filtrar por canal específico
            device_filter: Filtrar por dispositivo (desktop, mobile, tablet)
            country_filter: Filtrar por país
            city_filter: Filtrar por ciudad
            include_channel_breakdown: Si incluir desglose por tipo de tráfico

        Returns:
            DataFrame con todas las métricas diarias
        """
        from google.analytics.data_v1beta.types import (
            DateRange, Dimension, Metric, RunReportRequest,
            FilterExpression, Filter
        )

        property_id = self._format_property_id(property_id)

        # Métricas a extraer
        metrics_to_extract = [
            'sessions',
            'activeUsers',
            'conversions',
            'bounceRate',
            'screenPageViewsPerSession',
            'newUsers',
            'engagedSessions'
        ]

        # Construir métricas
        metrics = [Metric(name=m) for m in metrics_to_extract]

        # Dimensiones base
        dimensions = [Dimension(name='date')]

        # Construir filtros
        dimension_filter = self._build_filters(
            channel_filter, device_filter, country_filter, city_filter
        )

        # Request principal
        request = RunReportRequest(
            property=property_id,
            dimensions=dimensions,
            metrics=metrics,
            date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
            dimension_filter=dimension_filter
        )

        # Ejecutar request
        response = self.client.run_report(request)

        # Convertir a DataFrame
        df = self._response_to_dataframe(response, metrics_to_extract)

        # Renombrar columnas a español
        df = self._rename_columns(df)

        # Si se requiere desglose por canal, obtener datos adicionales
        if include_channel_breakdown:
            df = self._add_channel_breakdown(
                df, property_id, start_date, end_date,
                device_filter, country_filter, city_filter
            )

        return df

    def _build_filters(
        self,
        channel_filter: Optional[str],
        device_filter: Optional[str],
        country_filter: Optional[str],
        city_filter: Optional[str]
    ) -> Optional[object]:
        """
        Construir expresión de filtros para la API
        """
        from google.analytics.data_v1beta.types import (
            FilterExpression, FilterExpressionList, Filter
        )

        filter_expressions = []

        if channel_filter and channel_filter != 'Todos':
            filter_expressions.append(
                FilterExpression(
                    filter=Filter(
                        field_name='sessionDefaultChannelGroup',
                        string_filter=Filter.StringFilter(
                            value=channel_filter,
                            match_type=Filter.StringFilter.MatchType.EXACT
                        )
                    )
                )
            )

        if device_filter and device_filter != 'Todos':
            filter_expressions.append(
                FilterExpression(
                    filter=Filter(
                        field_name='deviceCategory',
                        string_filter=Filter.StringFilter(
                            value=device_filter,
                            match_type=Filter.StringFilter.MatchType.EXACT
                        )
                    )
                )
            )

        if country_filter and country_filter != 'Todos':
            filter_expressions.append(
                FilterExpression(
                    filter=Filter(
                        field_name='country',
                        string_filter=Filter.StringFilter(
                            value=country_filter,
                            match_type=Filter.StringFilter.MatchType.EXACT
                        )
                    )
                )
            )

        if city_filter and city_filter != 'Todos':
            filter_expressions.append(
                FilterExpression(
                    filter=Filter(
                        field_name='city',
                        string_filter=Filter.StringFilter(
                            value=city_filter,
                            match_type=Filter.StringFilter.MatchType.EXACT
                        )
                    )
                )
            )

        if not filter_expressions:
            return None

        if len(filter_expressions) == 1:
            return filter_expressions[0]

        # Combinar con AND
        return FilterExpression(
            and_group=FilterExpressionList(expressions=filter_expressions)
        )

    def _response_to_dataframe(
        self,
        response,
        metrics_names: List[str]
    ) -> pd.DataFrame:
        """
        Convertir respuesta de la API a DataFrame
        """
        data = []

        for row in response.rows:
            row_data = {}

            # Dimensiones
            for i, dim_value in enumerate(row.dimension_values):
                dim_name = response.dimension_headers[i].name
                row_data[dim_name] = dim_value.value

            # Métricas
            for i, metric_value in enumerate(row.metric_values):
                metric_name = response.metric_headers[i].name
                row_data[metric_name] = float(metric_value.value)

            data.append(row_data)

        df = pd.DataFrame(data)

        # Convertir fecha
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df = df.sort_values('date').reset_index(drop=True)

        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renombrar columnas de inglés a español
        """
        rename_map = {k: v for k, v in self.METRICS_MAP.items() if k in df.columns}
        return df.rename(columns=rename_map)

    def _add_channel_breakdown(
        self,
        df: pd.DataFrame,
        property_id: str,
        start_date: str,
        end_date: str,
        device_filter: Optional[str],
        country_filter: Optional[str],
        city_filter: Optional[str]
    ) -> pd.DataFrame:
        """
        Añadir columnas con desglose de tráfico por tipo de canal
        """
        from google.analytics.data_v1beta.types import (
            DateRange, Dimension, Metric, RunReportRequest
        )

        # Request con dimensión de canal
        request = RunReportRequest(
            property=property_id,
            dimensions=[
                Dimension(name='date'),
                Dimension(name='sessionDefaultChannelGroup')
            ],
            metrics=[Metric(name='sessions')],
            date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
            dimension_filter=self._build_filters(
                None, device_filter, country_filter, city_filter
            )
        )

        response = self.client.run_report(request)

        # Procesar respuesta
        channel_data = {}
        for row in response.rows:
            date_str = row.dimension_values[0].value
            channel = row.dimension_values[1].value
            sessions = float(row.metric_values[0].value)

            date = pd.to_datetime(date_str, format='%Y%m%d')

            if date not in channel_data:
                channel_data[date] = {
                    'trafico_organico': 0,
                    'trafico_directo': 0,
                    'trafico_pago': 0,
                    'trafico_otros': 0
                }

            # Clasificar canal
            if channel in self.ORGANIC_CHANNELS:
                channel_data[date]['trafico_organico'] += sessions
            elif channel in self.DIRECT_CHANNELS:
                channel_data[date]['trafico_directo'] += sessions
            elif channel in self.PAID_CHANNELS:
                channel_data[date]['trafico_pago'] += sessions
            else:
                channel_data[date]['trafico_otros'] += sessions

        # Crear DataFrame de canales
        channel_df = pd.DataFrame.from_dict(channel_data, orient='index')
        channel_df.index.name = 'date'
        channel_df = channel_df.reset_index()

        # Merge con DataFrame principal
        df = df.merge(channel_df, on='date', how='left')

        # Rellenar NaN con 0
        channel_cols = ['trafico_organico', 'trafico_directo', 'trafico_pago', 'trafico_otros']
        for col in channel_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def get_available_filters(
        self,
        property_id: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, List[str]]:
        """
        Obtener valores disponibles para filtros

        Returns:
            Dict con listas de valores disponibles para cada filtro
        """
        from google.analytics.data_v1beta.types import (
            DateRange, Dimension, Metric, RunReportRequest
        )

        property_id = self._format_property_id(property_id)

        filters = {
            'canales': ['Todos'],
            'dispositivos': ['Todos'],
            'paises': ['Todos'],
            'ciudades': ['Todos']
        }

        dimension_names = [
            ('sessionDefaultChannelGroup', 'canales'),
            ('deviceCategory', 'dispositivos'),
            ('country', 'paises'),
            ('city', 'ciudades')
        ]

        for dim_name, filter_key in dimension_names:
            try:
                request = RunReportRequest(
                    property=property_id,
                    dimensions=[Dimension(name=dim_name)],
                    metrics=[Metric(name='sessions')],
                    date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                    limit=100
                )

                response = self.client.run_report(request)

                values = [row.dimension_values[0].value for row in response.rows]
                filters[filter_key].extend(sorted(values))

            except Exception as e:
                print(f"Error obteniendo filtros para {dim_name}: {e}")

        return filters

    def prepare_for_causal_impact(
        self,
        df: pd.DataFrame,
        response_variable: str = 'conversiones'
    ) -> pd.DataFrame:
        """
        Preparar DataFrame para análisis de CausalImpact

        La primera columna debe ser 'y' (variable respuesta) y las demás
        son variables de control.

        Args:
            df: DataFrame con métricas extraídas
            response_variable: Nombre de la columna a usar como variable respuesta

        Returns:
            DataFrame preparado con índice datetime y columna 'y' primero
        """
        # Crear copia
        ci_df = df.copy()

        # Establecer índice de fecha
        if 'date' in ci_df.columns:
            ci_df.set_index('date', inplace=True)

        # Asegurar frecuencia diaria
        ci_df = ci_df.asfreq('D')

        # Verificar que existe la variable respuesta
        if response_variable not in ci_df.columns:
            raise ValueError(f"La columna '{response_variable}' no existe en los datos")

        # Renombrar variable respuesta a 'y' y moverla al principio
        ci_df['y'] = ci_df[response_variable]

        # Reordenar columnas: 'y' primero, luego las demás
        cols = ['y'] + [c for c in ci_df.columns if c != 'y' and c != response_variable]
        ci_df = ci_df[cols]

        # Eliminar columna original de respuesta si no es 'y'
        if response_variable != 'y' and response_variable in ci_df.columns:
            ci_df = ci_df.drop(columns=[response_variable])

        # Manejar NaN
        ci_df = ci_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        return ci_df

    def calculate_correlations(
        self,
        df: pd.DataFrame,
        response_variable: str = 'conversiones'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Calcular matriz de correlación y correlaciones con variable respuesta

        Args:
            df: DataFrame con métricas
            response_variable: Variable respuesta para calcular correlaciones

        Returns:
            Tuple con (matriz de correlación completa, serie de correlaciones con respuesta)
        """
        # Seleccionar solo columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Calcular matriz de correlación
        corr_matrix = df[numeric_cols].corr()

        # Correlaciones con variable respuesta
        if response_variable in corr_matrix.columns:
            corr_with_response = corr_matrix[response_variable].drop(response_variable)
            corr_with_response = corr_with_response.sort_values(ascending=False)
        else:
            corr_with_response = pd.Series()

        return corr_matrix, corr_with_response

    def get_recommended_controls(
        self,
        df: pd.DataFrame,
        response_variable: str = 'conversiones',
        min_correlation: float = 0.5
    ) -> List[str]:
        """
        Obtener variables recomendadas como control basándose en correlación

        Args:
            df: DataFrame con métricas
            response_variable: Variable respuesta
            min_correlation: Correlación mínima para recomendar

        Returns:
            Lista de variables recomendadas
        """
        _, corr_with_response = self.calculate_correlations(df, response_variable)

        # Filtrar por correlación mínima (valor absoluto)
        recommended = corr_with_response[abs(corr_with_response) >= min_correlation]

        return recommended.index.tolist()


def generate_sample_data(days: int = 365) -> pd.DataFrame:
    """
    Generar datos de ejemplo para pruebas

    Args:
        days: Número de días de datos a generar

    Returns:
        DataFrame con datos simulados de GA4
    """
    np.random.seed(2023)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Tendencia base
    trend = np.linspace(10000, 12000, days)

    # Efecto día de la semana
    day_of_week = dates.dayofweek
    weekly_effect = np.where(day_of_week >= 5, -2000, 500)

    # Sesiones base
    sesiones = trend + weekly_effect + np.random.normal(0, 500, days)
    sesiones = np.maximum(sesiones, 1000)

    # Otras métricas derivadas
    usuarios = sesiones * np.random.uniform(0.70, 0.80, days)
    trafico_org = sesiones * np.random.uniform(0.55, 0.65, days)
    trafico_dir = sesiones * np.random.uniform(0.20, 0.30, days)
    trafico_pago = sesiones * np.random.uniform(0.05, 0.15, days)
    bounce_rate = np.random.uniform(38, 52, days)
    paginas = np.random.uniform(2.5, 4.5, days)

    # Conversiones con relación a otras métricas
    conversion_rate_base = 0.025
    conversiones = (
        sesiones * conversion_rate_base +
        0.0003 * usuarios +
        0.0002 * trafico_org -
        0.5 * bounce_rate +
        15 * paginas +
        np.random.normal(0, 30, days)
    )
    conversiones = np.maximum(conversiones, 50)

    # Crear DataFrame
    data = pd.DataFrame({
        'date': dates,
        'sesiones_totales': sesiones,
        'usuarios_unicos': usuarios,
        'conversiones': conversiones,
        'trafico_organico': trafico_org,
        'trafico_directo': trafico_dir,
        'trafico_pago': trafico_pago,
        'bounce_rate': bounce_rate,
        'paginas_por_sesion': paginas
    })

    return data
