"""
Motor de Análisis Avanzado de Causal Impact
AccurateMetrics - Módulo de Causal Impact Avanzado

Este módulo implementa análisis de Causal Impact con:
- Soporte para múltiples variables de control
- Análisis de hasta 2 intervenciones
- Soporte para intervenciones PUNTUALES y PROLONGADAS
- Comparación de intervenciones
- Interpretación automática de resultados
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Literal
import warnings
warnings.filterwarnings('ignore')

# Verificar disponibilidad de CausalImpact
try:
    from causalimpact import CausalImpact
    CAUSALIMPACT_AVAILABLE = True
except ImportError:
    CAUSALIMPACT_AVAILABLE = False
    print("pycausalimpact no está instalado")


# Tipos de intervención
InterventionType = Literal['puntual', 'prolongada']


class CausalImpactAdvancedAnalyzer:
    """
    Analizador avanzado de impacto causal para datos de Google Analytics

    Características:
    - Soporte para múltiples variables de control
    - Análisis de hasta 2 intervenciones
    - Soporte para intervenciones PUNTUALES y PROLONGADAS
    - Comparación automática de intervenciones
    - Interpretación de resultados en español
    """

    def __init__(
        self,
        data: pd.DataFrame,
        response_variable: str = 'y',
        control_variables: Optional[List[str]] = None
    ):
        """
        Inicializar el analizador

        Args:
            data: DataFrame con variable respuesta ('y') y variables de control
            response_variable: Nombre de la variable respuesta (debe ser 'y' para CausalImpact)
            control_variables: Lista de variables de control a usar
        """
        self.original_data = data.copy()
        self.response_variable = response_variable
        self.control_variables = control_variables

        # Preparar datos
        self.data = self._prepare_data(data, control_variables)

        # Resultados de análisis
        self.results = {}
        self.impact_objects = {}

    def _prepare_data(
        self,
        data: pd.DataFrame,
        control_variables: Optional[List[str]]
    ) -> pd.DataFrame:
        """
        Preparar datos para CausalImpact

        Args:
            data: DataFrame original
            control_variables: Variables de control a incluir

        Returns:
            DataFrame preparado con índice datetime y frecuencia diaria
        """
        df = data.copy()

        # Verificar que 'y' existe
        if 'y' not in df.columns:
            raise ValueError("El DataFrame debe contener una columna 'y' (variable respuesta)")

        # Establecer índice si no está
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        elif df.index.name != 'date' and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("El DataFrame debe tener un índice datetime o columna 'date'")

        # Ordenar por fecha
        df.sort_index(inplace=True)

        # Establecer frecuencia diaria
        if df.index.freq is None:
            df = df.asfreq('D')

        # Seleccionar columnas
        cols_to_keep = ['y']
        if control_variables:
            cols_to_keep.extend([c for c in control_variables if c in df.columns and c != 'y'])
        df = df[cols_to_keep]

        # Manejar NaN
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        return df

    def analyze_intervention(
        self,
        intervention_date: str,
        intervention_name: str = 'Intervención',
        intervention_type: InterventionType = 'puntual',
        intervention_end_date: Optional[str] = None,
        analyze_residual: bool = False,
        pre_period_days: Optional[int] = None,
        post_period_days: Optional[int] = None,
        use_seasonality: bool = True
    ) -> Dict[str, Any]:
        """
        Analizar una intervención (puntual o prolongada)

        Args:
            intervention_date: Fecha de inicio de la intervención (YYYY-MM-DD)
            intervention_name: Nombre descriptivo de la intervención
            intervention_type: 'puntual' o 'prolongada'
            intervention_end_date: Fecha fin de intervención (solo para prolongada)
            analyze_residual: Si analizar efecto residual post-campaña (solo prolongada)
            pre_period_days: Días de período pre (None = usar todo lo disponible)
            post_period_days: Días de período post (None = usar todo lo disponible)
            use_seasonality: Si usar estacionalidad semanal

        Returns:
            Diccionario con resultados del análisis
        """
        if intervention_type == 'prolongada':
            return self._analyze_prolonged_intervention(
                intervention_start=intervention_date,
                intervention_end=intervention_end_date,
                intervention_name=intervention_name,
                analyze_residual=analyze_residual,
                pre_period_days=pre_period_days,
                use_seasonality=use_seasonality
            )
        else:
            return self._analyze_punctual_intervention(
                intervention_date=intervention_date,
                intervention_name=intervention_name,
                pre_period_days=pre_period_days,
                post_period_days=post_period_days,
                use_seasonality=use_seasonality
            )

    def _analyze_punctual_intervention(
        self,
        intervention_date: str,
        intervention_name: str = 'Intervención',
        pre_period_days: Optional[int] = None,
        post_period_days: Optional[int] = None,
        use_seasonality: bool = True
    ) -> Dict[str, Any]:
        """
        Analizar una intervención PUNTUAL (un solo día, efecto permanente)

        Matemáticamente:
        - Pre-período: [inicio_datos, día_antes_intervención]
        - Post-período: [fecha_intervención, fin_datos]
        - El efecto se asume PERMANENTE desde la fecha de intervención

        Args:
            intervention_date: Fecha de la intervención (YYYY-MM-DD)
            intervention_name: Nombre descriptivo
            pre_period_days: Días de período pre (None = usar todo)
            post_period_days: Días de período post (None = usar todo)
            use_seasonality: Si usar estacionalidad semanal

        Returns:
            Diccionario con resultados del análisis
        """
        if not CAUSALIMPACT_AVAILABLE:
            raise ImportError("pycausalimpact no está instalado")

        intervention_ts = pd.Timestamp(intervention_date)

        # Validar fecha
        if intervention_ts <= self.data.index.min():
            raise ValueError("La fecha de intervención debe ser posterior al inicio de los datos")
        if intervention_ts >= self.data.index.max():
            raise ValueError("La fecha de intervención debe ser anterior al final de los datos")

        # Calcular períodos
        if pre_period_days:
            pre_start = intervention_ts - pd.Timedelta(days=pre_period_days)
            pre_start = max(pre_start, self.data.index.min())
        else:
            pre_start = self.data.index.min()

        pre_end = intervention_ts - pd.Timedelta(days=1)
        post_start = intervention_ts

        if post_period_days:
            post_end = intervention_ts + pd.Timedelta(days=post_period_days)
            post_end = min(post_end, self.data.index.max())
        else:
            post_end = self.data.index.max()

        # Normalizar timestamps
        pre_start = pd.Timestamp(pre_start.date())
        pre_end = pd.Timestamp(pre_end.date())
        post_start = pd.Timestamp(post_start.date())
        post_end = pd.Timestamp(post_end.date())

        # Períodos para CausalImpact
        pre_period = [pre_start, pre_end]
        post_period = [post_start, post_end]

        # Filtrar datos
        analysis_data = self.data.loc[pre_start:post_end].copy()

        # Asegurar frecuencia
        if analysis_data.index.freq is None:
            analysis_data = analysis_data.asfreq('D')
            analysis_data = analysis_data.fillna(method='ffill').fillna(method='bfill')

        print(f"\n{'='*60}")
        print(f"ANÁLISIS PUNTUAL: {intervention_name}")
        print(f"{'='*60}")
        print(f"Tipo: PUNTUAL (efecto permanente desde {intervention_date})")
        print(f"Pre-período: {pre_start.date()} a {pre_end.date()} ({(pre_end - pre_start).days + 1} días)")
        print(f"Post-período: {post_start.date()} a {post_end.date()} ({(post_end - post_start).days + 1} días)")
        print(f"Variables de control: {[c for c in analysis_data.columns if c != 'y']}")

        # Ejecutar CausalImpact
        ci = self._run_causal_impact(analysis_data, pre_period, post_period, use_seasonality)

        # Guardar objeto para gráficos
        self.impact_objects[intervention_name] = ci

        # Extraer resultados
        results = self._extract_results(
            ci, intervention_name, intervention_date, pre_period, post_period,
            intervention_type='puntual'
        )
        self.results[intervention_name] = results

        return results

    def _analyze_prolonged_intervention(
        self,
        intervention_start: str,
        intervention_end: str,
        intervention_name: str = 'Campaña',
        analyze_residual: bool = False,
        pre_period_days: Optional[int] = None,
        use_seasonality: bool = True
    ) -> Dict[str, Any]:
        """
        Analizar una intervención PROLONGADA (campaña con duración)

        Matemáticamente:
        - Pre-período: [inicio_datos, día_antes_inicio_campaña]
        - Post-período (durante): [inicio_campaña, fin_campaña]
        - El efecto se mide SOLO durante la duración de la campaña

        Si analyze_residual=True, también analiza:
        - Efecto residual post-campaña: [día_después_fin_campaña, fin_datos]
        - Usando el mismo pre-período (excluyendo la campaña)

        Args:
            intervention_start: Fecha inicio de la campaña (YYYY-MM-DD)
            intervention_end: Fecha fin de la campaña (YYYY-MM-DD)
            intervention_name: Nombre descriptivo
            analyze_residual: Si analizar efecto residual después de la campaña
            pre_period_days: Días de período pre (None = usar todo)
            use_seasonality: Si usar estacionalidad semanal

        Returns:
            Diccionario con resultados del análisis (incluye residual si aplica)
        """
        if not CAUSALIMPACT_AVAILABLE:
            raise ImportError("pycausalimpact no está instalado")

        if not intervention_end:
            raise ValueError("Para intervención prolongada se requiere fecha de fin")

        start_ts = pd.Timestamp(intervention_start)
        end_ts = pd.Timestamp(intervention_end)

        # Validaciones
        if start_ts >= end_ts:
            raise ValueError("La fecha de inicio debe ser anterior a la fecha de fin")
        if start_ts <= self.data.index.min():
            raise ValueError("La fecha de inicio debe ser posterior al inicio de los datos")
        if end_ts >= self.data.index.max() and analyze_residual:
            raise ValueError("Para analizar efecto residual, la campaña debe terminar antes del fin de los datos")

        # Calcular pre-período (todo antes de la campaña)
        if pre_period_days:
            pre_start = start_ts - pd.Timedelta(days=pre_period_days)
            pre_start = max(pre_start, self.data.index.min())
        else:
            pre_start = self.data.index.min()

        pre_end = start_ts - pd.Timedelta(days=1)

        # Post-período = duración de la campaña
        post_start = start_ts
        post_end = end_ts

        # Normalizar timestamps
        pre_start = pd.Timestamp(pre_start.date())
        pre_end = pd.Timestamp(pre_end.date())
        post_start = pd.Timestamp(post_start.date())
        post_end = pd.Timestamp(post_end.date())

        # Períodos para CausalImpact
        pre_period = [pre_start, pre_end]
        post_period = [post_start, post_end]

        # Filtrar datos (pre + durante campaña)
        analysis_data = self.data.loc[pre_start:post_end].copy()

        # Asegurar frecuencia
        if analysis_data.index.freq is None:
            analysis_data = analysis_data.asfreq('D')
            analysis_data = analysis_data.fillna(method='ffill').fillna(method='bfill')

        campaign_days = (end_ts - start_ts).days + 1

        print(f"\n{'='*60}")
        print(f"ANÁLISIS PROLONGADO: {intervention_name}")
        print(f"{'='*60}")
        print(f"Tipo: PROLONGADA (campaña de {campaign_days} días)")
        print(f"Campaña: {start_ts.date()} a {end_ts.date()}")
        print(f"Pre-período: {pre_start.date()} a {pre_end.date()} ({(pre_end - pre_start).days + 1} días)")
        print(f"Post-período (durante campaña): {post_start.date()} a {post_end.date()} ({campaign_days} días)")
        print(f"Variables de control: {[c for c in analysis_data.columns if c != 'y']}")

        # Ejecutar CausalImpact para período de campaña
        ci = self._run_causal_impact(analysis_data, pre_period, post_period, use_seasonality)

        # Guardar objeto para gráficos
        self.impact_objects[intervention_name] = ci

        # Extraer resultados del período de campaña
        results = self._extract_results(
            ci, intervention_name, intervention_start, pre_period, post_period,
            intervention_type='prolongada',
            intervention_end_date=intervention_end
        )

        # Añadir información específica de campaña prolongada
        results['tipo_intervencion'] = 'prolongada'
        results['campana'] = {
            'fecha_inicio': intervention_start,
            'fecha_fin': intervention_end,
            'duracion_dias': campaign_days
        }

        # Analizar efecto residual si se solicita
        if analyze_residual:
            residual_results = self._analyze_residual_effect(
                intervention_name=intervention_name,
                campaign_end=end_ts,
                pre_start=pre_start,
                pre_end=pre_end,
                use_seasonality=use_seasonality
            )
            results['efecto_residual'] = residual_results

        self.results[intervention_name] = results

        return results

    def _analyze_residual_effect(
        self,
        intervention_name: str,
        campaign_end: pd.Timestamp,
        pre_start: pd.Timestamp,
        pre_end: pd.Timestamp,
        use_seasonality: bool
    ) -> Dict[str, Any]:
        """
        Analizar el efecto RESIDUAL después de que termina la campaña

        Matemáticamente:
        - Pre-período: MISMO que el análisis principal (antes de la campaña)
        - Post-período: [día_después_fin_campaña, fin_datos]

        IMPORTANTE: Excluimos el período de la campaña del análisis para que
        el modelo aprenda solo de datos "normales" (pre-campaña)

        Args:
            intervention_name: Nombre de la intervención
            campaign_end: Fecha fin de la campaña
            pre_start: Inicio del pre-período original
            pre_end: Fin del pre-período original
            use_seasonality: Si usar estacionalidad

        Returns:
            Diccionario con resultados del efecto residual
        """
        # Post-período residual = después de la campaña
        residual_start = campaign_end + pd.Timedelta(days=1)
        residual_end = self.data.index.max()

        residual_start = pd.Timestamp(residual_start.date())
        residual_end = pd.Timestamp(residual_end.date())

        # Verificar que hay datos suficientes
        residual_days = (residual_end - residual_start).days + 1
        if residual_days < 7:
            return {
                'disponible': False,
                'mensaje': f'Solo hay {residual_days} días después de la campaña. Se necesitan al menos 7.'
            }

        print(f"\n{'='*60}")
        print(f"ANÁLISIS EFECTO RESIDUAL: {intervention_name}")
        print(f"{'='*60}")
        print(f"Post-período residual: {residual_start.date()} a {residual_end.date()} ({residual_days} días)")

        # Para el análisis residual, usamos solo datos PRE-campaña + POST-campaña
        # Excluimos el período de la campaña para no contaminar el modelo
        pre_data = self.data.loc[pre_start:pre_end].copy()
        post_data = self.data.loc[residual_start:residual_end].copy()

        # Concatenar (saltando el período de campaña)
        analysis_data = pd.concat([pre_data, post_data])

        # Asegurar frecuencia (habrá un gap, lo manejamos)
        if analysis_data.index.freq is None:
            # Reindexar para tener todas las fechas y luego filtrar
            full_range = pd.date_range(start=pre_start, end=residual_end, freq='D')
            analysis_data = analysis_data.reindex(full_range)
            analysis_data = analysis_data.fillna(method='ffill').fillna(method='bfill')

        pre_period = [pre_start, pre_end]
        post_period = [residual_start, residual_end]

        try:
            ci_residual = self._run_causal_impact(analysis_data, pre_period, post_period, use_seasonality)

            # Extraer métricas del residual
            results = self._extract_results(
                ci_residual,
                f"{intervention_name} (Residual)",
                residual_start.strftime('%Y-%m-%d'),
                pre_period,
                post_period,
                intervention_type='residual'
            )

            # Guardar para gráficos
            self.impact_objects[f"{intervention_name}_residual"] = ci_residual

            return {
                'disponible': True,
                'periodo_inicio': residual_start.strftime('%Y-%m-%d'),
                'periodo_fin': residual_end.strftime('%Y-%m-%d'),
                'dias': residual_days,
                'resultados': results
            }

        except Exception as e:
            return {
                'disponible': False,
                'mensaje': f'Error analizando efecto residual: {str(e)}'
            }

    def _run_causal_impact(
        self,
        data: pd.DataFrame,
        pre_period: List[pd.Timestamp],
        post_period: List[pd.Timestamp],
        use_seasonality: bool
    ) -> CausalImpact:
        """
        Ejecutar análisis de CausalImpact con fallback

        Args:
            data: Datos para el análisis
            pre_period: Período pre-intervención
            post_period: Período post-intervención
            use_seasonality: Si usar estacionalidad

        Returns:
            Objeto CausalImpact con resultados
        """
        # Primer intento: con estacionalidad
        if use_seasonality:
            try:
                ci = CausalImpact(
                    data,
                    pre_period,
                    post_period,
                    model_args={
                        'nseasons': [{'period': 7}],
                        'standardize': True
                    }
                )
                print("CausalImpact ejecutado con estacionalidad semanal")
                return ci
            except TypeError:
                print("nseasons no soportado, reintentando sin estacionalidad...")
            except Exception as e:
                print(f"Error con estacionalidad: {e}")

        # Segundo intento: sin estacionalidad
        try:
            ci = CausalImpact(
                data,
                pre_period,
                post_period,
                model_args={'standardize': True}
            )
            print("CausalImpact ejecutado sin estacionalidad")
            return ci
        except TypeError:
            pass

        # Tercer intento: sin model_args
        ci = CausalImpact(data, pre_period, post_period)
        print("CausalImpact ejecutado con configuración básica")
        return ci

    def _extract_results(
        self,
        ci: CausalImpact,
        intervention_name: str,
        intervention_date: str,
        pre_period: List[pd.Timestamp],
        post_period: List[pd.Timestamp],
        intervention_type: str = 'puntual',
        intervention_end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extraer resultados del objeto CausalImpact

        Args:
            ci: Objeto CausalImpact
            intervention_name: Nombre de la intervención
            intervention_date: Fecha de intervención
            pre_period: Período pre
            post_period: Período post
            intervention_type: 'puntual', 'prolongada' o 'residual'
            intervention_end_date: Fecha fin (solo para prolongada)

        Returns:
            Diccionario con resultados procesados
        """
        # Intentar obtener de summary_data
        try:
            summary = ci.summary_data

            # Verificar si tiene datos válidos
            if not summary.empty:
                avg_actual = summary.loc['actual', 'average'] if 'actual' in summary.index else 0
                avg_predicted = summary.loc['predicted', 'average'] if 'predicted' in summary.index else 0

                if avg_actual > 0 or avg_predicted > 0:
                    return self._extract_from_summary(
                        ci, intervention_name, intervention_date, pre_period, post_period,
                        intervention_type, intervention_end_date
                    )
        except Exception:
            pass

        # Fallback: extraer de inferences
        return self._extract_from_inferences(
            ci, intervention_name, intervention_date, pre_period, post_period,
            intervention_type, intervention_end_date
        )

    def _extract_from_summary(
        self,
        ci: CausalImpact,
        intervention_name: str,
        intervention_date: str,
        pre_period: List[pd.Timestamp],
        post_period: List[pd.Timestamp],
        intervention_type: str = 'puntual',
        intervention_end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extraer resultados desde summary_data"""
        summary = ci.summary_data

        def safe_get(row, col, default=0.0):
            try:
                val = summary.loc[row, col]
                return float(val) if pd.notna(val) else default
            except:
                return default

        # Calcular métricas
        avg_actual = safe_get('actual', 'average')
        avg_predicted = safe_get('predicted', 'average')
        cum_actual = safe_get('actual', 'cumulative')
        cum_predicted = safe_get('predicted', 'cumulative')

        efecto_diario = avg_actual - avg_predicted
        efecto_total = cum_actual - cum_predicted
        cambio_porcentual = (efecto_diario / avg_predicted * 100) if avg_predicted != 0 else 0

        p_value = float(ci.p_value) if hasattr(ci, 'p_value') else 0.5

        return {
            'nombre': intervention_name,
            'fecha': intervention_date,
            'fecha_fin': intervention_end_date,
            'tipo_intervencion': intervention_type,
            'periodos': {
                'pre_inicio': pre_period[0].strftime('%Y-%m-%d'),
                'pre_fin': pre_period[1].strftime('%Y-%m-%d'),
                'pre_dias': (pre_period[1] - pre_period[0]).days + 1,
                'post_inicio': post_period[0].strftime('%Y-%m-%d'),
                'post_fin': post_period[1].strftime('%Y-%m-%d'),
                'post_dias': (post_period[1] - post_period[0]).days + 1
            },
            'metricas': {
                'promedio_observado': avg_actual,
                'promedio_predicho': avg_predicted,
                'efecto_diario': efecto_diario,
                'efecto_total': efecto_total,
                'cambio_porcentual': cambio_porcentual,
                'total_observado': cum_actual,
                'total_predicho': cum_predicted
            },
            'estadisticas': {
                'p_value': p_value,
                'es_significativo': p_value < 0.05,
                'nivel_significancia': self._get_significance_level(p_value)
            },
            'interpretacion': self._generate_interpretation(
                efecto_diario, efecto_total, cambio_porcentual, p_value,
                intervention_name, intervention_type
            )
        }

    def _extract_from_inferences(
        self,
        ci: CausalImpact,
        intervention_name: str,
        intervention_date: str,
        pre_period: List[pd.Timestamp],
        post_period: List[pd.Timestamp],
        intervention_type: str = 'puntual',
        intervention_end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extraer resultados desde inferences (fallback)"""
        inferences = ci.inferences
        intervention_ts = pd.Timestamp(intervention_date)

        # Filtrar período post
        post_mask = inferences.index >= post_period[0]
        if intervention_type == 'prolongada' and intervention_end_date:
            post_mask = post_mask & (inferences.index <= pd.Timestamp(intervention_end_date))

        post_data = inferences[post_mask]

        # Detectar columna de predicción
        pred_col = 'point_pred' if 'point_pred' in inferences.columns else 'preds'

        # Obtener valores reales
        actual = self.data.loc[post_data.index, 'y'].values
        predicted = post_data[pred_col].values

        # Calcular métricas
        avg_actual = float(np.mean(actual))
        avg_predicted = float(np.mean(predicted))
        cum_actual = float(np.sum(actual))
        cum_predicted = float(np.sum(predicted))

        efecto_diario = avg_actual - avg_predicted
        efecto_total = cum_actual - cum_predicted
        cambio_porcentual = (efecto_diario / avg_predicted * 100) if avg_predicted != 0 else 0

        p_value = float(ci.p_value) if hasattr(ci, 'p_value') else 0.5

        return {
            'nombre': intervention_name,
            'fecha': intervention_date,
            'fecha_fin': intervention_end_date,
            'tipo_intervencion': intervention_type,
            'periodos': {
                'pre_inicio': pre_period[0].strftime('%Y-%m-%d'),
                'pre_fin': pre_period[1].strftime('%Y-%m-%d'),
                'pre_dias': (pre_period[1] - pre_period[0]).days + 1,
                'post_inicio': post_period[0].strftime('%Y-%m-%d'),
                'post_fin': post_period[1].strftime('%Y-%m-%d'),
                'post_dias': (post_period[1] - post_period[0]).days + 1
            },
            'metricas': {
                'promedio_observado': avg_actual,
                'promedio_predicho': avg_predicted,
                'efecto_diario': efecto_diario,
                'efecto_total': efecto_total,
                'cambio_porcentual': cambio_porcentual,
                'total_observado': cum_actual,
                'total_predicho': cum_predicted
            },
            'estadisticas': {
                'p_value': p_value,
                'es_significativo': p_value < 0.05,
                'nivel_significancia': self._get_significance_level(p_value)
            },
            'interpretacion': self._generate_interpretation(
                efecto_diario, efecto_total, cambio_porcentual, p_value,
                intervention_name, intervention_type
            )
        }

    def _get_significance_level(self, p_value: float) -> str:
        """Obtener nivel de significancia en español"""
        if p_value < 0.01:
            return "muy_significativo"
        elif p_value < 0.05:
            return "significativo"
        elif p_value < 0.10:
            return "marginalmente_significativo"
        else:
            return "no_significativo"

    def _generate_interpretation(
        self,
        efecto_diario: float,
        efecto_total: float,
        cambio_porcentual: float,
        p_value: float,
        intervention_name: str,
        intervention_type: str = 'puntual'
    ) -> Dict[str, str]:
        """
        Generar interpretación automática de resultados

        Returns:
            Diccionario con interpretaciones en español
        """
        # Texto según tipo de intervención
        if intervention_type == 'prolongada':
            tipo_texto = "La campaña"
            periodo_texto = "durante el período de la campaña"
        elif intervention_type == 'residual':
            tipo_texto = "El efecto residual"
            periodo_texto = "después de finalizar la campaña"
        else:
            tipo_texto = "La intervención"
            periodo_texto = "desde la fecha de intervención"

        # Significancia
        if p_value < 0.01:
            sig_texto = "MUY SIGNIFICATIVO (p < 0.01)"
            sig_emoji = "+++"
        elif p_value < 0.05:
            sig_texto = "SIGNIFICATIVO (p < 0.05)"
            sig_emoji = "+"
        elif p_value < 0.10:
            sig_texto = "MARGINALMENTE SIGNIFICATIVO (p < 0.10)"
            sig_emoji = "~"
        else:
            sig_texto = "NO SIGNIFICATIVO (p >= 0.10)"
            sig_emoji = "-"

        # Dirección del efecto
        if efecto_diario > 0:
            direccion = "POSITIVO"
            direccion_emoji = "^"
        else:
            direccion = "NEGATIVO"
            direccion_emoji = "v"

        # Conclusión
        if p_value < 0.05:
            if efecto_diario > 0:
                conclusion = (
                    f"{tipo_texto} '{intervention_name}' tuvo un impacto POSITIVO estadísticamente significativo {periodo_texto}. "
                    f"Se observó un aumento de {abs(efecto_diario):.1f} unidades por día en promedio, "
                    f"representando un incremento del {abs(cambio_porcentual):.1f}%. "
                    f"El efecto acumulado total fue de {abs(efecto_total):.0f} unidades adicionales."
                )
            else:
                conclusion = (
                    f"{tipo_texto} '{intervention_name}' tuvo un impacto NEGATIVO estadísticamente significativo {periodo_texto}. "
                    f"Se observó una disminución de {abs(efecto_diario):.1f} unidades por día en promedio, "
                    f"representando una caída del {abs(cambio_porcentual):.1f}%. "
                    f"El efecto acumulado total fue de {abs(efecto_total):.0f} unidades perdidas."
                )
        else:
            conclusion = (
                f"{tipo_texto} '{intervention_name}' NO mostró un efecto estadísticamente significativo {periodo_texto}. "
                f"Aunque se observó un cambio del {cambio_porcentual:+.1f}%, "
                f"no hay evidencia suficiente para afirmar que este cambio se deba a la intervención "
                f"y no a la variabilidad natural de los datos."
            )

        return {
            'significancia': sig_texto,
            'significancia_emoji': sig_emoji,
            'direccion': direccion,
            'direccion_emoji': direccion_emoji,
            'conclusion': conclusion
        }

    def compare_interventions(self) -> Optional[pd.DataFrame]:
        """
        Comparar todas las intervenciones analizadas

        Returns:
            DataFrame con comparación de intervenciones
        """
        if len(self.results) < 2:
            return None

        comparison_data = []
        for name, result in self.results.items():
            tipo = result.get('tipo_intervencion', 'puntual')

            # Para intervenciones prolongadas, mostrar período
            if tipo == 'prolongada' and 'campana' in result:
                fecha = f"{result['campana']['fecha_inicio']} a {result['campana']['fecha_fin']}"
            else:
                fecha = result['fecha']

            comparison_data.append({
                'Intervención': name,
                'Tipo': tipo.title(),
                'Fecha/Período': fecha,
                'Efecto Diario': result['metricas']['efecto_diario'],
                'Efecto Total': result['metricas']['efecto_total'],
                'Cambio %': result['metricas']['cambio_porcentual'],
                'P-value': result['estadisticas']['p_value'],
                'Significativo': 'Sí' if result['estadisticas']['es_significativo'] else 'No'
            })

        return pd.DataFrame(comparison_data)

    def get_winner(self) -> Optional[Dict[str, Any]]:
        """
        Determinar la intervención con mayor impacto

        Returns:
            Diccionario con información del ganador
        """
        if len(self.results) < 2:
            return None

        comparison = self.compare_interventions()

        # Filtrar solo significativos
        significativos = comparison[comparison['Significativo'] == 'Sí']

        if significativos.empty:
            return {
                'hay_ganador': False,
                'mensaje': 'Ninguna intervención mostró resultados significativos'
            }

        # Encontrar el mayor cambio porcentual absoluto
        idx_max = significativos['Cambio %'].abs().idxmax()
        ganador = significativos.loc[idx_max]

        return {
            'hay_ganador': True,
            'nombre': ganador['Intervención'],
            'tipo': ganador['Tipo'],
            'cambio_porcentual': ganador['Cambio %'],
            'efecto_diario': ganador['Efecto Diario'],
            'efecto_total': ganador['Efecto Total'],
            'mensaje': f"La intervención '{ganador['Intervención']}' ({ganador['Tipo']}) tuvo el mayor impacto ({ganador['Cambio %']:+.1f}%)"
        }

    def get_plot_data(self, intervention_name: str) -> pd.DataFrame:
        """
        Obtener datos para graficar una intervención específica

        Args:
            intervention_name: Nombre de la intervención

        Returns:
            DataFrame con datos para gráficos
        """
        if intervention_name not in self.impact_objects:
            raise ValueError(f"No hay datos para la intervención '{intervention_name}'")

        ci = self.impact_objects[intervention_name]
        result = self.results[intervention_name]

        # Obtener inferences
        inferences = ci.inferences.copy()

        # Detectar columna de predicción
        if 'point_pred' in inferences.columns:
            inferences['preds'] = inferences['point_pred']
        if 'point_pred_lower' in inferences.columns:
            inferences['preds_lower'] = inferences['point_pred_lower']
            inferences['preds_upper'] = inferences['point_pred_upper']
        elif 'preds_lower' not in inferences.columns:
            std = inferences['preds'].std()
            inferences['preds_lower'] = inferences['preds'] - 2 * std
            inferences['preds_upper'] = inferences['preds'] + 2 * std

        # Añadir valores reales
        inferences['response'] = self.data.loc[inferences.index, 'y']

        # Añadir columna de período
        intervention_date = pd.Timestamp(result['fecha'])
        inferences['period'] = 'pre'
        inferences.loc[inferences.index >= intervention_date, 'period'] = 'post'

        return inferences

    def validate_data(self) -> Tuple[bool, str]:
        """
        Validar que los datos cumplen requisitos mínimos

        Returns:
            Tupla (es_válido, mensaje)
        """
        # Verificar días mínimos
        n_days = len(self.data)
        if n_days < 21:
            return False, f"Se necesitan al menos 21 días de datos. Tienes {n_days}."

        # Verificar variabilidad
        if self.data['y'].std() == 0:
            return False, "La variable respuesta no tiene variabilidad."

        # Verificar valores negativos
        if (self.data['y'] < 0).any():
            return False, "La variable respuesta contiene valores negativos."

        # Verificar NaN
        nan_pct = self.data['y'].isna().sum() / len(self.data) * 100
        if nan_pct > 20:
            return False, f"La variable respuesta tiene {nan_pct:.1f}% de valores faltantes."

        return True, "Los datos cumplen todos los requisitos."


def analyze_multiple_interventions(
    data: pd.DataFrame,
    interventions: List[Dict[str, Any]],
    control_variables: Optional[List[str]] = None
) -> Tuple[CausalImpactAdvancedAnalyzer, Dict[str, Any]]:
    """
    Función auxiliar para analizar múltiples intervenciones

    Args:
        data: DataFrame preparado para CausalImpact
        interventions: Lista de diccionarios con configuración de intervención
            - 'date': Fecha de inicio
            - 'name': Nombre
            - 'type': 'puntual' o 'prolongada' (opcional, default='puntual')
            - 'end_date': Fecha fin (requerido si type='prolongada')
            - 'analyze_residual': bool (opcional, default=False)
        control_variables: Variables de control a usar

    Returns:
        Tupla con (analizador, resultados)
    """
    analyzer = CausalImpactAdvancedAnalyzer(data, control_variables=control_variables)

    # Validar datos
    is_valid, msg = analyzer.validate_data()
    if not is_valid:
        raise ValueError(msg)

    # Analizar cada intervención
    for intervention in interventions:
        analyzer.analyze_intervention(
            intervention_date=intervention['date'],
            intervention_name=intervention['name'],
            intervention_type=intervention.get('type', 'puntual'),
            intervention_end_date=intervention.get('end_date'),
            analyze_residual=intervention.get('analyze_residual', False)
        )

    # Generar resumen
    summary = {
        'resultados': analyzer.results,
        'comparacion': analyzer.compare_interventions(),
        'ganador': analyzer.get_winner()
    }

    return analyzer, summary
