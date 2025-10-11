"""
Motor de análisis de Causal Impact para AccurateMetrics
Versión básica con soporte para una intervención
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from causalimpact import CausalImpact
    CAUSALIMPACT_AVAILABLE = True
except ImportError:
    CAUSALIMPACT_AVAILABLE = False
    print("⚠️ pycausalimpact no está instalado. Instala con: pip install pycausalimpact")


class CausalImpactAnalyzer:
    """
    Analizador de impacto causal para datos de Google Analytics
    """
    
    def __init__(self, data: pd.DataFrame, metric_column: str = 'sessions'):
        """
        Inicializa el analizador con datos de GA4
        
        Args:
            data: DataFrame con columnas 'date' y métricas
            metric_column: Columna a analizar ('sessions' o 'conversions')
        """
        self.original_data = data.copy()
        self.metric_column = metric_column
        self.data = self._prepare_data(data)
        self.impact_result = None
        self.intervention_date = None
        
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara los datos para el análisis
        
        Args:
            data: DataFrame original
            
        Returns:
            DataFrame preparado con índice de fecha
        """
        df = data.copy()
        
        # Asegurar que 'date' sea datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Establecer fecha como índice
        df.set_index('date', inplace=True)
        
        # Ordenar por fecha
        df.sort_index(inplace=True)
        
        # Verificar que la métrica existe
        if self.metric_column not in df.columns:
            raise ValueError(f"La columna '{self.metric_column}' no existe en los datos")
        
        # Seleccionar solo la métrica de interés
        df = df[[self.metric_column]]
        
        # Manejar valores faltantes
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def analyze_single_intervention(
        self,
        intervention_date: str,
        pre_period_days: Optional[int] = None,
        post_period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analiza el impacto de una única intervención
        
        Args:
            intervention_date: Fecha de la intervención (formato: 'YYYY-MM-DD')
            pre_period_days: Días antes de la intervención para el período pre (None = todos)
            post_period_days: Días después de la intervención para el período post (None = todos)
            
        Returns:
            Diccionario con resultados del análisis
        """
        if not CAUSALIMPACT_AVAILABLE:
            raise ImportError("pycausalimpact no está instalado")
        
        # Convertir fecha de intervención
        self.intervention_date = pd.to_datetime(intervention_date)
        
        # Validar que la fecha está en el rango de datos
        if self.intervention_date <= self.data.index.min():
            raise ValueError("La fecha de intervención debe ser posterior al inicio de los datos")
        if self.intervention_date >= self.data.index.max():
            raise ValueError("La fecha de intervención debe ser anterior al final de los datos")
        
        # Definir períodos pre y post
        if pre_period_days:
            pre_start = self.intervention_date - timedelta(days=pre_period_days)
            pre_start = max(pre_start, self.data.index.min())
        else:
            pre_start = self.data.index.min()
        
        pre_end = self.intervention_date - timedelta(days=1)
        
        post_start = self.intervention_date
        
        if post_period_days:
            post_end = self.intervention_date + timedelta(days=post_period_days)
            post_end = min(post_end, self.data.index.max())
        else:
            post_end = self.data.index.max()
        
        # Crear tuplas de períodos para CausalImpact
        pre_period = [pre_start, pre_end]
        post_period = [post_start, post_end]
        
        # Ejecutar análisis
        try:
            self.impact_result = CausalImpact(
                self.data,
                pre_period,
                post_period,
                model_args={'nseasons': 7}  # Estacionalidad semanal
            )
        except Exception as e:
            raise Exception(f"Error en el análisis CausalImpact: {str(e)}")
        
        # Extraer resultados principales
        summary = self._extract_summary()
        
        # Añadir información de períodos
        summary['periods'] = {
            'pre_period': {
                'start': pre_start.strftime('%Y-%m-%d'),
                'end': pre_end.strftime('%Y-%m-%d'),
                'days': (pre_end - pre_start).days + 1
            },
            'post_period': {
                'start': post_start.strftime('%Y-%m-%d'),
                'end': post_end.strftime('%Y-%m-%d'),
                'days': (post_end - post_start).days + 1
            },
            'intervention_date': intervention_date
        }
        
        return summary
    
    def _extract_summary(self) -> Dict[str, Any]:
        """
        Extrae un resumen de los resultados del análisis
        
        Returns:
            Diccionario con métricas clave
        """
        if not self.impact_result:
            return {}
        
        # Obtener el summary dataframe
        summary_df = self.impact_result.summary_df
        
        # Extraer métricas principales
        summary = {
            'average': {
                'actual': summary_df.loc['average', 'actual'],
                'predicted': summary_df.loc['average', 'predicted'],
                'predicted_lower': summary_df.loc['average', 'predicted_lower'],
                'predicted_upper': summary_df.loc['average', 'predicted_upper'],
                'abs_effect': summary_df.loc['average', 'abs_effect'],
                'abs_effect_lower': summary_df.loc['average', 'abs_effect_lower'],
                'abs_effect_upper': summary_df.loc['average', 'abs_effect_upper'],
                'rel_effect': summary_df.loc['average', 'rel_effect'],
                'rel_effect_lower': summary_df.loc['average', 'rel_effect_lower'],
                'rel_effect_upper': summary_df.loc['average', 'rel_effect_upper']
            },
            'cumulative': {
                'actual': summary_df.loc['cumulative', 'actual'],
                'predicted': summary_df.loc['cumulative', 'predicted'],
                'predicted_lower': summary_df.loc['cumulative', 'predicted_lower'],
                'predicted_upper': summary_df.loc['cumulative', 'predicted_upper'],
                'abs_effect': summary_df.loc['cumulative', 'abs_effect'],
                'abs_effect_lower': summary_df.loc['cumulative', 'abs_effect_lower'],
                'abs_effect_upper': summary_df.loc['cumulative', 'abs_effect_upper'],
                'rel_effect': summary_df.loc['cumulative', 'rel_effect'],
                'rel_effect_lower': summary_df.loc['cumulative', 'rel_effect_lower'],
                'rel_effect_upper': summary_df.loc['cumulative', 'rel_effect_upper']
            },
            'p_value': self.impact_result.p_value,
            'is_significant': self.impact_result.p_value < 0.05,
            'metric': self.metric_column
        }
        
        return summary
    
    def get_plot_data(self) -> pd.DataFrame:
        """
        Obtiene los datos para graficar
        
        Returns:
            DataFrame con datos originales y predicciones
        """
        if not self.impact_result:
            return pd.DataFrame()
        
        # Obtener series temporales del resultado
        result_df = self.impact_result.inferences.copy()
        
        # Renombrar columnas para claridad
        result_df.columns = [
            'predicted',
            'predicted_lower',
            'predicted_upper',
            'actual',
            'residuals',
            'cumulative_residuals'
        ]
        
        # Añadir columna de período
        result_df['period'] = 'pre'
        if self.intervention_date:
            result_df.loc[result_df.index >= self.intervention_date, 'period'] = 'post'
        
        return result_df
    
    def get_summary_text(self) -> str:
        """
        Genera un resumen en texto de los resultados
        
        Returns:
            String con el resumen narrativo
        """
        if not self.impact_result:
            return "No hay resultados de análisis disponibles"
        
        summary = self._extract_summary()
        
        # Construir narrativa
        text_parts = []
        
        # Título
        text_parts.append(f"📊 **Análisis de Impacto Causal - {self.metric_column.title()}**\n")
        
        # Efecto promedio
        avg_effect = summary['average']['rel_effect']
        avg_lower = summary['average']['rel_effect_lower']
        avg_upper = summary['average']['rel_effect_upper']
        
        text_parts.append(f"**Efecto Promedio:** {avg_effect:.1%}")
        text_parts.append(f"Intervalo de confianza: [{avg_lower:.1%}, {avg_upper:.1%}]\n")
        
        # Efecto acumulado
        cum_effect = summary['cumulative']['abs_effect']
        cum_actual = summary['cumulative']['actual']
        cum_predicted = summary['cumulative']['predicted']
        
        text_parts.append(f"**Efecto Acumulado:**")
        text_parts.append(f"- {self.metric_column.title()} observadas: {cum_actual:,.0f}")
        text_parts.append(f"- {self.metric_column.title()} esperadas: {cum_predicted:,.0f}")
        text_parts.append(f"- Diferencia: {cum_effect:,.0f} ({summary['cumulative']['rel_effect']:.1%})\n")
        
        # Significancia
        p_value = summary['p_value']
        if summary['is_significant']:
            text_parts.append(f"✅ **Resultado estadísticamente significativo** (p-value: {p_value:.3f})")
            
            if avg_effect > 0:
                text_parts.append("La intervención tuvo un **impacto positivo**.")
            else:
                text_parts.append("La intervención tuvo un **impacto negativo**.")
        else:
            text_parts.append(f"⚠️ **Resultado NO significativo** (p-value: {p_value:.3f})")
            text_parts.append("No hay evidencia suficiente de que la intervención haya tenido un impacto real.")
        
        return "\n".join(text_parts)
    
    def validate_data_requirements(self) -> Tuple[bool, str]:
        """
        Valida que los datos cumplan los requisitos mínimos
        
        Returns:
            Tupla (es_válido, mensaje)
        """
        # Verificar cantidad mínima de datos
        n_days = len(self.data)
        if n_days < 21:
            return False, f"Se necesitan al menos 21 días de datos. Tienes {n_days} días."
        
        # Verificar variabilidad en los datos
        if self.data[self.metric_column].std() == 0:
            return False, "Los datos no tienen variabilidad (todos los valores son iguales)."
        
        # Verificar valores negativos
        if (self.data[self.metric_column] < 0).any():
            return False, "Los datos contienen valores negativos, lo cual no es válido para esta métrica."
        
        return True, "Los datos cumplen todos los requisitos para el análisis."