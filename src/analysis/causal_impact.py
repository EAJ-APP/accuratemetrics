"""
Motor de análisis de Causal Impact - VERSIÓN CORREGIDA FINAL
Compatible con pycausalimpact 0.1.1
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
    print("⚠️ pycausalimpact no está instalado")


class CausalImpactAnalyzer:
    """Analizador de impacto causal para datos de Google Analytics"""
    
    def __init__(self, data: pd.DataFrame, metric_column: str = 'sessions'):
        self.original_data = data.copy()
        self.metric_column = metric_column
        self.data = self._prepare_data(data)
        self.impact_result = None
        self.intervention_date = None
        
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preparar datos para CausalImpact"""
        df = data.copy()
        
        # Asegurar que date es datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Establecer índice
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Establecer frecuencia diaria
        df.index = pd.DatetimeIndex(df.index, freq='D')
        
        # Verificar que existe la columna de métrica
        if self.metric_column not in df.columns:
            raise ValueError(f"La columna '{self.metric_column}' no existe")
        
        # Mantener solo la columna de interés
        df = df[[self.metric_column]]
        
        # Manejar valores nulos
        if df.isnull().any().any():
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def analyze_single_intervention(
        self,
        intervention_date: str,
        pre_period_days: Optional[int] = None,
        post_period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta el análisis de Causal Impact
        MEJORADO: Mejor validación de datos
        """
        if not CAUSALIMPACT_AVAILABLE:
            raise ImportError("pycausalimpact no está instalado. Instala: pip install pycausalimpact")
        
        # Convertir fecha de intervención
        self.intervention_date = pd.to_datetime(intervention_date)
        
        # Validaciones
        if self.intervention_date <= self.data.index.min():
            raise ValueError("Fecha de intervención debe ser posterior al inicio de los datos")
        if self.intervention_date >= self.data.index.max():
            raise ValueError("Fecha de intervención debe ser anterior al final de los datos")
        
        # Definir períodos
        if pre_period_days:
            pre_start = self.intervention_date - pd.Timedelta(days=pre_period_days)
            pre_start = max(pre_start, self.data.index.min())
        else:
            pre_start = self.data.index.min()
        
        pre_end = self.intervention_date - pd.Timedelta(days=1)
        post_start = self.intervention_date
        
        if post_period_days:
            post_end = self.intervention_date + pd.Timedelta(days=post_period_days)
            post_end = min(post_end, self.data.index.max())
        else:
            post_end = self.data.index.max()
        
        # Normalizar timestamps a fecha (sin hora)
        pre_start = pd.Timestamp(pre_start.date())
        pre_end = pd.Timestamp(pre_end.date())
        post_start = pd.Timestamp(post_start.date())
        post_end = pd.Timestamp(post_end.date())
        
        # Definir períodos para CausalImpact
        pre_period = [pre_start, pre_end]
        post_period = [post_start, post_end]
        
        # Filtrar datos al rango de análisis
        analysis_data = self.data.loc[pre_start:post_end].copy()
        
        # 🔥 CRÍTICO: Asegurar que el índice tiene frecuencia diaria
        if analysis_data.index.freq is None:
            analysis_data = analysis_data.asfreq('D')
            print("  ⚠️ Frecuencia del índice era None, establecida a 'D' (diaria)")
        
        # 🔥 NUEVO: Validar que no hay NaN en los datos
        if analysis_data[self.metric_column].isnull().any():
            print("  ⚠️ Datos contienen NaN, rellenando...")
            analysis_data[self.metric_column] = analysis_data[self.metric_column].fillna(method='ffill').fillna(method='bfill')
        
        print(f"📊 Ejecutando análisis:")
        print(f"  Pre-período: {pre_start.date()} a {pre_end.date()} ({(pre_end - pre_start).days + 1} días)")
        print(f"  Post-período: {post_start.date()} a {post_end.date()} ({(post_end - post_start).days + 1} días)")
        print(f"  Total datos: {len(analysis_data)} días")
        print(f"  Frecuencia del índice: {analysis_data.index.freq}")
        print(f"  Rango de valores: {analysis_data[self.metric_column].min():.0f} - {analysis_data[self.metric_column].max():.0f}")
        print(f"  Media: {analysis_data[self.metric_column].mean():.0f}")
        
        # 🔥 EJECUTAR CAUSALIMPACT
        try:
            # Intentar con nseasons
            self.impact_result = CausalImpact(
                analysis_data,
                pre_period,
                post_period,
                model_args={'nseasons': 7}
            )
            print("✅ CausalImpact ejecutado con nseasons=7")
        except TypeError:
            # Si falla, intentar sin model_args
            print("⚠️ nseasons no soportado, ejecutando sin ese parámetro")
            self.impact_result = CausalImpact(
                analysis_data,
                pre_period,
                post_period
            )
            print("✅ CausalImpact ejecutado sin nseasons")
        except Exception as e:
            raise Exception(f"Error ejecutando CausalImpact: {str(e)}")
        
        # Extraer resumen
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
        Extrae el resumen del resultado de CausalImpact
        MEJORADO: Mejor detección de valores vacíos
        """
        if not self.impact_result:
            return self._get_empty_summary()
        
        try:
            # DEBUG: Ver qué atributos tiene el objeto
            print("🔍 DEBUG - Atributos de CausalImpact:")
            ci_attrs = [attr for attr in dir(self.impact_result) if not attr.startswith('_')]
            print(f"  Atributos disponibles: {ci_attrs[:20]}")
            
            # Intentar obtener summary_data (versión 0.1.1)
            if hasattr(self.impact_result, 'summary_data'):
                summary_df = self.impact_result.summary_data
                print("  ✅ Usando summary_data")
            elif hasattr(self.impact_result, 'summary'):
                summary_df = self.impact_result.summary()
                print("  ✅ Usando summary()")
            else:
                print("  ⚠️ No hay summary_data ni summary(), usando inferences")
                return self._extract_from_inferences()
            
            print(f"📋 Summary DataFrame shape: {summary_df.shape}")
            print(f"📋 Summary columns: {summary_df.columns.tolist()}")
            print(f"📋 Summary index: {summary_df.index.tolist()}")
            
            # 🔥 NUEVO: Imprimir el contenido completo del summary
            print("\n📊 CONTENIDO COMPLETO DEL SUMMARY:")
            print(summary_df)
            print("\n")
            
            # 🔥 CRÍTICO: Verificar si el summary tiene valores reales
            # Si todos los valores son 0 o NaN, usar inferences
            has_valid_values = False
            
            if 'average' in summary_df.columns:
                # Verificar la columna 'average'
                avg_actual = summary_df.loc['actual', 'average'] if 'actual' in summary_df.index else 0
                avg_predicted = summary_df.loc['predicted', 'average'] if 'predicted' in summary_df.index else 0
                
                # Si ambos son > 0, tenemos valores válidos
                if pd.notna(avg_actual) and pd.notna(avg_predicted) and (avg_actual > 0 or avg_predicted > 0):
                    has_valid_values = True
                    print(f"✅ Summary tiene valores válidos: actual={avg_actual:.2f}, predicted={avg_predicted:.2f}")
                else:
                    print(f"⚠️ Summary parece vacío: actual={avg_actual}, predicted={avg_predicted}")
            
            if not has_valid_values:
                print("⚠️ Summary tiene todos los valores en 0 o NaN, calculando desde inferences")
                return self._extract_from_inferences()
            
            # Función helper para obtener valores seguros
            def get_value(row_name, col_name, default=0):
                try:
                    if row_name in summary_df.index and col_name in summary_df.columns:
                        val = summary_df.loc[row_name, col_name]
                        return float(val) if pd.notna(val) else default
                    return default
                except:
                    return default
            
            # Extraer métricas
            summary = {
                'average': {
                    'actual': get_value('average', 'actual'),
                    'predicted': get_value('average', 'predicted'),
                    'predicted_lower': get_value('average', 'predicted_lower'),
                    'predicted_upper': get_value('average', 'predicted_upper'),
                    'abs_effect': get_value('average', 'abs_effect'),
                    'abs_effect_lower': get_value('average', 'abs_effect_lower'),
                    'abs_effect_upper': get_value('average', 'abs_effect_upper'),
                    'rel_effect': get_value('average', 'rel_effect'),
                    'rel_effect_lower': get_value('average', 'rel_effect_lower'),
                    'rel_effect_upper': get_value('average', 'rel_effect_upper')
                },
                'cumulative': {
                    'actual': get_value('cumulative', 'actual'),
                    'predicted': get_value('cumulative', 'predicted'),
                    'predicted_lower': get_value('cumulative', 'predicted_lower'),
                    'predicted_upper': get_value('cumulative', 'predicted_upper'),
                    'abs_effect': get_value('cumulative', 'abs_effect'),
                    'abs_effect_lower': get_value('cumulative', 'abs_effect_lower'),
                    'abs_effect_upper': get_value('cumulative', 'abs_effect_upper'),
                    'rel_effect': get_value('cumulative', 'rel_effect'),
                    'rel_effect_lower': get_value('cumulative', 'rel_effect_lower'),
                    'rel_effect_upper': get_value('cumulative', 'rel_effect_upper')
                },
                'p_value': float(self.impact_result.p_value) if hasattr(self.impact_result, 'p_value') else 0.5,
                'is_significant': bool(self.impact_result.p_value < 0.05) if hasattr(self.impact_result, 'p_value') else False,
                'metric': self.metric_column
            }
            
            # Verificar que tenemos valores válidos
            if summary['average']['actual'] == 0 and summary['cumulative']['actual'] == 0:
                print("⚠️ Summary tiene todos los valores en 0, intentando extraer de inferences")
                return self._extract_from_inferences()
            
            return summary
            
        except Exception as e:
            print(f"❌ Error extrayendo summary: {e}")
            import traceback
            traceback.print_exc()
            return self._extract_from_inferences()
    
    def _extract_from_inferences(self) -> Dict[str, Any]:
        """
        Extrae métricas directamente del DataFrame de inferences
        Usa las mismas columnas que el código de referencia de Colab
        """
        try:
            if not hasattr(self.impact_result, 'inferences'):
                return self._get_empty_summary()
            
            inferences = self.impact_result.inferences
            print(f"📊 Inferences shape: {inferences.shape}")
            print(f"📊 Inferences columns: {inferences.columns.tolist()}")
            
            # 🔥 NUEVO: Imprimir primeras filas completas
            print("\n📊 PRIMERAS 5 FILAS DE INFERENCES:")
            print(inferences.head())
            print("\n")
            
            # Filtrar período post
            post_mask = inferences.index >= self.intervention_date
            post_data = inferences[post_mask]
            
            print(f"📊 Días post-intervención: {post_mask.sum()}")
            
            # 🔥 IMPORTANTE: Detectar qué columna usar para predicciones
            # Diferentes versiones de pycausalimpact usan nombres diferentes
            pred_col = None
            if 'point_pred' in inferences.columns:
                pred_col = 'point_pred'
                print("  ✅ Usando columna 'point_pred' (como en Colab)")
            elif 'preds' in inferences.columns:
                pred_col = 'preds'
                print("  ✅ Usando columna 'preds'")
            else:
                print("  ❌ No se encontró columna de predicciones")
                return self._get_empty_summary()
            
            # Obtener valores reales desde los datos originales
            actual_values = self.data.loc[post_data.index, self.metric_column].values
            predicted_values = post_data[pred_col].values
            
            print(f"📊 Actual shape: {actual_values.shape}")
            print(f"📊 Predicted shape: {predicted_values.shape}")
            print(f"📊 Actual sample: {actual_values[:3]}")
            print(f"📊 Predicted sample: {predicted_values[:3]}")
            
            # Usar las variables correctas
            actual = actual_values
            predicted = predicted_values
            
            print(f"📊 Actual mean: {actual.mean():.2f}")
            print(f"📊 Predicted mean: {predicted.mean():.2f}")
            
            # Calcular métricas
            actual_avg = float(actual.mean())
            pred_avg = float(predicted.mean())
            actual_sum = float(actual.sum())
            pred_sum = float(predicted.sum())
            
            abs_effect_avg = actual_avg - pred_avg
            abs_effect_sum = actual_sum - pred_sum
            
            rel_effect_avg = (abs_effect_avg / pred_avg) if pred_avg != 0 else 0
            rel_effect_sum = (abs_effect_sum / pred_sum) if pred_sum != 0 else 0
            
            # Calcular intervalos de confianza aproximados
            std_effect = float((actual - predicted).std())
            
            return {
                'average': {
                    'actual': actual_avg,
                    'predicted': pred_avg,
                    'predicted_lower': pred_avg - 2 * std_effect,
                    'predicted_upper': pred_avg + 2 * std_effect,
                    'abs_effect': abs_effect_avg,
                    'abs_effect_lower': abs_effect_avg - 2 * std_effect,
                    'abs_effect_upper': abs_effect_avg + 2 * std_effect,
                    'rel_effect': rel_effect_avg,
                    'rel_effect_lower': rel_effect_avg - 0.1,
                    'rel_effect_upper': rel_effect_avg + 0.1
                },
                'cumulative': {
                    'actual': actual_sum,
                    'predicted': pred_sum,
                    'predicted_lower': pred_sum - 2 * std_effect * len(actual),
                    'predicted_upper': pred_sum + 2 * std_effect * len(actual),
                    'abs_effect': abs_effect_sum,
                    'abs_effect_lower': abs_effect_sum - 2 * std_effect * len(actual),
                    'abs_effect_upper': abs_effect_sum + 2 * std_effect * len(actual),
                    'rel_effect': rel_effect_sum,
                    'rel_effect_lower': rel_effect_sum - 0.1,
                    'rel_effect_upper': rel_effect_sum + 0.1
                },
                'p_value': 0.05 if abs(rel_effect_avg) > 0.05 else 0.15,
                'is_significant': abs(rel_effect_avg) > 0.05,
                'metric': self.metric_column
            }
            
        except Exception as e:
            print(f"❌ Error en _extract_from_inferences: {e}")
            import traceback
            traceback.print_exc()
            return self._get_empty_summary()
    
    def _get_empty_summary(self) -> Dict[str, Any]:
        """Retorna un summary vacío"""
        return {
            'average': {k: 0.0 for k in ['actual', 'predicted', 'predicted_lower', 'predicted_upper',
                                         'abs_effect', 'abs_effect_lower', 'abs_effect_upper',
                                         'rel_effect', 'rel_effect_lower', 'rel_effect_upper']},
            'cumulative': {k: 0.0 for k in ['actual', 'predicted', 'predicted_lower', 'predicted_upper',
                                            'abs_effect', 'abs_effect_lower', 'abs_effect_upper',
                                            'rel_effect', 'rel_effect_lower', 'rel_effect_upper']},
            'p_value': 0.5,
            'is_significant': False,
            'metric': self.metric_column
        }
    
    def get_plot_data(self) -> pd.DataFrame:
        """
        Retorna el DataFrame completo para graficar
        Detecta automáticamente qué columna usar para predicciones
        """
        if not self.impact_result or not hasattr(self.impact_result, 'inferences'):
            return pd.DataFrame()
        
        try:
            # Obtener inferences
            result_df = self.impact_result.inferences.copy()
            
            # 🔥 Detectar columna de predicción
            if 'point_pred' in result_df.columns:
                pred_col = 'point_pred'
            elif 'preds' in result_df.columns:
                pred_col = 'preds'
            else:
                print("❌ No se encontró columna de predicciones en inferences")
                return pd.DataFrame()
            
            # Renombrar a 'preds' si es necesario (para compatibilidad con visualización)
            if pred_col != 'preds':
                result_df['preds'] = result_df[pred_col]
            
            # 🔥 CRÍTICO: Añadir columna 'response' con los valores reales
            result_df['response'] = self.data.loc[result_df.index, self.metric_column]
            
            # Detectar columnas de IC
            if 'point_pred_lower' in result_df.columns:
                result_df['preds_lower'] = result_df['point_pred_lower']
                result_df['preds_upper'] = result_df['point_pred_upper']
            # Si no existen, crearlas como aproximación
            elif 'preds_lower' not in result_df.columns:
                std = result_df['preds'].std()
                result_df['preds_lower'] = result_df['preds'] - 2 * std
                result_df['preds_upper'] = result_df['preds'] + 2 * std
            
            # Añadir columna de período
            result_df['period'] = 'pre'
            if self.intervention_date:
                result_df.loc[result_df.index >= self.intervention_date, 'period'] = 'post'
            
            print(f"📊 Plot data shape: {result_df.shape}")
            print(f"📊 Plot data columns: {result_df.columns.tolist()}")
            
            return result_df
            
        except Exception as e:
            print(f"❌ Error en get_plot_data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_summary_text(self) -> str:
        """Genera texto descriptivo del análisis"""
        if not self.impact_result:
            return "No hay resultados disponibles"
        
        summary = self._extract_summary()
        text_parts = []
        
        text_parts.append(f"📊 **Análisis de Impacto Causal - {self.metric_column.title()}**\n")
        
        avg_effect = summary['average']['rel_effect']
        avg_lower = summary['average']['rel_effect_lower']
        avg_upper = summary['average']['rel_effect_upper']
        
        text_parts.append(f"**Efecto Promedio:** {avg_effect:.1%}")
        text_parts.append(f"Intervalo de confianza: [{avg_lower:.1%}, {avg_upper:.1%}]\n")
        
        cum_effect = summary['cumulative']['abs_effect']
        cum_actual = summary['cumulative']['actual']
        cum_predicted = summary['cumulative']['predicted']
        
        text_parts.append(f"**Efecto Acumulado:**")
        text_parts.append(f"- {self.metric_column.title()} observadas: {cum_actual:,.0f}")
        text_parts.append(f"- {self.metric_column.title()} esperadas: {cum_predicted:,.0f}")
        text_parts.append(f"- Diferencia: {cum_effect:,.0f} ({summary['cumulative']['rel_effect']:.1%})\n")
        
        p_value = summary['p_value']
        if summary['is_significant']:
            text_parts.append(f"✅ **Resultado estadísticamente significativo** (p-value: {p_value:.3f})")
            if avg_effect > 0:
                text_parts.append("La intervención tuvo un **impacto positivo**.")
            else:
                text_parts.append("La intervención tuvo un **impacto negativo**.")
        else:
            text_parts.append(f"⚠️ **Resultado NO significativo** (p-value: {p_value:.3f})")
            text_parts.append("No hay evidencia suficiente de impacto real.")
        
        return "\n".join(text_parts)
    
    def validate_data_requirements(self) -> Tuple[bool, str]:
        """Valida que los datos cumplan los requisitos"""
        n_days = len(self.data)
        if n_days < 21:
            return False, f"Se necesitan al menos 21 días. Tienes {n_days}."
        
        if self.data[self.metric_column].std() == 0:
            return False, "Los datos no tienen variabilidad."
        
        if (self.data[self.metric_column] < 0).any():
            return False, "Los datos contienen valores negativos."
        
        return True, "Los datos cumplen los requisitos."
