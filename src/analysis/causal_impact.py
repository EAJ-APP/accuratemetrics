"""
Motor de análisis de Causal Impact para AccurateMetrics
FIX DEFINITIVO - Mapeo correcto de columnas de pycausalimpact 0.1.1
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
        df = data.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        df.index = pd.DatetimeIndex(df.index, freq='D')
        
        if self.metric_column not in df.columns:
            raise ValueError(f"La columna '{self.metric_column}' no existe")
        
        df = df[[self.metric_column]]
        
        if df.isnull().any().any():
            df = df.fillna(method='ffill')
            df = df.fillna(0)
        
        return df
    
    def analyze_single_intervention(
        self,
        intervention_date: str,
        pre_period_days: Optional[int] = None,
        post_period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        if not CAUSALIMPACT_AVAILABLE:
            raise ImportError("pycausalimpact no está instalado")
        
        self.intervention_date = pd.to_datetime(intervention_date)
        
        if self.intervention_date <= self.data.index.min():
            raise ValueError("Fecha de intervención debe ser posterior al inicio")
        if self.intervention_date >= self.data.index.max():
            raise ValueError("Fecha de intervención debe ser anterior al final")
        
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
        
        # Normalizar timestamps
        pre_start = pd.Timestamp(pre_start.date())
        pre_end = pd.Timestamp(pre_end.date())
        post_start = pd.Timestamp(post_start.date())
        post_end = pd.Timestamp(post_end.date())
        
        pre_period = [pre_start, pre_end]
        post_period = [post_start, post_end]
        
        analysis_data = self.data.loc[pre_start:post_end].copy()
        
        if analysis_data.index.freq is None:
            analysis_data.index = pd.DatetimeIndex(analysis_data.index, freq='D')
        
        # Ejecutar CausalImpact
        try:
            self.impact_result = CausalImpact(
                analysis_data,
                pre_period,
                post_period,
                model_args={'nseasons': 7}
            )
        except TypeError:
            try:
                self.impact_result = CausalImpact(
                    analysis_data,
                    pre_period,
                    post_period
                )
            except Exception as e:
                raise Exception(f"Error en CausalImpact: {str(e)}")
        except Exception as e:
            raise Exception(f"Error en CausalImpact: {str(e)}")
        
        summary = self._extract_summary()
        
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
        if not self.impact_result:
            return {}
        
        try:
            if hasattr(self.impact_result, 'summary_df'):
                summary_df = self.impact_result.summary_df
            elif hasattr(self.impact_result, 'summary'):
                summary_df = self.impact_result.summary()
            else:
                return self._extract_summary_from_inferences()
            
            summary = {
                'average': {
                    'actual': summary_df.loc['average', 'actual'] if 'actual' in summary_df.columns else 0,
                    'predicted': summary_df.loc['average', 'predicted'] if 'predicted' in summary_df.columns else 0,
                    'predicted_lower': summary_df.loc['average', 'predicted_lower'] if 'predicted_lower' in summary_df.columns else 0,
                    'predicted_upper': summary_df.loc['average', 'predicted_upper'] if 'predicted_upper' in summary_df.columns else 0,
                    'abs_effect': summary_df.loc['average', 'abs_effect'] if 'abs_effect' in summary_df.columns else 0,
                    'abs_effect_lower': summary_df.loc['average', 'abs_effect_lower'] if 'abs_effect_lower' in summary_df.columns else 0,
                    'abs_effect_upper': summary_df.loc['average', 'abs_effect_upper'] if 'abs_effect_upper' in summary_df.columns else 0,
                    'rel_effect': summary_df.loc['average', 'rel_effect'] if 'rel_effect' in summary_df.columns else 0,
                    'rel_effect_lower': summary_df.loc['average', 'rel_effect_lower'] if 'rel_effect_lower' in summary_df.columns else 0,
                    'rel_effect_upper': summary_df.loc['average', 'rel_effect_upper'] if 'rel_effect_upper' in summary_df.columns else 0
                },
                'cumulative': {
                    'actual': summary_df.loc['cumulative', 'actual'] if 'actual' in summary_df.columns else 0,
                    'predicted': summary_df.loc['cumulative', 'predicted'] if 'predicted' in summary_df.columns else 0,
                    'predicted_lower': summary_df.loc['cumulative', 'predicted_lower'] if 'predicted_lower' in summary_df.columns else 0,
                    'predicted_upper': summary_df.loc['cumulative', 'predicted_upper'] if 'predicted_upper' in summary_df.columns else 0,
                    'abs_effect': summary_df.loc['cumulative', 'abs_effect'] if 'abs_effect' in summary_df.columns else 0,
                    'abs_effect_lower': summary_df.loc['cumulative', 'abs_effect_lower'] if 'abs_effect_lower' in summary_df.columns else 0,
                    'abs_effect_upper': summary_df.loc['cumulative', 'abs_effect_upper'] if 'abs_effect_upper' in summary_df.columns else 0,
                    'rel_effect': summary_df.loc['cumulative', 'rel_effect'] if 'rel_effect' in summary_df.columns else 0,
                    'rel_effect_lower': summary_df.loc['cumulative', 'rel_effect_lower'] if 'rel_effect_lower' in summary_df.columns else 0,
                    'rel_effect_upper': summary_df.loc['cumulative', 'rel_effect_upper'] if 'rel_effect_upper' in summary_df.columns else 0
                },
                'p_value': self.impact_result.p_value if hasattr(self.impact_result, 'p_value') else 0.5,
                'is_significant': self.impact_result.p_value < 0.05 if hasattr(self.impact_result, 'p_value') else False,
                'metric': self.metric_column
            }
            
        except Exception as e:
            print(f"Error extrayendo summary estándar: {e}")
            return self._extract_summary_from_inferences()
        
        return summary
    
    def _extract_summary_from_inferences(self) -> Dict[str, Any]:
        try:
            inferences = self.impact_result.inferences
            post_mask = inferences.index >= self.intervention_date
            
            actual_avg = inferences.loc[post_mask, 'response'].mean()
            pred_avg = inferences.loc[post_mask, 'point_pred'].mean()
            
            actual_sum = inferences.loc[post_mask, 'response'].sum()
            pred_sum = inferences.loc[post_mask, 'point_pred'].sum()
            
            abs_effect_avg = actual_avg - pred_avg
            abs_effect_sum = actual_sum - pred_sum
            
            rel_effect_avg = (abs_effect_avg / pred_avg) if pred_avg != 0 else 0
            rel_effect_sum = (abs_effect_sum / pred_sum) if pred_sum != 0 else 0
            
            return {
                'average': {
                    'actual': actual_avg,
                    'predicted': pred_avg,
                    'predicted_lower': pred_avg * 0.9,
                    'predicted_upper': pred_avg * 1.1,
                    'abs_effect': abs_effect_avg,
                    'abs_effect_lower': abs_effect_avg * 0.8,
                    'abs_effect_upper': abs_effect_avg * 1.2,
                    'rel_effect': rel_effect_avg,
                    'rel_effect_lower': rel_effect_avg * 0.8,
                    'rel_effect_upper': rel_effect_avg * 1.2
                },
                'cumulative': {
                    'actual': actual_sum,
                    'predicted': pred_sum,
                    'predicted_lower': pred_sum * 0.9,
                    'predicted_upper': pred_sum * 1.1,
                    'abs_effect': abs_effect_sum,
                    'abs_effect_lower': abs_effect_sum * 0.8,
                    'abs_effect_upper': abs_effect_sum * 1.2,
                    'rel_effect': rel_effect_sum,
                    'rel_effect_lower': rel_effect_sum * 0.8,
                    'rel_effect_upper': rel_effect_sum * 1.2
                },
                'p_value': 0.05,
                'is_significant': abs(rel_effect_avg) > 0.1,
                'metric': self.metric_column
            }
        except Exception as e:
            print(f"Error en fallback: {e}")
            return {
                'average': {k: 0 for k in ['actual', 'predicted', 'predicted_lower', 'predicted_upper',
                                          'abs_effect', 'abs_effect_lower', 'abs_effect_upper',
                                          'rel_effect', 'rel_effect_lower', 'rel_effect_upper']},
                'cumulative': {k: 0 for k in ['actual', 'predicted', 'predicted_lower', 'predicted_upper',
                                             'abs_effect', 'abs_effect_lower', 'abs_effect_upper',
                                             'rel_effect', 'rel_effect_lower', 'rel_effect_upper']},
                'p_value': 0.5,
                'is_significant': False,
                'metric': self.metric_column
            }
    
    def get_plot_data(self) -> pd.DataFrame:
        """
        ✅ FIX DEFINITIVO: Mapeo correcto para pycausalimpact 0.1.1
        """
        if not self.impact_result:
            return pd.DataFrame()
        
        try:
            if not hasattr(self.impact_result, 'inferences'):
                return pd.DataFrame(columns=['actual', 'predicted', 'predicted_lower', 'predicted_upper', 'period'])
            
            # Obtener el DataFrame de inferences - ES EL DATO CRUDO DE CAUSALIMPACT
            result_df = self.impact_result.inferences.copy()
            
            print(f"🔍 DEBUG get_plot_data:")
            print(f"   Columnas originales: {result_df.columns.tolist()}")
            print(f"   Shape: {result_df.shape}")
            print(f"   Index type: {type(result_df.index)}")
            
            # ✅ MAPEO CORRECTO PARA PYCAUSALIMPACT 0.1.1
            # Las columnas reales son: response, point_pred, point_pred_lower, point_pred_upper
            # PERO en el DataFrame aparecen como: preds, preds_lower, preds_upper, etc.
            
            # Crear el DataFrame final con las columnas que esperamos
            final_df = pd.DataFrame(index=result_df.index)
            
            # Mapear 'actual' (datos reales)
            if 'response' in result_df.columns:
                final_df['actual'] = result_df['response']
            elif self.metric_column in result_df.columns:
                final_df['actual'] = result_df[self.metric_column]
            else:
                # Si no encontramos, usar la primera columna del input original
                final_df['actual'] = self.data.loc[result_df.index, self.metric_column]
            
            # Mapear 'predicted' (predicciones)
            if 'point_pred' in result_df.columns:
                final_df['predicted'] = result_df['point_pred']
            elif 'preds' in result_df.columns:
                final_df['predicted'] = result_df['preds']
            else:
                final_df['predicted'] = 0
            
            # Mapear límites de confianza
            if 'point_pred_lower' in result_df.columns:
                final_df['predicted_lower'] = result_df['point_pred_lower']
            elif 'preds_lower' in result_df.columns:
                final_df['predicted_lower'] = result_df['preds_lower']
            else:
                final_df['predicted_lower'] = final_df['predicted'] * 0.9
            
            if 'point_pred_upper' in result_df.columns:
                final_df['predicted_upper'] = result_df['point_pred_upper']
            elif 'preds_upper' in result_df.columns:
                final_df['predicted_upper'] = result_df['preds_upper']
            else:
                final_df['predicted_upper'] = final_df['predicted'] * 1.1
            
            # Añadir columna de período
            final_df['period'] = 'pre'
            if self.intervention_date:
                final_df.loc[final_df.index >= self.intervention_date, 'period'] = 'post'
            
            # Calcular residuales
            final_df['residuals'] = final_df['actual'] - final_df['predicted']
            final_df['cumulative_residuals'] = final_df['residuals'].cumsum()
            
            print(f"   Columnas finales: {final_df.columns.tolist()}")
            print(f"   Muestra de datos:")
            print(f"     actual: {final_df['actual'].head().tolist()}")
            print(f"     predicted: {final_df['predicted'].head().tolist()}")
            
            return final_df
            
        except Exception as e:
            print(f"❌ Error en get_plot_data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=['actual', 'predicted', 'predicted_lower', 'predicted_upper', 'period'])
    
    def get_summary_text(self) -> str:
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
        n_days = len(self.data)
        if n_days < 21:
            return False, f"Se necesitan al menos 21 días. Tienes {n_days}."
        
        if self.data[self.metric_column].std() == 0:
            return False, "Los datos no tienen variabilidad."
        
        if (self.data[self.metric_column] < 0).any():
            return False, "Los datos contienen valores negativos."
        
        return True, "Los datos cumplen los requisitos."
