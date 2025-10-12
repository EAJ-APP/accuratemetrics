"""
Visualizaciones para Causal Impact - VERSIÓN COMPLETAMENTE CORREGIDA
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


class ImpactVisualizer:
    """
    Clase para crear visualizaciones de Causal Impact
    """
    
    @staticmethod
    def plot_impact_analysis(
        plot_data: pd.DataFrame,
        intervention_date: pd.Timestamp,
        metric_name: str = "Métrica",
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Crea el gráfico principal del análisis de impacto
        """
        # ✅ DEBUG: Imprimir información del DataFrame
        import streamlit as st
        st.write("🔍 DEBUG - Información del DataFrame:")
        st.write(f"Columnas disponibles: {plot_data.columns.tolist()}")
        st.write(f"Shape: {plot_data.shape}")
        st.write("Primeras 3 filas:")
        st.dataframe(plot_data.head(3))
        
        # Verificar y mapear columnas
        column_mapping = {
            'actual': ['response', 'actual', 'y', 'sessions'],
            'predicted': ['point_pred', 'preds', 'predicted', 'point_prediction'],
            'predicted_lower': ['point_pred_lower', 'preds_lower', 'predicted_lower'],
            'predicted_upper': ['point_pred_upper', 'preds_upper', 'predicted_upper']
        }
        
        # Función helper
        def get_column(df, key):
            """Busca la columna en el DataFrame"""
            for col_name in column_mapping[key]:
                if col_name in df.columns:
                    st.write(f"✅ {key}: Encontrada columna '{col_name}'")
                    st.write(f"   Valores: {df[col_name].head(3).tolist()}")
                    return df[col_name]
            
            st.error(f"❌ {key}: NO encontrada. Buscaba: {column_mapping[key]}")
            return pd.Series(0, index=df.index)
        
        # Obtener las columnas necesarias
        actual_data = get_column(plot_data, 'actual')
        predicted_data = get_column(plot_data, 'predicted')
        predicted_lower = get_column(plot_data, 'predicted_lower')
        predicted_upper = get_column(plot_data, 'predicted_upper')
        
        # Si no hay límites de confianza, crear aproximaciones
        if (predicted_lower == 0).all():
            predicted_lower = predicted_data * 0.9
        if (predicted_upper == 0).all():
            predicted_upper = predicted_data * 1.1
        
        # Crear subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Observado vs Predicho',
                'Efecto Puntual (Diferencia)',
                'Efecto Acumulado'
            ),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # ✅ CORRECCIÓN: Convertir índice a lista de datetime de Python
        dates = plot_data.index
        
        # Convertir a datetime de Python (no pandas Timestamp)
        if isinstance(dates, pd.DatetimeIndex):
            dates_python = dates.to_pydatetime()
            dates_list = dates_python.tolist()
        elif hasattr(dates, 'tolist'):
            dates_list = dates.tolist()
        else:
            dates_list = list(dates)
        
        # ===================================================================
        # Panel 1: Observado vs Predicho
        # ===================================================================
        
        # Línea de valores observados
        fig.add_trace(
            go.Scatter(
                x=dates_list,  # Usar lista de datetime de Python
                y=actual_data,
                mode='lines',
                name='Observado',
                line=dict(color='black', width=2),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Línea de predicción
        fig.add_trace(
            go.Scatter(
                x=dates_list,
                y=predicted_data,
                mode='lines',
                name='Predicho',
                line=dict(color='blue', width=2, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Banda de confianza
        upper_values = predicted_upper.values.tolist()
        lower_values = predicted_lower.values.tolist()
        
        fig.add_trace(
            go.Scatter(
                x=dates_list + dates_list[::-1],
                y=upper_values + lower_values[::-1],
                fill='toself',
                fillcolor='rgba(0, 100, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='IC 95%'
            ),
            row=1, col=1
        )
        
        # ===================================================================
        # Panel 2: Efecto Puntual
        # ===================================================================
        
        effect = actual_data - predicted_data
        effect_upper = actual_data - predicted_lower
        effect_lower = actual_data - predicted_upper
        
        fig.add_trace(
            go.Scatter(
                x=dates_list,
                y=effect,
                mode='lines',
                name='Efecto',
                line=dict(color='green', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Banda de confianza del efecto
        effect_upper_values = effect_upper.values.tolist()
        effect_lower_values = effect_lower.values.tolist()
        
        fig.add_trace(
            go.Scatter(
                x=dates_list + dates_list[::-1],
                y=effect_upper_values + effect_lower_values[::-1],
                fill='toself',
                fillcolor='rgba(0, 255, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Línea en cero
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
        
        # ===================================================================
        # Panel 3: Efecto Acumulado
        # ===================================================================
        
        cumulative_effect = effect.cumsum()
        cumulative_upper = effect_upper.cumsum()
        cumulative_lower = effect_lower.cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=dates_list,
                y=cumulative_effect,
                mode='lines',
                name='Efecto Acumulado',
                line=dict(color='orange', width=2),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Banda de confianza acumulada
        cum_upper_values = cumulative_upper.values.tolist()
        cum_lower_values = cumulative_lower.values.tolist()
        
        fig.add_trace(
            go.Scatter(
                x=dates_list + dates_list[::-1],
                y=cum_upper_values + cum_lower_values[::-1],
                fill='toself',
                fillcolor='rgba(255, 165, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Línea en cero
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
        
        # ===================================================================
        # Línea vertical de intervención
        # ===================================================================
        
        # ✅ CORRECCIÓN: Usar add_shape en lugar de add_vline
        if isinstance(intervention_date, pd.Timestamp):
            intervention_dt = intervention_date.to_pydatetime()
        elif isinstance(intervention_date, str):
            intervention_dt = pd.Timestamp(intervention_date).to_pydatetime()
        else:
            intervention_dt = intervention_date
        
        # Añadir línea vertical manualmente en cada subplot
        # Nota: el primer eje Y es 'y', los demás son 'y2', 'y3', etc.
        y_refs = ['y domain', 'y2 domain', 'y3 domain']
        
        for idx, (row_num, yref) in enumerate(zip([1, 2, 3], y_refs)):
            fig.add_shape(
                type="line",
                x0=intervention_dt,
                x1=intervention_dt,
                y0=0,
                y1=1,
                yref=yref,
                line=dict(color="red", width=2, dash="dash"),
                row=row_num,
                col=1
            )
        
        # Añadir anotación solo en el primer panel
        fig.add_annotation(
            x=intervention_dt,
            y=1.05,
            yref="y domain",
            text="Intervención",
            showarrow=False,
            font=dict(color="red", size=12),
            row=1,
            col=1
        )
        
        # ===================================================================
        # Layout final
        # ===================================================================
        
        fig.update_layout(
            title=title or f"Análisis de Impacto Causal - {metric_name}",
            height=900,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Actualizar ejes
        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_xaxes(title_text="", row=2, col=1)
        fig.update_xaxes(title_text="Fecha", row=3, col=1)
        
        fig.update_yaxes(title_text=metric_name, row=1, col=1)
        fig.update_yaxes(title_text="Diferencia", row=2, col=1)
        fig.update_yaxes(title_text="Acumulado", row=3, col=1)
        
        return fig
    
    @staticmethod
    def plot_summary_metrics(summary: Dict[str, Any]) -> go.Figure:
        """
        Crea un gráfico de barras con las métricas principales
        
        Args:
            summary: Diccionario con el resumen del análisis
            
        Returns:
            Figura de Plotly
        """
        # Extraer datos
        avg_effect = summary['average']['rel_effect'] * 100
        avg_lower = summary['average']['rel_effect_lower'] * 100
        avg_upper = summary['average']['rel_effect_upper'] * 100
        
        cum_effect = summary['cumulative']['rel_effect'] * 100
        cum_lower = summary['cumulative']['rel_effect_lower'] * 100
        cum_upper = summary['cumulative']['rel_effect_upper'] * 100
        
        # Crear figura
        fig = go.Figure()
        
        # Barras principales
        fig.add_trace(go.Bar(
            x=['Efecto Promedio', 'Efecto Acumulado'],
            y=[avg_effect, cum_effect],
            marker_color=['blue', 'green'],
            text=[f"{avg_effect:.1f}%", f"{cum_effect:.1f}%"],
            textposition='outside',
            name='Efecto'
        ))
        
        # Barras de error (intervalos de confianza)
        fig.add_trace(go.Scatter(
            x=['Efecto Promedio', 'Efecto Acumulado'],
            y=[avg_effect, cum_effect],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[avg_upper - avg_effect, cum_upper - cum_effect],
                arrayminus=[avg_effect - avg_lower, cum_effect - cum_lower],
                color='gray',
                thickness=2,
                width=10
            ),
            mode='markers',
            marker=dict(size=0.01, color='rgba(0,0,0,0)'),
            showlegend=False
        ))
        
        # Línea en cero
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
        
        # Actualizar layout
        fig.update_layout(
            title="Resumen del Impacto (%)",
            yaxis_title="Cambio Porcentual",
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_period_comparison(
        data: pd.DataFrame,
        intervention_date: pd.Timestamp,
        metric_column: str
    ) -> go.Figure:
        """
        Compara los períodos pre y post intervención
        
        Args:
            data: DataFrame original con los datos
            intervention_date: Fecha de intervención
            metric_column: Columna de la métrica
            
        Returns:
            Figura de Plotly
        """
        # Separar períodos
        data_copy = data.copy()
        if 'date' in data_copy.columns:
            data_copy['date'] = pd.to_datetime(data_copy['date'])
        elif data_copy.index.name == 'date':
            data_copy.reset_index(inplace=True)
            data_copy['date'] = pd.to_datetime(data_copy['date'])
        
        # Convertir intervention_date a Timestamp si es necesario
        if not isinstance(intervention_date, pd.Timestamp):
            intervention_date = pd.Timestamp(intervention_date)
        
        pre_data = data_copy[data_copy['date'] < intervention_date]
        post_data = data_copy[data_copy['date'] >= intervention_date]
        
        # Crear figura con subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distribución Pre-Intervención', 'Distribución Post-Intervención'),
            specs=[[{'type': 'box'}, {'type': 'box'}]]
        )
        
        # Box plot pre
        fig.add_trace(
            go.Box(
                y=pre_data[metric_column],
                name='Pre',
                marker_color='lightblue',
                boxmean='sd'
            ),
            row=1, col=1
        )
        
        # Box plot post
        fig.add_trace(
            go.Box(
                y=post_data[metric_column],
                name='Post',
                marker_color='lightgreen',
                boxmean='sd'
            ),
            row=1, col=2
        )
        
        # Actualizar layout
        fig.update_layout(
            title=f"Comparación Pre vs Post - {metric_column.title()}",
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text=metric_column.title(), row=1, col=1)
        fig.update_yaxes(title_text=metric_column.title(), row=1, col=2)
        
        return fig