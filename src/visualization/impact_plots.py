"""
Visualizaciones para Causal Impact - VERSIÓN FINAL CORREGIDA
Compatible con estructura de columnas de pycausalimpact 0.1.1
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


class ImpactVisualizer:
    """Clase para crear visualizaciones de Causal Impact"""
    
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
        # Verificar que tenemos datos
        if plot_data.empty:
            raise ValueError("El DataFrame está vacío")
        
        # Obtener las columnas necesarias
        # pycausalimpact 0.1.1 usa: response, preds, preds_lower, preds_upper
        actual_data = plot_data['response']
        predicted_data = plot_data['preds']
        predicted_lower = plot_data['preds_lower']
        predicted_upper = plot_data['preds_upper']
        
        # Convertir índice a lista de datetime
        dates = plot_data.index
        if isinstance(dates, pd.DatetimeIndex):
            dates_list = dates.to_pydatetime().tolist()
        else:
            dates_list = list(dates)
        
        # Convertir intervention_date
        if isinstance(intervention_date, pd.Timestamp):
            intervention_dt = intervention_date.to_pydatetime()
        elif isinstance(intervention_date, str):
            intervention_dt = pd.Timestamp(intervention_date).to_pydatetime()
        else:
            intervention_dt = intervention_date
        
        # Crear figura con 3 subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Observado vs Predicho',
                'Efecto Puntual',
                'Efecto Acumulado'
            ),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # ================================================================
        # PANEL 1: Observado vs Predicho
        # ================================================================
        
        # Línea de valores observados (reales)
        fig.add_trace(
            go.Scatter(
                x=dates_list,
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
                name='Predicho (contrafactual)',
                line=dict(color='#1E88E5', width=2, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Banda de confianza (95%)
        fig.add_trace(
            go.Scatter(
                x=dates_list + dates_list[::-1],
                y=predicted_upper.tolist() + predicted_lower.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(30, 136, 229, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='IC 95%',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # ================================================================
        # PANEL 2: Efecto Puntual (diferencia día a día)
        # ================================================================
        
        effect = actual_data - predicted_data
        effect_upper = actual_data - predicted_lower
        effect_lower = actual_data - predicted_upper
        
        # Línea de efecto
        fig.add_trace(
            go.Scatter(
                x=dates_list,
                y=effect,
                mode='lines',
                name='Efecto',
                line=dict(color='#43A047', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Banda de confianza del efecto
        fig.add_trace(
            go.Scatter(
                x=dates_list + dates_list[::-1],
                y=effect_upper.tolist() + effect_lower.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(67, 160, 71, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        # Línea en cero
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
        
        # ================================================================
        # PANEL 3: Efecto Acumulado
        # ================================================================
        
        cumulative_effect = effect.cumsum()
        cumulative_upper = effect_upper.cumsum()
        cumulative_lower = effect_lower.cumsum()
        
        # Línea de efecto acumulado
        fig.add_trace(
            go.Scatter(
                x=dates_list,
                y=cumulative_effect,
                mode='lines',
                name='Efecto Acumulado',
                line=dict(color='#FB8C00', width=2),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Banda de confianza acumulada
        fig.add_trace(
            go.Scatter(
                x=dates_list + dates_list[::-1],
                y=cumulative_upper.tolist() + cumulative_lower.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(251, 140, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=3, col=1
        )
        
        # Línea en cero
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
        
        # ================================================================
        # Línea vertical de intervención en los 3 paneles
        # ================================================================
        
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
        
        # Anotación de intervención
        fig.add_annotation(
            x=intervention_dt,
            y=1.05,
            yref="y domain",
            text="↓ Intervención",
            showarrow=False,
            font=dict(color="red", size=12, family="Arial Black"),
            row=1,
            col=1
        )
        
        # ================================================================
        # Layout final
        # ================================================================
        
        fig.update_layout(
            title={
                'text': title or f"Análisis de Impacto Causal - {metric_name}",
                'x': 0.5,
                'xanchor': 'center'
            },
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
        
        # Determinar colores según si es positivo o negativo
        avg_color = '#43A047' if avg_effect >= 0 else '#E53935'
        cum_color = '#43A047' if cum_effect >= 0 else '#E53935'
        
        # Barras
        fig.add_trace(go.Bar(
            x=['Efecto Promedio', 'Efecto Acumulado'],
            y=[avg_effect, cum_effect],
            marker_color=[avg_color, cum_color],
            text=[f"{avg_effect:+.1f}%", f"{cum_effect:+.1f}%"],
            textposition='outside',
            textfont=dict(size=14, family="Arial Black"),
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
        
        # Layout
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
        """
        # Preparar datos
        data_copy = data.copy()
        if 'date' in data_copy.columns:
            data_copy['date'] = pd.to_datetime(data_copy['date'])
        elif data_copy.index.name == 'date':
            data_copy.reset_index(inplace=True)
            data_copy['date'] = pd.to_datetime(data_copy['date'])
        
        # Convertir intervention_date
        if not isinstance(intervention_date, pd.Timestamp):
            intervention_date = pd.Timestamp(intervention_date)
        
        # Separar períodos
        pre_data = data_copy[data_copy['date'] < intervention_date]
        post_data = data_copy[data_copy['date'] >= intervention_date]
        
        # Crear figura con subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Pre-Intervención', 'Post-Intervención'),
            specs=[[{'type': 'box'}, {'type': 'box'}]]
        )
        
        # Box plot pre
        fig.add_trace(
            go.Box(
                y=pre_data[metric_column],
                name='Pre',
                marker_color='#90CAF9',
                boxmean='sd',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Box plot post
        fig.add_trace(
            go.Box(
                y=post_data[metric_column],
                name='Post',
                marker_color='#A5D6A7',
                boxmean='sd',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Calcular estadísticas
        pre_mean = pre_data[metric_column].mean()
        post_mean = post_data[metric_column].mean()
        change_pct = ((post_mean - pre_mean) / pre_mean * 100) if pre_mean != 0 else 0
        
        # Layout
        fig.update_layout(
            title=f"Comparación Pre vs Post - {metric_column.title()}<br><sub>Cambio: {change_pct:+.1f}%</sub>",
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text=metric_column.title(), row=1, col=1)
        fig.update_yaxes(title_text=metric_column.title(), row=1, col=2)
        
        return fig
