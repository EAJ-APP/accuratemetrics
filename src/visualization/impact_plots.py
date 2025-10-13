"""
Visualizaciones para Causal Impact - GRÁFICOS MEJORADOS
Enfoque en claridad visual y mostrar TODAS las líneas
Corrección de escalas absurdas en intervalos de confianza
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import streamlit as st


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
        MEJORADO: Muestra TODAS las líneas claramente
        """
        # Verificar datos
        if plot_data.empty:
            st.error("❌ El DataFrame está vacío")
            raise ValueError("El DataFrame está vacío")
        
        st.write("🔍 DEBUG - Datos para graficar:")
        st.write(f"  Shape: {plot_data.shape}")
        st.write(f"  Columnas: {plot_data.columns.tolist()}")
        st.write(f"  Index min: {plot_data.index.min()}")
        st.write(f"  Index max: {plot_data.index.max()}")
        st.write(f"  Intervention: {intervention_date}")
        
        # Verificar columnas necesarias
        required_cols = ['response', 'preds', 'preds_lower', 'preds_upper']
        missing_cols = [col for col in required_cols if col not in plot_data.columns]
        
        if missing_cols:
            st.error(f"❌ Faltan columnas: {missing_cols}")
            raise ValueError(f"Faltan columnas necesarias: {missing_cols}")
        
        st.success(f"✅ Todas las columnas necesarias están presentes")
        
        # Extraer datos
        actual_data = plot_data['response'].values
        predicted_data = plot_data['preds'].values
        predicted_lower_raw = plot_data['preds_lower'].values
        predicted_upper_raw = plot_data['preds_upper'].values
        
        # 🔥 CRÍTICO: Limitar intervalos de confianza a valores razonables
        # A veces pycausalimpact genera ICs absurdos (millones cuando datos son miles)
        
        # Calcular desviación estándar de los datos para tener una referencia
        data_std = actual_data.std()
        data_mean = actual_data.mean()
        
        # Los ICs no deberían estar más allá de ±4 desviaciones estándar de la predicción
        max_deviation = 4 * data_std
        
        predicted_lower = np.maximum(predicted_lower_raw, predicted_data - max_deviation)
        predicted_upper = np.minimum(predicted_upper_raw, predicted_data + max_deviation)
        
        # Asegurar que lower < upper
        predicted_lower = np.minimum(predicted_lower, predicted_data)
        predicted_upper = np.maximum(predicted_upper, predicted_data)
        
        st.write(f"  📊 Estadísticas de datos:")
        st.write(f"    - Media: {data_mean:.0f}")
        st.write(f"    - Std: {data_std:.0f}")
        st.write(f"    - Max deviation permitida: ±{max_deviation:.0f}")
        st.write(f"  📊 ICs originales vs ajustados:")
        st.write(f"    - Lower raw (primeros 3): {predicted_lower_raw[:3]}")
        st.write(f"    - Lower ajustado (primeros 3): {predicted_lower[:3]}")
        st.write(f"    - Upper raw (primeros 3): {predicted_upper_raw[:3]}")
        st.write(f"    - Upper ajustado (primeros 3): {predicted_upper[:3]}")
        
        # Fechas
        dates = plot_data.index.to_pydatetime()
        dates_list = list(dates)
        
        st.write(f"  Actual values (primeros 3): {actual_data[:3]}")
        st.write(f"  Predicted values (primeros 3): {predicted_data[:3]}")
        st.write(f"  Fechas (primeras 3): {dates_list[:3]}")
        
        # Convertir intervention_date
        if isinstance(intervention_date, pd.Timestamp):
            intervention_dt = intervention_date.to_pydatetime()
        else:
            intervention_dt = pd.Timestamp(intervention_date).to_pydatetime()
        
        # Crear figura con 3 paneles
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                '1️⃣ Observado vs Predicho (Contrafactual)',
                '2️⃣ Efecto Puntual (Diferencia Diaria)',
                '3️⃣ Efecto Acumulado'
            ),
            vertical_spacing=0.12,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # ================================================================
        # PANEL 1: Observado vs Predicho
        # ================================================================
        
        # Línea NEGRA GRUESA: Valores Reales (Observados)
        fig.add_trace(
            go.Scatter(
                x=dates_list,
                y=actual_data,
                mode='lines',
                name='📊 Observado (Real)',
                line=dict(color='black', width=3),
                showlegend=True,
                hovertemplate='Observado: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Línea AZUL PUNTEADA: Predicción (Contrafactual)
        fig.add_trace(
            go.Scatter(
                x=dates_list,
                y=predicted_data,
                mode='lines',
                name='🔮 Predicho (sin intervención)',
                line=dict(color='#1E88E5', width=3, dash='dash'),
                showlegend=True,
                hovertemplate='Predicho: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Banda de confianza del predicho (AZUL CLARO)
        fig.add_trace(
            go.Scatter(
                x=dates_list + dates_list[::-1],
                y=list(predicted_upper) + list(predicted_lower[::-1]),
                fill='toself',
                fillcolor='rgba(30, 136, 229, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='📏 IC 95% Predicción',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # ================================================================
        # PANEL 2: Efecto Puntual (Diferencia = Real - Predicho)
        # ================================================================
        
        effect = actual_data - predicted_data
        effect_upper = actual_data - predicted_lower
        effect_lower = actual_data - predicted_upper
        
        # Línea VERDE: Efecto positivo/negativo
        effect_colors = ['green' if e >= 0 else 'red' for e in effect]
        
        fig.add_trace(
            go.Scatter(
                x=dates_list,
                y=effect,
                mode='lines',
                name='💥 Efecto Diario',
                line=dict(color='#43A047', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(67, 160, 71, 0.2)',
                showlegend=True,
                hovertemplate='Efecto: %{y:+,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Banda de confianza del efecto
        fig.add_trace(
            go.Scatter(
                x=dates_list + dates_list[::-1],
                y=list(effect_upper) + list(effect_lower[::-1]),
                fill='toself',
                fillcolor='rgba(67, 160, 71, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        # Línea de referencia en cero
        fig.add_hline(
            y=0, 
            line_dash="solid", 
            line_color="gray", 
            line_width=1,
            row=2, col=1
        )
        
        # ================================================================
        # PANEL 3: Efecto Acumulado (REINICIADO EN LA INTERVENCIÓN)
        # ================================================================
        
        # 🔥 CORRECCIÓN: Reiniciar el acumulado en la fecha de intervención
        # Para que empiece en 0 y muestre solo el efecto POST-intervención
        
        # Crear máscara de intervención
        intervention_mask = pd.Series(dates) >= intervention_dt
        
        # Calcular acumulado solo desde la intervención
        cumulative_effect = np.zeros(len(effect))
        cumulative_upper = np.zeros(len(effect_upper))
        cumulative_lower = np.zeros(len(effect_lower))
        
        # Encontrar el índice de la intervención
        intervention_idx = np.where(intervention_mask)[0][0] if intervention_mask.any() else 0
        
        # Acumular solo desde la intervención hacia adelante
        if intervention_idx < len(effect):
            cumulative_effect[intervention_idx:] = np.cumsum(effect[intervention_idx:])
            cumulative_upper[intervention_idx:] = np.cumsum(effect_upper[intervention_idx:])
            cumulative_lower[intervention_idx:] = np.cumsum(effect_lower[intervention_idx:])
        
        st.write(f"  📊 Efecto acumulado reiniciado desde intervención (índice {intervention_idx})")
        
        # Línea NARANJA: Efecto acumulado
        fig.add_trace(
            go.Scatter(
                x=dates_list,
                y=cumulative_effect,
                mode='lines',
                name='📈 Efecto Acumulado',
                line=dict(color='#FB8C00', width=3),
                fill='tozeroy',
                fillcolor='rgba(251, 140, 0, 0.2)',
                showlegend=True,
                hovertemplate='Acumulado: %{y:+,.0f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Banda de confianza acumulada
        fig.add_trace(
            go.Scatter(
                x=dates_list + dates_list[::-1],
                y=list(cumulative_upper) + list(cumulative_lower[::-1]),
                fill='toself',
                fillcolor='rgba(251, 140, 0, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=3, col=1
        )
        
        # Línea de referencia en cero
        fig.add_hline(
            y=0, 
            line_dash="solid", 
            line_color="gray", 
            line_width=1,
            row=3, col=1
        )
        
        # ================================================================
        # LÍNEA VERTICAL ROJA: Marca de Intervención
        # ================================================================
        
        # Añadir en los 3 paneles
        for row_num in [1, 2, 3]:
            fig.add_shape(
                type="line",
                x0=intervention_dt,
                x1=intervention_dt,
                y0=0,
                y1=1,
                yref=f"y{row_num} domain" if row_num > 1 else "y domain",
                line=dict(color="red", width=3, dash="dash"),
                row=row_num,
                col=1
            )
        
        # Anotación solo en el primer panel
        fig.add_annotation(
            x=intervention_dt,
            y=1.08,
            yref="y domain",
            text="⚡ INTERVENCIÓN",
            showarrow=False,
            font=dict(color="red", size=14, family="Arial Black"),
            row=1,
            col=1
        )
        
        # ================================================================
        # Áreas de fondo PRE y POST
        # ================================================================
        
        # Fondo verde muy tenue para POST
        fig.add_vrect(
            x0=intervention_dt,
            x1=dates_list[-1],
            fillcolor="rgba(67, 160, 71, 0.05)",
            layer="below",
            line_width=0,
            row=1,
            col=1
        )
        
        # ================================================================
        # Layout final con EJES AJUSTADOS
        # ================================================================
        
        # Calcular rangos óptimos para cada panel
        # Solo usar valores razonables (no los ICs crudos que pueden ser absurdos)
        
        # Panel 1: Observado vs Predicho (usar solo actual y predicho base)
        panel1_values = np.concatenate([actual_data, predicted_data])
        panel1_min = panel1_values.min()
        panel1_max = panel1_values.max()
        panel1_range = panel1_max - panel1_min
        panel1_padding = panel1_range * 0.15  # 15% padding
        
        # Panel 2: Efecto puntual
        panel2_values = effect
        panel2_min = panel2_values.min()
        panel2_max = panel2_values.max()
        panel2_range = max(abs(panel2_min), abs(panel2_max))
        panel2_padding = panel2_range * 0.2  # 20% padding, simétrico
        
        # Panel 3: Efecto acumulado
        panel3_values = cumulative_effect
        panel3_min = panel3_values.min()
        panel3_max = panel3_values.max()
        panel3_range = max(abs(panel3_min), abs(panel3_max))
        panel3_padding = panel3_range * 0.2  # 20% padding, simétrico
        
        st.write(f"📊 Rangos calculados (ajustados):")
        st.write(f"  Panel 1: {panel1_min:.0f} a {panel1_max:.0f} (range: {panel1_range:.0f})")
        st.write(f"  Panel 2: {panel2_min:.0f} a {panel2_max:.0f}")
        st.write(f"  Panel 3: {panel3_min:.0f} a {panel3_max:.0f}")
        
        fig.update_layout(
            title={
                'text': title or f"<b>Análisis de Impacto Causal - {metric_name}</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=950,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1
            ),
            margin=dict(r=200)  # Más espacio para la leyenda
        )
        
        # Actualizar ejes X
        fig.update_xaxes(title_text="", showgrid=True, row=1, col=1)
        fig.update_xaxes(title_text="", showgrid=True, row=2, col=1)
        fig.update_xaxes(title_text="<b>Fecha</b>", showgrid=True, row=3, col=1)
        
        # Actualizar ejes Y con RANGOS AJUSTADOS (sin ICs absurdos)
        fig.update_yaxes(
            title_text=f"<b>{metric_name}</b>", 
            showgrid=True, 
            range=[panel1_min - panel1_padding, panel1_max + panel1_padding],
            row=1, 
            col=1
        )
        
        # Panel 2: Hacer simétrico alrededor de cero
        fig.update_yaxes(
            title_text="<b>Diferencia</b>", 
            showgrid=True,
            range=[-(panel2_range + panel2_padding), (panel2_range + panel2_padding)],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            row=2, 
            col=1
        )
        
        # Panel 3: Hacer simétrico alrededor de cero
        fig.update_yaxes(
            title_text="<b>Acumulado</b>", 
            showgrid=True,
            range=[-(panel3_range + panel3_padding), (panel3_range + panel3_padding)],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            row=3, 
            col=1
        )
        
        st.success("✅ Gráfico generado exitosamente")
        
        return fig
    
    @staticmethod
    def plot_summary_metrics(summary: Dict[str, Any]) -> go.Figure:
        """
        Crea un gráfico de barras con las métricas principales
        CORREGIDO: Mostrar average vs cumulative correctamente
        """
        # Extraer datos CORRECTAMENTE
        # 🔥 IMPORTANTE: average y cumulative son DIFERENTES
        
        # Efecto PROMEDIO (por día)
        avg_effect = summary['average']['rel_effect'] * 100
        avg_lower = summary['average']['rel_effect_lower'] * 100
        avg_upper = summary['average']['rel_effect_upper'] * 100
        
        # Efecto ACUMULADO (total del período)
        cum_effect = summary['cumulative']['rel_effect'] * 100
        cum_lower = summary['cumulative']['rel_effect_lower'] * 100
        cum_upper = summary['cumulative']['rel_effect_upper'] * 100
        
        # 🔥 DEBUG: Verificar que sean diferentes
        import streamlit as st
        st.write(f"🔍 DEBUG Resumen:")
        st.write(f"  Average effect: {avg_effect:.2f}%")
        st.write(f"  Cumulative effect: {cum_effect:.2f}%")
        st.write(f"  ¿Son iguales? {avg_effect == cum_effect}")
        
        if avg_effect == cum_effect:
            st.warning("⚠️ ADVERTENCIA: Average y Cumulative son iguales. Esto puede ser un bug del summary_data.")
            st.info("💡 Esto ocurre cuando CausalImpact no calcula correctamente el summary. Los valores se calculan desde inferences.")
        
        # Colores según si es positivo o negativo
        avg_color = '#43A047' if avg_effect >= 0 else '#E53935'
        cum_color = '#43A047' if cum_effect >= 0 else '#E53935'
        
        # Crear figura
        fig = go.Figure()
        
        # Barras principales con VALORES ABSOLUTOS también
        fig.add_trace(go.Bar(
            x=['Efecto Promedio<br>Diario', 'Efecto Total<br>Acumulado'],
            y=[avg_effect, cum_effect],
            marker_color=[avg_color, cum_color],
            text=[
                f"{avg_effect:+.2f}%<br>({summary['average']['abs_effect']:+,.0f})",
                f"{cum_effect:+.2f}%<br>({summary['cumulative']['abs_effect']:+,.0f})"
            ],
            textposition='outside',
            textfont=dict(size=14, family="Arial"),
            name='Efecto',
            width=0.5,
            hovertemplate='%{x}<br>Relativo: %{y:.2f}%<extra></extra>'
        ))
        
        # Barras de error (intervalos de confianza)
        fig.add_trace(go.Scatter(
            x=['Efecto Promedio<br>Diario', 'Efecto Total<br>Acumulado'],
            y=[avg_effect, cum_effect],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[avg_upper - avg_effect, cum_upper - cum_effect],
                arrayminus=[avg_effect - avg_lower, cum_effect - cum_lower],
                color='gray',
                thickness=3,
                width=15
            ),
            mode='markers',
            marker=dict(size=0.01, color='rgba(0,0,0,0)'),
            showlegend=False,
            name='IC 95%'
        ))
        
        # Línea en cero
        fig.add_hline(
            y=0, 
            line_dash="solid", 
            line_color="black", 
            line_width=2
        )
        
        # Layout
        fig.update_layout(
            title="<b>Resumen del Impacto (%)</b>",
            yaxis_title="<b>Cambio Porcentual (%)</b>",
            height=450,
            template='plotly_white',
            showlegend=False,
            font=dict(size=13)
        )
        
        # Añadir anotación explicativa
        fig.add_annotation(
            text="Las barras grises muestran el intervalo de confianza del 95%",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.15,
            showarrow=False,
            font=dict(size=11, color="gray")
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
        MEJORADO: Más informativo
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
            subplot_titles=(
                f'<b>PRE</b><br>{len(pre_data)} días', 
                f'<b>POST</b><br>{len(post_data)} días'
            ),
            specs=[[{'type': 'box'}, {'type': 'box'}]]
        )
        
        # Box plot PRE (azul)
        fig.add_trace(
            go.Box(
                y=pre_data[metric_column],
                name='Pre',
                marker_color='#90CAF9',
                boxmean='sd',
                showlegend=False,
                hovertemplate='%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Box plot POST (verde)
        fig.add_trace(
            go.Box(
                y=post_data[metric_column],
                name='Post',
                marker_color='#A5D6A7',
                boxmean='sd',
                showlegend=False,
                hovertemplate='%{y:,.0f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Calcular estadísticas
        pre_mean = pre_data[metric_column].mean()
        post_mean = post_data[metric_column].mean()
        change_pct = ((post_mean - pre_mean) / pre_mean * 100) if pre_mean != 0 else 0
        
        # Añadir líneas de media
        fig.add_hline(
            y=pre_mean,
            line_dash="dash",
            line_color="blue",
            line_width=2,
            row=1,
            col=1,
            annotation_text=f"Media: {pre_mean:,.0f}",
            annotation_position="right"
        )
        
        fig.add_hline(
            y=post_mean,
            line_dash="dash",
            line_color="green",
            line_width=2,
            row=1,
            col=2,
            annotation_text=f"Media: {post_mean:,.0f}",
            annotation_position="right"
        )
        
        # Layout
        change_color = "green" if change_pct >= 0 else "red"
        
        fig.update_layout(
            title=f"<b>Comparación Pre vs Post - {metric_column.title()}</b><br>" + 
                  f"<span style='color:{change_color};font-size:16px'>Cambio: {change_pct:+.1f}%</span>",
            height=450,
            showlegend=False,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text=f"<b>{metric_column.title()}</b>", row=1, col=1)
        fig.update_yaxes(title_text=f"<b>{metric_column.title()}</b>", row=1, col=2)
        
        return fig
