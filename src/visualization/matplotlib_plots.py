"""
Visualizaciones con Matplotlib y Seaborn para Causal Impact Avanzado
AccurateMetrics - Módulo de Visualización

Este módulo proporciona gráficos de alta calidad para:
- Análisis exploratorio de datos
- Matriz de correlación
- Gráficos de CausalImpact personalizados
- Comparación de intervenciones
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import io


# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Colores personalizados
COLORS = {
    'primary': '#2ecc71',       # Verde
    'secondary': '#3498db',     # Azul
    'tertiary': '#9b59b6',      # Púrpura
    'quaternary': '#f39c12',    # Naranja
    'positive': '#27ae60',      # Verde oscuro
    'negative': '#e74c3c',      # Rojo
    'neutral': '#95a5a6',       # Gris
    'highlight': '#e74c3c',     # Rojo para intervenciones
    'intervention_1': '#e74c3c',  # Rojo
    'intervention_2': '#f39c12',  # Naranja
}


def plot_exploratory_analysis(
    data: pd.DataFrame,
    intervention_dates: Optional[List[str]] = None,
    response_variable: str = 'conversiones',
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Crear gráfico de análisis exploratorio con 6 paneles

    Args:
        data: DataFrame con los datos
        intervention_dates: Lista de fechas de intervención
        response_variable: Variable respuesta
        figsize: Tamaño de la figura

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Asegurar que tenemos índice de fecha
    if 'date' in data.columns:
        plot_data = data.set_index('date')
    else:
        plot_data = data.copy()

    # Panel 1: Variable respuesta (conversiones) con intervenciones
    ax1 = axes[0, 0]
    ax1.plot(plot_data.index, plot_data[response_variable],
             linewidth=1.5, color=COLORS['primary'], label=response_variable.replace('_', ' ').title())

    if intervention_dates:
        for i, int_date in enumerate(intervention_dates):
            int_ts = pd.Timestamp(int_date)
            color = COLORS['intervention_1'] if i == 0 else COLORS['intervention_2']
            ax1.axvline(int_ts, color=color, linestyle='--', linewidth=2,
                       alpha=0.7, label=f'Intervención {i+1}')
            # Sombra para período post
            ax1.axvspan(int_ts, plot_data.index.max(),
                       alpha=0.1, color=color)

    ax1.set_title('Variable Respuesta con Intervenciones', fontweight='bold', fontsize=13)
    ax1.set_ylabel(response_variable.replace('_', ' ').title())
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 2: Sesiones totales
    ax2 = axes[0, 1]
    if 'sesiones_totales' in plot_data.columns:
        ax2.plot(plot_data.index, plot_data['sesiones_totales'],
                 linewidth=1.5, color=COLORS['secondary'])
        ax2.set_title('Sesiones Totales (Variable de Control)', fontweight='bold', fontsize=13)
        ax2.set_ylabel('Sesiones')
    else:
        ax2.text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center', transform=ax2.transAxes)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 3: Usuarios únicos
    ax3 = axes[1, 0]
    if 'usuarios_unicos' in plot_data.columns:
        ax3.plot(plot_data.index, plot_data['usuarios_unicos'],
                 linewidth=1.5, color=COLORS['tertiary'])
        ax3.set_title('Usuarios Únicos (Variable de Control)', fontweight='bold', fontsize=13)
        ax3.set_ylabel('Usuarios')
    else:
        ax3.text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center', transform=ax3.transAxes)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 4: Tráfico orgánico
    ax4 = axes[1, 1]
    if 'trafico_organico' in plot_data.columns:
        ax4.plot(plot_data.index, plot_data['trafico_organico'],
                 linewidth=1.5, color=COLORS['quaternary'])
        ax4.set_title('Tráfico Orgánico (Variable de Control)', fontweight='bold', fontsize=13)
        ax4.set_ylabel('Sesiones Orgánicas')
    else:
        ax4.text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center', transform=ax4.transAxes)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 5: Correlaciones con variable respuesta (barras horizontales)
    ax5 = axes[2, 0]
    numeric_cols = plot_data.select_dtypes(include=[np.number]).columns
    if response_variable in numeric_cols:
        corr = plot_data[numeric_cols].corr()[response_variable].drop(response_variable)
        corr = corr.sort_values(ascending=True)

        # Colores según correlación
        colors = [COLORS['positive'] if x > 0.5 else COLORS['negative'] if x < -0.5 else COLORS['neutral']
                  for x in corr.values]

        bars = ax5.barh(range(len(corr)), corr.values, color=colors)
        ax5.set_yticks(range(len(corr)))
        ax5.set_yticklabels([c.replace('_', ' ').title() for c in corr.index], fontsize=9)
        ax5.set_xlabel('Correlación')
        ax5.set_title('Correlación con Variable Respuesta', fontweight='bold', fontsize=13)
        ax5.axvline(0, color='black', linewidth=1)
        ax5.axvline(0.5, color='green', linewidth=1, linestyle='--', alpha=0.5)
        ax5.axvline(-0.5, color='red', linewidth=1, linestyle='--', alpha=0.5)
        ax5.set_xlim(-1, 1)
        ax5.grid(True, alpha=0.3, axis='x')

        # Añadir valores en las barras
        for bar, val in zip(bars, corr.values):
            ax5.text(val + 0.02 if val >= 0 else val - 0.02,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', ha='left' if val >= 0 else 'right',
                    fontsize=8)

    # Panel 6: Matriz de correlación (heatmap)
    ax6 = axes[2, 1]
    if len(numeric_cols) > 1:
        corr_matrix = plot_data[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        # Crear heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, ax=ax6, vmin=-1, vmax=1,
                    cbar_kws={'label': 'Correlación'},
                    annot_kws={'size': 8},
                    xticklabels=[c.replace('_', ' ')[:12] for c in corr_matrix.columns],
                    yticklabels=[c.replace('_', ' ')[:12] for c in corr_matrix.index])
        ax6.set_title('Matriz de Correlación', fontweight='bold', fontsize=13)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax6.yaxis.get_majorticklabels(), rotation=0, fontsize=8)

    plt.suptitle('Análisis Exploratorio de Datos de Tráfico Web',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_correlation_heatmap(
    data: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Crear matriz de correlación detallada

    Args:
        data: DataFrame con los datos
        figsize: Tamaño de la figura

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()

    # Crear heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax, vmin=-1, vmax=1,
                square=True,
                cbar_kws={'label': 'Correlación', 'shrink': 0.8},
                annot_kws={'size': 10},
                linewidths=0.5)

    ax.set_title('Matriz de Correlación Completa', fontweight='bold', fontsize=14, pad=20)

    # Formatear etiquetas
    labels = [c.replace('_', ' ').title() for c in corr_matrix.columns]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, rotation=0, fontsize=10)

    plt.tight_layout()
    return fig


def plot_causal_impact_custom(
    ci_result,
    data: pd.DataFrame,
    intervention_date: str,
    title: str = "Análisis de Causal Impact",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Crear gráfico personalizado de CausalImpact con 3 paneles

    Args:
        ci_result: Objeto CausalImpact o DataFrame de inferences
        data: DataFrame original con variable 'y'
        intervention_date: Fecha de intervención
        title: Título del gráfico
        figsize: Tamaño de la figura

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Obtener inferences
    if hasattr(ci_result, 'inferences'):
        inferences = ci_result.inferences.copy()
    else:
        inferences = ci_result.copy()

    intervention_ts = pd.Timestamp(intervention_date)

    # Detectar columnas
    pred_col = 'point_pred' if 'point_pred' in inferences.columns else 'preds'
    lower_col = 'point_pred_lower' if 'point_pred_lower' in inferences.columns else 'preds_lower'
    upper_col = 'point_pred_upper' if 'point_pred_upper' in inferences.columns else 'preds_upper'

    # Si no hay IC, crearlos
    if lower_col not in inferences.columns:
        std = inferences[pred_col].std()
        inferences['preds_lower'] = inferences[pred_col] - 2 * std
        inferences['preds_upper'] = inferences[pred_col] + 2 * std
        lower_col = 'preds_lower'
        upper_col = 'preds_upper'

    # Obtener datos observados
    if 'response' in inferences.columns:
        y_original = inferences['response'].values
    elif 'y' in data.columns:
        y_original = data.loc[inferences.index, 'y'].values
    else:
        raise ValueError("No se encontró variable respuesta en los datos")

    # Calcular límites razonables
    y_min = min(y_original.min(), inferences[pred_col].min())
    y_max = max(y_original.max(), inferences[pred_col].max())
    margin = (y_max - y_min) * 0.15

    # ==== PANEL 1: Observado vs Predicho ====
    ax1 = axes[0]

    # Datos observados (negro)
    ax1.plot(inferences.index, y_original, 'k-', linewidth=2, label='Observado', zorder=3)

    # Predicción (azul punteado)
    ax1.plot(inferences.index, inferences[pred_col], 'b--', linewidth=2,
             label='Predicho (contrafactual)', alpha=0.8, zorder=2)

    # Intervalo de confianza (banda azul)
    lower_clipped = np.clip(inferences[lower_col], y_min - margin, y_max + margin)
    upper_clipped = np.clip(inferences[upper_col], y_min - margin, y_max + margin)

    ax1.fill_between(inferences.index, lower_clipped, upper_clipped,
                     alpha=0.2, color='blue', label='IC 95%', zorder=1)

    # Línea de intervención
    ax1.axvline(intervention_ts, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Intervención')

    ax1.set_ylabel('Respuesta', fontweight='bold', fontsize=11)
    ax1.set_title('Observado vs Contrafactual', fontweight='bold', fontsize=12)
    ax1.set_ylim([y_min - margin, y_max + margin])
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # ==== PANEL 2: Efecto Puntual ====
    ax2 = axes[1]

    point_effects = y_original - inferences[pred_col].values
    effects_lower = y_original - inferences[upper_col].values
    effects_upper = y_original - inferences[lower_col].values

    effects_min = point_effects.min()
    effects_max = point_effects.max()
    margin_effects = max(abs(effects_min), abs(effects_max)) * 0.2

    ax2.plot(inferences.index, point_effects, 'g-', linewidth=2, label='Efecto Puntual')

    # Banda de confianza
    lower_eff_clipped = np.clip(effects_lower, -(abs(effects_min) + margin_effects), abs(effects_max) + margin_effects)
    upper_eff_clipped = np.clip(effects_upper, -(abs(effects_min) + margin_effects), abs(effects_max) + margin_effects)

    ax2.fill_between(inferences.index, lower_eff_clipped, upper_eff_clipped,
                     alpha=0.2, color='green', label='IC 95%')

    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axvline(intervention_ts, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax2.set_ylabel('Efecto Puntual', fontweight='bold', fontsize=11)
    ax2.set_title('Efectos Puntuales (Diarios)', fontweight='bold', fontsize=12)
    ax2.set_ylim([-(abs(effects_min) + margin_effects), abs(effects_max) + margin_effects])
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # ==== PANEL 3: Efecto Acumulado ====
    ax3 = axes[2]

    # Calcular acumulado solo después de la intervención
    cumulative_effect = np.zeros(len(point_effects))
    mask_pre = inferences.index < intervention_ts
    mask_post = inferences.index >= intervention_ts

    cumulative_effect[mask_post] = np.cumsum(point_effects[mask_post])

    # IC acumulado
    cum_lower = np.zeros(len(point_effects))
    cum_upper = np.zeros(len(point_effects))
    cum_lower[mask_post] = np.cumsum(effects_lower[mask_post])
    cum_upper[mask_post] = np.cumsum(effects_upper[mask_post])

    cum_min = min(cumulative_effect.min(), cum_lower.min())
    cum_max = max(cumulative_effect.max(), cum_upper.max())
    margin_cum = max(abs(cum_min), abs(cum_max)) * 0.2

    ax3.plot(inferences.index, cumulative_effect, color='purple', linewidth=2, label='Efecto Acumulado')

    # Banda de confianza
    cum_lower_clipped = np.clip(cum_lower, cum_min - margin_cum, cum_max + margin_cum)
    cum_upper_clipped = np.clip(cum_upper, cum_min - margin_cum, cum_max + margin_cum)

    ax3.fill_between(inferences.index, cum_lower_clipped, cum_upper_clipped,
                     alpha=0.2, color='purple', label='IC 95%')

    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.axvline(intervention_ts, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax3.set_ylabel('Efecto Acumulado', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Fecha', fontweight='bold', fontsize=11)
    ax3.set_title('Efectos Acumulativos', fontweight='bold', fontsize=12)
    ax3.set_ylim([cum_min - margin_cum, cum_max + margin_cum])
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_intervention_comparison(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Crear gráfico de comparación de intervenciones

    Args:
        comparison_df: DataFrame con comparación de intervenciones
        figsize: Tamaño de la figura

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Preparar datos
    nombres = comparison_df['Intervención'].tolist()
    n_interventions = len(nombres)

    # Colores según significancia
    colors = [COLORS['positive'] if sig == 'Sí' else COLORS['neutral']
              for sig in comparison_df['Significativo']]

    # Panel 1: Efecto Diario
    ax1 = axes[0]
    bars1 = ax1.barh(nombres, comparison_df['Efecto Diario'], color=colors, edgecolor='black')
    ax1.set_xlabel('Unidades/Día', fontsize=12)
    ax1.set_title('Efecto Diario Promedio', fontweight='bold', fontsize=13)
    ax1.axvline(0, color='black', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='x')

    # Añadir valores
    for bar, val in zip(bars1, comparison_df['Efecto Diario']):
        offset = 2 if val >= 0 else -2
        ax1.text(val + offset, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}', va='center', fontsize=10, fontweight='bold',
                ha='left' if val >= 0 else 'right')

    # Panel 2: Efecto Total
    ax2 = axes[1]
    bars2 = ax2.barh(nombres, comparison_df['Efecto Total'], color=colors, edgecolor='black')
    ax2.set_xlabel('Unidades Totales', fontsize=12)
    ax2.set_title('Efecto Acumulativo Total', fontweight='bold', fontsize=13)
    ax2.axvline(0, color='black', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars2, comparison_df['Efecto Total']):
        offset = max(abs(comparison_df['Efecto Total'])) * 0.02
        offset = offset if val >= 0 else -offset
        ax2.text(val + offset, bar.get_y() + bar.get_height()/2,
                f'{val:+.0f}', va='center', fontsize=10, fontweight='bold',
                ha='left' if val >= 0 else 'right')

    # Panel 3: Cambio Porcentual
    ax3 = axes[2]
    bars3 = ax3.barh(nombres, comparison_df['Cambio %'], color=colors, edgecolor='black')
    ax3.set_xlabel('Cambio (%)', fontsize=12)
    ax3.set_title('Efecto Relativo (%)', fontweight='bold', fontsize=13)
    ax3.axvline(0, color='black', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars3, comparison_df['Cambio %']):
        offset = max(abs(comparison_df['Cambio %'])) * 0.02
        offset = offset if val >= 0 else -offset
        ax3.text(val + offset, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', fontsize=10, fontweight='bold',
                ha='left' if val >= 0 else 'right')

    plt.suptitle('Comparación de Intervenciones\n(Verde = Significativo, Gris = No Significativo)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_recommended_variables(
    correlations: pd.Series,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Visualizar variables recomendadas como control

    Args:
        correlations: Serie con correlaciones
        threshold: Umbral de correlación para recomendar
        figsize: Tamaño de la figura

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ordenar correlaciones
    correlations = correlations.sort_values(ascending=True)

    # Colores según umbral
    colors = [COLORS['positive'] if abs(x) >= threshold else COLORS['neutral']
              for x in correlations.values]

    # Crear barras
    bars = ax.barh(range(len(correlations)), correlations.values, color=colors, edgecolor='black')

    # Configurar ejes
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels([c.replace('_', ' ').title() for c in correlations.index], fontsize=11)
    ax.set_xlabel('Correlación con Variable Respuesta', fontsize=12)
    ax.set_title('Variables de Control Recomendadas', fontweight='bold', fontsize=14)

    # Líneas de referencia
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(threshold, color='green', linewidth=2, linestyle='--', alpha=0.7, label=f'Umbral ({threshold})')
    ax.axvline(-threshold, color='green', linewidth=2, linestyle='--', alpha=0.7)

    # Añadir valores y emojis
    for bar, val, color in zip(bars, correlations.values, colors):
        emoji = ' (Recomendado)' if abs(val) >= threshold else ''
        ax.text(val + 0.02 if val >= 0 else val - 0.02,
                bar.get_y() + bar.get_height()/2,
                f'{val:.2f}{emoji}',
                va='center', ha='left' if val >= 0 else 'right',
                fontsize=10, fontweight='bold' if abs(val) >= threshold else 'normal')

    ax.set_xlim(-1.1, 1.1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def create_results_summary_card(
    results: Dict[str, Any],
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Crear tarjeta resumen de resultados

    Args:
        results: Diccionario con resultados del análisis
        figsize: Tamaño de la figura

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Extraer datos
    metricas = results['metricas']
    stats = results['estadisticas']
    interp = results['interpretacion']

    # Determinar color según significancia
    if stats['es_significativo']:
        bg_color = '#d4edda' if metricas['efecto_diario'] > 0 else '#f8d7da'
        header_color = COLORS['positive'] if metricas['efecto_diario'] > 0 else COLORS['negative']
        emoji = '^' if metricas['efecto_diario'] > 0 else 'v'
    else:
        bg_color = '#f8f9fa'
        header_color = COLORS['neutral']
        emoji = '~'

    # Fondo
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=bg_color,
                                edgecolor='black', linewidth=2, transform=ax.transAxes))

    # Contenido
    content = f"""
{results['nombre']}
{'='*40}

Fecha de Intervención: {results['fecha']}

MÉTRICAS PRINCIPALES
--------------------
Efecto Diario: {metricas['efecto_diario']:+.1f} unidades
Efecto Total:  {metricas['efecto_total']:+.0f} unidades
Cambio:        {metricas['cambio_porcentual']:+.1f}%

ESTADÍSTICAS
------------
P-value: {stats['p_value']:.4f}
Estado:  {interp['significancia']}

CONCLUSIÓN
----------
{interp['conclusion'][:200]}...
"""

    ax.text(0.5, 0.5, content, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', wrap=True)

    plt.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure, format: str = 'png', dpi: int = 150) -> bytes:
    """
    Convertir figura de matplotlib a bytes para descarga

    Args:
        fig: Figura de matplotlib
        format: Formato de imagen (png, pdf, svg)
        dpi: Resolución de la imagen

    Returns:
        Bytes de la imagen
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return buf.getvalue()
