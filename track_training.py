#!/usr/bin/env python3
"""
FPCA Analysis - Análisis de Evolución Temporal
Análisis de la progresión del entrenamiento a través del tiempo
Basado en: https://doi.org/10.1123/ijspp.2024-0548
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from datetime import datetime, timedelta
import seaborn as sns
import warnings
import argparse
import sys
import os
warnings.filterwarnings('ignore')

def load_temporal_data(file_path):
    """
    Carga datos con información temporal (fechas o períodos)
    Espera columnas: id, date/period, year, weight, y duraciones (1, 5, 10, etc.)
    """
    print(f"Cargando datos temporales desde: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo '{file_path}' no existe")
    
    try:
        data = pd.read_csv(file_path)
        print(f"Archivo cargado: {data.shape[0]} filas, {data.shape[1]} columnas")
    except Exception as e:
        raise Exception(f"Error al leer el archivo CSV: {e}")
    
    # Identificar columnas temporales y de metadatos
    metadata_cols = ['id', 'date', 'period', 'year', 'month', 'week', 'weight']
    available_metadata = [col for col in metadata_cols if col in data.columns]
    
    # Identificar columnas de tiempo (duraciones)
    time_columns = [col for col in data.columns if col not in metadata_cols]
    
    if not time_columns:
        raise ValueError("No se encontraron columnas de duración válidas")
    
    # Ordenar columnas por valor numérico
    try:
        time_columns_sorted = sorted(time_columns, key=lambda x: float(x))
        time_values = np.array([float(col) for col in time_columns_sorted])
    except ValueError:
        print("Advertencia: Usando orden original de columnas")
        time_columns_sorted = time_columns
        time_values = np.arange(len(time_columns))
    
    # Preparar datos de potencia
    power_data = data[time_columns_sorted].dropna()
    metadata = data[available_metadata]
    
    print(f"Columnas de metadata disponibles: {available_metadata}")
    print(f"Datos limpios: {power_data.shape[0]} observaciones")
    print(f"Rango temporal: {time_values[0]} - {time_values[-1]} segundos")
    
    return power_data, metadata, time_values, available_metadata

def create_temporal_periods(metadata, period_type='month'):
    """
    Crea períodos temporales para el análisis
    """
    if 'date' in metadata.columns:
        # Convertir fechas si están disponibles
        try:
            metadata['date'] = pd.to_datetime(metadata['date'])
            if period_type == 'month':
                metadata['period'] = metadata['date'].dt.strftime('%Y-%m')
            elif period_type == 'quarter':
                metadata['period'] = metadata['date'].dt.quarter.astype(str) + 'Q' + metadata['date'].dt.year.astype(str)
            elif period_type == 'season':
                # Definir estaciones (hemisferio norte)
                month = metadata['date'].dt.month
                metadata['period'] = np.where(month.isin([12, 1, 2]), 'Invierno',
                                   np.where(month.isin([3, 4, 5]), 'Primavera',
                                   np.where(month.isin([6, 7, 8]), 'Verano', 'Otoño'))) + '-' + metadata['date'].dt.year.astype(str)
        except:
            print("Error procesando fechas, usando columna 'period' si está disponible")
    
    if 'period' not in metadata.columns:
        # Si no hay información temporal, usar year o crear períodos artificiales
        if 'year' in metadata.columns:
            metadata['period'] = metadata['year'].astype(str)
        else:
            print("Advertencia: No hay información temporal, creando períodos secuenciales")
            metadata['period'] = 'Periodo-' + (metadata.index // 50 + 1).astype(str)
    
    return metadata

def temporal_fpca_analysis(power_data, metadata, time_points, n_components=3):
    """
    Realiza FPCA para cada período temporal
    """
    periods = metadata['period'].unique()
    periods = sorted(periods)
    
    print(f"\nAnalizando {len(periods)} períodos temporales:")
    for p in periods:
        count = sum(metadata['period'] == p)
        print(f"  {p}: {count} observaciones")
    
    temporal_results = {}
    
    for period in periods:
        period_mask = metadata['period'] == period
        period_data = power_data[period_mask]
        
        if len(period_data) < 5:  # Mínimo de observaciones
            print(f"Advertencia: {period} tiene muy pocas observaciones ({len(period_data)})")
            continue
        
        # Aplicar FPCA para este período
        log_data = np.log(period_data.clip(lower=1e-6))
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(log_data)
        
        pca = PCA(n_components=min(n_components, len(period_data)-1))
        scores = pca.fit_transform(data_scaled)
        
        temporal_results[period] = {
            'pca': pca,
            'scaler': scaler,
            'scores': scores,
            'mean_function': np.mean(log_data, axis=0),
            'log_data': log_data,
            'n_obs': len(period_data),
            'explained_variance': pca.explained_variance_ratio_
        }
    
    return temporal_results

def plot_temporal_evolution(temporal_results, time_points):
    """
    Visualiza la evolución temporal de las funciones medias
    """
    periods = sorted(temporal_results.keys())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Panel superior: Evolución de la función media
    colors = plt.cm.viridis(np.linspace(0, 1, len(periods)))
    
    for i, period in enumerate(periods):
        mean_func = temporal_results[period]['mean_function']
        power_curve = np.exp(mean_func)
        ax1.semilogx(time_points, power_curve, color=colors[i], 
                    linewidth=2, label=f'{period} (n={temporal_results[period]["n_obs"]})')
    
    ax1.set_title('A. Evolución temporal de las curvas de potencia media')
    ax1.set_ylabel('Potencia (W)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1500)
    
    # Panel inferior: Cambios relativos respecto al primer período
    if len(periods) > 1:
        baseline = temporal_results[periods[0]]['mean_function']
        baseline_power = np.exp(baseline)
        
        for i, period in enumerate(periods[1:], 1):
            current = temporal_results[period]['mean_function']
            current_power = np.exp(current)
            change = ((current_power - baseline_power) / baseline_power) * 100
            
            ax2.semilogx(time_points, change, color=colors[i], 
                        linewidth=2, label=f'{period}')
        
        ax2.set_title(f'B. Cambio porcentual respecto a {periods[0]}')
        ax2.set_xlabel('Duración (s)')
        ax2.set_ylabel('Cambio (%)')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_component_evolution(temporal_results, time_points, component=0):
    """
    Visualiza cómo evoluciona un componente principal específico
    """
    periods = sorted(temporal_results.keys())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(periods)))
    
    # Panel superior: Forma del componente a través del tiempo
    for i, period in enumerate(periods):
        if component < len(temporal_results[period]['pca'].components_):
            comp = temporal_results[period]['pca'].components_[component]
            ax1.semilogx(time_points, comp, color=colors[i], 
                        linewidth=2, label=f'{period}')
    
    ax1.set_title(f'A. Evolución del Componente Principal {component+1}')
    ax1.set_ylabel('Peso del componente')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Panel inferior: Varianza explicada del componente
    variances = []
    period_labels = []
    
    for period in periods:
        if component < len(temporal_results[period]['explained_variance']):
            variances.append(temporal_results[period]['explained_variance'][component] * 100)
            period_labels.append(period)
    
    if variances:
        bars = ax2.bar(range(len(variances)), variances, color=colors[:len(variances)])
        ax2.set_title(f'B. Varianza explicada por PC{component+1} en cada período')
        ax2.set_ylabel('Varianza explicada (%)')
        ax2.set_xticks(range(len(period_labels)))
        ax2.set_xticklabels(period_labels, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, val in zip(bars, variances):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_performance_metrics_evolution(temporal_results, time_points, metrics_durations=[5, 60, 300, 1200]):
    """
    Analiza métricas específicas de rendimiento a través del tiempo
    """
    periods = sorted(temporal_results.keys())
    
    # Encontrar los índices más cercanos a las duraciones deseadas
    metric_indices = []
    actual_durations = []
    
    for duration in metrics_durations:
        idx = np.argmin(np.abs(time_points - duration))
        metric_indices.append(idx)
        actual_durations.append(time_points[idx])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (duration, idx) in enumerate(zip(actual_durations, metric_indices)):
        if i >= 4:  # Solo 4 gráficas
            break
            
        powers = []
        period_names = []
        
        for period in periods:
            mean_func = temporal_results[period]['mean_function']
            power = np.exp(mean_func[idx])
            powers.append(power)
            period_names.append(period)
        
        # Gráfica de línea con tendencia
        axes[i].plot(range(len(powers)), powers, 'o-', linewidth=2, markersize=8)
        
        # Línea de tendencia
        if len(powers) > 2:
            z = np.polyfit(range(len(powers)), powers, 1)
            p = np.poly1d(z)
            axes[i].plot(range(len(powers)), p(range(len(powers))), 
                        "--", alpha=0.8, color='red')
            
            # Calcular pendiente (cambio por período)
            slope = z[0]
            axes[i].text(0.02, 0.98, f'Tendencia: {slope:+.1f} W/período',
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[i].set_title(f'Potencia a {duration:.0f}s')
        axes[i].set_ylabel('Potencia (W)')
        axes[i].set_xticks(range(len(period_names)))
        axes[i].set_xticklabels(period_names, rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Evolución de métricas específicas de potencia', fontsize=16)
    plt.tight_layout()
    return fig

def generate_progress_report(temporal_results, time_points):
    """
    Genera un reporte de progreso cuantitativo
    """
    periods = sorted(temporal_results.keys())
    
    if len(periods) < 2:
        print("Se necesitan al menos 2 períodos para generar reporte de progreso")
        return None
    
    print(f"\n{'='*60}")
    print("REPORTE DE PROGRESO EN EL ENTRENAMIENTO")
    print(f"{'='*60}")
    
    baseline_period = periods[0]
    latest_period = periods[-1]
    
    baseline_func = temporal_results[baseline_period]['mean_function']
    latest_func = temporal_results[latest_period]['mean_function']
    
    baseline_power = np.exp(baseline_func)
    latest_power = np.exp(latest_func)
    
    # Métricas clave
    key_durations = [5, 60, 300, 1200]  # 5s, 1min, 5min, 20min
    
    print(f"Comparación entre {baseline_period} y {latest_period}:")
    print("-" * 50)
    
    for duration in key_durations:
        idx = np.argmin(np.abs(time_points - duration))
        actual_duration = time_points[idx]
        
        baseline_val = baseline_power[idx]
        latest_val = latest_power[idx]
        change = ((latest_val - baseline_val) / baseline_val) * 100
        
        print(f"Potencia a {actual_duration:.0f}s: {baseline_val:.0f}W → {latest_val:.0f}W ({change:+.1f}%)")
    
    # Análisis de componentes principales
    print(f"\nAnálisis de patrones de entrenamiento:")
    print("-" * 50)
    
    baseline_var = temporal_results[baseline_period]['explained_variance']
    latest_var = temporal_results[latest_period]['explained_variance']
    
    for i in range(min(3, len(baseline_var), len(latest_var))):
        baseline_pct = baseline_var[i] * 100
        latest_pct = latest_var[i] * 100
        change = latest_pct - baseline_pct
        
        print(f"Componente {i+1}: {baseline_pct:.1f}% → {latest_pct:.1f}% ({change:+.1f}%)")
    
    # Tendencias generales
    print(f"\nTendencias generales:")
    print("-" * 50)
    
    total_change = np.mean(((latest_power - baseline_power) / baseline_power) * 100)
    print(f"Cambio promedio en todas las duraciones: {total_change:+.1f}%")
    
    # Identificar fortalezas y debilidades
    changes = ((latest_power - baseline_power) / baseline_power) * 100
    
    # Encontrar mejores y peores mejoras
    best_idx = np.argmax(changes)
    worst_idx = np.argmin(changes)
    
    print(f"Mayor mejora: {time_points[best_idx]:.0f}s ({changes[best_idx]:+.1f}%)")
    print(f"Menor mejora: {time_points[worst_idx]:.0f}s ({changes[worst_idx]:+.1f}%)")
    
    return {
        'baseline_period': baseline_period,
        'latest_period': latest_period,
        'total_change': total_change,
        'best_improvement': (time_points[best_idx], changes[best_idx]),
        'worst_improvement': (time_points[worst_idx], changes[worst_idx])
    }

def main():
    """
    Función principal para análisis temporal
    """
    parser = argparse.ArgumentParser(description='Análisis FPCA temporal')
    parser.add_argument('--f', '--file', dest='file', required=True,
                       help='Archivo CSV con datos temporales')
    parser.add_argument('--period-type', choices=['month', 'quarter', 'season'], 
                       default='month', help='Tipo de agrupación temporal')
    parser.add_argument('--components', type=int, default=3,
                       help='Número de componentes principales')
    parser.add_argument('--output-dir', default='.',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    try:
        print("=== Análisis FPCA de Evolución Temporal ===\n")
        
        # Cargar datos
        power_data, metadata, time_points, available_cols = load_temporal_data(args.file)
        
        # Crear períodos temporales
        metadata = create_temporal_periods(metadata, args.period_type)
        
        # Realizar análisis FPCA temporal
        temporal_results = temporal_fpca_analysis(power_data, metadata, time_points, args.components)
        
        if not temporal_results:
            print("Error: No se pudieron procesar los datos temporales")
            return
        
        # Crear visualizaciones
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        
        # 1. Evolución temporal
        fig1 = plot_temporal_evolution(temporal_results, time_points)
        output_path = os.path.join(args.output_dir, f'{base_name}_evolucion_temporal.png')
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_path}")
        
        # 2. Evolución de componentes
        fig2 = plot_component_evolution(temporal_results, time_points, component=0)
        output_path = os.path.join(args.output_dir, f'{base_name}_evolucion_componentes.png')
        fig2.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_path}")
        
        # 3. Métricas específicas
        fig3 = plot_performance_metrics_evolution(temporal_results, time_points)
        output_path = os.path.join(args.output_dir, f'{base_name}_metricas_evolucion.png')
        fig3.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_path}")
        
        # 4. Generar reporte
        progress_report = generate_progress_report(temporal_results, time_points)
        
        plt.show()
        
        print(f"\n=== Análisis temporal completado ===")
        
        return temporal_results, progress_report
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
