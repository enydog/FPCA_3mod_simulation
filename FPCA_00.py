#!/usr/bin/env python3
"""
FPCA Analysis - Versión simplificada
Análisis de Componentes Principales Funcionales para datos de potencia
*********************************************************************************************************
DOI: https://doi.org/10.1123/ijspp.2024-0548
Keywords: principal-component analysis; power output; cycling; performance; critical-power model
Michael J. Puchowicz and Philip F. Skiba
********************************************************************************************************
Uso:
    python FPCA_00.py --f archivo.csv
    python FPCA_00.py --file archivo.csv
    python FPCA_00.py -f archivo.csv
    python FPCA_00.py archivo.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import warnings
import argparse
import sys
import os
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """
    Carga y prepara los datos del archivo CSV
    """
    print(f"Cargando datos desde: {file_path}")
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo '{file_path}' no existe")
    
    # Cargar datos
    try:
        data = pd.read_csv(file_path)
        print(f"Archivo cargado: {data.shape[0]} filas, {data.shape[1]} columnas")
    except Exception as e:
        raise Exception(f"Error al leer el archivo CSV: {e}")
    
    # Identificar columnas de tiempo (excluyendo id, year, weight)
    non_time_cols = ['id', 'year', 'weight']
    time_columns = [col for col in data.columns if col not in non_time_cols]
    
    # Verificar que hay columnas de tiempo
    if not time_columns:
        raise ValueError("No se encontraron columnas de tiempo válidas")
    
    # Ordenar columnas por valor numérico
    try:
        time_columns_sorted = sorted(time_columns, key=lambda x: float(x))
        time_values = np.array([float(col) for col in time_columns_sorted])
    except ValueError:
        print("Advertencia: No todas las columnas son numéricas, usando orden original")
        time_columns_sorted = time_columns
        time_values = np.arange(len(time_columns))
    
    # Extraer datos de potencia
    power_data = data[time_columns_sorted].dropna()
    
    if power_data.empty:
        raise ValueError("No hay datos válidos después de eliminar valores faltantes")
    
    print(f"Datos limpios: {power_data.shape[0]} observaciones, {power_data.shape[1]} puntos temporales")
    print(f"Rango temporal: {time_values[0]} - {time_values[-1]} segundos")
    
    return power_data, time_values

def simple_fpca_analysis(data, time_points, n_components=3):
    """
    Realiza un análisis FPCA simplificado usando PCA estándar
    """
    print(f"\nRealizando análisis FPCA con {n_components} componentes...")
    
    # Aplicar transformación log
    log_data = np.log(data.clip(lower=1e-6))  # Evitar log(0)
    
    # PCA en los datos transformados
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Estandarizar los datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(log_data)
    
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data_scaled)
    
    # Calcular la función media
    mean_function = np.mean(log_data, axis=0)
    
    print(f"Varianza explicada: {pca.explained_variance_ratio_}")
    print(f"Varianza total explicada: {sum(pca.explained_variance_ratio_):.3f}")
    
    return {
        'pca': pca,
        'scaler': scaler,
        'scores': scores,
        'mean_function': mean_function,
        'log_data': log_data,
        'components': pca.components_
    }

def plot_raw_data(data, time_points, n_curves=50):
    """
    Grafica las curvas originales
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Panel superior: Curvas individuales
    sample_indices = np.random.choice(len(data), min(n_curves, len(data)), replace=False)
    for idx in sample_indices:
        ax1.semilogx(time_points, data.iloc[idx], color='black', alpha=0.1, linewidth=0.5)
    
    ax1.set_title('A. Curvas individuales de potencia')
    ax1.set_ylabel('Potencia (W)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 2000)
    
    # Panel inferior: Función media
    mean_curve = np.mean(data, axis=0)
    ax2.semilogx(time_points, mean_curve, color='red', linewidth=2)
    ax2.set_title('B. Función media')
    ax2.set_xlabel('Duración (s)')
    ax2.set_ylabel('Potencia (W)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 2000)
    
    plt.tight_layout()
    return fig

def plot_components(fpca_results, time_points, n_variations=30):
    """
    Visualiza los componentes principales
    """
    mean_func = fpca_results['mean_function']
    components = fpca_results['components']
    explained_var = fpca_results['pca'].explained_variance_
    
    n_comp = len(components)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(4, n_comp)):
        ax = axes[i]
        
        # Función media
        ax.semilogx(time_points, np.exp(mean_func), color='red', linewidth=2, label='Media')
        
        # Variaciones del componente
        std_comp = np.sqrt(explained_var[i])
        variations = np.linspace(-2*std_comp, 2*std_comp, n_variations)
        
        colors = plt.cm.gray(np.linspace(0.3, 1, n_variations))
        
        for j, var in enumerate(variations):
            curve_var = mean_func + components[i] * var
            ax.semilogx(time_points, np.exp(curve_var), color=colors[j], alpha=0.6, linewidth=0.8)
        
        ax.set_title(f'Componente Principal {i+1} ({explained_var[i]/sum(explained_var)*100:.1f}%)')
        ax.set_ylim(0, 1500)
        ax.grid(True, alpha=0.3)
        if i >= 2:  # Solo en la fila inferior
            ax.set_xlabel('Duración (s)')
        if i % 2 == 0:  # Solo en la columna izquierda
            ax.set_ylabel('Potencia (W)')
    
    plt.tight_layout()
    return fig

def plot_scores_distribution(scores):
    """
    Grafica la distribución de los scores de los componentes principales
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    colors = ['black', 'blue', 'red', 'green']
    
    for i in range(min(4, scores.shape[1])):
        pc_scores = scores[:, i]
        std_pc = np.std(pc_scores)
        
        # Estimación de densidad kernel
        kde = gaussian_kde(pc_scores)
        x_range = np.linspace(pc_scores.min(), pc_scores.max(), 200)
        density = kde(x_range)
        
        ax.plot(x_range, density, color=colors[i], linewidth=2, 
               label=f'PC{i+1} (SD={std_pc:.2f})')
    
    ax.set_xlabel('Score del Componente Principal')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de los Scores de los Componentes Principales')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def evaluate_model(fpca_results, data, time_points):
    """
    Evalúa la calidad del modelo FPCA
    """
    pca = fpca_results['pca']
    scaler = fpca_results['scaler']
    log_data = fpca_results['log_data']
    
    # Reconstruir los datos
    data_scaled = scaler.transform(log_data)
    reconstructed_scaled = pca.inverse_transform(pca.transform(data_scaled))
    reconstructed_log = scaler.inverse_transform(reconstructed_scaled)
    reconstructed = np.exp(reconstructed_log)
    
    # Calcular errores
    original = data.values
    errors = (original - reconstructed) / original * 100  # Errores porcentuales
    
    mean_errors = np.mean(errors, axis=0)
    std_errors = np.std(errors, axis=0)
    
    # Graficar residuales
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Residuales promedio
    ax1.semilogx(time_points, mean_errors, 'ko', markersize=4)
    ax1.fill_between(time_points, mean_errors - 2*std_errors, mean_errors + 2*std_errors, 
                    alpha=0.3, color='gray')
    ax1.set_title('Residuales del modelo FPCA')
    ax1.set_ylabel('Error porcentual (%)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Ejemplos de ajuste
    sample_indices = np.random.choice(len(data), 4, replace=False)
    for i, idx in enumerate(sample_indices):
        if i < 4:
            color = ['blue', 'green', 'orange', 'purple'][i]
            ax2.semilogx(time_points, original[idx], color='gray', alpha=0.7, 
                        linewidth=1, label='Original' if i == 0 else '')
            ax2.semilogx(time_points, reconstructed[idx], color=color, 
                        linewidth=2, label=f'Ajuste {i+1}')
    
    ax2.set_title('Ejemplos de ajuste del modelo')
    ax2.set_xlabel('Duración (s)')
    ax2.set_ylabel('Potencia (W)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Estadísticas de error
    rmse = np.sqrt(np.mean((original - reconstructed)**2))
    mae = np.mean(np.abs(original - reconstructed))
    
    print(f"\nEstadísticas del modelo:")
    print(f"RMSE: {rmse:.2f} W")
    print(f"MAE: {mae:.2f} W")
    print(f"Error porcentual medio: {np.mean(np.abs(errors)):.2f}%")
    
    return fig, {'rmse': rmse, 'mae': mae, 'errors': errors}

def parse_arguments():
    """
    Parsea los argumentos de línea de comandos
    """
    parser = argparse.ArgumentParser(
        description='Análisis FPCA de datos de potencia',
        epilog='''
Ejemplos de uso:
    python FPCA_00.py --f mmp.csv
    python FPCA_00.py --file datos_potencia.csv  
    python FPCA_00.py -f mi_archivo.csv
    python FPCA_00.py datos.csv
    python FPCA_00.py --f datos.csv --components 3
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'file', 
        nargs='?',  # Opcional
        help='Archivo CSV con los datos de potencia'
    )
    
    parser.add_argument(
        '--f', '--file', 
        dest='file_flag',
        help='Archivo CSV con los datos de potencia'
    )
    
    parser.add_argument(
        '--components', '-c',
        type=int,
        default=4,
        help='Número de componentes principales (por defecto: 4)'
    )
    
    parser.add_argument(
        '--test-size', '-t',
        type=float,
        default=0.2,
        help='Proporción de datos para prueba (por defecto: 0.2)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='.',
        help='Directorio de salida para las gráficas (por defecto: directorio actual)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='No mostrar las gráficas (solo guardar)'
    )
    
    return parser.parse_args()

def get_input_file(args):
    """
    Determina el archivo de entrada a partir de los argumentos
    """
    # Prioridad: --f/--file, luego argumento posicional
    input_file = args.file_flag or args.file
    
    if not input_file:
        # Si no se especifica archivo, buscar archivos CSV en el directorio
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        
        if not csv_files:
            print("Error: No se especificó archivo y no se encontraron archivos CSV")
            print("Uso: python FPCA_00.py --f archivo.csv")
            sys.exit(1)
        elif len(csv_files) == 1:
            input_file = csv_files[0]
            print(f"Usando archivo encontrado: {input_file}")
        else:
            print("Se encontraron múltiples archivos CSV:")
            for i, f in enumerate(csv_files):
                print(f"  {i+1}. {f}")
            print("Por favor especifica cuál usar: python FPCA_00.py --f archivo.csv")
            sys.exit(1)
    
    return input_file
def main():
    """
    Función principal
    """
    print("=== Análisis FPCA de datos de potencia ===\n")
    
    # Parsear argumentos de línea de comandos
    args = parse_arguments()
    
    try:
        # 1. Obtener archivo de entrada
        input_file = get_input_file(args)
        
        # 2. Crear directorio de salida si no existe
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directorio creado: {output_dir}")
        
        # 3. Cargar datos
        data, time_points = load_and_prepare_data(input_file)
        
        # 4. Dividir en entrenamiento y prueba
        train_data, test_data = train_test_split(data, test_size=args.test_size, random_state=42)
        print(f"\nDivisión de datos:")
        print(f"Entrenamiento: {train_data.shape}")
        print(f"Prueba: {test_data.shape}")
        
        # 5. Realizar FPCA
        fpca_results = simple_fpca_analysis(train_data, time_points, n_components=args.components)
        
        # 6. Crear visualizaciones
        print("\nGenerando gráficas...")
        
        # Obtener nombre base del archivo para las gráficas
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Gráfica de datos originales
        fig1 = plot_raw_data(train_data, time_points)
        output_path = os.path.join(output_dir, f'{base_name}_01_datos_originales.png')
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_path}")
        
        # Gráfica de componentes
        fig2 = plot_components(fpca_results, time_points)
        output_path = os.path.join(output_dir, f'{base_name}_02_componentes_principales.png')
        fig2.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_path}")
        
        # Distribución de scores
        fig3 = plot_scores_distribution(fpca_results['scores'])
        output_path = os.path.join(output_dir, f'{base_name}_03_distribucion_scores.png')
        fig3.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_path}")
        
        # Evaluación del modelo
        fig4, stats = evaluate_model(fpca_results, train_data, time_points)
        output_path = os.path.join(output_dir, f'{base_name}_04_evaluacion_modelo.png')
        fig4.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_path}")
        
        # Mostrar gráficas si no se especifica --no-plots
        if not args.no_plots:
            plt.show()
        else:
            plt.close('all')  # Cerrar todas las figuras para ahorrar memoria
        
        print(f"\n=== Análisis completado exitosamente ===")
        print(f"Archivo procesado: {input_file}")
        print(f"Componentes principales: {args.components}")
        print(f"Gráficas guardadas en: {output_dir}")
        
        return fpca_results, stats
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, statistics = main()
