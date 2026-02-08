import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Diccionario de etiquetas según el manual técnico
LABELS_DICT = {
    1: 'Baseline',
    2: 'Stress',
    3: 'Amusement',
    4: 'Meditation'
}

def conduct_wesad_eda(pkl_path):
    print(f"--- Iniciando EDA para: {os.path.basename(pkl_path)} ---")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Acceso a las señales de pecho
    chest_signals = data['signal']['chest']
    
    # IMPORTANTE: Se corrigen las llaves 'Resp' y 'Temp' para evitar el KeyError
    df = pd.DataFrame({
        'ACC_x': chest_signals['ACC'][:, 0],
        'ACC_y': chest_signals['ACC'][:, 1],
        'ACC_z': chest_signals['ACC'][:, 2],
        'ECG': chest_signals['ECG'].flatten(),
        'EDA': chest_signals['EDA'].flatten(),
        'EMG': chest_signals['EMG'].flatten(),
        'Resp': chest_signals['Resp'].flatten(), # Corregido de 'RESP' a 'Resp'
        'Temp': chest_signals['Temp'].flatten(), # Corregido de 'TEMP' a 'Temp'
        'Label': data['label']
    })

    # Filtrado de etiquetas válidas (1-4)
    df = df[df['Label'].isin([1, 2, 3, 4])].copy()
    df['Label_Name'] = df['Label'].map(LABELS_DICT)

    # 1. Estructura de los datos
    print(f"\nDimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("\nNombres y Tipos de Variables:")
    print(df.dtypes)

    # 2. Estadísticas Descriptivas completas
    # Describe genera automáticamente: media, std, min, max y percentiles (25, 50, 75)
    target_vars = ['EDA', 'Temp', 'Resp', 'ECG']
    stats = df[target_vars].describe(percentiles=[.25, .5, .75])
    
    # Agregamos la mediana explícitamente para mayor claridad
    stats.loc['median'] = df[target_vars].median()
    
    print("\nResumen Estadístico:")
    print(stats.round(4))

    # 3. Visualizaciones para la presentación
    # Histograma para ver la distribución
    df[target_vars].hist(bins=50, figsize=(15, 8), color='teal', edgecolor='black')
    plt.suptitle('Distribución de Señales Fisiológicas')
    plt.savefig('histogramas_wesad.png')

    # Boxplots para detección de outliers por estado
    plt.figure(figsize=(15, 6))
    for i, var in enumerate(['EDA', 'Temp'], 1):
        plt.subplot(1, 2, i)
        sns.boxplot(x='Label_Name', y=var, data=df)
        plt.title(f'Análisis de Outliers: {var}')
    plt.tight_layout()
    plt.savefig('boxplots_wesad.png')
    
    #plt.show()

# Ejecución
# conduct_wesad_eda('wesad/WESAD/S2/S2.pkl')
# Ejemplo de uso (Asegúrate de tener el archivo en la ruta correcta)
df_final = conduct_wesad_eda('wesad/WESAD/S2/S2.pkl')