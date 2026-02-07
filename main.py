import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re

files = [
    "Stress_Dataset",
    "StressLevelDataset"
]
file = files[0]
# Configuración de ruta
file_path = os.path.join("datasets", f"{file}.csv")

def limpiar_nombre_archivo(nombre):
    nombre_limpio = re.sub(r'[\\/*?:"<>| ]', '_', nombre)
    return nombre_limpio

def descripcion_datos_categoricos(df):
    output_dir = f'plots_categoricos/{file}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Diccionario para guardar qué etiquetas tiene cada columna
    mapeo_etiquetas = {}

    print("=== ANÁLISIS EXPLORATORIO: DATOS CATEGÓRICOS ===")
    
    # 1. MODA
    print("\n[1] MODA POR COLUMNA:")
    print(df.mode().iloc[0])

    # 2. TABLAS, HISTOGRAMAS Y MAPEO
    print("\n[2] DETALLE POR COLUMNA (Etiquetas y Frecuencias):")
    for col in df.columns:
        if col == 'Age':
            print(f"\n>>> Columna: {col} (TRATADA COMO NUMÉRICA)")
            # Descripción numérica
            desc = df[col].describe()
            print(f"Mínimo: {desc['min']}")
            print(f"Máximo: {desc['max']}")
            print(f"Media:  {desc['mean']:.2f}")
            print(f"Desviación Estándar: {desc['std']:.2f}")
            
            # Histograma de clases 
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True, color='skyblue')
            plt.title(f'Distribución Numérica de la Edad')
            plt.savefig(f'{output_dir}/distribucion_edad.png')
            plt.close()
        else:   
            raw_unique = df[col].unique()
            try:
                unique_vals = sorted([int(float(x)) for x in raw_unique if pd.notna(x)])
            except (ValueError, TypeError):
                unique_vals = sorted([str(x) for x in raw_unique if pd.notna(x)])
            mapeo_etiquetas[col] = unique_vals
            
            print(f"\n>>> Columna: {col}")
            print(f"Valores posibles (etiquetas): {unique_vals}")
            
            # Frecuencias ordenadas por etiqueta
            frec_abs = df[col].value_counts().sort_index()
            frec_rel = (df[col].value_counts(normalize=True) * 100).sort_index()
            
            tabla = pd.DataFrame({'Absoluta': frec_abs, 'Relativa (%)': frec_rel})
            tabla.index.name = 'Etiqueta'
            print(tabla)

            # Histograma de clases
            plt.figure(figsize=(8, 5))
            sns.countplot(
                data=df, 
                x=col, 
                hue=col, 
                palette='viridis', 
                order=unique_vals, 
                legend=False
            )
            plt.title(f'Histograma de Clases: {col}')
            nombre_archivo = limpiar_nombre_archivo(col)
            plt.savefig(f'{output_dir}/histograma_{nombre_archivo}.png')
            plt.close()
    
    return mapeo_etiquetas 

if __name__ == "__main__":
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        descripcion_datos_categoricos(df)
        print("\n¡Proceso completado! Los histogramas de clases se guardaron en 'plots_categoricos/'.")
    else:
        print(f"Error: No se encontró el archivo en {file_path}")