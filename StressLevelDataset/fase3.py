import pandas as pd
from plantilla import (
    dividir_train_test,
    entrenar_y_evaluar,
    graficar_matriz_confusion,
    guardar_resultados,
    ruta_resultado,
    imprimir_resultado,
    ranking_informacion_mutua,
    graficar_ranking
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def preprocesar_datos(df):
    """
    Aplica el preprocesamiento de las fases anteriores:
    1. Limpieza de nulos.
    2. Discretización de variables clave (Anxiety, Self-esteem, Depression).
    """
    df_limpio = df.copy()
    
    # 1. Limpieza básica
    df_limpio = df_limpio.dropna()
    
    # 2. Discretización por frecuencia (qcut) como en la Fase 2
    # Esto convierte los rangos numéricos en niveles 0, 1, 2 (Bajo, Medio, Alto)
    cols_a_discretizar = ['anxiety_level', 'self_esteem', 'depression']
    for col in cols_a_discretizar:
        if col in df_limpio.columns:
            # labels=False devuelve 0, 1, 2 directamente
            df_limpio[col] = pd.qcut(df_limpio[col], q=3, labels=False, duplicates='drop')
    
    # 3. Asegurar que todos los datos sean numéricos para los clasificadores
    return df_limpio.apply(pd.to_numeric)

# --- INICIO DEL SCRIPT ---

# 1. Cargar datos crudos
ruta_csv = "./datasets/StressLevelDataset.csv"
df_raw = pd.read_csv(ruta_csv)

# 2. Aplicar preprocesamiento de fases anteriores
df_listo = preprocesar_datos(df_raw)

# 3. Separar X e y
target = "stress_level"
X = df_listo.drop(columns=[target])
y = df_listo[target]

# Lista para guardar todas las métricas
todas_las_metricas = []

# ---------------------------------------------------------
# ACTIVIDAD 1: TODAS LAS CARACTERÍSTICAS
# ---------------------------------------------------------
print("\n--- EJECUTANDO ACTIVIDAD 1: TODAS LAS CARACTERÍSTICAS ---")

X_train, X_test, y_train, y_test = dividir_train_test(X, y)

# Entrenar Bayes
m_nb_full, _ = entrenar_y_evaluar(GaussianNB(), "Bayes_Todas", X_train, X_test, y_train, y_test)
imprimir_resultado(m_nb_full)
todas_las_metricas.append(m_nb_full)

# Entrenar Árbol
m_tree_full, _ = entrenar_y_evaluar(DecisionTreeClassifier(random_state=42), "Arbol_Todas", X_train, X_test, y_train, y_test)
imprimir_resultado(m_tree_full)
todas_las_metricas.append(m_tree_full)

# Guardar matrices de confusión
graficar_matriz_confusion(m_nb_full['matriz_confusion'], "Bayes Todas", ruta_resultado("cm_nb_todas.png"))
graficar_matriz_confusion(m_tree_full['matriz_confusion'], "Arbol Todas", ruta_resultado("cm_tree_todas.png"))


# ---------------------------------------------------------
# ACTIVIDAD 2: MEJORES CARACTERÍSTICAS (INFO MUTUA)
# ---------------------------------------------------------
print("\n--- EJECUTANDO ACTIVIDAD 2: MEJORES CARACTERÍSTICAS (INFO MUTUA) ---")

# 1. Calcular Ranking sobre los datos preprocesados
ranking_im = ranking_informacion_mutua(X, y)
graficar_ranking(ranking_im, "Ranking Información Mutua", ruta_resultado("ranking_im.png"))

# 2. Seleccionar las mejores (Top 5 según el ranking)
mejores_features = ranking_im.head(5).index.tolist()
print(f"Mejores características seleccionadas: {mejores_features}")
X_top = X[mejores_features]

# 3. Dividir y entrenar con el subconjunto
X_train_top, X_test_top, y_train_top, y_test_top = dividir_train_test(X_top, y)

# Bayes con top features
m_nb_top, _ = entrenar_y_evaluar(GaussianNB(), "Bayes_Top_IM", X_train_top, X_test_top, y_train_top, y_test_top)
imprimir_resultado(m_nb_top)
todas_las_metricas.append(m_nb_top)

# Árbol con top features
m_tree_top, _ = entrenar_y_evaluar(DecisionTreeClassifier(random_state=42), "Arbol_Top_IM", X_train_top, X_test_top, y_train_top, y_test_top)
imprimir_resultado(m_tree_top)
todas_las_metricas.append(m_tree_top)

# Guardar matrices
graficar_matriz_confusion(m_nb_top['matriz_confusion'], "Bayes Top IM", ruta_resultado("cm_nb_top.png"))
graficar_matriz_confusion(m_tree_top['matriz_confusion'], "Arbol Top IM", ruta_resultado("cm_tree_top.png"))

# ---------------------------------------------------------
# FINALIZACIÓN
# ---------------------------------------------------------
# Guardar resumen final en CSV para la comparativa de la presentación
guardar_resultados(todas_las_metricas, ruta_resultado("comparativa_estres_fase3.csv"))

print(f"\n¡Proceso completado! Los resultados están en la carpeta: {ruta_resultado('')}")