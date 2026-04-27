from plantilla import (
    cargar_dataset,
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

# 1. Cargar datos
X, y = cargar_dataset("./datasets/StressLevelDataset.csv", columna_target="stress_level")

# Lista para guardar todas las métricas y compararlas al final
todas_las_metricas = []

print("\n--- ACTIVIDAD 1: TODAS LAS CARACTERÍSTICAS ---")

# Dividir 70/30 
X_train, X_test, y_train, y_test = dividir_train_test(X, y)

# Entrenar Bayes
modelo_nb_full = GaussianNB()
m_nb_full, _ = entrenar_y_evaluar(modelo_nb_full, "Bayes_Todas", X_train, X_test, y_train, y_test)
todas_las_metricas.append(m_nb_full)

# Entrenar Árbol
modelo_tree_full = DecisionTreeClassifier(random_state=42)
m_tree_full, _ = entrenar_y_evaluar(modelo_tree_full, "Arbol_Todas", X_train, X_test, y_train, y_test)
todas_las_metricas.append(m_tree_full)

# Guardar matrices de confusión
graficar_matriz_confusion(m_nb_full['matriz_confusion'], "Bayes Todas", ruta_resultado("cm_nb_todas.png"))
graficar_matriz_confusion(m_tree_full['matriz_confusion'], "Arbol Todas", ruta_resultado("cm_tree_todas.png"))

print("\n--- EJECUTANDO ACTIVIDAD 2: MEJORES CARACTERÍSTICAS (INFO MUTUA) ---")

# 1. Calcular Ranking
ranking_im = ranking_informacion_mutua(X, y)
graficar_ranking(ranking_im, "Ranking Información Mutua", ruta_resultado("ranking_im.png"))

# 2. Seleccionar las mejores (ejemplo: las top 5)
mejores_features = ranking_im.head(5).index.tolist()
X_top = X[mejores_features]

# 3. Dividir y entrenar con el subconjunto
X_train_top, X_test_top, y_train_top, y_test_top = dividir_train_test(X_top, y)

# Bayes con top features
m_nb_top, _ = entrenar_y_evaluar(GaussianNB(), "Bayes_Top_IM", X_train_top, X_test_top, y_train_top, y_test_top)
todas_las_metricas.append(m_nb_top)

# Árbol con top features
m_tree_top, _ = entrenar_y_evaluar(DecisionTreeClassifier(random_state=42), "Arbol_Top_IM", X_train_top, X_test_top, y_train_top, y_test_top)
todas_las_metricas.append(m_tree_top)

# Guardar matrices
graficar_matriz_confusion(m_nb_top['matriz_confusion'], "Bayes Top IM", ruta_resultado("cm_nb_top.png"))

# Guardar resumen final en CSV
guardar_resultados(todas_las_metricas, ruta_resultado("comparativa_estres.csv"))

print("\n¡Proceso completado!")