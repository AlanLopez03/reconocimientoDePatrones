from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from plantilla import (
    cargar_dataset, dividir_train_test, entrenar_y_evaluar,
    ranking_informacion_mutua, guardar_resultados, ruta_resultado,
)

# --- Cargar y dividir ---
X, y = cargar_dataset("datasets/StressLevelDataset.csv", columna_target="stress_level")
X_tr, X_te, y_tr, y_te = dividir_train_test(X, y)

# Acumulador de métricas
resultados = []

# --- Tarea 1: todas las features ---
resultados.append(entrenar_y_evaluar(
    GaussianNB(), "T1_NB_todas", X_tr, X_te, y_tr, y_te
))
resultados.append(entrenar_y_evaluar(
    DecisionTreeClassifier(random_state=42), "T1_Arbol_todas", X_tr, X_te, y_tr, y_te
))

# --- Tarea 2: top-5 por información mutua ---
ranking = ranking_informacion_mutua(X, y)
top5 = ranking.head(5).index.tolist()
X_tr5, X_te5, y_tr5, y_te5 = dividir_train_test(X[top5], y)

resultados.append(entrenar_y_evaluar(
    GaussianNB(), "T2_NB_top5", X_tr5, X_te5, y_tr5, y_te5
))
resultados.append(entrenar_y_evaluar(
    DecisionTreeClassifier(random_state=42), "T2_Arbol_top5", X_tr5, X_te5, y_tr5, y_te5
))

# --- Guardar todo en un solo CSV ---
guardar_resultados(resultados, ruta_resultado("resultados_stress.csv"))