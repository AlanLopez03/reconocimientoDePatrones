"""
Plantilla común para las tareas de clasificación (Naive Bayes y Árboles de
decisión) sobre StressLevelDataset y WESAD.

Uso típico desde un script de tarea:

    from plantilla import (
        dividir_train_test,
        entrenar_y_evaluar,
        graficar_matriz_confusion,
        guardar_resultados,
        ruta_resultado,
    )

Todos los artefactos (CSVs, gráficas) se guardan dentro de la carpeta
`resultados_clasificadores/` mediante el helper `ruta_resultado(nombre)`.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


CARPETA_RESULTADOS = "resultados_clasificadores"


def ruta_resultado(nombre_archivo):
    """
    Devuelve la ruta `resultados_clasificadores/<nombre_archivo>` y se asegura
    de que la carpeta exista. Úsalo siempre que escribas un artefacto del
    experimento (CSV, PNG, etc.) para que todo quede centralizado.
    """
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    return os.path.join(CARPETA_RESULTADOS, nombre_archivo)


def cargar_dataset(ruta, columna_target, columnas_a_excluir=None):
    """
    Carga un CSV y devuelve (X, y).

    - ruta: ruta al archivo CSV.
    - columna_target: nombre de la columna que se usará como etiqueta.
    - columnas_a_excluir: lista de columnas extra a quitar de X
      (ej. ['Subject', 'Time'] para WESAD).
    """
    df = pd.read_csv(ruta)
    excluir = [columna_target] + (columnas_a_excluir or [])
    X = df.drop(columns=excluir)
    y = df[columna_target]
    return X, y


def dividir_train_test(X, y, test_size=0.30, random_state=42):
    """
    Split estratificado 70/30 (por defecto). Siempre estratifica para preservar
    la distribución de clases en train y test.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def entrenar_y_evaluar(modelo, nombre, X_train, X_test, y_train, y_test, average="weighted"):
    """
    Entrena `modelo` con (X_train, y_train), predice sobre X_test y devuelve
    un diccionario con las métricas estándar.

    `average` controla cómo se promedian precision/recall/F1 en multiclase.
    'weighted' funciona bien tanto para binario como multiclase.
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = {
        "modelo": nombre,
        "n_features": X_train.shape[1],
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "matriz_confusion": confusion_matrix(y_test, y_pred),
        "reporte": classification_report(y_test, y_pred, zero_division=0),
    }
    return metricas, modelo


def graficar_matriz_confusion(cm, nombre, ruta_salida, etiquetas=None):
    """
    Guarda un heatmap de la matriz de confusión en `ruta_salida`.
    Sugerencia: pasar `ruta_salida=ruta_resultado("cm_<nombre>.png")`.
    """
    os.makedirs(os.path.dirname(ruta_salida) or ".", exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=etiquetas if etiquetas is not None else "auto",
        yticklabels=etiquetas if etiquetas is not None else "auto",
        cbar=False,
    )
    plt.title(f"Matriz de confusión — {nombre}")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def imprimir_resultado(metricas):
    """
    Imprime un resumen legible de un diccionario de métricas.
    """
    print(f"\n=== {metricas['modelo']} (features={metricas['n_features']}) ===")
    print(f"  Accuracy : {metricas['accuracy']:.4f}")
    print(f"  Precision: {metricas['precision']:.4f}")
    print(f"  Recall   : {metricas['recall']:.4f}")
    print(f"  F1       : {metricas['f1']:.4f}")
    print(metricas["reporte"])


def guardar_resultados(lista_metricas, ruta_csv):
    """
    Vuelca los resultados (excepto la matriz de confusión y el reporte) a un
    CSV con una fila por modelo. Útil para que la persona encargada de la
    comparación final concatene los CSVs de todos los equipos.
    Sugerencia: pasar `ruta_csv=ruta_resultado("resultados_<dataset>.csv")`.
    """
    filas = []
    for m in lista_metricas:
        filas.append({
            "modelo": m["modelo"],
            "n_features": m["n_features"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })
    df = pd.DataFrame(filas)
    df.to_csv(ruta_csv, index=False)
    return df


def ranking_informacion_mutua(X, y, random_state=42):
    """
    Devuelve una Serie de pandas con la información mutua de cada feature
    respecto a `y`, ordenada de mayor a menor. Pensado para StressLevelDataset.
    """
    from sklearn.feature_selection import mutual_info_classif

    scores = mutual_info_classif(X, y, random_state=random_state)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)

import pandas as pd
import numpy as np

def ranking_fisher(X, y):
    clases = np.unique(y)
    scores = {}

    for col in X.columns:
        mu1 = X[y == clases[0]][col].mean()
        mu2 = X[y == clases[1]][col].mean()
        var1 = X[y == clases[0]][col].var()
        var2 = X[y == clases[1]][col].var()

        score = (mu1 - mu2)**2 / (var1 + var2)
        scores[col] = score

    return pd.Series(scores).sort_values(ascending=False)

def graficar_ranking(ranking, titulo, ruta_salida, top_k=None, resaltadas=None):
    """
    Gráfica de barras horizontales de un ranking de features.

    - ranking: pd.Series con scores indexados por nombre de feature.
    - top_k: si se indica, solo muestra las top-k.
    - resaltadas: lista de features a colorear distinto (ej. seleccionadas).
    """
    os.makedirs(os.path.dirname(ruta_salida) or ".", exist_ok=True)

    serie = ranking if top_k is None else ranking.head(top_k)
    serie = serie.sort_values(ascending=True)

    if resaltadas is not None:
        colores = ["salmon" if c in resaltadas else "lightgray" for c in serie.index]
    else:
        colores = "steelblue"

    plt.figure(figsize=(8, max(4, 0.35 * len(serie))))
    plt.barh(serie.index, serie.values, color=colores, edgecolor="black")
    plt.title(titulo)
    plt.xlabel("Score")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
