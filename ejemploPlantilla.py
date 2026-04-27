from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plantilla import (
    cargar_dataset, dividir_train_test, entrenar_y_evaluar,
    ranking_informacion_mutua, guardar_resultados, ruta_resultado,
    ranking_fisher, graficar_matriz_confusion,
)

# --- Cargar y dividir ---
ruta = r"./programas_wesad/wesad_preprocesado_listo.csv"
X, y = cargar_dataset(ruta, columna_target="Label",
columnas_a_excluir=['Subject', 'Time']
)

X_tr, X_te, y_tr, y_te = dividir_train_test(X, y)

# Acumulador de métricas
resultados = []

# --- Tarea 1: todas las features ---
metricas_nb, modelo_nb = (entrenar_y_evaluar(
    GaussianNB(), "T1_NB_todas", X_tr, X_te, y_tr, y_te
))
metricas_tree, modelo_tree =(entrenar_y_evaluar(
    DecisionTreeClassifier(random_state=42), "T1_Arbol_todas", X_tr, X_te, y_tr, y_te
))

graficar_matriz_confusion(metricas_nb["matriz_confusion"], "T1_NB_todas",
    ruta_resultado("cm_T1_NB_todas.png"), etiquetas=["No Estrés", "Estrés"])
graficar_matriz_confusion(metricas_tree["matriz_confusion"], "T1_Arbol_todas",
    ruta_resultado("cm_T1_Arbol_todas.png"), etiquetas=["No Estrés", "Estrés"])

resultados.append(metricas_nb)
resultados.append(metricas_tree)

#guarda el arbol de decision de la tarea 1
plt.figure(figsize=(24,16))
plot_tree(modelo_tree, feature_names=X_tr.columns, class_names=True, filled=True)
plt.title("Árbol de decisión Tarea 1")
plt.tight_layout()
plt.savefig(ruta_resultado("arbol_decision_T1.png"), dpi=300, bbox_inches='tight')
plt.close()

#guarda el bayes ingenuo de la tarea 1
df_nb = pd.DataFrame(modelo_nb.theta_, columns=X_tr.columns)
df_nb["Clase"] = modelo_nb.classes_

plt.figure(figsize=(12,6))
sns.heatmap(df_nb.drop("Clase", axis=1), annot=False)
plt.title("Parámetros Naive Bayes tarea 1")
plt.tight_layout()
plt.savefig(ruta_resultado("naive_bayes_heatmap_T1.png"), bbox_inches='tight')
plt.close()

priors = pd.DataFrame({
    "Clase": modelo_nb.classes_,
    "Prior": modelo_nb.class_prior_
})
medias = pd.DataFrame(modelo_nb.theta_, columns=X_tr.columns)
medias["Clase"] = modelo_nb.classes_
varianzas = pd.DataFrame(modelo_nb.var_, columns=X_tr.columns)
varianzas["Clase"] = modelo_nb.classes_

priors.to_csv(ruta_resultado("naive_bayes_modelo1_priors.csv"), index=False)
medias.to_csv(ruta_resultado("naive_bayes_modelo1_medias.csv"), index=False)
varianzas.to_csv(ruta_resultado("naive_bayes_modelo1_varianzas.csv"), index=False)


# --- Tarea 2: top-5 por información mutua ---
ranking = ranking_fisher(X, y)
top5 = ranking.head(5).index.tolist()
X_tr5, X_te5, y_tr5, y_te5 = dividir_train_test(X[top5], y)

metricas_nb, modelo_nb = (entrenar_y_evaluar(
    GaussianNB(), "T2_NB_top5", X_tr5, X_te5, y_tr5, y_te5
))
metricas_tree, modelo_tree =(entrenar_y_evaluar(
    DecisionTreeClassifier(random_state=42), "T2_Arbol_top5", X_tr5, X_te5, y_tr5, y_te5
))

graficar_matriz_confusion(metricas_nb["matriz_confusion"], "T2_NB_top5",
    ruta_resultado("cm_T2_NB_top5.png"), etiquetas=["No Estrés", "Estrés"])
graficar_matriz_confusion(metricas_tree["matriz_confusion"], "T2_Arbol_top5",
    ruta_resultado("cm_T2_Arbol_top5.png"), etiquetas=["No Estrés", "Estrés"])

resultados.append(metricas_nb)
resultados.append(metricas_tree)

#guarda el arbol de decision de la tarea 2
plt.figure(figsize=(24,16))
plot_tree(modelo_tree, feature_names=top5, class_names=True, filled=True)
plt.title("Árbol de decisión Tarea 2")
plt.tight_layout()
plt.savefig(ruta_resultado("arbol_decision_T2.png"), dpi=300, bbox_inches='tight')
plt.close()

#guarda el bayes ingenuo de la tarea 2
df_nb = pd.DataFrame(modelo_nb.theta_, columns=top5)
df_nb["Clase"] = modelo_nb.classes_

plt.figure(figsize=(12,6))
sns.heatmap(df_nb.drop("Clase", axis=1), annot=False)
plt.title("Parámetros Naive Bayes tarea 2")
plt.tight_layout()
plt.savefig(ruta_resultado("naive_bayes_heatmap_T2.png"), bbox_inches='tight')
plt.close()

priors = pd.DataFrame({
    "Clase": modelo_nb.classes_,
    "Prior": modelo_nb.class_prior_
})
medias = pd.DataFrame(modelo_nb.theta_, columns=top5)
medias["Clase"] = modelo_nb.classes_
varianzas = pd.DataFrame(modelo_nb.var_, columns=top5)
varianzas["Clase"] = modelo_nb.classes_

priors.to_csv(ruta_resultado("naive_bayes_modelo2_priors.csv"), index=False)
medias.to_csv(ruta_resultado("naive_bayes_modelo2_medias.csv"), index=False)
varianzas.to_csv(ruta_resultado("naive_bayes_modelo2_varianzas.csv"), index=False)
# --- Guardar todo en un solo CSV ---
guardar_resultados(resultados, ruta_resultado("resultados_stress.csv"))