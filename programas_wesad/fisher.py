import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1. CARGAR DATOS
# ============================================
# Asegúrate de que la ruta sea correcta en tu PC
try:
    datos = pd.read_csv(r'wesad_preprocesado_listo.csv')
except FileNotFoundError:
    print("Error: No se encontró el archivo CSV en la ruta especificada.")
    exit()

caracteristicas = datos.drop(columns=['Subject', 'Label', 'Time'])
etiquetas = datos['Label']

# ============================================
# 2. FACTOR DE FISHER
# ============================================
def calcular_factor_fisher(X, y):
    clases = np.unique(y)
    num_muestras = len(y)
    nombres_caracteristicas = list(X.columns)
    probabilidades_clase = {clase: np.sum(y == clase) / num_muestras for clase in clases}
    
    factores_fisher = []
    for nombre in nombres_caracteristicas:
        caracteristica = X[nombre]
        media_global = np.mean(caracteristica)
        
        # Numerador: Σ p_i * (μ_i - μ_global)^2
        numerador = sum(
            probabilidades_clase[clase] * (np.mean(caracteristica[y == clase]) - media_global) ** 2
            for clase in clases
        )
        # Denominador: Σ p_i * σ_i^2
        denominador = sum(
            probabilidades_clase[clase] * np.var(caracteristica[y == clase], ddof=1)
            for clase in clases
        )
        
        factores_fisher.append(numerador / (denominador + 1e-10))
    
    return pd.Series(factores_fisher, index=nombres_caracteristicas)

fisher_scores = calcular_factor_fisher(caracteristicas, etiquetas)
ranking_fisher = fisher_scores.sort_values(ascending=False)

# ============================================
# 3. SELECCIÓN ESCALAR HACIA ADELANTE
# ============================================
def seleccion_escalar_adelante(X, fisher_scores, M=5, alpha1=1.0, alpha2=0.5):
    nombres_todos = list(X.columns)
    seleccionadas = []
    nombres_restantes = nombres_todos.copy()
    
    for k in range(1, M + 1):
        mejor_score = -np.inf
        mejor_caracteristica = None
        
        for nombre in nombres_restantes:
            if k == 1:
                score = alpha1 * fisher_scores[nombre]
            else:
                termino_fisher = alpha1 * fisher_scores[nombre]
                penalizacion = sum(abs(np.corrcoef(X[s_r], X[nombre])[0, 1]) for s_r in seleccionadas)
                score = termino_fisher - (alpha2 / (k - 1)) * penalizacion
                
            if score > mejor_score:
                mejor_score = score
                mejor_caracteristica = nombre
        
        seleccionadas.append(mejor_caracteristica)
        nombres_restantes.remove(mejor_caracteristica)
    
    return seleccionadas

caracteristicas_seleccionadas = seleccion_escalar_adelante(caracteristicas, fisher_scores, M=5)

# ============================================
# 4. GRÁFICOS (RECTIFICADO)
# ============================================
# Creamos la figura con dos subgráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Gráfico 1: Ranking General ---
# Invertimos para que el mayor esté arriba en el barh
ranking_plot = ranking_fisher.sort_values(ascending=True) 
colores_todos = ['red' if c in caracteristicas_seleccionadas else 'lightgray' for c in ranking_plot.index]

ax1.barh(ranking_plot.index, ranking_plot.values, color=colores_todos, edgecolor='black')
ax1.set_title('Ranking de Fisher (Todas)')
ax1.set_xlabel('Score de Fisher')
ax1.grid(axis='x', linestyle='--', alpha=0.6)

# --- Gráfico 2: Características Seleccionadas ---
# Mostramos las M seleccionadas y su score original
scores_selec = [fisher_scores[c] for c in caracteristicas_seleccionadas]
# Invertimos el orden para que la primera seleccionada aparezca arriba
ax2.barh(caracteristicas_seleccionadas[::-1], scores_selec[::-1], color='salmon', edgecolor='black')
ax2.set_title(f'Top {len(caracteristicas_seleccionadas)} Seleccionadas\n(Orden de importancia + Descorrelación)')
ax2.set_xlabel('Score de Fisher Individual')
ax2.grid(axis='x', linestyle='--', alpha=0.6)

# Ajustar diseño y MOSTRAR
plt.tight_layout()
#plt.show() # <--- ESTA LÍNEA ES VITAL
plt.savefig("ranking_fisher.png")