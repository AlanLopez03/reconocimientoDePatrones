import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar los datos que extrajiste en el paso anterior
df = pd.read_csv("wesad_features_extraidas.csv")

print("\n--- 2. ANÁLISIS EXPLORATORIO (EDA) ---")

# A. Balance de Clases
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Label', palette="Set2")
plt.title("Distribución de Clases (0 = No Estrés, 1 = Estrés)")
plt.savefig("distribucion_clases.png")  # Guardar la figura

# B. Identificación de Outliers (Valores Atípicos) - ORIGINAL
features = df.drop(columns=["Time", "Label", "Subject"]).columns

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[features], orient="h", palette="viridis")
plt.title("Distribución de Características (Búsqueda de Outliers)")
plt.savefig("boxplot_outliers.png")  # Guardar la figura

# C. Matriz de Correlación
plt.figure(figsize=(10, 8))
corr = df[features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Matriz de Correlación entre Variables Fisiológicas")
plt.savefig("matriz_correlacion.png")  # Guardar la figura

# --- 3. PREPROCESAMIENTO ---
df_limpio = df.dropna().copy()
print(f"\nTamaño original: {df_limpio.shape[0]} ventanas de tiempo.")

# SEPARAR columnas numéricas para buscar outliers
columnas_fisiologicas = df_limpio.drop(columns=["Time", "Label", "Subject"]).columns

# ELIMINACIÓN DE OUTLIERS (Método IQR)
Q1 = df_limpio[columnas_fisiologicas].quantile(0.25)
Q3 = df_limpio[columnas_fisiologicas].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 3.0 * IQR 
limite_superior = Q3 + 3.0 * IQR

condicion = ~((df_limpio[columnas_fisiologicas] < limite_inferior) | (df_limpio[columnas_fisiologicas] > limite_superior)).any(axis=1)
df_sin_outliers = df_limpio[condicion].copy()

print(f"Tamaño después de limpiar outliers: {df_sin_outliers.shape[0]} ventanas de tiempo.")
print(f"Se eliminaron {df_limpio.shape[0] - df_sin_outliers.shape[0]} filas con valores anómalos.")

# AHORA SÍ, ESTANDARIZAR (Sobre los datos ya sin outliers)
X = df_sin_outliers.drop(columns=["Time", "Label", "Subject"])
etiquetas = df_sin_outliers[["Subject", "Label", "Time"]].reset_index(drop=True)

scaler = StandardScaler()
X_escalado = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# GUARDAR Y ENTREGAR
df_final = pd.concat([etiquetas, X_escalado], axis=1)
df_final.to_csv("wesad_preprocesado_listo.csv", index=False)
print(" ¡Preprocesamiento listo! Archivo generado para la Persona 3.")

# --- 4. NUEVO: GRÁFICAS DE COMPARACIÓN ANTES VS DESPUÉS ---
print("\n--- GENERANDO GRÁFICA DE COMPARACIÓN ANTES VS DESPUÉS ---")

# Crear una figura con 2 espacios (1 fila, 2 columnas)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Gráfica 1 (Izquierda): ANTES
sns.boxplot(data=df_limpio[columnas_fisiologicas], orient="h", palette="viridis", ax=axes[0])
axes[0].set_title("ANTES: Datos Crudos\n(Diferentes escalas y Outliers extremos)")
axes[0].set_xlabel("Valor Original")

# Gráfica 2 (Derecha): DESPUÉS
sns.boxplot(data=X_escalado, orient="h", palette="viridis", ax=axes[1])
axes[1].set_title("DESPUÉS: Datos Preprocesados\n(Sin Outliers y Estandarizados al mismo rango)")
axes[1].set_xlabel("Valor Estandarizado (Media=0, Desv=1)")

plt.tight_layout()
plt.savefig("comparacion_antes_despues.png", dpi=300) # dpi=300 le da alta resolución
print("Gráfica de comparación guardada como 'comparacion_antes_despues.png'")