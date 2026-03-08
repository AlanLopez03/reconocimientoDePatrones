import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import os

def discretize(df, columns, T=3, method="amplitude", labels=[1,2,3]):
    df_result = df.copy()

    if method == "amplitude":
        for col in columns:
            df_result[col] = pd.cut(df[col], bins=T, labels=labels)
    elif method == "frequency":
        for col in columns:
            df_result[col] = pd.qcut(df[col], q=T, labels=labels)
    
    return df_result

def ranking_by_mutual_information(X, y):
    probabilities = mutual_info_classif(X, y, discrete_features=True, random_state=42)

    ranking = pd.DataFrame({
        'feature': X.columns,
        'probabilitie': probabilities
    })

    ranking = ranking.sort_values(by="probabilitie", ascending=False)

    return ranking

def plot_distribution(df, col, title, output_path):
    plt.figure(figsize=(10, 6))
    counts = df[col].value_counts().sort_index(ascending=False)
    ax = counts.rename_axis(None).plot(kind='barh', color='skyblue', edgecolor='black')
    
    legend_text = "\n".join([f'Nivel {k}: {v}' for k, v in counts.sort_index().items()])
    plt.text(1.02, 0.5, f'Distribución:\n{legend_text}', transform=ax.transAxes, 
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'),
             verticalalignment='center')
    
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    output_dir = f"assets/fase2"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv("datasets/StressLevelDataset.csv")
    columns = ['anxiety_level', 'self_esteem', 'depression']
    T = 3
    labels = ['Bajo', 'Medio', 'Alto']

    # Equal Amplitude
    df_amplitude = discretize(df, columns, T, method="amplitude", labels=labels)
    #for col in columns:
    #    df_amplitude[col].value_counts().sort_index(ascending=False).rename_axis(None).plot(kind='barh')
    #    plt.savefig(f"{output_dir}/{col}_dist_ampl.png")
    #    print(f"Guardando {col}...")
    plot_distribution(df_amplitude, columns[0], "Nivel de ansiedad", f"{output_dir}/{columns[0]}_ampl.png")
    plot_distribution(df_amplitude, columns[1], "Autoestima", f"{output_dir}/{columns[1]}_ampl.png")
    plot_distribution(df_amplitude, columns[0], "Depresión", f"{output_dir}/{columns[2]}_ampl.png")

    # Equal Frequency
    df_frequency = discretize(df, columns, T, method="frequency", labels=labels)
    #for col in columns:
    #    df_frequency[col].value_counts().sort_index(ascending=False).rename_axis(None).plot(kind='barh')
    #    plt.savefig(f"{output_dir}/{col}_dist_freq.png")
    #    print(f"Guardando {col}...")
    plot_distribution(df_frequency, columns[0], "Nivel de ansiedad", f"{output_dir}/{columns[0]}_freq.png")
    plot_distribution(df_frequency, columns[1], "Autoestima", f"{output_dir}/{columns[1]}_freq.png")
    plot_distribution(df_frequency, columns[0], "Depresión", f"{output_dir}/{columns[2]}_freq.png")



    # Results
    print("Equal Amplitude")
    print(df_amplitude[columns].head())
    print("Counter: ")
    for col in columns:
        print(f"Distribution {col}")
        print(df_amplitude[col].value_counts().sort_index())
        
    print("Equal Frequency")
    print(df_frequency[columns].head())
    print("Counter: ")
    for col in columns:
        print(f"Distribution {col}")
        print(df_frequency[col].value_counts().sort_index())

    # Ranking
    # Equal Amplitude
    X = df_amplitude[columns]
    y = df["stress_level"]
    mut_inf_ranking = ranking_by_mutual_information(X, y)
    print(mut_inf_ranking)

    # Equal Frequency
    X = df_frequency[columns]
    y = df["stress_level"]
    mut_inf_ranking = ranking_by_mutual_information(X, y)
    print(mut_inf_ranking)

    # Min Max
    for col in columns:
        print(f"{col} MIN: {df[col].min()}")
        print(f"{col} MAX: {df[col].max()}")
    
    print("Initial")
    for col in columns:
        print(f"Distribution {col}")
        print(df[col].value_counts().sort_index())