import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import os

def discretize(df, columns, T=3, method="amplitude"):
    df_result = df.copy()

    if method == "amplitude":
        for col in columns:
            df_result[col] = pd.cut(df[col], bins=T, labels=[i for i in range(1, T+1)])
    elif method == "frequency":
        for col in columns:
            df_result[col] = pd.qcut(df[col], q=T, labels=[i for i in range(1, T+1)])
    
    return df_result

def ranking_by_mutual_information(X, y):
    probabilities = mutual_info_classif(X, y, discrete_features=True, random_state=42)

    ranking = pd.DataFrame({
        'feature': X.columns,
        'probabilitie': probabilities
    })

    ranking = ranking.sort_values(by="probabilitie", ascending=False)

    return ranking

if __name__ == "__main__":
    output_dir = f"assets/fase2"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv("datasets/StressLevelDataset.csv")
    columns = ['anxiety_level', 'self_esteem', 'depression']
    T = 3

    # Equal Amplitude
    df_amplitude = discretize(df, columns, T, method="amplitude")
    for col in columns:
        df_amplitude[col].value_counts().sort_index(ascending=False).rename_axis(None).plot(kind='barh')
        plt.savefig(f"{output_dir}/{col}_dist_ampl.png")
        print(f"Guardando {col}...")

    # Equal Frequency
    df_frequency = discretize(df, columns, T, method="frequency")
    for col in columns:
        df_frequency[col].value_counts().sort_index(ascending=False).rename_axis(None).plot(kind='barh')
        plt.savefig(f"{output_dir}/{col}_dist_freq.png")
        print(f"Guardando {col}...")


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