import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MAX_HIST_SAMPLES = 200000  # para no graficar millones de puntos (ajústalo si quieres)
HIST_BINS = 60

def _to_2d(arr):
    """Convierte la señal a matriz 2D (n_muestras, n_canales)."""
    x = np.asarray(arr)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    return x

def signal_stats(x_2d: np.ndarray, signal_name: str, group: str):
    """Calcula estadísticas por canal para una señal 2D."""
    rows = []
    n_samples = x_2d.shape[0]

    for ch in range(x_2d.shape[1]):
        col = x_2d[:, ch].astype(float)
        valid = col[~np.isnan(col)]

        if valid.size == 0:
            rows.append({
                "grupo": group,
                "senal": signal_name,
                "canal": ch,
                "total_muestras": n_samples,
                "muestras_validas": 0,
                "media": np.nan,
                "min": np.nan,
                "max": np.nan,
                "desv_estandar": np.nan,
            })
            continue

        rows.append({
            "grupo": group,
            "senal": signal_name,
            "canal": ch,
            "total_muestras": n_samples,
            "muestras_validas": int(valid.size),
            "media": float(np.mean(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "desv_estandar": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
        })

    return rows

def _sample_for_hist(x: np.ndarray, max_n: int):
    """Submuestrea para histogramas grandes sin sesgo fuerte."""
    if x.size <= max_n:
        return x
    idx = np.random.choice(x.size, size=max_n, replace=False)
    return x[idx]

def plot_histograms_selected(data, subject_name="S?", save_dir=None):
    """
    Genera histogramas para:
      - chest/EDA canal 0
      - chest/ECG canal 0
      - wrist/ACC canales 0,1,2
    """
    signals = data.get("signal", {})
    if not isinstance(signals, dict):
        print("No encontré data['signal'] como dict.")
        return

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    def plot_hist(vec, title, filename=None):
        vec = vec[~np.isnan(vec)].astype(float)
        if vec.size == 0:
            print(f"Sin datos válidos para {title}")
            return

        vec_plot = _sample_for_hist(vec, MAX_HIST_SAMPLES)

        plt.figure(figsize=(6, 4))
        plt.hist(vec_plot, bins=HIST_BINS)
        plt.title(title)
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.tight_layout()

        if filename and save_dir is not None:
            out_path = save_dir / filename
            plt.savefig(out_path, dpi=160)
            print(f"📌 Guardado: {out_path}")



    # 1) chest EDA
    try:
        chest_eda = _to_2d(signals["chest"]["EDA"])[:, 0]
        plot_hist(chest_eda, f"{subject_name} - chest/EDA (canal 0)", f"{subject_name}_chest_EDA_c0_hist.png")
    except KeyError:
        print("No encontré chest/EDA en este .pkl")

    # 2) chest ECG
    try:
        chest_ecg = _to_2d(signals["chest"]["ECG"])[:, 0]
        plot_hist(chest_ecg, f"{subject_name} - chest/ECG (canal 0)", f"{subject_name}_chest_ECG_c0_hist.png")
    except KeyError:
        print("No encontré chest/ECG en este .pkl")

    try:
        wrist_acc = _to_2d(signals["wrist"]["ACC"])
        plot_acc_multichannel(
            wrist_acc,
            group="wrist",
            subject_name=subject_name,
            max_samples=3000,
            save_path=save_dir / f"{subject_name}_wrist_ACC_multichannel.png" if save_dir else None
        )
    except KeyError:
        print("No encontré wrist/ACC para gráfica multicanal")
    try:
        chest_temp = _to_2d(signals["chest"]["Temp"])[:, 0]
        plot_hist(
            chest_temp,
            f"{subject_name} - chest/Temp (canal 0)",
            f"{subject_name}_chest_Temp_c0_hist.png"
        )
    except KeyError:
        print("No encontré chest/Temp en este .pkl")
def plot_acc_multichannel(acc_2d, group, subject_name, max_samples=5000, save_path=None):
    """
    Grafica los 3 ejes del acelerómetro en una sola gráfica.
    """
    acc_2d = np.asarray(acc_2d)
    acc_2d = acc_2d[:max_samples, :]  # limitar muestras

    plt.figure(figsize=(10, 4))
    plt.plot(acc_2d[:, 0], label="Eje X")
    plt.plot(acc_2d[:, 1], label="Eje Y")
    plt.plot(acc_2d[:, 2], label="Eje Z")

    plt.title(f"{subject_name} - {group}/ACC (3 ejes)")
    plt.xlabel("Muestras")
    plt.ylabel("Aceleración")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"📌 Guardado: {save_path}")

   # ()

def wesad_subject_stats(pkl_path: str, out_csv: str = None, make_hists=True, save_hists=True) -> pd.DataFrame:
    pkl_path = Path(pkl_path)

    with pkl_path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    all_rows = []
    signal_root = data.get("signal", {})

    for group_name, group_signals in signal_root.items():
        if not isinstance(group_signals, dict):
            continue
        for sig_name, sig_values in group_signals.items():
            x = _to_2d(sig_values)
            all_rows.extend(signal_stats(x, sig_name, group_name))

    df = pd.DataFrame(all_rows).sort_values(["grupo", "senal", "canal"]).reset_index(drop=True)

    if out_csv is None:
        out_csv = pkl_path.with_name(pkl_path.stem + "_stats.csv")#exportar la tabla de estadisticas

    df.to_csv(out_csv, index=False)
    print(f"✅ Estadísticas guardadas en: {out_csv}")
    print(df.to_string(index=False))

    # Histogramas seleccionados
    if make_hists:
        subject_name = pkl_path.parent.name  # "S4"
        save_dir = pkl_path.parent / "eda_plots" if save_hists else None
        plot_histograms_selected(data, subject_name=subject_name, save_dir=save_dir)

    return df

if __name__ == "__main__":
    wesad_subject_stats("WESAD/S4/S4.pkl", make_hists=True, save_hists=True)
