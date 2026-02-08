import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def _to_2d(arr):
    """Convierte la señal a matriz 2D (n_muestras, n_canales)."""
    x = np.asarray(arr)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 2:
        # caso raro: aplana todo menos el eje de muestras
        x = x.reshape(x.shape[0], -1)
    return x

def signal_stats(x_2d: np.ndarray, signal_name: str, group: str):
    """Calcula estadísticas por canal para una señal 2D."""
    rows = []
    n_samples = x_2d.shape[0]
    n_channels = x_2d.shape[1]

    for ch in range(n_channels):
        col = x_2d[:, ch].astype(float)

        # manejar NaNs de forma segura (por si aparecen)
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

def wesad_subject_stats(pkl_path: str, out_csv: str = None) -> pd.DataFrame:
    pkl_path = Path(pkl_path)

    with pkl_path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    all_rows = []

    # Estructura típica: data["signal"]["chest"] y data["signal"]["wrist"]
    signal_root = data.get("signal", {})

    for group_name, group_signals in signal_root.items():
        # group_name suele ser "chest" y/o "wrist"
        if not isinstance(group_signals, dict):
            continue

        for sig_name, sig_values in group_signals.items():
            x = _to_2d(sig_values)
            all_rows.extend(signal_stats(x, sig_name, group_name))

    df = pd.DataFrame(all_rows).sort_values(["grupo", "senal", "canal"]).reset_index(drop=True)

    if out_csv is None:
        out_csv = pkl_path.with_name(pkl_path.stem + "_stats.csv")

    df.to_csv(out_csv, index=False)
    print(f"✅ Estadísticas guardadas en: {out_csv}")
    print(df.to_string(index=False))

    return df

if __name__ == "__main__":
    # Cambia la ruta a tu S2.pkl
    wesad_subject_stats("wesad/WESAD/S10/S10.pkl")

