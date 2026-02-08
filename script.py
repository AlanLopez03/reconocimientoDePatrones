import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS = {
    0: "transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignore",
    6: "ignore",
    7: "ignore",
}

# -------------------------
# Helpers
# -------------------------
def load_subject_pkl(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")  # WESAD suele requerir latin1

def label_stats(labels: np.ndarray, fs: int) -> pd.DataFrame:
    values, counts = np.unique(labels, return_counts=True)
    df = pd.DataFrame({"label_id": values, "samples": counts})
    df["label_name"] = df["label_id"].map(LABELS)
    df["seconds"] = df["samples"] / fs
    df["minutes"] = df["seconds"] / 60
    return df.sort_values("minutes", ascending=False)

def window_features(x: np.ndarray, win_size: int, step: int) -> pd.DataFrame:
    feats = []
    if win_size <= 0 or step <= 0:
        raise ValueError("win_size y step deben ser > 0")

    for start in range(0, len(x) - win_size + 1, step):
        w = x[start:start + win_size]
        feats.append({
            "start": start,
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "min": float(np.min(w)),
            "max": float(np.max(w)),
        })
    return pd.DataFrame(feats)

def get_wrist_fs(subject_dict: dict, signal_name: str, default_fs: int) -> int:
    """
    Intenta obtener fs desde subject_dict['wrist_frequency'].
    Si no existe o no contiene la señal, usa default_fs.
    """
    wf = subject_dict.get("wrist_frequency", None)
    if isinstance(wf, dict) and signal_name in wf:
        return int(wf[signal_name])
    return int(default_fs)

def downsample_labels_to_fs(labels_700hz: np.ndarray, fs_labels: int, fs_target: int) -> np.ndarray:
    """
    Reduce labels desde fs_labels a fs_target tomando una muestra cada factor.
    Ej: 700 -> 4 Hz: factor = round(700/4) = 175
    """
    if fs_target <= 0 or fs_labels <= 0:
        raise ValueError("fs_labels y fs_target deben ser > 0")

    factor = int(round(fs_labels / fs_target))
    factor = max(1, factor)
    return labels_700hz[::factor]

def dominant_label(window_labels: np.ndarray) -> int:
    vals, cnts = np.unique(window_labels, return_counts=True)
    return int(vals[np.argmax(cnts)])

# -------------------------
# 1) Cargar todos los sujetos
# -------------------------
WESAD_ROOT = "wesad/WESAD"  # carpeta donde están S2/, S3/, ...
pkl_files = sorted(glob.glob(os.path.join(WESAD_ROOT, "S*", "S*.pkl")))

print("Sujetos encontrados:", len(pkl_files))
print("Ejemplos:", pkl_files[:3])

if not pkl_files:
    raise FileNotFoundError(
        "No encontré archivos .pkl. Verifica WESAD_ROOT y que exista wesad/WESAD/SX/SX.pkl"
    )

# -------------------------
# 2) EDA rápido por sujeto (labels a 700Hz)
# -------------------------
all_label_stats = []
for pkl_path in pkl_files:
    d = load_subject_pkl(pkl_path)
    sid = d.get("subject", os.path.basename(os.path.dirname(pkl_path)))
    labels = np.array(d["label"])

    fs_labels = int(d.get("chest_frequency", 700))  # labels están a 700Hz según README
    stats = label_stats(labels, fs=fs_labels)
    stats["subject"] = sid
    all_label_stats.append(stats)

label_summary = pd.concat(all_label_stats, ignore_index=True)

total_by_class = (
    label_summary.groupby(["label_id", "label_name"])["minutes"]
    .sum().reset_index().sort_values("minutes", ascending=False)
)
print(total_by_class)

plt.figure()
plt.bar(total_by_class["label_name"], total_by_class["minutes"])
plt.xticks(rotation=45, ha="right")
plt.title("Tiempo total por condición (min) - WESAD")
plt.tight_layout()
plt.savefig("wesad_time_by_class.png", dpi=300)

# -------------------------
# 3) Visualizar EDA wrist de 1 sujeto (sin asumir 700Hz)
# -------------------------
example = load_subject_pkl(pkl_files[0])
sid = example.get("subject", "S?")
labels_700 = np.array(example["label"])
fs_labels = int(example.get("chest_frequency", 700))

wrist = example["signal"]["wrist"]
if not (isinstance(wrist, dict) and "EDA" in wrist):
    if isinstance(wrist, dict):
        print("Claves wrist:", list(wrist.keys()))
    raise ValueError("No encontré wrist['EDA'] en el .pkl de ejemplo.")

eda = np.array(wrist["EDA"]).squeeze()

# Frecuencia real de EDA (en README: 4Hz si no viene en metadata)
fs_eda = get_wrist_fs(example, "EDA", default_fs=4)
print(f"Sujeto {sid}: fs_labels={fs_labels}, fs_eda={fs_eda}, len(labels)={len(labels_700)}, len(eda)={len(eda)}")

# Bajamos labels a fs_eda para alinear longitudes
labels_eda = downsample_labels_to_fs(labels_700, fs_labels=fs_labels, fs_target=fs_eda)

# Igualamos longitudes por seguridad
L = min(len(labels_eda), len(eda))
labels_eda = labels_eda[:L]
eda = eda[:L]

# Segmento de 60s en la frecuencia de EDA
t0 = 0
t1 = min(L, 60 * fs_eda)

seg_labels = labels_eda[t0:t1]
seg_eda = eda[t0:t1]

plt.figure()
plt.plot(seg_eda)
plt.title(f"{sid} - Wrist EDA (segmento 60s @ {fs_eda}Hz)")
plt.xlabel("Muestras")
plt.ylabel("EDA")
plt.tight_layout()
plt.savefig(f"wesad_eda_{sid}.png", dpi=300)

# -------------------------
# 4) Features por ventana + comparación por clase (sobre EDA @ fs_eda)
# -------------------------
# Ventanas de 10s, paso 5s (en muestras de fs_eda)
win = 10 * fs_eda
step = 5 * fs_eda

# Si el segmento es muy corto, evita error
if len(seg_eda) < win:
    raise ValueError(f"Segmento muy corto ({len(seg_eda)} muestras) para ventana de {win} muestras. "
                     f"Incrementa el segmento o reduce win.")

feat_df = window_features(seg_eda, win_size=win, step=step)

win_labels = []
for start in feat_df["start"].astype(int).values:
    wlbl = seg_labels[start:start + win]
    win_labels.append(dominant_label(wlbl))

feat_df["label_id"] = win_labels
feat_df["label_name"] = feat_df["label_id"].map(LABELS)

# Opcional: filtrar labels que no quieres en análisis (ignore + transient)
feat_df_clean = feat_df[~feat_df["label_id"].isin([5, 6, 7])].copy()

print("\nPromedios de features por condición (incluye transient si no lo filtras):")
print(feat_df_clean.groupby("label_name")[["mean", "std"]].mean())

# Boxplot simple (sin seaborn) de la media por clase
classes = feat_df_clean["label_name"].unique().tolist()
plt.figure()
data = [feat_df_clean.loc[feat_df_clean["label_name"] == c, "mean"].values for c in classes]
plt.boxplot(data, labels=classes)
plt.xticks(rotation=45, ha="right")
plt.title(f"{sid} - Distribución de mean(EDA) por condición (ventanas)")
plt.tight_layout()
plt.savefig(f"wesad_eda_mean_boxplot_{sid}.png", dpi=300)
