import os
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt, find_peaks

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    matthews_corrcoef, roc_auc_score
)

# ===================== CONFIG =====================
DATA_PATH = "./WESAD"
FS = 700

SUBJECT_IDS = (
    [f"S{i}" for i in range(2, 12)] +
    [f"S{i}" for i in range(13, 18)]
)

WINDOW_SEC = 120
STEP_SEC = 60
SEQ_LEN = 5

# ===================== LOAD SUBJECT =====================
def load_subject(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

# ===================== EDA FEATURES =====================
def tonic_phasic(eda, fs):
    eda = eda[~np.isnan(eda)]
    b, a = butter(4, 0.05 / (fs / 2), btype="low")
    tonic = filtfilt(b, a, eda)
    phasic = eda - tonic
    return tonic, phasic

def extract_eda_features(eda, fs):
    tonic, phasic = tonic_phasic(eda, fs)
    peaks, _ = find_peaks(phasic, height=0.01, distance=fs)
    return len(peaks), np.trapezoid(np.abs(phasic)) / fs, np.mean(tonic)

def build_eda_table(subject):
    eda = subject["signal"]["chest"]["EDA"]
    labels = subject["label"]

    WINDOW = FS * WINDOW_SEC
    STEP = FS * STEP_SEC
    rows = []

    for i in range(0, len(eda) - WINDOW, STEP):
        win_eda = eda[i:i + WINDOW]
        win_labels = labels[i:i + WINDOW]

        if np.isnan(win_eda).any():
            continue

        majority = np.bincount(win_labels).argmax()
        if majority not in [1, 2, 3, 4]:
            continue

        label_bin = 1 if majority == 2 else 0
        scr, auc, tonic = extract_eda_features(win_eda, FS)

        rows.append({
            "Time": i / FS,
            "Label": label_bin,
            "EDA_SCR_count": scr,
            "EDA_Phasic_AUC": auc,
            "EDA_Tonic_Mean": tonic
        })

    return pd.DataFrame(rows)

# ===================== HRV FEATURES =====================
def build_hrv_table(subject):
    ecg = subject["signal"]["chest"]["ECG"]
    labels = subject["label"]

    cleaned = nk.ecg_clean(ecg, sampling_rate=FS)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=FS)
    rpeaks_idx = rpeaks["ECG_R_Peaks"]

    WINDOW = FS * WINDOW_SEC
    STEP = FS * STEP_SEC
    rows = []

    for start in range(0, len(ecg) - WINDOW, STEP):
        end = start + WINDOW
        peaks = rpeaks_idx[(rpeaks_idx >= start) & (rpeaks_idx < end)] - start
        if len(peaks) <= 2:
            continue

        rr_ms = np.diff(peaks) / FS * 1000
        if len(rr_ms) < 2:
            continue

        try:
            rpeaks_clean = nk.intervals_to_peaks(rr_ms, sampling_rate=FS)
            hrv_t = nk.hrv_time(rpeaks_clean, sampling_rate=FS, show=False)
            hrv_f = nk.hrv_frequency(rpeaks_clean, sampling_rate=FS, show=False)

            label_bin = np.bincount((labels[start:end] == 2).astype(int)).argmax()

            rows.append({
                "Time": (start + end) / 2 / FS,
                "Label": label_bin,
                "HRV_RMSSD": hrv_t["HRV_RMSSD"].values[0],
                "HRV_SDNN": hrv_t["HRV_SDNN"].values[0],
                "HRV_MeanNN": hrv_t["HRV_MeanNN"].values[0],
                "HRV_LF": hrv_f["HRV_LF"].values[0],
                "HRV_HF": hrv_f["HRV_HF"].values[0],
                "HRV_LFHF": hrv_f["HRV_LFHF"].values[0],
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

# ===================== FEATURE EXTRACTION =====================
eda_all, hrv_all = [], []

for sid in SUBJECT_IDS:
    subject = load_subject(f"{DATA_PATH}/{sid}/{sid}.pkl")

    eda_df = build_eda_table(subject)
    hrv_df = build_hrv_table(subject)

    eda_df["Subject"] = sid
    hrv_df["Subject"] = sid

    eda_all.append(eda_df)
    hrv_all.append(hrv_df)

EDA_FEATURES = pd.concat(eda_all, ignore_index=True)
HRV_FEATURES = pd.concat(hrv_all, ignore_index=True)

merged_df = pd.merge(
    EDA_FEATURES,
    HRV_FEATURES,
    on=["Time", "Label", "Subject"],
    how="inner"
)

print("\n===== DATASET SUMMARY =====")
print("Total samples:", len(merged_df))
print("Total subjects:", merged_df["Subject"].nunique())
print("Subjects:", sorted(merged_df["Subject"].unique()))

print("\nLabel distribution:")
print(merged_df["Label"].value_counts())

print("\nWindows per subject:")
print(merged_df.groupby("Subject").size())

# ===================== TABULAR DATA =====================
X = merged_df.drop(columns=["Time", "Label", "Subject"])
y = merged_df["Label"].values
groups = merged_df["Subject"].values

logo = LeaveOneGroupOut()

print("\n===== FEATURE STATISTICS (mean / std) =====")
print(X.describe().T[["mean", "std"]])

merged_df.to_csv("wesad_features_extraidas.csv", index=False)
print("✅ Extracción completada y guardada en 'wesad_features_extraidas.csv'")
# # # ===================== MODEL EVALUATION HELPER =====================
# # def evaluate_model(model, X, y, groups, model_name):
# #     acc, f1, mcc, roc = [], [], [], []
# #     y_true, y_pred, y_prob = [], [], []

# #     for tr, te in logo.split(X, y, groups):
# #         model.fit(X.iloc[tr], y[tr])

# #         preds = model.predict(X.iloc[te])
# #         probs = (
# #             model.predict_proba(X.iloc[te])[:, 1]
# #             if hasattr(model, "predict_proba")
# #             else preds
# #         )

# #         acc.append(accuracy_score(y[te], preds))
# #         f1.append(f1_score(y[te], preds))
# #         mcc.append(matthews_corrcoef(y[te], preds))
# #         roc.append(roc_auc_score(y[te], probs))

# #         y_true.extend(y[te])
# #         y_pred.extend(preds)

# #     acc_m, acc_s = np.mean(acc), np.std(acc)
# #     f1_m = np.mean(f1)
# #     mcc_m = np.mean(mcc)
# #     roc_m = np.mean(roc)

# #     cm = confusion_matrix(y_true, y_pred)

# #     print(f"\n===== {model_name} RESULTS =====")
# #     print(f"Accuracy      : {acc_m:.3f} ± {acc_s:.3f}")
# #     print(f"F1-score      : {f1_m:.3f}")
# #     print(f"MCC           : {mcc_m:.3f}")
# #     print(f"ROC-AUC       : {roc_m:.3f}")

# #     sns.heatmap(
# #         cm, annot=True, fmt="d", cmap="Blues",
# #         xticklabels=["Non-Stress", "Stress"],
# #         yticklabels=["Non-Stress", "Stress"]
# #     )
# #     plt.title(f"{model_name} Confusion Matrix (LOSO)")
# #     #plt.show()
# #     plt.savefig(f"./imagenes/{model_name}_confusion_matrix.png")

# #     return acc_m, acc_s, f1_m, mcc_m, roc_m



# # # ===================== CLASSICAL MODELS =====================
# # results = {}

# # results["SVM"] = evaluate_model(
# #     Pipeline([("scaler", StandardScaler()),
# #               ("svm", SVC(kernel="rbf", class_weight="balanced",probability=True))]),
# #     X, y, groups, "SVM"
# # )

# # results["RF"] = evaluate_model(
# #     RandomForestClassifier(
# #         n_estimators=300,
# #         min_samples_leaf=5,
# #         class_weight="balanced",
# #         random_state=42
# #     ),
# #     X, y, groups, "Random Forest"
# # )

# # results["XGBoost"] = evaluate_model(
# #     XGBClassifier(
# #         n_estimators=300,
# #         max_depth=4,
# #         learning_rate=0.05,
# #         subsample=0.8,
# #         colsample_bytree=0.8,
# #         scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
# #         eval_metric="logloss"
# #     ),
# #     X, y, groups, "XGBoost"
# # )

# # results["LR"] = evaluate_model(
# #     Pipeline([("scaler", StandardScaler()),
# #               ("lr", LogisticRegression(
# #                   class_weight="balanced",
# #                   max_iter=1000))]),
# #     X, y, groups, "Logistic Regression"
# # )

# # results["AdaBoost"] = evaluate_model(
# #     AdaBoostClassifier(
# #         estimator=DecisionTreeClassifier(
# #             max_depth=2,
# #             min_samples_leaf=5,
# #             class_weight="balanced"
# #         ),
# #         n_estimators=200,
# #         learning_rate=0.5
# #     ),
# #     X, y, groups, "AdaBoost"
# # )

# # results["LDA"] = evaluate_model(
# #     Pipeline([("scaler", StandardScaler()),
# #               ("lda", LinearDiscriminantAnalysis())]),
# #     X, y, groups, "LDA"
# # )

# # results["KNN"] = evaluate_model(
# #     Pipeline([("scaler", StandardScaler()),
# #               ("knn", KNeighborsClassifier(
# #                   n_neighbors=7,
# #                   weights="distance"))]),
# #     X, y, groups, "KNN"
# # )

# # # ===================== LSTM & CNN DATA =====================
# # FEATURE_COLS = X.columns
# # X_seq, y_seq, groups_seq = [], [], []

# # for s in merged_df["Subject"].unique():
# #     df_s = merged_df[merged_df["Subject"] == s].sort_values("Time")
# #     Xs, ys = df_s[FEATURE_COLS].values, df_s["Label"].values

# #     for i in range(len(df_s) - SEQ_LEN + 1):
# #         X_seq.append(Xs[i:i + SEQ_LEN])
# #         y_seq.append(ys[i + SEQ_LEN - 1])
# #         groups_seq.append(s)

# # X_seq = np.array(X_seq)
# # y_seq = np.array(y_seq)
# # groups_seq = np.array(groups_seq)

# # # ===================== DL MODEL EVALUATION HELPER =====================
# # def evaluate_dl_model(build_model_fn, X_seq, y_seq, groups_seq, model_name):
# #     acc, f1, mcc, roc = [], [], [], []
# #     y_true, y_pred, y_prob = [], [], []

# #     for tr, te in logo.split(X_seq, y_seq, groups_seq):
# #         model = build_model_fn((X_seq.shape[1], X_seq.shape[2]))

# #         model.fit(X_seq[tr], y_seq[tr],
# #                   epochs=20, batch_size=16, verbose=0)

# #         probs = model.predict(X_seq[te]).ravel()
# #         preds = (probs >= 0.5).astype(int)

# #         acc.append(accuracy_score(y_seq[te], preds))
# #         f1.append(f1_score(y_seq[te], preds))
# #         mcc.append(matthews_corrcoef(y_seq[te], preds))
# #         roc.append(roc_auc_score(y_seq[te], probs))

# #         y_true.extend(y_seq[te])
# #         y_pred.extend(preds)

# #     acc_m, acc_s = np.mean(acc), np.std(acc)

# #     print(f"\n===== {model_name} RESULTS =====")
# #     print(f"Accuracy      : {acc_m:.3f} ± {acc_s:.3f}")
# #     print(f"F1-score      : {np.mean(f1):.3f}")
# #     print(f"MCC           : {np.mean(mcc):.3f}")
# #     print(f"ROC-AUC       : {np.mean(roc):.3f}")

# #     cm = confusion_matrix(y_true, y_pred)
# #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# #     plt.title(f"{model_name} Confusion Matrix (LOSO)")
# #     #plt.show()
# #     plt.savefig(f"./imagenes/{model_name}_confusion_matrix.png")

# #     return acc_m, acc_s, np.mean(f1), np.mean(mcc), np.mean(roc)

    
# # def build_lstm(input_shape):
# #     model = Sequential([
# #         LSTM(32, input_shape=input_shape),
# #         Dropout(0.3),
# #         Dense(1, activation="sigmoid")
# #     ])

# #     model.compile(
# #         optimizer=Adam(learning_rate=0.001),
# #         loss="binary_crossentropy",
# #         metrics=["accuracy"]
# #     )
# #     return model

# # def build_cnn(input_shape):
# #     model = Sequential([
# #         Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape),
# #         MaxPooling1D(pool_size=2),
# #         Dropout(0.3),
# #         Flatten(),
# #         Dense(32, activation="relu"),
# #         Dropout(0.3),
# #         Dense(1, activation="sigmoid")
# #     ])

# #     model.compile(
# #         optimizer=Adam(learning_rate=0.001),
# #         loss="binary_crossentropy",
# #         metrics=["accuracy"]
# #     )
# #     return model

# # # ===================== LSTM =====================
# # results["LSTM"] = evaluate_dl_model(
# #     build_lstm, X_seq, y_seq, groups_seq, "LSTM"
# # )

# # # ===================== CNN =====================
# # results["CNN"] = evaluate_dl_model(
# #     build_cnn, X_seq, y_seq, groups_seq, "CNN (1D)"
# # )

# # # ===================== FINAL COMPARISON PLOT =====================
# # print("\n===== FINAL MODEL COMPARISON =====")
# # print("Model      | Acc(mean±std) | F1   | MCC  | ROC-AUC")
# # print("-" * 55)

# # for model, (acc_m, acc_s, f1, mcc, roc) in results.items():
# #     print(f"{model:10s} | {acc_m:.3f}±{acc_s:.3f} | {f1:.3f} | {mcc:.3f} | {roc:.3f}")

# # models = list(results.keys())
# # f1_vals  = [results[m][2] for m in models]
# # mcc_vals = [results[m][3] for m in models]
# # roc_vals = [results[m][4] for m in models]

# # # -------- F1-SCORE PLOT --------
# # plt.figure(figsize=(10, 5))
# # plt.bar(models, f1_vals)
# # plt.ylabel("F1-score (LOSO)")
# # plt.title("F1-score Comparison Across Models")
# # plt.ylim(0, 1)
# # plt.grid(axis="y")
# # #plt.show()
# # plt.savefig("./imagenes/f1_score_comparison.png")

# # print(
# #     "\nF1-score Interpretation:\n"
# #     "- F1-score balances Precision and Recall.\n"
# #     "- Higher F1 indicates better stress vs non-stress classification,\n"
# #     "  especially important for imbalanced datasets.\n"
# #     "- Models with low F1 but high accuracy may be biased toward the majority class."
# # )

# # # -------- MCC PLOT --------
# # plt.figure(figsize=(10, 5))
# # plt.bar(models, mcc_vals)
# # plt.ylabel("MCC (LOSO)")
# # plt.title("Matthews Correlation Coefficient (MCC) Comparison")
# # plt.ylim(-1, 1)
# # plt.grid(axis="y")
# # #plt.show()
# # plt.savefig("./imagenes/mcc_comparison.png")
# # print(
# #     "\nMCC Interpretation:\n"
# #     "- MCC measures overall classification quality using all confusion matrix terms.\n"
# #     "- Range: -1 (total disagreement) to +1 (perfect prediction).\n"
# #     "- MCC is robust to class imbalance and is one of the most reliable single metrics.\n"
# #     "- Higher MCC indicates better generalization across subjects."
# # )

# # # -------- ROC-AUC PLOT --------
# # plt.figure(figsize=(10, 5))
# # plt.bar(models, roc_vals)
# # plt.ylabel("ROC-AUC (LOSO)")
# # plt.title("ROC-AUC Comparison Across Models")
# # plt.ylim(0.5, 1.0)
# # plt.grid(axis="y")
# # #plt.show()
# # plt.savefig("./imagenes/roc_auc_comparison.png")
# # print(
# #     "\nROC-AUC Interpretation:\n"
# #     "- ROC-AUC measures how well a model separates stress vs non-stress classes.\n"
# #     "- It is threshold-independent and evaluates ranking quality.\n"
# #     "- High ROC-AUC with low F1/MCC may indicate good probability estimates\n"
# #     "  but suboptimal decision threshold."
# # )

# # # ===================== TOP-3 MODEL SELECTION =====================

# # # Convert results to DataFrame
# # results_df = pd.DataFrame.from_dict(
# #     results,
# #     orient="index",
# #     columns=["Acc_mean", "Acc_std", "F1", "MCC", "ROC_AUC"]
# # )

# # # Composite score (weighted)
# # results_df["Composite_Score"] = (
# #     0.4 * results_df["MCC"] +
# #     0.4 * results_df["ROC_AUC"] +
# #     0.2 * results_df["F1"]
# # )

# # # Sort models by composite score
# # top3_models = results_df.sort_values(
# #     by="Composite_Score",
# #     ascending=False
# # ).head(3)

# # print("\n===== TOP 3 BEST MODELS =====")
# # print(top3_models[["F1", "MCC", "ROC_AUC", "Composite_Score"]])
# # print(
# #     "\nInterpretation:\n"
# #     "- MCC captures balanced prediction quality even under class imbalance.\n"
# #     "- ROC-AUC measures the model’s ability to separate stress vs non-stress across thresholds.\n"
# #     "- F1-score reflects the trade-off between precision and recall.\n\n"
# #     "The composite score prioritizes robustness (MCC), discriminative power (ROC-AUC), "
# #     "and classification balance (F1). Models ranking in the top three show consistent "
# #     "performance across subjects, making them reliable for real-world, subject-independent "
# #     "stress detection."
# # )