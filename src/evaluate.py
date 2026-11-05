"""
evaluate.py
Modul untuk mengevaluasi model pada test set, menghitung metrik evaluasi
(Accuracy, Precision, Recall, F1-Score), serta menghasilkan visualisasi publikasi.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

TARGET_NAMES = ["Negatif", "Netral", "Positif"]


def plot_training_history(histories: dict, save_path: str):
    """
    Plot kurva perbandingan loss dan akurasi training vs validation untuk semua model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"Simple RNN": "#e74c3c", "GRU": "#2980b9", "LSTM": "#27ae60"}

    # Plot Accuracy
    for name, hist in histories.items():
        epochs = range(1, len(hist["accuracy"]) + 1)
        c = colors.get(name, "#34495e")
        axes[0].plot(epochs, hist["accuracy"], label=f"{name} (Train)", color=c, linestyle="--", alpha=0.8)
        axes[0].plot(epochs, hist["val_accuracy"], label=f"{name} (Val)", color=c, linewidth=2)

    axes[0].set_title("Perbandingan Akurasi (Training vs Validation)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Akurasi")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot Loss
    for name, hist in histories.items():
        epochs = range(1, len(hist["loss"]) + 1)
        c = colors.get(name, "#34495e")
        axes[1].plot(epochs, hist["loss"], label=f"{name} (Train)", color=c, linestyle="--", alpha=0.8)
        axes[1].plot(epochs, hist["val_loss"], label=f"{name} (Val)", color=c, linewidth=2)

    axes[1].set_title("Perbandingan Loss (Training vs Validation)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Training history plot tersimpan di: {save_path}")


def plot_confusion_matrices(eval_results: dict, save_path: str):
    """
    Plot Confusion Matrix heatmap berdampingan untuk model yang dievaluasi.
    """
    n_models = len(eval_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, eval_results.items()):
        cm = res["confusion_matrix"]
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=TARGET_NAMES,
            yticklabels=TARGET_NAMES,
            ax=ax,
            cbar=False
        )
        ax.set_title(f"Confusion Matrix - {name}\nAkurasi: {res['accuracy'] * 100:.2f}%", fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix plot tersimpan di: {save_path}")


def plot_metrics_comparison(eval_results: dict, save_path: str):
    """
    Plot diagram batang perbandingan Accuracy, Macro F1, Macro Precision, dan Macro Recall.
    """
    records = []
    for name, res in eval_results.items():
        records.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Macro F1": res["macro_f1"],
            "Macro Precision": res["macro_precision"],
            "Macro Recall": res["macro_recall"],
            "Training Time (s)": res.get("train_time_sec", 0.0)
        })

    df = pd.DataFrame(records)

    # Plot metrics
    metrics = ["Accuracy", "Macro F1", "Macro Precision", "Macro Recall"]
    df_melt = df.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model", palette="Set2", ax=ax)
    ax.set_title("Perbandingan Kinerja Evaluasi Model pada Test Set", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Skor (0.0 - 1.0)")
    ax.legend(title="Arsitektur Model", loc="lower right")

    # Tambahkan angka di atas bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=9, color='black',
                        xytext=(0, 2), textcoords='offset points')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Metrics comparison plot tersimpan di: {save_path}")


def evaluate_model(model, X_test, y_test, model_name: str, train_time_sec: float = 0.0) -> dict:
    """
    Menghitung metrik lengkap untuk satu model.
    """
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True, zero_division=0)

    print(f"\n{'='*50}")
    print(f"Hasil Evaluasi Model: {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES, zero_division=0))

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "train_time_sec": float(train_time_sec)
    }
