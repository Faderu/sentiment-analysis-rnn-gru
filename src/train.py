"""
train.py
Skrip utama untuk melatih model Simple RNN dan GRU pada dataset IndoNLU SmSA,
menyimpan model terbaik, serta mengevaluasi dan menyimpan hasil perbandingan.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from data_preprocessing import load_dataset, create_vectorizer, LABEL_MAP
from models import build_simple_rnn_model, build_gru_model, build_lstm_model
from evaluate import evaluate_model, plot_training_history, plot_confusion_matrices, plot_metrics_comparison

# Setup path direktori
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameter
VOCAB_SIZE = 8000
MAX_LEN = 80
EMBEDDING_DIM = 64
RECURRENT_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 8


def main():
    print("[1/5] Memuat dan memproses dataset...")
    df_train, df_valid, df_test = load_dataset(DATA_DIR)
    print(f"Data Latih: {len(df_train)} baris")
    print(f"Data Validasi: {len(df_valid)} baris")
    print(f"Data Uji: {len(df_test)} baris")

    print("[2/5] Membangun Text Vectorizer...")
    vectorizer = create_vectorizer(df_train["cleaned_text"].values, max_tokens=VOCAB_SIZE, max_len=MAX_LEN)

    X_train = vectorizer(df_train["cleaned_text"].values).numpy()
    y_train = df_train["label_id"].values

    X_val = vectorizer(df_valid["cleaned_text"].values).numpy()
    y_val = df_valid["label_id"].values

    X_test = vectorizer(df_test["cleaned_text"].values).numpy()
    y_test = df_test["label_id"].values

    # Simpan kosakata vectorizer untuk keperluan inferensi nanti
    vocab = vectorizer.get_vocabulary()
    vocab_path = os.path.join(MODELS_DIR, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Kosakata ({len(vocab)} kata) tersimpan di {vocab_path}")

    histories = {}
    eval_results = {}

    # ==========================================
    # 1. Melatih Simple RNN
    # ==========================================
    print("\n" + "="*50)
    print("[3/5] Melatih Model 1: Simple RNN...")
    print("="*50)
    rnn_model = build_simple_rnn_model(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_len=MAX_LEN,
        rnn_units=RECURRENT_UNITS,
        num_classes=3
    )
    rnn_model.summary()

    rnn_ckpt_path = os.path.join(MODELS_DIR, "simple_rnn_best.keras")
    callbacks_rnn = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(rnn_ckpt_path, monitor="val_accuracy", save_best_only=True)
    ]

    t0 = time.time()
    history_rnn = rnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_rnn,
        verbose=1
    )
    rnn_time = time.time() - t0
    print(f"Waktu pelatihan Simple RNN: {rnn_time:.2f} detik")

    histories["Simple RNN"] = {k: [float(v) for v in vals] for k, vals in history_rnn.history.items()}
    rnn_eval = evaluate_model(rnn_model, X_test, y_test, "Simple RNN", train_time_sec=rnn_time)
    eval_results["Simple RNN"] = rnn_eval

    # ==========================================
    # 2. Melatih GRU (Gated Recurrent Unit)
    # ==========================================
    print("\n" + "="*50)
    print("[4/5] Melatih Model 2: GRU (Gated Recurrent Unit)...")
    print("="*50)
    gru_model = build_gru_model(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_len=MAX_LEN,
        gru_units=RECURRENT_UNITS,
        num_classes=3
    )
    gru_model.summary()

    gru_ckpt_path = os.path.join(MODELS_DIR, "gru_best.keras")
    callbacks_gru = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(gru_ckpt_path, monitor="val_accuracy", save_best_only=True)
    ]

    t0 = time.time()
    history_gru = gru_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_gru,
        verbose=1
    )
    gru_time = time.time() - t0
    print(f"Waktu pelatihan GRU: {gru_time:.2f} detik")

    histories["GRU"] = {k: [float(v) for v in vals] for k, vals in history_gru.history.items()}
    gru_eval = evaluate_model(gru_model, X_test, y_test, "GRU", train_time_sec=gru_time)
    eval_results["GRU"] = gru_eval

    # ==========================================
    # 3. Visualisasi & Penyimpanan Metrik
    # ==========================================
    print("\n" + "="*50)
    print("[5/5] Membuat Visualisasi & Menyimpan Hasil...")
    print("="*50)

    # Plot grafik
    plot_training_history(histories, os.path.join(RESULTS_DIR, "training_curves.png"))
    plot_confusion_matrices(eval_results, os.path.join(RESULTS_DIR, "confusion_matrices.png"))
    plot_metrics_comparison(eval_results, os.path.join(RESULTS_DIR, "metrics_comparison.png"))

    # Simpan JSON metrik
    summary_export = {}
    for name, res in eval_results.items():
        summary_export[name] = {
            "accuracy": res["accuracy"],
            "macro_precision": res["macro_precision"],
            "macro_recall": res["macro_recall"],
            "macro_f1": res["macro_f1"],
            "train_time_sec": res["train_time_sec"]
        }

    with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=4)

    with open(os.path.join(RESULTS_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=4)

    # Simpan tabel CSV ringkasan
    df_summary = pd.DataFrame([
        {
            "Model": k,
            "Accuracy": f"{v['accuracy']*100:.2f}%",
            "Macro F1": f"{v['macro_f1']:.4f}",
            "Macro Precision": f"{v['macro_precision']:.4f}",
            "Macro Recall": f"{v['macro_recall']:.4f}",
            "Training Time (s)": f"{v['train_time_sec']:.2f}"
        }
        for k, v in summary_export.items()
    ])
    df_summary.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)
    print("\nTabel Perbandingan Akhir:")
    print(df_summary.to_string(index=False))

    print("\nPelatihan dan evaluasi selesai dengan sukses!")


if __name__ == "__main__":
    main()
