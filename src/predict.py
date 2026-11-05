"""
predict.py
Skrip inferensi untuk menguji teks kustom / kalimat baru menggunakan model terlatih.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from data_preprocessing import clean_text, INV_LABEL_MAP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")


def load_inference_pipeline(model_type: str = "gru"):
    vocab_path = os.path.join(MODELS_DIR, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vectorizer = layers.TextVectorization(
        vocabulary=vocab,
        output_mode="int",
        output_sequence_length=80,
        standardize=None
    )

    if model_type.lower() == "rnn":
        model_path = os.path.join(MODELS_DIR, "simple_rnn_best.keras")
    else:
        model_path = os.path.join(MODELS_DIR, "gru_best.keras")

    model = keras.models.load_model(model_path)
    return vectorizer, model


def predict_sentiment(text: str, vectorizer, model):
    cleaned = clean_text(text)
    vec = vectorizer([cleaned]).numpy()
    probs = model.predict(vec, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = INV_LABEL_MAP[pred_idx]

    return {
        "text": text,
        "cleaned_text": cleaned,
        "sentiment": pred_label,
        "probabilities": {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2])
        }
    }


if __name__ == "__main__":
    sample_texts = [
        "Tempat makan ini sangat nyaman, makanannya luar biasa lezat dan pelayanannya ramah!",
        "Pelayanan lambat sekali, pesanan dingin dan kasirnya tidak sopan sama sekali. Kecewa parah.",
        "Kemarin saya memesan paket nasi goreng dan es teh manis di cabang jalan riau."
    ]

    print("=== Demo Inferensi Sentimen (Menggunakan Model GRU) ===")
    vec, model = load_inference_pipeline("gru")

    for text in sample_texts:
        res = predict_sentiment(text, vec, model)
        print(f"\nTeks Asli : {res['text']}")
        print(f"Prediksi  : {res['sentiment'].upper()} (P={res['probabilities'][res['sentiment']]:.4f})")
        print(f"Detail P  : Negatif={res['probabilities']['negative']:.3f}, Netral={res['probabilities']['neutral']:.3f}, Positif={res['probabilities']['positive']:.3f}")
