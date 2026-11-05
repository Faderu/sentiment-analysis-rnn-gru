"""
data_preprocessing.py
Modul untuk memuat dataset, membersihkan teks bahasa Indonesia,
serta melakukan tokenisasi dan vektorisasi sekuens teks.
"""

import os
import re
import string
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

LABEL_MAP = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def clean_text(text: str) -> str:
    """
    Membersihkan teks dari URL, mention, tanda baca, angka, dan whitespace berlebih.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Hapus mention Twitter / tag (@user / <USER_MENTION>)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    # Hapus angka
    text = re.sub(r"\d+", "", text)
    # Hapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalisasi spasi
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(data_dir: str):
    """
    Memuat dataset IndoNLU SmSA (train, valid, test) dari folder data/raw.
    """
    train_path = os.path.join(data_dir, "train_preprocess.tsv")
    valid_path = os.path.join(data_dir, "valid_preprocess.tsv")
    test_path = os.path.join(data_dir, "test_preprocess.tsv")

    df_train = pd.read_csv(train_path, sep="\t", header=None, names=["text", "label"])
    df_valid = pd.read_csv(valid_path, sep="\t", header=None, names=["text", "label"])
    df_test = pd.read_csv(test_path, sep="\t", header=None, names=["text", "label"])

    # Bersihkan teks
    df_train["cleaned_text"] = df_train["text"].apply(clean_text)
    df_valid["cleaned_text"] = df_valid["text"].apply(clean_text)
    df_test["cleaned_text"] = df_test["text"].apply(clean_text)

    # Encode label
    df_train["label_id"] = df_train["label"].map(LABEL_MAP)
    df_valid["label_id"] = df_valid["label"].map(LABEL_MAP)
    df_test["label_id"] = df_test["label"].map(LABEL_MAP)

    return df_train, df_valid, df_test


def create_vectorizer(train_texts, max_tokens: int = 10000, max_len: int = 80):
    """
    Membuat dan mengadaptasi Keras TextVectorization layer pada data latih.
    """
    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_len,
        standardize=None # pembersihan sudah dilakukan oleh clean_text
    )
    vectorizer.adapt(train_texts)
    return vectorizer
