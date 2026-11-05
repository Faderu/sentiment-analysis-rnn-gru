"""
models.py
Definisi arsitektur Deep Learning untuk Analisis Sentimen:
1. Simple RNN (Recurrent Neural Network)
2. GRU (Gated Recurrent Unit)
3. LSTM (Long Short-Term Memory) - Arsitektur Pembanding Tambahan
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers


def build_simple_rnn_model(
    vocab_size: int = 10000,
    embedding_dim: int = 64,
    max_len: int = 80,
    rnn_units: int = 64,
    num_classes: int = 3,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3
) -> keras.Model:
    """
    Membangun model berbasis Simple RNN standar.
    """
    model = keras.Sequential([
        layers.Input(shape=(max_len,), name="input_ids"),
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="embedding"
        ),
        layers.SpatialDropout1D(dropout_rate, name="spatial_dropout"),
        layers.SimpleRNN(
            units=rnn_units,
            dropout=dropout_rate,
            return_sequences=False,
            name="simple_rnn"
        ),
        layers.Dense(32, activation="relu", name="dense_hidden"),
        layers.Dropout(dropout_rate, name="dense_dropout"),
        layers.Dense(num_classes, activation="softmax", name="output_layer")
    ], name="Simple_RNN_Model")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_gru_model(
    vocab_size: int = 10000,
    embedding_dim: int = 64,
    max_len: int = 80,
    gru_units: int = 64,
    num_classes: int = 3,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3
) -> keras.Model:
    """
    Membangun model berbasis GRU (Gated Recurrent Unit).
    Memanfaatkan Reset Gate dan Update Gate untuk menangani vanishing gradient.
    """
    model = keras.Sequential([
        layers.Input(shape=(max_len,), name="input_ids"),
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="embedding"
        ),
        layers.SpatialDropout1D(dropout_rate, name="spatial_dropout"),
        layers.GRU(
            units=gru_units,
            dropout=dropout_rate,
            return_sequences=False,
            name="gru"
        ),
        layers.Dense(32, activation="relu", name="dense_hidden"),
        layers.Dropout(dropout_rate, name="dense_dropout"),
        layers.Dense(num_classes, activation="softmax", name="output_layer")
    ], name="GRU_Model")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_lstm_model(
    vocab_size: int = 10000,
    embedding_dim: int = 64,
    max_len: int = 80,
    lstm_units: int = 64,
    num_classes: int = 3,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3
) -> keras.Model:
    """
    Membangun model berbasis LSTM (Long Short-Term Memory) sebagai arsitektur pembanding.
    """
    model = keras.Sequential([
        layers.Input(shape=(max_len,), name="input_ids"),
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="embedding"
        ),
        layers.SpatialDropout1D(dropout_rate, name="spatial_dropout"),
        layers.LSTM(
            units=lstm_units,
            dropout=dropout_rate,
            return_sequences=False,
            name="lstm"
        ),
        layers.Dense(32, activation="relu", name="dense_hidden"),
        layers.Dropout(dropout_rate, name="dense_dropout"),
        layers.Dense(num_classes, activation="softmax", name="output_layer")
    ], name="LSTM_Model")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
