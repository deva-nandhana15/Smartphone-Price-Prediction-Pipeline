"""
Train TensorFlow regression model from preprocessed Parquet file.
Saves model to models/price_predictor/{version}/

Run (after spark preprocessing):
python train.py \
  --preprocessed-parquet ../preprocessed/phones.parquet \
  --metadata ../preprocess/metadata.json \
  --scaler ../preprocess/scaler.joblib \
  --output-dir ../models/price_predictor \
  --epochs 20 \
  --batch-size 64
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear', name="price")
    model = tf.keras.Sequential([inputs, x, outputs])
    # compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse', metrics=['mae', 'mse'])
    return model

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading preprocessed parquet:", args.preprocessed_parquet)
    df = pd.read_parquet(args.preprocessed_parquet)
    with open(args.metadata, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_columns"]

    X = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y = df['price'].astype(np.float32).values.reshape(-1, 1)

    # optionally load scaler
    if args.scaler and os.path.exists(args.scaler):
        scaler = joblib.load(args.scaler)
        try:
            X = scaler.transform(X)
            print("Applied loaded scaler")
        except Exception:
            print("Loaded scaler could not be applied; skipping.")
    else:
        # fit local scaler and save
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        if args.scaler:
            joblib.dump(scaler, args.scaler)
            print("Saved new scaler to:", args.scaler)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.12, random_state=42)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.output_dir, "ckpt-{epoch}"), save_weights_only=False, save_best_only=True, monitor='val_loss')
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Versioning: increment folder numbers
    version = 1
    existing = [d for d in os.listdir(args.output_dir) if d.isdigit()]
    if existing:
        version = max(map(int, existing)) + 1
    model_path = os.path.join(args.output_dir, str(version))
    model.save(model_path)
    print("Saved model to:", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-parquet", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--scaler", required=False, default="../preprocess/scaler.joblib")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    main(args)
