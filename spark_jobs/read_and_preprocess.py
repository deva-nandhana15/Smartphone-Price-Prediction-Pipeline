"""
Spark job: Read mobile phone JSON documents from MongoDB Atlas, preprocess,
extract features, save preprocessed Parquet and metadata (JSON + scaler.joblib).

Run example (local dev):
spark-submit \
  --master local[*] \
  --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
  read_and_preprocess.py \
  --mongo-uri "$MONGO_URI" \
  --mongo-db "$MONGO_DB" \
  --mongo-collection "$MONGO_COLLECTION" \
  --output-parquet ../preprocessed/phones.parquet \
  --metadata ../preprocess/metadata.json \
  --scaler-path ../preprocess/scaler.joblib
"""

import argparse
import json
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, coalesce
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, IntegerType
import joblib
import numpy as np
import pandas as pd

def build_spark(mongo_uri, app_name="mob-preprocess"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.mongodb.input.uri", mongo_uri) \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    return spark

def normalize_booleans(df, colnames):
    for c in colnames:
        df = df.withColumn(c,
            when(col(c).isin('TRUE','True','true', True, 1, '1'), 1).otherwise(0)
        )
    return df

def cast_numerics(df, numeric_cols):
    for c in numeric_cols:
        df = df.withColumn(c, col(c).cast(DoubleType()))
    return df

def fill_numeric_median(df, numeric_cols):
    # compute medians locally (small number of columns) using approxQuantile
    medians = {}
    for c in numeric_cols:
        try:
            med = df.approxQuantile(c, [0.5], 0.01)[0]
            if med is None:
                med = 0.0
        except Exception:
            med = 0.0
        medians[c] = med
    df = df.fillna(medians)
    return df, medians

def generate_derived(df):
    # pixels and total_camera_count
    df = df.withColumn('pixels', (coalesce(col('resolution_width'), col('resolution_height')) * coalesce(col('resolution_height'), col('resolution_width'))).cast(DoubleType()))
    df = df.withColumn('total_cameras', (coalesce(col('num_rear_cameras'), col('num_front_cameras')) + coalesce(col('num_front_cameras'), col('num_rear_cameras'))).cast(DoubleType()))
    return df

def select_and_order(df, numeric_cols, categorical_cols, bool_cols, target_col='price'):
    # ensure columns exist; add missing as nulls
    cols = []
    for c in ([target_col] + numeric_cols + list(categorical_cols) + bool_cols):
        if c in df.columns:
            cols.append(c)
        else:
            df = df.withColumn(c, col(c))  # will be null
            cols.append(c)
    return df.select(*cols)

def compute_top_k_mapping(df, col, k=30):
    # return mapping dict: value->idx, with '__OTHER__' for rare
    counts = df.groupBy(col).count().orderBy('count', ascending=False).limit(k).collect()
    mapping = {}
    idx = 0
    for r in counts:
        val = r[0]
        if val is None:
            continue
        mapping[val] = idx
        idx += 1
    mapping['__OTHER__'] = idx
    return mapping

def map_categorical_series(series, mapping):
    return [mapping.get(x, mapping['__OTHER__']) for x in series]

def main(args):
    spark = build_spark(args.mongo_uri)
    print("Reading from MongoDB:", args.mongo_uri)
    df = spark.read.format("mongodb").option("uri", args.mongo_uri).load()

    # columns expected (from sample)
    numeric_cols = [
        "rating", "num_cores", "processor_speed", "battery_capacity",
        "fast_charging", "fast_charging_available", "ram_capacity",
        "internal_memory", "screen_size", "refresh_rate",
        "num_rear_cameras", "num_front_cameras",
        "primary_camera_rear", "primary_camera_front", "pixels"
    ]
    categorical_cols = ["brand_name", "processor_brand", "os"]
    bool_cols = ["has_5g", "has_nfc", "has_ir_blaster", "extended_memory_available"]

    # Normalize booleans
    df = normalize_booleans(df, bool_cols)

    # Cast numerics
    df = cast_numerics(df, numeric_cols)

    # Derived
    df = generate_derived(df)

    # Fill numeric missing with median
    df, medians = fill_numeric_median(df, numeric_cols + ['pixels'])

    # Compute top-k mapping for categorical columns
    cat_mappings = {}
    for c in categorical_cols:
        mapping = compute_top_k_mapping(df, c, k=50)
        cat_mappings[c] = mapping

    # Convert Spark DataFrame to Pandas (careful: do this if dataset fits memory)
    # For large data, write parquet partitioned and process in TF.
    pandas_df = df.toPandas()
    # apply categorical mapping (map to ints)
    for c in categorical_cols:
        mapping = cat_mappings[c]
        pandas_df[c] = pandas_df[c].map(lambda x: mapping.get(x, mapping.get('__OTHER__')))

    # booleans already 0/1 from spark; ensure numeric types
    for c in numeric_cols:
        if c in pandas_df.columns:
            pandas_df[c] = pandas_df[c].astype(float).fillna(medians.get(c, 0.0))
    for c in bool_cols:
        if c in pandas_df.columns:
            pandas_df[c] = pandas_df[c].astype(int).fillna(0)

    # target
    if 'price' not in pandas_df.columns:
        raise RuntimeError("No 'price' column found in documents")

    # select final feature order
    feature_cols = []
    # keep numerical features then categorical then bools
    numeric_keep = [c for c in numeric_cols if c in pandas_df.columns]
    feature_cols.extend(numeric_keep)
    feature_cols.extend([c for c in categorical_cols if c in pandas_df.columns])
    feature_cols.extend([c for c in bool_cols if c in pandas_df.columns])

    X = pandas_df[feature_cols].fillna(0.0)
    y = pandas_df['price'].astype(float).fillna(0.0)

    # Save Parquet for TF training
    out_parquet = args.output_parquet
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df_out = pandas_df[[*feature_cols, 'price']]
    df_out.to_parquet(out_parquet, index=False)
    print("Saved preprocessed parquet to:", out_parquet)

    # Save metadata (feature order, mappings, medians)
    metadata = {
        "feature_columns": feature_cols,
        "categorical_mappings": cat_mappings,
        "numeric_medians": medians,
        "bool_columns": bool_cols,
    }
    os.makedirs(os.path.dirname(args.metadata), exist_ok=True)
    with open(args.metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to:", args.metadata)

    # Fit scaler on numeric columns and save via joblib
    # Use StandardScaler from scikit-learn in joblib. If not installed, use simple mean-std
    try:
        from sklearn.preprocessing import StandardScaler
        numeric_indices = [i for i, c in enumerate(feature_cols) if c in numeric_keep]
        scaler = StandardScaler()
        scaler.fit(X.values)
        joblib.dump(scaler, args.scaler_path)
        print("Saved scaler to:", args.scaler_path)
    except Exception as ex:
        print("skipping scaler save (scikit-learn not available):", ex)

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo-uri", required=True)
    parser.add_argument("--mongo-db", default=None)
    parser.add_argument("--mongo-collection", default=None)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--scaler-path", required=True)
    args = parser.parse_args()
    main(args)
