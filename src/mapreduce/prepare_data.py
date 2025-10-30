#!/usr/bin/env python3
"""
Convert Parquet data to TSV format for MapReduce training
"""
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
import sys

def convert_to_tsv(spark, input_path, output_path):
    """Convert Parquet with feature vectors to TSV format"""
    print(f"Loading data from {input_path}")
    df = spark.read.parquet(input_path)
    
    # Select only what we need: item_id, features (vector), price
    df_selected = df.select('item_id', 'features', 'price')
    
    # Convert vector to string of comma-separated values
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType
    
    def vector_to_string(vector):
        if vector is not None:
            return ','.join([str(float(v)) for v in vector.toArray()])
        return ''
    
    vector_udf = udf(vector_to_string, StringType())
    
    df_tsv = df_selected.withColumn('features_str', vector_udf('features'))
    df_final = df_tsv.select('item_id', 'features_str', 'price')
    
    # Save as TSV
    print(f"Saving to {output_path}")
    df_final.write.mode('overwrite').option('sep', '\t').csv(output_path, header=False)
    
    print(f"Conversion complete!")
    return df_final.count()

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName('Prepare_MapReduce_Data') \
        .getOrCreate()
    
    train_input = 'hdfs://namenode:9000/data/processed/train'
    test_input = 'hdfs://namenode:9000/data/processed/test'
    
    train_output = 'hdfs://namenode:9000/data/mapreduce/train'
    test_output = 'hdfs://namenode:9000/data/mapreduce/test'
    
    print("Converting training data...")
    train_count = convert_to_tsv(spark, train_input, train_output)
    print(f"Training records: {train_count}")
    
    print("\nConverting test data...")
    test_count = convert_to_tsv(spark, test_input, test_output)
    print(f"Test records: {test_count}")
    
    print("\n✅ Data preparation complete!")
    spark.stop()
