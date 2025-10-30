#!/usr/bin/env python3
"""
Simple Linear Regression Training using Python (simulating MapReduce)
This runs the gradient descent algorithm on the prepared data.
"""
from pyspark.sql import SparkSession
import numpy as np
import sys

def train_model(spark, train_path, iterations=50, learning_rate=0.01):
    """Train linear regression model using gradient descent"""
    
    print(f"Loading training data from {train_path}")
    
    # Load data
    df = spark.read.csv(train_path, sep='\t', inferSchema=True)
    df = df.toDF('item_id', 'features_str', 'price')
    
    # Convert to RDD for processing
    data_rdd = df.rdd.map(lambda row: (
        np.array([float(x) for x in row.features_str.split(',')]),
        float(row.price)
    ))
    
    # Collect a sample to get feature count
    sample = data_rdd.first()
    n_features = len(sample[0]) + 1  # +1 for bias
    
    print(f"Number of features: {n_features - 1} + bias = {n_features}")
    print(f"Training samples: {data_rdd.count()}")
    
    # Initialize weights
    weights = np.zeros(n_features)
    
    # Training loop
    for iteration in range(iterations):
        # Map phase: compute gradients
        gradients_rdd = data_rdd.map(lambda x: compute_gradient(x, weights, learning_rate))
        
        # Reduce phase: aggregate gradients
        total_gradient, total_error, count = gradients_rdd.reduce(
            lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])
        )
        
        # Update weights
        avg_gradient = total_gradient / count
        weights = weights - (learning_rate * avg_gradient)
        
        # Calculate metrics
        rmse = np.sqrt(total_error / count)
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{iterations} - RMSE: {rmse:.2f}")
    
    print(f"\n✅ Training complete!")
    print(f"Final RMSE: {rmse:.2f}")
    
    return weights

def compute_gradient(data, weights, learning_rate):
    """Compute gradient for a single data point"""
    features, actual_price = data
    
    # Add bias term
    features_with_bias = np.append([1.0], features)
    
    # Predict
    prediction = np.dot(weights, features_with_bias)
    
    # Compute error
    error = prediction - actual_price
    
    # Compute gradient
    gradient = error * features_with_bias
    
    return (gradient, error ** 2, 1)

def evaluate_model(spark, test_path, weights):
    """Evaluate model on test data"""
    
    print(f"\nEvaluating on test data from {test_path}")
    
    # Load test data
    df = spark.read.csv(test_path, sep='\t', inferSchema=True)
    df = df.toDF('item_id', 'features_str', 'price')
    
    # Convert to RDD
    data_rdd = df.rdd.map(lambda row: (
        np.array([float(x) for x in row.features_str.split(',')]),
        float(row.price)
    ))
    
    # Make predictions
    predictions_rdd = data_rdd.map(lambda x: predict_and_evaluate(x, weights))
    
    # Calculate metrics
    total_se, total_ae, count = predictions_rdd.reduce(
        lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])
    )
    
    rmse = np.sqrt(total_se / count)
    mae = total_ae / count
    
    # Calculate R²
    predictions_and_actuals = predictions_rdd.map(lambda x: (x[3], x[4])).collect()
    predictions = [p[0] for p in predictions_and_actuals]
    actuals = [p[1] for p in predictions_and_actuals]
    
    mean_actual = np.mean(actuals)
    ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
    ss_res = sum((a - p) ** 2 for a, p in zip(actuals, predictions))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\n=== Model Evaluation Metrics ===")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Total Test Samples: {count}")
    
    return rmse, mae, r2

def predict_and_evaluate(data, weights):
    """Make prediction and compute error"""
    features, actual_price = data
    
    # Add bias term
    features_with_bias = np.append([1.0], features)
    
    # Predict
    prediction = np.dot(weights, features_with_bias)
    
    # Compute errors
    squared_error = (prediction - actual_price) ** 2
    absolute_error = abs(prediction - actual_price)
    
    return (squared_error, absolute_error, 1, prediction, actual_price)

def save_model(weights, output_path):
    """Save model weights to HDFS"""
    import json
    from hdfs import InsecureClient
    
    hdfs_client = InsecureClient('http://namenode:9870', user='spark')
    
    # Create models directory
    try:
        hdfs_client.makedirs('/models')
    except:
        pass
    
    # Save weights
    weights_dict = {
        'weights': weights.tolist(),
        'n_features': len(weights),
        'timestamp': str(np.datetime64('now'))
    }
    
    hdfs_client.write(output_path, json.dumps(weights_dict, indent=2), encoding='utf-8', overwrite=True)
    print(f"\n✅ Model saved to {output_path}")

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName('MapReduce_Training') \
        .getOrCreate()
    
    train_path = 'hdfs://namenode:9000/data/mapreduce/train'
    test_path = 'hdfs://namenode:9000/data/mapreduce/test'
    model_output = '/models/linear_regression_weights.json'
    
    print("="*60)
    print("SMARTPHONE PRICE PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Train model
    weights = train_model(spark, train_path, iterations=50, learning_rate=0.01)
    
    # Evaluate model
    rmse, mae, r2 = evaluate_model(spark, test_path, weights)
    
    # Save model
    save_model(weights, model_output)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to HDFS: {model_output}")
    print(f"Final Test RMSE: ${rmse:.2f}")
    print(f"Final Test R²: {r2:.4f}")
    
    spark.stop()
