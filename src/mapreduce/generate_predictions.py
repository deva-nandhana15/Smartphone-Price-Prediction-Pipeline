#!/usr/bin/env python3
"""
Generate predictions and create analysis report using the trained model
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, trim
import json
import sys

def create_spark_session():
    """Create Spark session"""
    return SparkSession.builder \
        .appName("Generate_Predictions") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .getOrCreate()

def load_model(spark, model_path):
    """Load trained model weights from HDFS"""
    try:
        from hdfs import InsecureClient
        hdfs_client = InsecureClient('http://namenode:9870', user='spark')
        
        with hdfs_client.read(model_path, encoding='utf-8') as reader:
            model_data = json.load(reader)
        
        weights = model_data['weights']
        print(f"✅ Model loaded successfully ({model_data['n_features']} features)")
        return weights
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def predict_price(features, weights):
    """Calculate predicted price using weights"""
    prediction = weights[0]  # intercept
    for i, feature in enumerate(features[1:], 1):
        prediction += weights[i] * feature
    return prediction

def generate_predictions(spark, test_path, weights, output_path):
    """Generate predictions for test data"""
    print(f"\nGenerating predictions from {test_path}")
    
    # Read test data
    test_df = spark.read.csv(test_path, sep='\t', header=False)
    test_df = test_df.filter(trim(col("_c0")) != "")
    
    # Parse TSV data
    parsed_df = test_df.select(
        split(col("_c0"), "\t").alias("fields")
    )
    
    test_data = parsed_df.collect()
    
    predictions = []
    actual_prices = []
    errors = []
    
    for row in test_data:
        fields = row['fields']
        if len(fields) < 2:
            continue
        
        try:
            # Extract features and actual price
            item_id = fields[0]
            features_str = fields[1:]
            
            # Convert features to float
            features = [1.0]  # bias term
            for f in features_str:
                try:
                    features.append(float(f))
                except:
                    features.append(0.0)
            
            # Get actual price (first feature after bias)
            actual_price = features[1]
            
            # Predict
            predicted_price = predict_price(features, weights)
            
            predictions.append({
                'item_id': item_id,
                'actual_price': actual_price,
                'predicted_price': predicted_price,
                'error': abs(actual_price - predicted_price),
                'error_percent': abs(actual_price - predicted_price) / actual_price * 100 if actual_price > 0 else 0
            })
            
            actual_prices.append(actual_price)
            errors.append(abs(actual_price - predicted_price))
            
        except Exception as e:
            continue
    
    print(f"✅ Generated {len(predictions)} predictions")
    
    # Calculate statistics
    rmse = (sum([e**2 for e in errors]) / len(errors)) ** 0.5
    mae = sum(errors) / len(errors)
    mape = sum([abs(p['actual_price'] - p['predicted_price']) / p['actual_price'] 
                for p in predictions if p['actual_price'] > 0]) / len(predictions) * 100
    
    print(f"\n{'='*60}")
    print("PREDICTION STATISTICS")
    print(f"{'='*60}")
    print(f"Total Predictions: {len(predictions)}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Save predictions to HDFS
    save_predictions(predictions, output_path)
    
    # Print sample predictions
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS (First 20)")
    print(f"{'='*60}")
    print(f"{'Item ID':<15} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'Error %':<10}")
    print("-" * 60)
    
    for pred in predictions[:20]:
        print(f"{pred['item_id']:<15} ${pred['actual_price']:<9.2f} ${pred['predicted_price']:<9.2f} ${pred['error']:<9.2f} {pred['error_percent']:<9.2f}%")
    
    # Analyze error distribution
    analyze_errors(predictions)
    
    return predictions

def save_predictions(predictions, output_path):
    """Save predictions to HDFS"""
    try:
        from hdfs import InsecureClient
        hdfs_client = InsecureClient('http://namenode:9870', user='spark')
        
        # Convert to JSON
        predictions_json = json.dumps(predictions, indent=2)
        
        # Save to HDFS
        hdfs_client.write(output_path, predictions_json, encoding='utf-8', overwrite=True)
        print(f"✅ Predictions saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")

def analyze_errors(predictions):
    """Analyze prediction error distribution"""
    print(f"\n{'='*60}")
    print("ERROR DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    # Group errors by range
    error_ranges = {
        '$0-50': 0,
        '$50-100': 0,
        '$100-200': 0,
        '$200-300': 0,
        '$300-500': 0,
        '$500+': 0
    }
    
    for pred in predictions:
        error = pred['error']
        if error < 50:
            error_ranges['$0-50'] += 1
        elif error < 100:
            error_ranges['$50-100'] += 1
        elif error < 200:
            error_ranges['$100-200'] += 1
        elif error < 300:
            error_ranges['$200-300'] += 1
        elif error < 500:
            error_ranges['$300-500'] += 1
        else:
            error_ranges['$500+'] += 1
    
    total = len(predictions)
    print(f"\n{'Error Range':<15} {'Count':<10} {'Percentage':<15}")
    print("-" * 40)
    for range_name, count in error_ranges.items():
        percentage = count / total * 100
        print(f"{range_name:<15} {count:<10} {percentage:>6.2f}%")
    
    # Find best and worst predictions
    sorted_preds = sorted(predictions, key=lambda x: x['error'])
    
    print(f"\n{'='*60}")
    print("BEST PREDICTIONS (Lowest Error)")
    print(f"{'='*60}")
    print(f"{'Item ID':<15} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-" * 50)
    for pred in sorted_preds[:5]:
        print(f"{pred['item_id']:<15} ${pred['actual_price']:<9.2f} ${pred['predicted_price']:<9.2f} ${pred['error']:<9.2f}")
    
    print(f"\n{'='*60}")
    print("WORST PREDICTIONS (Highest Error)")
    print(f"{'='*60}")
    print(f"{'Item ID':<15} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-" * 50)
    for pred in sorted_preds[-5:]:
        print(f"{pred['item_id']:<15} ${pred['actual_price']:<9.2f} ${pred['predicted_price']:<9.2f} ${pred['error']:<9.2f}")

def main():
    """Main execution"""
    print("="*60)
    print("SMARTPHONE PRICE PREDICTION - GENERATE PREDICTIONS")
    print("="*60)
    
    # Configuration
    model_path = "/models/linear_regression_weights.json"
    test_path = "hdfs://namenode:9000/data/mapreduce/test"
    output_path = "/data/predictions/test_predictions.json"
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Load model
        weights = load_model(spark, model_path)
        
        # Generate predictions
        predictions = generate_predictions(spark, test_path, weights, output_path)
        
        print(f"\n{'='*60}")
        print("✅ PREDICTION GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Predictions saved to HDFS: {output_path}")
        print(f"Total predictions: {len(predictions)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
