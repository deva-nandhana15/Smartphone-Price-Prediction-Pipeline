#!/bin/bash

set -e

echo "=== Smartphone Price Prediction Pipeline ==="
echo ""

# Step 1: Start Docker containers
echo "[1/6] Starting Docker containers..."
docker-compose up -d
sleep 30  # Wait for services to be ready

# Step 2: Check HDFS health
echo "[2/6] Checking HDFS health..."
docker exec namenode hdfs dfsadmin -report

# Step 3: Create HDFS directories
echo "[3/6] Creating HDFS directories..."
docker exec namenode hdfs dfs -mkdir -p /data/raw
docker exec namenode hdfs dfs -mkdir -p /data/processed
docker exec namenode hdfs dfs -mkdir -p /models
docker exec namenode hdfs dfs -mkdir -p /logs

# Step 4: Run data ingestion
echo "[4/6] Running data ingestion..."
docker exec spark-master python3 /opt/pipeline/ingestion/ebay_ingestion.py

# Step 5: Run PySpark preprocessing
echo "[5/6] Running PySpark preprocessing..."
docker exec spark-master spark-submit \
    --master spark://spark-master:7077 \
    --executor-memory 2g \
    --executor-cores 2 \
    /opt/pipeline/preprocessing/spark_preprocessing.py

# Step 6: Run MapReduce training
echo "[6/6] Running MapReduce model training..."
./scripts/train_model.sh

echo ""
echo "=== Pipeline completed successfully! ==="
