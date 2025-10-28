# Mobile Price Prediction Pipeline

See repository for Spark preprocessing, TensorFlow training, and FastAPI inference.

## Quick local flow

1. Create virtualenvs or use conda.
2. Place Mongo URI into environment variable `MONGO_URI` or edit `config/config.yaml`.
3. Run spark preprocessing:
   - `cd spark_jobs`
   - `spark-submit --master local[*] --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 read_and_preprocess.py --mongo-uri "$MONGO_URI" --output-parquet ../preprocessed/phones.parquet --metadata ../preprocess/metadata.json --scaler-path ../preprocess/scaler.joblib`
4. Train model:
   - `cd tf_training`
   - `python train.py --preprocessed-parquet ../preprocessed/phones.parquet --metadata ../preprocess/metadata.json --scaler ../preprocess/scaler.joblib --output-dir ../models/price_predictor --epochs 20 --batch-size 64`
5. Run API:
   - `cd api`
   - `pip install -r requirements.txt`
   - `uvicorn app:app --reload --port 8000`
6. Test:
   - `POST http://localhost:8000/predict` with a JSON payload.
