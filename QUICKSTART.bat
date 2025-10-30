@echo off
echo Quick Start Guide for Smartphone Price Prediction Pipeline
echo ==========================================================
echo.
echo Step 1: Set up eBay API credentials
echo ------------------------------------
echo Run: scripts\setup_env.bat
echo Or manually set:
echo   set EBAY_APP_ID=your_app_id
echo   set EBAY_CLIENT_SECRET=your_client_secret
echo.
echo Step 2: Start Docker containers
echo --------------------------------
echo Run: docker-compose up -d
echo.
echo Step 3: Wait for services to initialize (30 seconds)
echo.
echo Step 4: Run the full pipeline
echo ------------------------------
echo Run: scripts\run_pipeline.bat
echo.
echo Alternative: Run individual components
echo ---------------------------------------
echo Ingestion only:
echo   docker exec spark-master python3 /opt/pipeline/ingestion/ebay_ingestion.py
echo.
echo Preprocessing only:
echo   docker exec spark-master spark-submit /opt/pipeline/preprocessing/spark_preprocessing.py
echo.
echo Training only:
echo   scripts\train_model.bat
echo.
echo Monitoring URLs:
echo ----------------
echo HDFS NameNode:      http://localhost:9870
echo Yarn ResourceMgr:   http://localhost:8088
echo Spark UI:           http://localhost:4040
echo.
echo Stop all containers:
echo --------------------
echo Run: docker-compose down
echo.
pause
