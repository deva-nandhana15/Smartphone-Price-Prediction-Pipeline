@echo off
setlocal enabledelayedexpansion

set ITERATIONS=50
set HDFS_TRAIN_DATA=/data/processed/train
set HDFS_MODEL_PATH=/models

echo Training Linear Regression model using MapReduce...

REM Initialize weights to zero
docker exec namenode bash -c "echo '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0' > /tmp/model_weights.txt"
docker exec namenode hdfs dfs -put -f /tmp/model_weights.txt %HDFS_MODEL_PATH%/model_weights.txt

for /l %%i in (1,1,%ITERATIONS%) do (
    echo Iteration %%i/%ITERATIONS%
    
    REM Run MapReduce training job
    docker exec resourcemanager hadoop jar /opt/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar -files hdfs://namenode:9000%HDFS_MODEL_PATH%/model_weights.txt -mapper "python3 /opt/pipeline/mapreduce/mapper_train.py" -reducer "python3 /opt/pipeline/mapreduce/reducer_train.py" -input %HDFS_TRAIN_DATA% -output /tmp/model_updates_iter_%%i
    
    REM Update weights
    docker exec namenode bash -c "hdfs dfs -cat /tmp/model_updates_iter_%%i/part-* | python3 /opt/pipeline/mapreduce/update_weights.py > /tmp/new_weights.txt"
    docker exec namenode hdfs dfs -put -f /tmp/new_weights.txt %HDFS_MODEL_PATH%/model_weights.txt
    
    REM Clean up
    docker exec namenode hdfs dfs -rm -r /tmp/model_updates_iter_%%i
)

echo Training complete! Model saved to %HDFS_MODEL_PATH%/model_weights.txt
