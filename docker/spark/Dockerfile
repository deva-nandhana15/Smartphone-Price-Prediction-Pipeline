FROM apache/spark:3.5.0

USER root

RUN pip install --no-cache-dir \
    requests \
    hdfs \
    numpy \
    pandas \
    python-dotenv

USER ${spark_uid}
