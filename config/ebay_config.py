import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

EBAY_CONFIG = {
    'app_id': os.getenv('EBAY_APP_ID'),
    'dev_id': os.getenv('EBAY_DEV_ID'),
    'client_secret': os.getenv('EBAY_CLIENT_SECRET'),
    'category_id': '15032',  # Cell Phones & Accessories
    'search_keywords': [
        'iPhone',
        'Samsung Galaxy',
        'Google Pixel',
        'OnePlus',
        'Xiaomi',
        'Huawei',
        'Motorola',
        'LG smartphone',
        'Sony Xperia'
    ],
    'items_per_page': 200,
    'max_pages': 200,
    'rate_limit_delay': 0.5,  # seconds between API calls
    'max_retries': 3
}

HDFS_CONFIG = {
    'namenode_host': 'namenode',
    'namenode_port': 9000,
    'raw_data_path': '/data/raw',
    'processed_data_path': '/data/processed',
    'model_path': '/models',
    'logs_path': '/logs'
}

SPARK_CONFIG = {
    'app_name': 'Smartphone_Price_Prediction',
    'master': 'spark://spark-master:7077',
    'executor_memory': '2g',
    'executor_cores': 2
}
