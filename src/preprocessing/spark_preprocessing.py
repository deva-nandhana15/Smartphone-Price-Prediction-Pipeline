from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, regexp_extract, lower, trim, 
    explode, split, avg, stddev, lit
)
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import logging
import re
import sys
sys.path.append('/opt/config')
from ebay_config import HDFS_CONFIG, SPARK_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartphoneDataPreprocessor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName(SPARK_CONFIG['app_name']) \
            .master(SPARK_CONFIG['master']) \
            .config("spark.executor.memory", SPARK_CONFIG['executor_memory']) \
            .config("spark.executor.cores", SPARK_CONFIG['executor_cores']) \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark session created")
    
    def load_raw_data(self, input_path: str):
        """Load raw JSON data from HDFS"""
        logger.info(f"Loading data from {input_path}")
        # Load JSON with multiLine option to handle array of objects
        df = self.spark.read.option("multiLine", "true").json(f"hdfs://namenode:9000{input_path}")
        logger.info(f"Loaded {df.count()} records")
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate items based on item_id"""
        initial_count = df.count()
        df = df.dropDuplicates(['item_id'])
        final_count = df.count()
        logger.info(f"Removed {initial_count - final_count} duplicates")
        return df
    
    def filter_smartphones(self, df):
        """Filter to keep only smartphones (exclude accessories)"""
        # Keywords that indicate accessories
        accessory_keywords = [
            'case', 'cover', 'charger', 'cable', 'screen protector',
            'tempered glass', 'holder', 'mount', 'stand', 'adapter',
            'battery', 'headphone', 'earphone', 'stylus', 'pen'
        ]
        
        # Create regex pattern
        pattern = '|'.join(accessory_keywords)
        
        # Filter out accessories
        df = df.filter(~lower(col('title')).rlike(pattern))
        
        # Keep items with phone-related keywords
        phone_keywords = [
            'iphone', 'samsung', 'galaxy', 'pixel', 'oneplus',
            'xiaomi', 'huawei', 'motorola', 'lg', 'sony', 'nokia',
            'smartphone', 'mobile phone'
        ]
        phone_pattern = '|'.join(phone_keywords)
        df = df.filter(lower(col('title')).rlike(phone_pattern))
        
        logger.info(f"Filtered to {df.count()} smartphones")
        return df
    
    def extract_brand(self, df):
        """Extract brand from title"""
        brands = {
            'Apple': 'iphone',
            'Samsung': 'samsung',
            'Google': 'pixel',
            'OnePlus': 'oneplus',
            'Xiaomi': 'xiaomi|redmi|poco',
            'Huawei': 'huawei|honor',
            'Motorola': 'motorola|moto',
            'LG': 'lg ',
            'Sony': 'sony|xperia',
            'Nokia': 'nokia'
        }
        
        brand_col = when(col('title').isNull(), 'Unknown')
        for brand, pattern in brands.items():
            brand_col = brand_col.when(
                lower(col('title')).rlike(pattern), brand
            )
        brand_col = brand_col.otherwise('Other')
        
        df = df.withColumn('brand', brand_col)
        return df
    
    def extract_storage(self, df):
        """Extract storage capacity in GB"""
        # Pattern: 64GB, 128GB, etc.
        df = df.withColumn(
            'storage_gb',
            regexp_extract(col('title'), r'(\d+)\s*GB', 1).cast(IntegerType())
        )
        
        # Handle TB
        df = df.withColumn(
            'storage_tb',
            regexp_extract(col('title'), r'(\d+)\s*TB', 1).cast(IntegerType())
        )
        
        df = df.withColumn(
            'storage_gb',
            when(col('storage_tb').isNotNull(), col('storage_tb') * 1024)
            .otherwise(col('storage_gb'))
        ).drop('storage_tb')
        
        return df
    
    def extract_ram(self, df):
        """Extract RAM in GB"""
        # Pattern: 4GB RAM, 6GB, etc.
        df = df.withColumn(
            'ram_gb',
            regexp_extract(col('title'), r'(\d+)\s*GB\s*RAM', 1).cast(IntegerType())
        )
        
        # Alternative pattern: RAM 4GB
        df = df.withColumn(
            'ram_gb',
            when(col('ram_gb').isNull(),
                 regexp_extract(col('title'), r'RAM\s*(\d+)\s*GB', 1).cast(IntegerType()))
            .otherwise(col('ram_gb'))
        )
        
        return df
    
    def extract_camera(self, df):
        """Extract camera megapixels"""
        # Pattern: 12MP, 48MP camera, etc.
        df = df.withColumn(
            'camera_mp',
            regexp_extract(col('title'), r'(\d+)\s*MP', 1).cast(IntegerType())
        )
        
        return df
    
    def extract_screen_size(self, df):
        """Extract screen size in inches"""
        # Pattern: 6.1", 6.7 inch, etc.
        df = df.withColumn(
            'screen_inches',
            regexp_extract(col('title'), r'(\d+\.?\d*)\s*["\']|(\d+\.?\d*)\s*inch', 1)
            .cast(DoubleType())
        )
        
        return df
    
    def clean_price_data(self, df):
        """Clean and filter price data"""
        # Remove null prices
        df = df.filter(col('price').isNotNull())
        
        # Remove outliers
        df = df.filter((col('price') >= 50) & (col('price') <= 5000))
        
        logger.info(f"After price filtering: {df.count()} records")
        return df
    
    def handle_missing_values(self, df):
        """Impute missing values with medians"""
        numeric_cols = ['storage_gb', 'ram_gb', 'camera_mp', 'screen_inches']
        
        for col_name in numeric_cols:
            # Calculate median
            median_value = df.approxQuantile(col_name, [0.5], 0.01)[0]
            
            # Fill nulls with median
            df = df.withColumn(
                col_name,
                when(col(col_name).isNull(), lit(median_value))
                .otherwise(col(col_name))
            )
            
            logger.info(f"Imputed {col_name} with median: {median_value}")
        
        return df
    
    def create_derived_features(self, df):
        """Create derived features"""
        # Price per GB storage
        df = df.withColumn(
            'price_per_gb',
            when(col('storage_gb') > 0, col('price') / col('storage_gb'))
            .otherwise(0)
        )
        
        # Camera quality score (normalized)
        df = df.withColumn(
            'camera_score',
            when(col('camera_mp') > 0, col('camera_mp') / 100.0)
            .otherwise(0)
        )
        
        # Condition score
        condition_score = when(col('condition').like('%New%'), 1.0) \
            .when(col('condition').like('%Refurbished%'), 0.7) \
            .when(col('condition').like('%Used%'), 0.5) \
            .otherwise(0.3)
        
        df = df.withColumn('condition_score', condition_score)
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        categorical_cols = ['brand', 'condition']
        
        indexers = [
            StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index", 
                         handleInvalid='keep')
            for col_name in categorical_cols
        ]
        
        encoders = [
            OneHotEncoder(inputCol=f"{col_name}_index", 
                         outputCol=f"{col_name}_encoded")
            for col_name in categorical_cols
        ]
        
        pipeline = Pipeline(stages=indexers + encoders)
        model = pipeline.fit(df)
        df = model.transform(df)
        
        return df
    
    def create_feature_vector(self, df):
        """Create feature vector for modeling"""
        numeric_features = [
            'storage_gb', 'ram_gb', 'camera_mp', 'screen_inches',
            'price_per_gb', 'camera_score', 'condition_score'
        ]
        
        categorical_features = ['brand_encoded', 'condition_encoded']
        
        assembler = VectorAssembler(
            inputCols=numeric_features + categorical_features,
            outputCol='features_raw'
        )
        
        df = assembler.transform(df)
        
        # Normalize features
        scaler = StandardScaler(
            inputCol='features_raw',
            outputCol='features',
            withMean=True,
            withStd=True
        )
        
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)
        
        return df
    
    def split_data(self, df, train_ratio=0.8):
        """Split data into train and test sets"""
        train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=42)
        
        logger.info(f"Train set: {train_df.count()} records")
        logger.info(f"Test set: {test_df.count()} records")
        
        return train_df, test_df
    
    def save_processed_data(self, train_df, test_df):
        """Save processed data to HDFS as Parquet"""
        train_path = f"hdfs://namenode:9000{HDFS_CONFIG['processed_data_path']}/train"
        test_path = f"hdfs://namenode:9000{HDFS_CONFIG['processed_data_path']}/test"
        
        train_df.write.mode('overwrite').parquet(train_path)
        test_df.write.mode('overwrite').parquet(test_path)
        
        logger.info(f"Train data saved to {train_path}")
        logger.info(f"Test data saved to {test_path}")
    
    def run_pipeline(self, input_path: str):
        """Run the complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        # Load data
        df = self.load_raw_data(input_path)
        
        # Cleaning
        df = self.remove_duplicates(df)
        df = self.filter_smartphones(df)
        df = self.clean_price_data(df)
        
        # Feature extraction
        df = self.extract_brand(df)
        df = self.extract_storage(df)
        df = self.extract_ram(df)
        df = self.extract_camera(df)
        df = self.extract_screen_size(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.create_derived_features(df)
        df = self.encode_categorical_features(df)
        df = self.create_feature_vector(df)
        
        # Select final columns
        final_df = df.select(
            'item_id', 'title', 'price', 'brand', 'condition',
            'storage_gb', 'ram_gb', 'camera_mp', 'screen_inches',
            'features', 'price_per_gb', 'camera_score', 'condition_score'
        )
        
        # Split and save
        train_df, test_df = self.split_data(final_df)
        self.save_processed_data(train_df, test_df)
        
        logger.info("Preprocessing pipeline completed successfully!")
        
        return train_df, test_df


def main():
    preprocessor = SmartphoneDataPreprocessor()
    preprocessor.run_pipeline(f"{HDFS_CONFIG['raw_data_path']}/*.json")


if __name__ == '__main__':
    main()
