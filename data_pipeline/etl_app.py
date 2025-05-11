import os
import logging
from flask import Flask, request
from prometheus_client import Counter, Gauge, generate_latest
import pandas as pd
import numpy as np
import requests
import sqlite3
import boto3
import threading

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = "etl-data"

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

# Prometheus metrics definition
# Data quality metrics
data_quality_missing_values = Gauge('data_quality_missing_values', 'Number of missing values', ['task', 'field'])
data_quality_duplicates = Gauge('data_quality_duplicates', 'Number of duplicate entries', ['field'])
data_quality_processing_errors = Counter('data_quality_processing_errors', 'Number of data processing errors', ['error_type'])

# Logging configuration
log_file = "/app/etl_data/app.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

app = Flask(__name__)


def check_and_clean_data(data, table_name):
    """Check data quality and record metrics, while removing rows with missing values"""
    try:
        task_map = {
            "task1_data": "text_summary",
            "task2_data": "fake_news_detection",
            "task3_data": "text_classification"
        }

        if table_name not in task_map:
            return None

        task_label = task_map[table_name]

        # Specify field lists
        if table_name == "task1_data":
            fields_to_check = ['label_input', 'label_label1']
        elif table_name == "task2_data":
            fields_to_check = ['label_input', 'label_label2']
        elif table_name == "task3_data":
            fields_to_check = ['label_input', 'label_label3']

        # Record missing values count (per column) and remove these rows
        for field in fields_to_check:
            if field in data.columns:
                missing_count = data[field].isna().sum()
                if missing_count > 0:
                    data_quality_missing_values.labels(task=task_label, field=field).inc(missing_count)

        # Remove rows with missing values (only considering specified fields)
        data.dropna(subset=fields_to_check, inplace=True)

        # Check for duplicate rows (full row duplicates)
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            data_quality_duplicates.labels(field=table_name).set(duplicates)
            # Remove duplicate rows
            data.drop_duplicates(inplace=True)
            logging.info(f"Removed {duplicates} duplicate rows from {table_name}")

        return data

    except Exception as e:
        data_quality_processing_errors.labels(error_type='quality_check_and_clean_error').inc()
        logging.error(f"Error checking data quality and clean for {table_name}: {str(e)}")
        return None


def save_to_volume(data, table_name):
    """Append processed data to shared volume CSV for rclone to upload"""
    try:
        # build save path
        save_dir = f"/app/etl_data/{table_name}"
        os.makedirs(save_dir, exist_ok=True)

        # fixed filename
        if table_name == "task1_data":  
            file_path = os.path.join(save_dir, "summary_train.csv")
        elif table_name == "task2_data":
            file_path = os.path.join(save_dir, "welfake_train.csv")
        elif table_name == "task3_data":
            file_path = os.path.join(save_dir, "classification_train.csv")

        # check if file exists
        write_header = not os.path.exists(file_path)

        if write_header:
            logging.info(f"{file_path} not found, creating new CSV file")

        # save as CSV (append mode)
        data.to_csv(file_path, mode='a', header=write_header, index=False, encoding='utf-8')

        logging.info(f"Appended data to {file_path}")
        return True

    except Exception as e:
        logging.error(f"Error appending to volume: {str(e)}")
        return False

def read_from_minio():
    """Read SQLite data from MinIO"""
    try:
        # Build object name
        object_name = "evaluation.db"
        temp_db = "/tmp/evaluation_temp.db"
        
        # check if MinIO bucket exists
        existing_buckets = s3_client.list_buckets()
        bucket_names = [b['Name'] for b in existing_buckets['Buckets']]
        if MINIO_BUCKET not in bucket_names:
            logging.error(f"MinIO bucket '{MINIO_BUCKET}' not found.")
            return None
        
        # check if evaluation.db file exists in the bucket
        objects = s3_client.list_objects_v2(Bucket=MINIO_BUCKET)
        if "Contents" not in objects or not any(obj["Key"] == object_name for obj in objects["Contents"]):
            logging.error(f"File '{object_name}' not found in bucket '{MINIO_BUCKET}'.")
            return None

        logging.info(f"Found '{object_name}' in bucket '{MINIO_BUCKET}', starting download...")

        # Download file from MinIO
        s3_client.download_file(MINIO_BUCKET, object_name, temp_db)
        
        # Read data from SQLite
        conn = sqlite3.connect(temp_db)
        
        # Get all table names
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Read all table data
        all_data = {}
        for table in tables: 
            table_name = table[0]
            query = f"SELECT * FROM {table_name}"
            data = pd.read_sql_query(query, conn)
            all_data[table_name] = data
            
        conn.close()
        
        # Delete temporary file
        os.remove(temp_db)
        
        return all_data
    except Exception as e:
        logging.error(f"Error reading from MinIO: {str(e)}")
        return None
    
def read_from_minio():
    """Read SQLite data from MinIO with bucket and file existence check"""
    object_name = "evaluation.db"
    temp_db = "/app/evaluation_temp.db"

    try:
        # check if MinIO bucket exists
        existing_buckets = s3_client.list_buckets()
        bucket_names = [b['Name'] for b in existing_buckets['Buckets']]
        if MINIO_BUCKET not in bucket_names:
            logging.error(f"MinIO bucket '{MINIO_BUCKET}' not found.")
            return None
        
        # check if evaluation.db file exists in the bucket
        objects = s3_client.list_objects_v2(Bucket=MINIO_BUCKET)
        if "Contents" not in objects or not any(obj["Key"] == object_name for obj in objects["Contents"]):
            logging.error(f"File '{object_name}' not found in bucket '{MINIO_BUCKET}'.")
            return None

        logging.info(f"Found '{object_name}' in bucket '{MINIO_BUCKET}', starting download...")

        # download file
        s3_client.download_file(MINIO_BUCKET, object_name, temp_db)

        # read SQLite file
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        all_data = {}
        for table in tables:
            table_name = table[0]
            query = f"SELECT * FROM {table_name}"
            data = pd.read_sql_query(query, conn)
            all_data[table_name] = data

        conn.close()
        os.remove(temp_db)
        logging.info("Successfully read and parsed evaluation.db from MinIO.")
        return all_data

    except Exception as e:
        logging.error(f"Error reading from MinIO: {str(e)}")
        if os.path.exists(temp_db):
            os.remove(temp_db)
        return None    

@app.route("/etl", methods=["POST"])
def trigger_etl():
    data = request.get_json(force=True)
    task = data.get("task")

    if not task:
        return {"error": "Missing task field"}, 400

    # Create thread for data processing
    def process_data():
        try:
            # Read data from MinIO
            evaluation_data = read_from_minio()
            if evaluation_data is None:
                logging.error("Failed to read evaluation data")
                return

            # Process data for each table
            for table_name, table_data in evaluation_data.items():
                # Process data
                processed_data = check_and_clean_data(table_data, table_name)
                if processed_data is None:
                    logging.error(f"Failed to process evaluation data for table {table_name}")
                    continue
                    
                # Save processed data to shared volume
                if not save_to_volume(processed_data, table_name):
                    logging.error(f"Failed to save data for table {table_name}")
                    continue

            # Trigger train service after ETL success
            max_retries = 5
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    train_response = requests.post(
                        "http://train:8000/train",
                        json={"task": "etl_process"},
                        timeout=5
                    )
                    if train_response.status_code == 200:
                        success = True
                        logging.info("Successfully triggered train service")
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            logging.warning(f"Failed to trigger train service (attempt {retry_count}/{max_retries}): {train_response.text}")
                        else:
                            logging.error(f"Failed to trigger train service after {max_retries} attempts: {train_response.text}")
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logging.warning(f"Error triggering train service (attempt {retry_count}/{max_retries}): {str(e)}")
                    else:
                        logging.error(f"Error triggering train service after {max_retries} attempts: {str(e)}")

        except Exception as e:
            logging.exception("ETL failed")

    # Start thread
    threading.Thread(target=process_data).start()

    return {"status": "ETL process started", "task": task}, 200

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose Prometheus metrics"""
    return generate_latest(), 200, {'Content-Type': 'text/plain'}    # text specifies return type as plain text
