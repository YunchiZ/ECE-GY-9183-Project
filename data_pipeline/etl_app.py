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

# MinIO 配置
MINIO_ENDPOINT = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = "etl-data"

# 初始化 S3 客户端
s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

# Prometheus 指标定义
# 数据质量指标
data_quality_missing_values = Gauge('data_quality_missing_values', 'Number of missing values', ['task', 'field'])
data_quality_duplicates = Gauge('data_quality_duplicates', 'Number of duplicate entries', ['field'])
data_quality_processing_errors = Counter('data_quality_processing_errors', 'Number of data processing errors', ['error_type'])

# 日志配置
log_file = "/app/etl_data/app.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

app = Flask(__name__)


def check_and_clean_data(data, table_name):
    """检查数据质量并记录指标，同时删除缺失值所在行"""
    try:
        task_map = {
            "task1_data": "text_summary",
            "task2_data": "fake_news_detection",
            "task3_data": "text_classification"
        }

        if table_name not in task_map:
            return None

        task_label = task_map[table_name]

        # 指定字段列表
        if table_name == "task1_data":
            fields_to_check = ['label_input', 'label_label1']
        elif table_name == "task2_data":
            fields_to_check = ['label_input', 'label_label2']
        elif table_name == "task3_data":
            fields_to_check = ['label_input', 'label_label3']

        # 记录缺失值数量（每列）并删除这些行
        for field in fields_to_check:
            if field in data.columns:
                missing_count = data[field].isna().sum()
                if missing_count > 0:
                    data_quality_missing_values.labels(task=task_label, field=field).inc(missing_count)

        # 删除含缺失值的整行（只考虑指定字段）
        data.dropna(subset=fields_to_check, inplace=True)

        # 检查重复行（全行重复）
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            data_quality_duplicates.labels(field=table_name).set(duplicates)
            # 删除重复行
            data.drop_duplicates(inplace=True)
            logging.info(f"Removed {duplicates} duplicate rows from {table_name}")

        return data

    except Exception as e:
        data_quality_processing_errors.labels(error_type='quality_check_and_clean_error').inc()
        logging.error(f"Error checking data quality and clean for {table_name}: {str(e)}")
        return None


def save_to_volume(data, table_name):
    """保存处理后的数据到共享卷，供 rclone 服务上传到对象存储"""
    try:
        # 构建保存路径
        save_dir = f"/app/etl_data/{table_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 使用固定文件名
        file_path = os.path.join(save_dir, "evaluation.csv")
        
        # 保存为 CSV
        data.to_csv(file_path, index=False, encoding='utf-8')
        
        logging.info(f"Data saved to {file_path}, waiting for rclone to upload to object storage")
        return True
    
    except Exception as e:
        logging.error(f"Error saving to volume: {str(e)}")
        return False

def read_from_minio():
    """从 MinIO 读取 SQLite 数据"""
    try:
        # 构建对象名称
        object_name = "evaluation.db"
        temp_db = "/tmp/evaluation_temp.db"
        
        # 从 MinIO 下载文件
        s3_client.download_file(MINIO_BUCKET, object_name, temp_db)
        
        # 从 SQLite 读取数据
        conn = sqlite3.connect(temp_db)
        
        # 获取所有表单名称
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # 读取所有表单数据
        all_data = {}
        for table in tables: 
            table_name = table[0]
            query = f"SELECT * FROM {table_name}"
            data = pd.read_sql_query(query, conn)
            all_data[table_name] = data
            
        conn.close()
        
        # 删除临时文件
        os.remove(temp_db)
        
        return all_data
    except Exception as e:
        logging.error(f"Error reading from MinIO: {str(e)}")
        return None


@app.route("/etl", methods=["POST"])
def trigger_etl():
    data = request.get_json(force=True)
    task = data.get("task")

    if not task:
        return {"error": "Missing task field"}, 400

    # 创建第一个线程用于立即返回响应
    def send_response():
        try:
            requests.post(   ##这样对吗？？？
                "http://trigger:8000/trigger",
                json={"status": "received", "task": task},
                timeout=5
            )
        except Exception as e:
            logging.error(f"Error sending response to trigger service: {str(e)}")

    # 创建第二个线程用于处理数据
    def process_data():
        try:
            # 从 MinIO 读取数据
            evaluation_data = read_from_minio()
            if evaluation_data is None:
                logging.error("Failed to read evaluation data")
                return

            # 处理每个表单的数据
            for table_name, table_data in evaluation_data.items():
                # 处理数据
                processed_data = check_and_clean_data(table_data, table_name)
                if processed_data is None:
                    logging.error(f"Failed to process evaluation data for table {table_name}")
                    continue
                    
                # 保存处理后的数据到共享卷
                if not save_to_volume(processed_data, table_name):
                    logging.error(f"Failed to save data for table {table_name}")
                    continue

            # ETL 成功后触发 train 服务
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

    # 启动两个线程
    threading.Thread(target=send_response).start()
    threading.Thread(target=process_data).start()

    return {"status": "ETL process started", "task": task}, 200

@app.route("/metrics", methods=["GET"])
def metrics():
    """暴露 Prometheus 指标"""
    return generate_latest(), 200, {'Content-Type': 'text/plain'}    # text指定返回类型为纯文本
