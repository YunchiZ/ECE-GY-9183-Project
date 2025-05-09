import os
import logging
from flask import Flask, request
from prometheus_client import Counter, Gauge, start_http_server, generate_latest
import time
import json
import pandas as pd
import numpy as np
import requests
import sqlite3
import boto3
from io import BytesIO

# MinIO 配置
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
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
data_quality_duplicates = Gauge('data_quality_duplicates', 'Number of duplicate entries', ['task'])
data_quality_processing_errors = Counter('data_quality_processing_errors', 'Number of data processing errors', ['task', 'error_type'])

# 数据目录配置 待修改...
MONITOR_DATA_DIR = '/app/monitor_data'  # 容器内的部署数据目录
PROCESSED_DATA_DIR = '/app/etl_data'  # 容器内的处理后数据目录
etl_data_dir = '/app/etl_data'  # ETL 工作目录
log_file = os.path.join(etl_data_dir, "app.log")

# 确保目录存在
os.makedirs(MONITOR_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(etl_data_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

app = Flask(__name__)

# 启动 Prometheus 指标服务器
start_http_server(8001)

def clean_text(df):
    df.dropna(inplace=True)
    return df

def check_data_quality(data, task_id):
    """检查数据质量并记录指标"""
    try:
        # 检查缺失值
        for field in ['predictions_input', 'predictions_pred', 'label_input']:
            if pd.isna(data[field]):
                data_quality_missing_values.labels(task=f'task{task_id}', field='batch').inc()
                return False
        
        # 检查重复值
        if isinstance(data, pd.DataFrame):
            duplicates = data.duplicated().sum()
            if duplicates > 0:
                data_quality_duplicates.labels(task=f'task{task_id}').set(duplicates)
                return False
        
        return True
    except Exception as e:
        data_quality_processing_errors.labels(task=f'task{task_id}', error_type='quality_check_error').inc()
        logging.error(f"Error checking data quality: {str(e)}")
        return False

def process_evaluation_data(data, task_id):
    """处理评估数据"""
    try:
        # 检查数据质量
        if not check_data_quality(data, task_id):
            return None
        
        # 根据任务类型处理数据
        if task_id == 1:  # 文本摘要
            data['predictions_input'] = clean_text(data['predictions_input'])
            data['label_input'] = clean_text(data['label_input'])
            data['label_label1'] = clean_text(data['label_label1'])
        elif task_id == 2:  # 假新闻检测
            data['predictions_input'] = clean_text(data['predictions_input'])
            data['label_input'] = clean_text(data['label_input'])
            data['label_label2'] = int(data['label_label2'])  # 确保是整数
        elif task_id == 3:  # 文本分类
            data['predictions_input'] = clean_text(data['predictions_input'])
            data['label_input'] = clean_text(data['label_input'])
            data['label_label3'] = int(data['label_label3'])  # 确保是整数
        
        return data
    except Exception as e:
        data_quality_processing_errors.labels(task=f'task{task_id}', error_type='processing_error').inc()
        logging.error(f"Error processing evaluation data: {str(e)}")
        raise

def save_to_volume(data, task_id):
    """保存处理后的数据到共享卷"""
    try:
        # 构建保存路径
        save_dir = f"/data/processed/task{task_id}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 使用固定文件名
        file_path = os.path.join(save_dir, "evaluation.csv")
        
        # 保存为 CSV
        data.to_csv(file_path, index=False, encoding='utf-8')
        
        return True
    except Exception as e:
        logging.error(f"Error saving to volume: {str(e)}")
        return False

def read_from_minio(task_id):
    """从 MinIO 读取 SQLite 数据"""
    try:
        # 构建对象名称
        object_name = f"task{task_id}/evaluation.db"
        temp_db = f"/tmp/task{task_id}_temp.db"
        
        # 从 MinIO 下载文件
        s3_client.download_file(MINIO_BUCKET, object_name, temp_db)
        
        # 从 SQLite 读取数据
        conn = sqlite3.connect(temp_db)
        query = "SELECT * FROM evaluation"
        data = pd.read_sql_query(query, conn)
        conn.close()
        
        # 删除临时文件
        os.remove(temp_db)
        
        return data
    except Exception as e:
        logging.error(f"Error reading from MinIO: {str(e)}")
        return None

def save_to_minio(data, task_id):
    """保存处理后的数据到 MinIO"""
    try:
        # 将数据保存为临时 CSV 文件
        temp_csv = f"/tmp/task{task_id}_processed.csv"
        data.to_csv(temp_csv, index=False, encoding='utf-8')
        
        # 构建对象名称
        object_name = f"task{task_id}/evaluation_processed.csv"
        
        # 上传到 MinIO
        s3_client.upload_file(temp_csv, MINIO_BUCKET, object_name)
        
        # 删除临时文件
        os.remove(temp_csv)
        
        return True
    except Exception as e:
        logging.error(f"Error saving to MinIO: {str(e)}")
        return False

def process_online_evaluation(task_id, data):
    try:
        # 处理数据
        data = process_evaluation_data(data, task_id)
        if data is None:
            return False
        
        # 保存到共享卷
        if not save_to_volume(data, task_id):
            return False
        
        return True
    except Exception as e:
        logging.error(f"Error processing online evaluation for task {task_id}: {str(e)}")
        return False

@app.route("/etl", methods=["POST"])
def trigger_etl():
    data = request.get_json(force=True)
    task = data.get("task")

    if not task:
        return {"error": "Missing task field"}, 400

    try:
        # 处理所有三个任务
        for task_id in range(1, 4):
            # 从 MinIO 读取数据
            evaluation_data = read_from_minio(task_id)
            if evaluation_data is None:
                logging.error(f"Failed to read evaluation data for task {task_id}")
                continue

            # 处理数据
            success = process_online_evaluation(task_id, evaluation_data)
            if not success:
                logging.error(f"Failed to process evaluation data for task {task_id}")
                continue
                
            # ETL 成功后触发 train 服务
            try:
                train_response = requests.post(
                    "http://train:8000/train",
                    json={},  # 移除 task 参数
                    timeout=5
                )
                if train_response.status_code != 200:
                    logging.error(f"Failed to trigger train service: {train_response.text}")
            except Exception as e:
                logging.error(f"Error triggering train service: {str(e)}")

        return {"status": "ETL process completed for all tasks", "task": task}, 200
    except Exception as e:
        logging.exception("ETL failed")
        return {"error": str(e)}, 500

@app.route("/metrics", methods=["GET"])
def metrics():
    """暴露 Prometheus 指标"""
    return generate_latest(), 200, {'Content-Type': 'text/plain'}    # text指定返回类型为纯文本
