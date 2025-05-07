import os
import logging
from flask import Flask, request
from prometheus_client import Counter, Gauge, start_http_server
import time
import json
import pandas as pd
import numpy as np
import requests


# Prometheus 指标定义
# 数据质量指标
data_quality_missing_values = Gauge('data_quality_missing_values', 'Number of missing values', ['task', 'field'])
data_quality_duplicates = Gauge('data_quality_duplicates', 'Number of duplicate entries', ['task'])
data_quality_processing_errors = Counter('data_quality_processing_errors', 'Number of data processing errors', ['task', 'error_type'])

# 数据目录配置 待修改...
DEPLOY_DATA_DIR = '/app/deploy_data'  # 容器内的部署数据目录
PROCESSED_DATA_DIR = '/app/processed_data'  # 容器内的处理后数据目录
etl_data_dir = '/app/etl_data'  # ETL 工作目录
log_file = os.path.join(etl_data_dir, "app.log")

# 确保目录存在
os.makedirs(DEPLOY_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(etl_data_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

app = Flask(__name__)

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

def save_to_volume(data, filepath):
    """保存数据到挂载卷"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 如果是 DataFrame，转换为 JSON
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        
        # 保存为 JSON 文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error saving to volume: {str(e)}")
        return False

def read_from_deploy_volume(task_id):
    """从部署数据卷读取数据"""
    try:
        # 构建文件路径
        file_path = os.path.join(DEPLOY_DATA_DIR, f"task{task_id}/evaluation.json")
        
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        logging.error(f"Error reading from deploy volume: {str(e)}")
        return None

def process_online_evaluation(task_id, data):
    try:
        # 处理数据
        data = process_evaluation_data(data, task_id)
        if data is None:
            return False
        
        # 生成文件路径
        timestamp = int(time.time())
        file_path = f"task{task_id}/evaluation_{timestamp}.json"
        full_path = os.path.join(PROCESSED_DATA_DIR, file_path)
        
        # 保存到挂载卷
        if not save_to_volume(data, full_path):
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
        task_id = int(task.replace("task", ""))
        if 1 <= task_id <= 3:
            # 从部署数据卷读取数据
            evaluation_data = read_from_deploy_volume(task_id)
            if evaluation_data is None:
                return {"error": "Failed to read evaluation data"}, 500

            # 处理数据
            success = process_online_evaluation(task_id, evaluation_data)
            if not success:
                return {"error": "Failed to process evaluation data"}, 500
                
            # ETL 成功后触发 train 服务
            try:
                train_response = requests.post(
                    "http://train:8000/train",
                    json={"task": task},
                    timeout=5
                )
                if train_response.status_code != 200:
                    logging.error(f"Failed to trigger train service: {train_response.text}")
                    return {"status": "ETL success but train trigger failed", "task": task}, 200
            except Exception as e:
                logging.error(f"Error triggering train service: {str(e)}")
                return {"status": "ETL success but train trigger failed", "task": task}, 200
        else:
            return {"error": "Invalid task ID for evaluation"}, 400
    except Exception as e:
        logging.exception("ETL failed")
        return {"error": str(e)}, 500

    return {"status": "ETL success and train triggered", "task": task}, 200

if __name__ == "__main__":
    # 启动 Prometheus 指标服务器
    start_http_server(8001)
    # 启动 Flask 应用
    app.run(host="0.0.0.0", port=8000)
