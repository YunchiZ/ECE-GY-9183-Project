import threading
import json
import requests
import random
import os
import sqlite3
from flask import Flask, request, jsonify
import logging
import subprocess
from fakenews_app import *
from classification_app import *
from summary_app import *

# Configure logging
logging.basicConfig(level=logging.INFO, 
    format='[%(levelname)s] %(message)s',
    filename='train.log',
    filemode='w' )

app = Flask(__name__)
train_data_dir = '/app/models'

base_url = os.environ.get("DEPLOY_ENDPOINT", "http://deploy:8000")
DEPLOY_ENDPOINT = base_url.rstrip('/') + "/notify"
WANDB_KEY = os.environ.get("WANDB_LICENSE")

TRAINING_SCRIPTS = {
    1: "classification_app",
    2: "fakenews_app.py",
    3: "summary_app.py"
}

@app.route("/train", methods=["POST"])
def start_training():
    data = request.json
    task_name = data.get("task")

    thread = threading.Thread(target=run_all_training)
    thread.start()

    return jsonify({"msg": f"Training for task '{task_name}' started"}), 200

def run_all_training():
    try:
        logging.info("Starting training for: classification")
        path, model = classification_app.classification_run(WANDB_KEY)
        if path != "fail":
            upload_to_s3(str(path), "candidate", f"{model}.onnx")
            notify_deploy(2, model)

        logging.info("Starting training for: fakenews")
        path, model = fakenews_app.fakenews_run(WANDB_KEY)
        if path != "fail":
            upload_to_s3(str(path), "candidate", f"{model}.onnx")
            notify_deploy(1, model)

        logging.info("Starting training for: summary")
        path, model = summary_app.summary_run(WANDB_KEY)
        if path != "fail":
            upload_to_s3(str(path), "candidate", f"{model}.onnx")
            notify_deploy(0, model)

    except Exception as e:
        logging.error(f"Training failed: {e}")

def notify_deploy(task_name, model_name=None, max_retries=3):
    payload = {
        "type": "shadow",
        "index": task_name,
        "model_name": model_name
    }

    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Attempt {attempt}: Notifying deploy service at {DEPLOY_ENDPOINT} ...")
            response = requests.post(DEPLOY_ENDPOINT, json=payload, timeout=10)

            if response.status_code == 200:
                logging.info("Deploy acknowledged training completion.")
                break
            else:
                logging.warning(f"Deploy responded with status {response.status_code}: {response.text}")

        except Exception as e:
            logging.error(f"Attempt {attempt} failed: {e}")

        if attempt == max_retries:
            logging.error("All attempts to notify deploy failed.")

def upload_to_s3(local_path, bucket_name, s3_key):
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_URL"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
        )

        # check if bucket
        try:
            s3.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                s3.create_bucket(Bucket=bucket_name)
                logging.info(f"Created bucket: {bucket_name}")
            else:
                raise

        s3.upload_file(local_path, bucket_name, s3_key)
        logging.info(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logging.error(f"Failed to upload to S3: {e}")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000)
