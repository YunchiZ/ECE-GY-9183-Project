import threading
import json
import requests
import random
import os
import sqlite3
# from flask import Flask, request, jsonify
import logging
import subprocess
# import time

app = Flask(__name__)
train_data_dir = '/app/models'  # train容器管理的卷的路径

DEPLOY_ENDPOINT = "http://deploy:8000/notify" 

TRAINING_SCRIPTS = {
    "0": "classification_app.py",
    "1": "fakenews_app.py",
    "2": "summary_app.py"
}


@app.route("/train", methods=["POST"])
def start_training():
    data = request.json
    task_name = data.get("task")

    if not task_name or task_name not in TRAINING_SCRIPTS:
        return jsonify({"error": "Invalid or missing task name."}), 400

    thread = threading.Thread(target=run_all_training)
    thread.start()

    return jsonify({"msg": f"Training for task '{task_name}' started"}), 200

def run_all_training():
    for task_name in TRAINING_SCRIPTS.keys():
        print(f"[INFO] Starting training for: {task_name}")
        try:
            result = subprocess.run(
                ["python", os.path.join(train_data_dir, TRAINING_SCRIPTS[task_name])],
                capture_output=True,
                text=True
            )

            print(result.stdout)
            if result.stderr:
                print(f"[WARN] stderr:\n{result.stderr}")
            
            if "test failed" in result.stdout.lower(): # skip message if train fail
                print(f"[WARN] Training for task {task_name} failed. Skipping deploy notification.")
                continue

            model_name = parse_model_name(result.stdout)

            # 训练完成立即通知 deploy
            notify_deploy(task_name, model_name)

        except Exception as e:
            print(f"[ERROR] Training failed for task '{task_name}': {e}")



def notify_deploy(task_name, model_name=None):
    payload = {
        "type":"shadow",
        "index": task_name,
        "model_name": model_name
    }

    try:
        print(f"[INFO] Notifying deploy service at {DEPLOY_ENDPOINT} ...")
        response = requests.post(DEPLOY_ENDPOINT, json=payload, timeout=10)

        if response.status_code == 200:
            print("[INFO] Deploy acknowledged training completion.")
        else:
            print(f"[WARN] Deploy responded with status {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[ERROR] Failed to notify deploy: {e}")

def parse_model_name(output_text):
    lines = [line.strip() for line in output_text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else None






