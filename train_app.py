import threading
import json
import requests
import random
import os
import sqlite3
from flask import Flask, request, jsonify
import logging
import subprocess
# import time

app = Flask(__name__)
train_data_dir = '/app/models'  # train容器管理的卷的路径

DEPLOY_ENDPOINT = "http://deploy:8000/training-complete" 

TRAINING_SCRIPTS = {
    "classification": "classification_app.py",
    "fakenews": "fakenews_app.py",
    "summary": "summary_app.py"
}
# ---------------- Logging Initialization ---------------- logging信息所需要的基本函数
# log_file = os.path.join(_data_dir, "app.log")  # 把logging文件放置在你的卷里面

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_file),
#         logging.StreamHandler()
#     ]
# )

# ---------------- Global Variables ---------------- 你包会用到的全局变量
# ---------------- SQLite 相关函数
# ---------------- Associated Functions 在app.route会用到的函数
# ---------------- Flask App Routes ----------------

@app.route("/start-training", methods=["POST"])
def start_training():
    data = request.json
    task_name = data.get("task")

    if not task_name or task_name not in TRAINING_SCRIPTS:
        return jsonify({"error": "Invalid or missing task name."}), 400

    thread = threading.Thread(target=run_training_pipeline, args=(task_name,))
    thread.start()

    return jsonify({"msg": f"Training for task '{task_name}' started"}), 202

def run_training_pipeline(task_name):
    script = TRAINING_SCRIPTS[task_name]
    script_path = os.path.join(train_data_dir, script)

    print(f"[INFO] 🚀 Starting training for '{task_name}' using script: {script_path}")
    
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True
        )

        print(f"[INFO] Finished training '{task_name}'")
        print(result.stdout)

        if result.stderr:
            print(f"[WARN] stderr from training:\n{result.stderr}")

        notify_deploy(task_name)

    except Exception as e:
        print(f"[ERROR] Training failed for task '{task_name}': {e}")

def notify_deploy(task_name):
    payload = {"task": task_name}

    try:
        print(f"[INFO] 📡 Notifying deploy service at {DEPLOY_ENDPOINT} ...")
        response = requests.post(DEPLOY_ENDPOINT, json=payload, timeout=10)

        if response.status_code == 200:
            print("[INFO] Deploy acknowledged training completion.")
        else:
            print(f"[WARN] Deploy responded with status {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[ERROR] Failed to notify deploy: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

    # 要按顺序运行的脚本列表
    scripts = [
        "classification_app.py",
        "fakenews_app.py",
        "summary_app.py"
    ]

    for script in scripts:
        print(f"running: {script}")
        result = subprocess.run(["python", script], capture_output=True, text=True)

        print(f"{script} finished: ")
        print(result.stdout)

        if result.stderr:
            print(f"{script} error: ")
            print(result.stderr)





