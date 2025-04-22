import threading
import json
import requests
import random
import os
import sqlite3
from flask import Flask, request
import logging
import subprocess
# import time

app = Flask(__name__)
# etl_data_dir = '/app/etl_data'   # etl容器管理的卷的路径
train_data_dir = '/app/models'  # train容器管理的卷的路径
# model_status_file = '/app/models/model_status.json' train容器需要的记录模型状态的json文件

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





