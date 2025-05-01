from flask import Flask, request
import logging
import threading
import json
import requests
import os
import sqlite3
import numpy as np
from typing import List

from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score

from prometheus_update import UpdateAgent

# ！！！！！！！
# 全篇代码还差很多异常处理(try except)的框架
# 例如对于指标s_metrics; c_metrics的操作等； 任何文件的读取与写入等等
# 全篇代码还差大量关于logging日志输出的代码 需要全部进行微调
# 对于json消息的解析还需要全面优化 request.get_json(force=True) 被gpt标注为是存在问题的
# 而且需要补足backoff或队列机制
# 所有涉及数据库操作的部分 是否需要线程锁？
# 关于数据库的某些设置 请参照 deploy_app.py
# 对于整个代码中全局变量的使用 需要检查一遍是否使用线程锁
# !考虑引入kafka来管理http消息队列(待定)
# 考虑使用MonitorStateManager的类来管理所有监控状态
"""
import threading

class MonitorStateManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.serving = ["BART-v0", "XLN-v0", "BERT-v0"]
        self.candidate = [None, None, None]
        self.stage = ["normal", "normal", "normal"]
        self.s_metrics = [[], [], []]
        self.c_metrics = [[], [], []]
        self.error_flags = [False, False, False]

    def set_stage(self, index, new_stage):
        with self._lock:
            self.stage[index] = new_stage

    def get_stage(self, index):
        with self._lock:
            return self.stage[index]

    def update_metrics(self, index, metric_s, metric_c):
        with self._lock:
            self.s_metrics[index].append(metric_s)
            self.c_metrics[index].append(metric_c)
        # ...更多操作...

    # 其他类似的操作
"""


app = Flask(__name__)
thread_lock = threading.Lock()
# =====================================
# 文件地址；
# LOCK_file = "./.../.../"
# 记录一个主机的全局变量的json文件 如果LOCK=True 那么说明正在进行CI/CD流程 已经锁止
# 避免频繁触发CI/CD流程
deploy_data_dir = "/app/deploy_data"
monitor_data_dir = "/app/monitor_data"
label_dir = "/app/label"

deploy_database = os.path.join(deploy_data_dir, "serving_data.db")
label_database = os.path.join(label_dir, "label.db")

LOCK_file = "/app/LOCK.json"
# =====================================
# 全局变量 Global Variables
# 目前这里设置的逻辑是 为了简化过程 训练时同时启动三种模型流程 而非单独一个模型
# candidate1, candidate2, candidate3 = None, None, None
serving = ["BART-v0", "XLN-v0", "BERT-v0"]
candidate = [None, None, None]
stage = ["normal", "normal", "normal"]  # 这个stage是跟任务绑定起来的
# 因为candidate模型需要经历 shadow -> canary -> normal的阶段
# 不需要针对serving设置stage 因为它本身的stage状态就是"serving"
# 在本项目中 任何一个模型的fail或者git actions操作 均会触发同时对于三个模型的训练和部署 并且使得LOCK锁止直至最后一个模型完成其最终部署决策
# 由于样本数量的限制 训练和部署不一定处于同期 顺序: DistilBERT -> XLNet-> BART
# =====================================
# 临界条件 Threshold Values
# 所有list中的三个值都意味着三个模型及其任务 [BART, XLNet, DistilBERT][文本总结, 真假鉴别, 文本分类]
critical_decay = [
    0.05,
    0.05,
    0.05,
]  # 这个值代表了模型性能的衰退 具体是多少需要进行实验和测试
critical_sample = [
    10000,
    5000,
    5000,
]  # 需要通过x次样本的结果评估来决定是否从shadow到canary或者从canary到serving
sample_num = [0, 0, 0]
t = [0.02, 0.03, 0.03]  # 需要模型有n%的性能提升才认为通过测试
# SLA = 1500 # ms 平均响应时间需要低于500ms 无论是shadow模式下模型的推理时间还是canary模式下前端的总响应时间
critical_err = (
    0.02  # 无论是shadow阶段还是canary阶段还是normal阶段 错误率都不得超过此占比
)
# shadow阶段 由于API并不返回candidate的预测值 所以不需要及时下架candidate 错误率交由monitor容器进行分析 降低deploy容器的负担
# 此时错误类型包括 1) 模型推理响应时间超过LSA时间(会被记录在SQlite数据结构中) 2) 服务器API未知原因错误(错误码5xx)
# canary阶段 由于API需要返回candidate的预测值 不仅需要满足实时性 也需要包括前端的错误 所以此时错误率交由前端容器frontend进行分析
# 此时错误类型包括 1) 前端响应时间超过LSA时间(由前端自己记录) 2) 服务器API错误(错误码5xx) 3) 客户端http错误(4xx)
# serving_err = [0, 0, 0]  # 三个服役模型的错误率统计
# candidate_err = [0, 0, 0]  # 三个候选模型的错误率统计
# 由于错误率统计不再基于往期记录 而是每一轮(每k次API调用)的记录 所以不使用这两个list
# 因为基于每一轮的错误率记录能够及时反应模型错误率突然上升的情况(包括但不限于GPU临时故障等)
error_ = [False, False, False]  # 有无frontend汇报的错误率超标问题 分别对应三个模型
# ======================================
# 还有一些指标列表需要写在这里
s_metrics: List[List[float]] = [[], [], []]  # 暂时使用列表 保证不同长度时的灵活性
# 分别存放三个模型的指标记录 对于文本总结任务 记录ROUGE数值 对于另外两个任务 记录ACC数值
c_metrics: List[List[float]] = [[], [], []]
# 关于数据指标的初始化??? 暂时未知 需要ETL那边的分析函数
# ？？？


# ================================================ 板块隔离带
# def report(message, task_id, model_name, model_type):  # 报告给prometheus容器的通用函数
#     # 该函数内容在项目最后需要大改 因为该函数需要适应下面两个函数hardware_status和docker_status
#     # message = 1 -> data shift 外部数据分布漂移
#     # message = 2 -> error normal阶段 前端容器统计serving报警率超标
#     # message = 3 -> performance decay 模型性能衰退
#     # message = 4 -> shadow/canary阶段 候选模型错误率过高 直接被淘汰(消息播报 而非状态展示)

#     # 放到一个端口上让Prometheus抓取

#     prometheus_url = "http://prometheus:9090"  # 是这样的吗?
#     payload = {
#         "warning_type": message,
#         "task_type": task_id,
#         "model_name": model_name,
#         "model_type": model_type,
#     }
#     # 该payload后期需要大改
#     # 根据prometheus和grafana的具体工作机制来决定这里是否需要更详细的payload信息封装
#     try:
#         response = requests.post(prometheus_url, json=payload, timeout=5)
#         logging.info(
#             f"Notified prometheus docker with {payload}, status: {response.status_code}"
#         )

#     except requests.RequestException as e:
#         logging.error(f"Error notifying prometheus docker: {e}")


# ================================================


def notify(docker, index, notification_type):
    # 这里的index就是任务(模型)ID 对应文本总结、真假鉴别、分类三个NLP任务
    docker_url = "http://" + docker + ":8000/notify"  # 内容网络访问容器端口
    payload = {"type": notification_type, "index": index}
    try:
        response = requests.post(docker_url, json=payload, timeout=5)
        logging.info(
            f"Notified {docker} docker with {payload}, status: {response.status_code}"
        )
        return 200
    except requests.RequestException as e:
        logging.error(f"Error notifying {docker} docker: {e}")
        return 500


# ================================================ 数据库相关的所有函数
# 实际上前端容器 会根据标签的id 将标签放进不同任务名称的数据库中 所以不需要再让Monitor容器进行匹配
# 这些函数都要确保:
# 1) 检查文件存在性
# 2) 访问试探 try except
# 3) 系列异常处理

# deploy_data schema:
# CREATE TABLE IF NOT EXISTs predictions (id INTEGER PRIMARY KEY AUTOINCREMENT
# input TEXT
# pred TEXT/INTEGER


# label schema:
# id INTEGER PRIMARY KEY
# input TEXT
# label TEXT/INTEGER


def database_merge(root_name):

    # index 确定任务类型，创建对应表单
    # $ 根据deploy_data_dir和serving[index]确定database访问路径
    # $ 根据label_dir以及? 确定用户行为标签库/文件的访问路径
    # $ 生成对应的临时数据库放置的地址 有可能这个步骤不需要 因为有可能直接在原地进行拼接 但还是建议拼接成一个新的数据库 避免冲突

    new_db_path = os.path.join(monitor_data_dir, root_name)

    deploy_database = os.path.join(deploy_database, "serving_data.db")
    label_database = os.path.join(label_database, "label.db")

    deploy_conn = sqlite3.connect(deploy_database)
    deploy_cur = deploy_conn.cursor()

    label_conn = sqlite3.connect(label_database)
    label_cur = label_conn.cursor()

    deploy_cur.execute(f"ATTACH DATABASE '{label_database}' AS label_db")

    new_conn = sqlite3.connect(new_db_path)
    new_cur = new_conn.cursor()

    # 在新数据库中创建三个表，分别对应三个任务
    new_cur.execute(
        """
    CREATE TABLE IF NOT EXISTS task1_data (
        id INTEGER PRIMARY KEY,
        predictions_input TEXT,
        predictions_pred TEXT,
        label_input TEXT,
        label_label1 TEXT
    )
    """
    )

    new_cur.execute(
        """
    CREATE TABLE IF NOT EXISTS task2_data (
        id INTEGER PRIMARY KEY,
        predictions_input TEXT,
        predictions_pred TEXT,
        label_input TEXT,
        label_label2 INTEGER
    )
    """
    )

    new_cur.execute(
        """
    CREATE TABLE IF NOT EXISTS task3_data (
        id INTEGER PRIMARY KEY,
        predictions_input TEXT,
        predictions_pred TEXT,
        label_input TEXT,
        label_label3 INTEGER
    )
    """
    )

    # 执行基本查询获取所有匹配记录
    base_query = """
    SELECT 
        p.id,
        p.input AS predictions_input,
        p.pred AS predictions_pred,
        l.input AS label_input,
        l.pred1 AS label_label1,
        l.pred2 AS label_label2,
        l.pred3 AS label_label3
    FROM 
        predictions p
    JOIN 
        label_db.label l ON p.id = l.id
    """

    results = deploy_cur.execute(base_query).fetchall()

    task1_insert = """
    INSERT INTO task1_data (
        id, predictions_input, predictions_pred, label_input, label_label1
    ) VALUES (?, ?, ?, ?, ?)
    """

    task2_insert = """
    INSERT INTO task2_data (
        id, predictions_input, predictions_pred, label_input, label_label2
    ) VALUES (?, ?, ?, ?, ?)
    """

    task3_insert = """
    INSERT INTO task3_data (
        id, predictions_input, predictions_pred, label_input, label_label3
    ) VALUES (?, ?, ?, ?, ?)
    """

    task1_data = [(row[0], row[1], row[2], row[3], row[4]) for row in results]
    task2_data = [(row[0], row[1], row[2], row[3], row[5]) for row in results]
    task3_data = [(row[0], row[1], row[2], row[3], row[6]) for row in results]

    new_cur.executemany(task1_insert, task1_data)
    new_cur.executemany(task2_insert, task2_data)
    new_cur.executemany(task3_insert, task3_data)

    new_conn.commit()

    deploy_cur.close()
    deploy_conn.close()
    label_cur.close()
    label_conn.close()
    new_cur.close()
    new_conn.close()

    # $ (数据库行为) 根据prediction ID 进行数据库内容匹配和整合
    # $ 将新数据库放置在database_dir中 等待下面的metric_analysis函数进行阶段定制化分析
    return new_db_path


def clear_labels(label_type):  # 删除/app/labels中被访问过的数据库
    # 访问/app/labels中的数据库 注意是否需要使用线程锁
    # 根据label_type检测当前任务数据库
    # 删除对应的任务的数据库

    # 考虑前端直接覆盖文件
    return None


def metric_analysis(database, itype, data_analysis, e_analysis):

    conn = sqlite3.connect(database)
    cur = conn.cursor()

    error_status, data_info = False, 0
    # $ 根据传入的目标数据库地址"database"进行访问
    # $ 通过prediction条目与label条目进行对应的指标分析
    try:
        # 任务1: 文本摘要评估 (ROUGE)
        if itype == 0:
            # 从task1_data表中获取所有预测和标签
            query = "SELECT id, predictions_pred, label_label1 FROM task1_data"
            results = cur.execute(query).fetchall()

            # 初始化ROUGE评分器和分数列表
            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            rouge_scores = []

            # 计算每个样本的ROUGE分数
            for id, pred, label in results:
                if pred and label:  # 确保预测和标签都不为空
                    scores = scorer.score(label, pred)
                    # 使用F1分数作为主要指标
                    rouge_f1 = {k: v.fmeasure for k, v in scores.items()}
                    rouge_scores.append(rouge_f1)

            # 计算平均ROUGE分数
            avg_scores = {}
            for key in ["rouge1", "rouge2", "rougeL"]:
                avg_scores[key] = np.mean([score[key] for score in rouge_scores])
            # 比例待定
            metric1 = (
                0.2 * avg_scores["rouge1"]
                + 0.3 * avg_scores["rouge2"]
                + 0.5 * avg_scores["rougeL"]
            )

        # 任务2: 分类任务1的ACC评估
        elif itype == 1:
            # 从task2_data表中获取所有预测和标签
            query = "SELECT id, predictions_pred, label_label2 FROM task2_data"
            results = cur.execute(query).fetchall()

            # 提取预测和真实标签
            y_pred = []
            y_true = []

            for id, pred, label in results:
                # 确保预测值和标签是整数
                try:
                    pred_val = int(pred)
                    label_val = int(label)
                    y_pred.append(pred_val)
                    y_true.append(label_val)
                except (ValueError, TypeError):
                    continue  # 跳过无法转换为整数的值

            # 计算准确率
            if y_pred and y_true:
                accuracy = accuracy_score(y_true, y_pred)

                metric2 = accuracy

        # 任务3: 分类任务2的ACC评估
        elif itype == 2:
            # 从task3_data表中获取所有预测和标签
            query = "SELECT id, predictions_pred, label_label3 FROM task3_data"
            results = cur.execute(query).fetchall()

            # 提取预测和真实标签
            y_pred = []
            y_true = []

            for id, pred, label in results:
                # 确保预测值和标签是整数
                try:
                    pred_val = int(pred)
                    label_val = int(label)
                    y_pred.append(pred_val)
                    y_true.append(label_val)
                except (ValueError, TypeError):
                    continue  # 跳过无法转换为整数的值

            # 计算准确率
            if y_pred and y_true:
                accuracy = accuracy_score(y_true, y_pred)

                metric3 = accuracy

        else:
            error_status = True
            data_info = f"无效的任务类型: {itype}"

    finally:
        # 关闭数据库连接
        cur.close()
        conn.close()

    metric = (metric1, metric2, metric3)

    # $ 在normal阶段需要分析数据分布的改变:
    if data_analysis:
        # $ 对应的关于数据分布的分析 这个函数可能来源于ETL部分的代码改写
        # 直接返回关于data shift的指标 而不是数据分布 直接就在这里进行分析
        data_info = 0
        pass

    # $ 在shadow阶段还需要统计candidate的错误率
    if e_analysis:
        error_rate = 0  # 统计数据库error条目占比
        # $ 然后判断error_rate有无超过critical_err这个值
        # $ 如果超过了 那么 error_status = True
        if error_rate > critical_err:
            error_status = True
        else:
            error_status = False

    return metric, data_info, error_status


def database_output(s_database, c_database, database_index, operation):
    # $ 对s_database进行修改 - 删除ID 模型预测值 cls 一系列不被最终数据集需要的键值对
    # $ 将s_database根据database_index写入monitor data中的目标任务数据库
    # $ 删除该临时数据库s_database
    if operation == "canary":
        # $ 对c_database 进行同样的修改操作
        # $ 然后把candidate数据库也写进去
        pass
    if operation == "shadow" or operation == "canary":
        pass
        # $ 在shadow和canary阶段 都需要删除c_database临时数据库


@app.route("/init", methods=["POST"])
def init():
    # 收到消息只有两种类型: candidate / serving
    # candidate代表 此时train触发deploy 需要monitor监控对应的模型的数据库
    # serving 代表 deploy容器完成canary阶段 需要monitor监控对应的数据库
    # 另外 如上所说 不可能三个模型是异步进行部署的 所以不存在多个模型恰好在同一个时间点都需要执行相同指令的情况
    # 为什么这么说? 因为1.train容器同时只支持最多一个模型的训练和离线测试 2.从shadow到canary到serving是需要样本流入的 不单是时间决定
    # 就算是是真的存在这种情况 deploy容器会发送2~3此请求 因为init()每次只执行对于一个模型的状态修改 已使用线程锁进行保护
    # init()的触发大概率不会与monitor()同时发生 但是为了避免小概率存在的这种情况 已使用线程锁进行保护
    global serving, candidate, stage, s_metrics, c_metrics

    data = request.get_json(force=True)
    message = data.get("type")
    model = data.get("model")  # 这里的model是真实模型名称 不是索引数字
    model_index = data.get("index")

    if not message:
        logging.error("No notification type provided in the request.")
        return "No notification type provided", 400

    if model_index not in [0, 1, 2]:
        logging.error("Invalid model index.")
        return "Invalid model index", 400

    if message == "candidate":
        with thread_lock:
            candidate[model_index] = model
            stage[model_index] = "shadow"
            # $ 需要重置serving相关的指标 因为shadow阶段需要比较接下来同期的serving和candidate表现：
            s_metrics[model_index] = []
            # 需要重置serving相关的指标 目前好像指标就只有一个
            c_metrics[model_index] = []  # 连同candidate也一起重置了
            # 之前的serving监控记录已经没有意义了 因为现在的重点考察对象是candidate
        # 不需要进行报告 deploy会将此条部署汇报给prometheus
        pass  # 这里是否还有逻辑内容遗漏?

    elif message == "serving":
        with thread_lock:
            candidate[model_index] = None
            serving[model_index] = model
            stage[model_index] = "normal"
            # $ 将各类全局指标的值进行迁移:
            # 例如将candidate(canary阶段产生的)的记录覆盖到serving中 并将candidate中对应位进行清零:
            s_metrics[model_index] = c_metrics[model_index]
            c_metrics[model_index] = []  # 重置
            # 不需要额外对"canary"阶段进行选取 因为从shadow到canary的时候 所有记录都会发生一次重置
            # $ 还需要对数据漂移的监控指标[model_index]初始化为[]

    else:
        logging.error(f"Unknown notification type: {message}")
        return "Unknown notification type", 400


@app.route("/monitor", methods=["POST"])  # 逻辑最复杂的模型监控部分 核心机制部分
def monitor():
    global serving, candidate, stage, sample_num, s_metrics, c_metrics, error_
    data = request.get_json(force=True)
    index = data.get("index")
    status = data.get("status")
    # 来自deploy容器告知monitor进行监控的http消息中包含index键 对应模型的id 0~2 也就是任务的id
    # 同样地 一次消息不会同时发两个任务的监控请求
    # 该消息产生的机制:  deploy容器中的每次API调用后产生的结果不会被立马写入SQlite中 而是存入内存变量中
    # 当内存中的样本数量达到例如1000个之后 便会将之前的记录删除(默认monitor容器在上一轮已经将其取走)
    # 然后一次性写入deploy data 并且http通知monitor容器对这批数据进行监控 - 读取、匹配与转移
    # 在获取index之后还需要获取status 因为monitor容器在本程序中也需要即时并且同步维护关于deploy容器中模型状态的全局变量
    # 只有status与本地记录匹配 才能进行对应的监控操作 该种机制的设置为了防止任何潜在的信息冲突 不过从逻辑上 已经确保了deploy和monitor环环相扣
    # ！不管所监控的状态是怎么样的 都要确保流程中包含以下内容中的多项:
    # 1) 临时数据库拼接(从deploy data读 -> 缓存拼接 -> 放置于monitor data)
    # 2) 指标计算
    # 3) 全局指标记录更新
    # 4) 数据库写入 临时数据库删除
    # 5) 判断临界条件(error)
    # 6) 容器http通知(最优优先级)(deploy&frontend&etl)
    # 7) 指标的更新或者重置
    # 8) 全局状态变量修改 / 修改状态文件
    # 9) 向prometheus报告所有相关内容

    if stage[index] != status:  # 如果出现不匹配的消息 直接报错 并且忽略
        return "Unmatched model stage status", 400

    if status == "normal":
        # 1）首先将对应database与label卷中的数据进行数据库匹配和拼接 并且放置于monitor data中
        with thread_lock:
            serving_name = serving[index]
        db_dir = database_merge(serving_name, index)
        clear_labels(index)
        # 2）接下来遍历该数据库 进行匹配、指标计算、整合、迁移; normal阶段的错误统计由前端统计 不在这里统计:
        metric, data, _ = metric_analysis(db_dir, index, True, False)
        # 3) 随后对临时数据库进行无关条目的删除 并且添加至对应任务的database中
        database_output(db_dir, None, index, "normal")
        # 4) 进行全局指标变量的更新和存储:
        # 使用线程锁访问指标全局变量:
        with thread_lock:
            err_info = error_[index]  # 全局变量error_中对应的内容有无出现告警
            avg_metric = np.mean(s_metrics[index]) if len(s_metrics[index]) > 0 else 0
            s_metrics[index].append(metric)
            # $ 访问关于数据分布data_distribution的全局指标变量
            # $ 进行类似的操作: 分析之前的指标的平均数 然后把data这个变量也加进去成为记录的一部分
        data_shift = False  # 是为了代码完整性 先写成这样
        # 5) 接下来进入临界条件的判断: 有无frontend统计错误率超标 / 有无数据漂移 / 有无性能衰退
        # 实际上为了此部分的流程效率 临界条件判断是有优先级的 但是由于需要向prometheus全部上报 所以三者都需要进行判断
        # 如果临界条件被触发 则根据LOCK.json触发ETL并且告警给prometheus
        if data_shift or err_info or (avg_metric - metric < critical_decay[index]):
            with thread_lock:
                # 读取LOCK.json文件中的LOCK键的值
                lock = False
                try:
                    with open(LOCK_file, "r") as f:
                        lock_data = json.load(f)
                        lock = lock_data.get("LOCK", False)
                except Exception as e:
                    print(f"读取锁文件时出错: {e}")

                if lock:
                    code = notify(
                        "etl", None, "trigger"
                    )  # 其实消息内容不重要 重要的是触发
                    if code == 200:  # 说明ETL触发成功
                        # 修改LOCK.json文件中的LOCK键值为True
                        try:
                            with open(LOCK_file, "r") as f:
                                lock_data = json.load(f)

                            lock_data["LOCK"] = True

                            with open(LOCK_file, "w") as f:
                                json.dump(lock_data, f)
                        except Exception as e:
                            print(f"修改锁文件时出错: {e}")
        # 6) 给prometheus报告相关情况
        if data_shift:
            prometheus_agent.report_warning(
                "data shift", index, serving_name, "serving"
            )

        # if err_info:
        #     report("2", index, "serving")
        if avg_metric - metric >= critical_decay[index]:
            prometheus_agent.report_warning(
                "performance decay", index, serving_name, "serving"
            )

    elif status == "shadow":
        fail = False
        # 两个数据库的分析结束后需要删除bind amount中的数据
        # 1) 数据库拼接(serving+candidate)
        with thread_lock:
            serving_name = serving[index]
            candidate_name = candidate[index]
        db_dir_s = database_merge(serving_name)
        db_dir_c = database_merge(candidate_name)
        clear_labels(index)
        # 2) 匹配、指标计算
        metric_s, _, _ = metric_analysis(db_dir_s, index, False, False)
        metric_c, _, error = metric_analysis(db_dir_c, index, False, True)
        # 3) 将临时数据库添加入对应任务的database中
        database_output(db_dir_s, db_dir_c, index, "shadow")
        # 4) 全局变量操作
        with thread_lock:
            s_metrics[index].append(metric_s)
            c_metrics[index].append(metric_c)
            sample_num[index] = (
                sample_num[index] + 1000
            )  # 这里的1000可能会被改 取决于deploy容器进行通知的频率
            count = sample_num[index]
        # 5) 进入临界条件的判断:
        if error > critical_err:  # 首先判断错误率 若超标 直接进入部署撤销流程
            notify("deploy", index, "normal")
            fail = True
            # 等待立即进入shadow fail的标准阶段
        elif count >= critical_sample[index]:  # 进入临界判断状态
            s_metric_avg = np.mean(s_metrics[index]) if len(s_metrics[index]) > 0 else 0
            c_metric_avg = np.mean(c_metrics[index]) if len(c_metrics[index]) > 0 else 0
            if (
                c_metric_avg - s_metric_avg >= t[index]
            ):  # 通过online evaluation 从shadow进入canary阶段
                notify("deploy", index, "canary")
                notify("frontend", index, "canary")
                with thread_lock:
                    stage[index] = "canary"
                    s_metrics[index], c_metrics[index] = (
                        [],
                        [],
                    )  # 重置指标 供canary阶段重新计算
                    sample_num[index] = 0
            else:  # 未通过测试 进入shadow fail标准阶段
                notify("deploy", index, "normal")
                fail = True
                # 等待立即进入shadow(online evaluation) fail的标准阶段
        if fail:
            # $ 报告给prometheus - candidate模型错误率超标:(下面的代码可能要改)
            prometheus_agent.report_warning(
                "excessive error", index, candidate_name, "candidate"
            )
            with thread_lock:
                stage[index] = "normal"
                # s_metric[index] = []  这行代码可以不用 因为既然已经失败了 可以保留serving记录 其实不用清除
                c_metrics[index] = []  # 重置 为下一次做准备
                sample_num[index] = 0
                # 不需要对数据漂移的监控指标[model_index]进行初始化

    elif status == "canary":
        # 不再需要计算两者的指标 因为serving和candidate模型收到的流量不同
        # 1) 数据库拼接、删除label数据:
        with thread_lock:
            serving_name = serving[index]
            candidate_name = candidate[index]
        db_dir_s = database_merge(serving_name)
        db_dir_c = database_merge(candidate_name)
        clear_labels(index)
        # 2) 不需要进行指标计算 将临时数据库添加入对应任务的database中
        database_output(db_dir_s, db_dir_c, index, "canary")
        # 3) 全局指标变量获取:
        with thread_lock:
            err_info = error_[index]
        # 4) 临界条件判断 并执行对应操作:
        if err_info:
            # 由于warning线程已及时汇报 此处不再需要向prometheus上报
            notify("deploy", index, "normal")
            with thread_lock:
                stage[index] = "normal"
                # 不需要对数据漂移的监控指标[model_index]进行初始化

    else:
        pass  # 不可能出现的情况 为了保持代码完整性 写pass在这里


@app.route("/warning", methods=["POST"])
def warning():
    # 来自前端的http请求 主要有两种缘由:
    # 1) 某个normal阶段的模型 前端统计的错误率超过e=2% 需要由monitor容器触发ETL(为避免冲突 仅monitor有此操作权限)
    # 2) 在canary阶段 candidate模型在前端统计的错误率超过e=2% 需要通知deploy及时下架此candidate模型以及执行系列操作
    # (极小概率下)如果两个模型恰好同时告警 那么前端容器会发两次post 而不是在一次post中放置两个信息
    global serving, candidate, stage, error_
    data = request.get_json(force=True)
    ind = data.get("index")
    if ind is None:
        logging.error("No error type provided in the request.")
        return "No error type provided", 400
    # model_type = data.get("type") # model_type在这里有两种: "serving"/"candidate"

    # 来自font-end前端容器的错误率超标标识
    try:
        if 0 <= ind <= 2 and isinstance(ind, int):
            with thread_lock:
                error_[ind] = True
                current_status = stage[ind]
                serving_name = serving[ind]
                candidate_name = candidate[ind]
            # 使用线程锁修改error_对应的模型为True (等待下一次monitor被触发时 检验到此状态变量 并执行对应操作)
            # 为什么不直接进行修改其他全局变量?
            # 1. 防止该线程对于众多全局变量的修改正好与monitor线程冲突 (即使两个事件同时发生具有极小概率)
            # 2. 直接进行修改 会使得部分数据库没有来得及被搬运至monitor data卷供后续新模型训练
            # $ 向prometheus汇报此error信息 汇报时不需要使用线程锁:
            if current_status == "normal":
                prometheus_agent.report_warning(
                    "excessive error", ind, serving_name, "serving"
                )

            elif current_status == "canary":
                prometheus_agent.report_warning(
                    "excessive error", ind, candidate_name, "candidate"
                )

        else:
            logging.error(f"Error index type out of range: {ind}")
            return "Error index type out of range", 400
    except Exception as e:
        logging.exception(
            f"An unexpected error occurred: {type(e).__name__} - {str(e)}"
        )
        return "Internal Server Error", 500

    return "", 200


# ---------------- Main ----------------
if __name__ == "__main__":
    init()
    prometheus_agent = UpdateAgent()
    app.run(
        host="0.0.0.0", port=8000, threaded=True
    )  # 外部触发的关于模型部署的监控 线路3 主路
