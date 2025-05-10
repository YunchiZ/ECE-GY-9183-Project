import logging
import threading
import json
import requests
import os
import sqlite3
import numpy as np
from typing import List

from flask import Flask, Response, request
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score

from prometheus_update import UpdateAgent
from prometheus_client import generate_latest
from minio import Minio

# ！！！！！！！
# The entire code still lacks many exception handling frameworks (try except)
# For example, operations on metrics s_metrics; c_metrics; any file reading and writing, etc.
# The entire code still lacks a large amount of logging output code that needs to be fine-tuned
# The parsing of JSON messages needs to be fully optimized. request.get_json(force=True) has been marked by GPT as problematic
# And we need to add backoff or queue mechanisms?
# Do all database operations need thread locks?
# For some database settings, please refer to deploy_app.py
# The use of global variables throughout the code needs to be checked for thread locks
# !Consider introducing Kafka to manage HTTP message queues (pending)
# Consider using MonitorStateManager class to manage all monitoring states


app = Flask(__name__)
thread_lock = threading.Lock()
status_lock = threading.Lock()
metric_lock = threading.Lock()

etl_url = os.getenv("ETL_URL")
minio_endpoint = os.getenv("MINIO_URL")
# Agent for updating prometheus metrics
update_agent = None


def get_update_agent() -> UpdateAgent:
    """return: UpdateAgent"""
    global update_agent
    if update_agent is None:
        update_agent = UpdateAgent()
    return update_agent


# =====================================
# File addresses:
# LOCK_file = "./.../.../"
# A global variable JSON file for a host. If LOCK=True, it means CI/CD process is in progress and has been locked
# To avoid frequent triggering of CI/CD processes
deploy_data_dir = "/app/deploy_data"


deploy_database = os.path.join(deploy_data_dir, "serving_data.db")
candidate_database = os.path.join(deploy_data_dir, "candidate_data.db")

LOCK_file = os.path.join(deploy_data_dir, "LOCK.json")

# =====================================
# Global Variables
# The logic set here is to simplify the process by starting three model processes during training rather than a single model
# candidate1, candidate2, candidate3 = None, None, None
serving = ["BART-v1", "XLN-v1", "BERT-v1"]
candidate = [None, None, None]
stage = ["normal", "normal", "normal"]  # This stage is tied to the task
# Because candidate models need to go through shadow -> canary -> normal stages
# No need to set stage for serving because its stage status is "serving" itself
# In this project, any model failure or git actions operation will trigger simultaneous training and deployment for all three models, and will lock LOCK until the last model completes its final deployment decision
# Due to sample size limitations, training and deployment may not be in the same period. Order: DistilBERT -> XLNet-> BART
# =====================================
# Threshold Values
# All three values in the list correspond to three models and their tasks [BART, XLNet, DistilBERT][text summarization, fact verification, text classification]
critical_decay = [
    0.05,
    0.05,
    0.05,
]  # This value represents the performance decay of the model. The specific value needs experiment and testing
critical_sample = [
    6000,
    6000,
    6000,
]  # Need to evaluate through x sample results to decide whether to go from shadow to canary or from canary to serving
sample_num = [0, 0, 0]
t = [
    0.02,
    0.03,
    0.03,
]  # The model needs n% performance improvement to be considered passing the test
SLA = 1500  # ms The average response time needs to be less than 500ms, whether it's the inference time of the model in shadow mode or the total response time at the frontend in canary mode
critical_err = 0.02
# Whether in shadow, canary, or normal stage, the error rate must not exceed this ratio
# In shadow stage, since the API does not return the candidate's prediction value, there's no need to immediately remove the candidate. The error rate is analyzed by the monitor container to reduce the burden on the deploy container.
# Error types include: 1) Model inference response time exceeds LSA time (will be recorded in SQLite data structure) 2) Server API unknown error (error code 5xx)
# In canary stage, since the API needs to return the candidate's prediction value, it needs not only real-time performance but also includes frontend errors, so the error rate is analyzed by the frontend container
# Error types include: 1) Frontend response time exceeds LSA time (recorded by the frontend itself) 2) Server API error (error code 5xx) 3) Client http error (4xx)
# serving_err = [0, 0, 0]  # Error rate statistics for three serving models
# candidate_err = [0, 0, 0]  # Error rate statistics for three candidate models
# Since error rate statistics are no longer based on past records, but on each round (every k API calls), these two lists are not used
# Because error rate records based on each round can promptly reflect situations where model error rates suddenly increase (including but not limited to temporary GPU failures, etc.)
# ======================================
# Some metric lists need to be written here
s_metrics: List[List[float]] = [
    [],
    [],
    [],
]  # Temporarily use lists to ensure flexibility for different lengths
# Store metrics records for three models. For text summarization tasks, record ROUGE values. For the other two tasks, record ACC values
c_metrics: List[List[float]] = [[], [], []]
# About metric initialization??? Unknown for now, need analysis functions from ETL side
data_shift_metrics = {"word_counts": [], "label_counts": []}
# ================================================ Section Divider


def write_to_minio(file_path, object_name=None, bucket_name="etl_data"):
    """
    :param file_path: local file path
    :param bucket_name: minio bucket name
    :param object_name: object name on minio"""

    if object_name is None:
        object_name = os.path.basename(file_path)
    access_key = os.getenv("minio-access-key")
    secret_key = os.getenv("minio-secret-key")
    secure = os.getenv("minio-secure", "false").lower() == "true"
    try:
        client = Minio(
            endpoint=minio_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logging.info(f"created bucket '{bucket_name}'")

        if os.path.exists(file_path):
            client.fput_object(
                bucket_name,
                object_name,
                file_path,
            )
            logging.info(f"'{file_path}' uploaded to  '{object_name}'")
        else:
            logging.error(f"Error: file '{file_path}' does not exist")
        return True
    except Exception as e:
        logging.error(f"Error uploading '{file_path}': {e}")
        return False


def read_from_minio(object_name, file_path, bucket_name="frontend"):
    """
    :param object_name: object name on minio
    :param file_path: local file path
    :param bucket_name: minio bucket name"""

    access_key = os.getenv("minio-access-key")
    secret_key = os.getenv("minio-secret-key")
    secure = os.getenv("minio-secure", "false").lower() == "true"
    try:
        client = Minio(
            endpoint=minio_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        client.fget_object(bucket_name, object_name, file_path)
        logging.info(f"'{object_name}' downloaded to '{file_path}'")
        return os.path.abspath(file_path)
    except Exception as e:
        logging.error(f"Error downloading '{object_name}': {e}")
        return False


def notify(docker, index, notification_type):
    # Here index is the task (model) ID, corresponding to text summarization, fact verification, and classification - three NLP tasks
    if docker == "etl":
        response_url = etl_url + "/etl"
    if docker == "deploy":
        response_url = "http://deploy:8000/notify"

    payload = {"type": notification_type, "index": index}
    try:
        response = requests.post(response_url, json=payload, timeout=5)
        logging.info(
            f"Notified {docker} docker with {payload}, status: {response.status_code}"
        )
        return 200
    except requests.RequestException as e:
        logging.error(f"Error notifying {docker} docker: {e}")
        return 500


# ================================================ All database-related functions
# Actually, the frontend container will put labels into different task name databases according to the label ID, so the Monitor container doesn't need to match them
# These functions must ensure:
# 1) Check file existence
# 2) Access probing with try except
# 3) Series of exception handling

# deploy_data schema:
# CREATE TABLE IF NOT EXISTs predictions (id INTEGER PRIMARY KEY AUTOINCREMENT
# input TEXT
# pred TEXT/INTEGER
# time REAL


# label schema:
# id INTEGER PRIMARY KEY
# input TEXT
# label TEXT/INTEGER


def database_merge(type: str):
    """
    return : new_db_path
    """
    global deploy_database, candidate_database
    # index determines task type, create corresponding form
    # $ Determine database access path based on deploy_data_dir and serving[index]
    # $ Determine user behavior label library/file access path based on label_dir and ?

    # with open(LOCK_file_db, "r") as f:
    #     lock_data = json.load(f)
    #     lock = lock_data.get("LOCK", False)
    # while lock:
    #     with open(LOCK_file_db, "r") as f:
    #         lock_data = json.load(f)
    #         lock = lock_data.get("LOCK", False)
    #         time.sleep(1)

    new_db_path = os.path.join(deploy_data_dir, "evaluation.db")

    if type == "candidate":
        merge_database = os.path.join(candidate_database, "candidate_data.db")
    elif type == "serving":
        merge_database = os.path.join(deploy_database, "serving_data.db")

    label_database = read_from_minio("label.db", "label.db", "frontend")

    deploy_conn = sqlite3.connect(merge_database)
    deploy_cur = deploy_conn.cursor()

    label_conn = sqlite3.connect(label_database)
    label_cur = label_conn.cursor()

    deploy_cur.execute(f"ATTACH DATABASE '{label_database}' AS label_db")

    new_conn = sqlite3.connect(new_db_path)
    new_cur = new_conn.cursor()

    # Create three tables in the new database, corresponding to three tasks
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

    base_query = """
    SELECT 
        p.id,
        p.text AS predictions_input,
        p.pred AS predictions_pred,
        l.text AS label_input,
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
        id , predictions_input, predictions_pred, label_input, label_label3
    ) VALUES (?, ?, ?, ?, ?)
    """

    task1_data = [(row[0], row[1], row[2], row[3], row[4]) for row in results]
    task2_data = [(row[0], row[1], row[2], row[3], row[5]) for row in results]
    task3_data = [(row[0], row[1], row[2], row[3], row[6]) for row in results]

    # TO THE SAME DB
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

    write_to_minio(new_db_path)
    # $ (Database behavior) Match and integrate database content according to prediction ID
    # $ Place the new database in database_dir, waiting for the metric_analysis function below to perform stage-customized analysis
    return new_db_path


def metric_analysis(database: str, itype: int, e_analysis: bool = False):
    """return: metric, error_status"""
    metric = 0
    conn = sqlite3.connect(database)
    cur = conn.cursor()

    error_status = False
    # $ Access according to the target database address "database" passed in
    # $ Perform corresponding metric analysis by matching prediction entries with label entries
    try:
        # Task 1: Text summarization evaluation (ROUGE)
        if itype == 0:
            # Get all predictions and labels from task1_data table
            query = "SELECT id, predictions_pred, label_label1 FROM task1_data"
            results = cur.execute(query).fetchall()

            # Initialize ROUGE scorer and score list
            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            rouge_scores = []

            # Calculate ROUGE score for each sample
            for id, pred, label in results:
                if (
                    pred and label
                ):  # Ensure that both prediction and label are not empty
                    scores = scorer.score(label, pred)
                    # Use F1 score as the main metric
                    rouge_f1 = {k: v.fmeasure for k, v in scores.items()}
                    rouge_scores.append(rouge_f1)

            # Calculate average ROUGE score
            avg_scores = {}
            for key in ["rouge1", "rouge2", "rougeL"]:
                avg_scores[key] = np.mean([score[key] for score in rouge_scores])
            # Ratio to be determined
            metric = (
                0.2 * avg_scores["rouge1"]
                + 0.3 * avg_scores["rouge2"]
                + 0.5 * avg_scores["rougeL"]
            )

        # Task 2: Classification task 1 ACC evaluation
        elif itype == 1:
            # Get all predictions and labels from task2_data table
            query = "SELECT id, predictions_pred, label_label2 FROM task2_data"
            results = cur.execute(query).fetchall()

            # Extract predictions and true labels
            y_pred = []
            y_true = []

            for id, pred, label in results:
                # Ensure that prediction value and label are integers
                try:
                    pred_val = int(pred)
                    label_val = int(label)
                    y_pred.append(pred_val)
                    y_true.append(label_val)
                except (ValueError, TypeError):
                    continue

            # Calculate accuracy
            if y_pred and y_true:
                accuracy = accuracy_score(y_true, y_pred)

                metric = accuracy

        # Task 3: Classification task 2 ACC evaluation
        elif itype == 2:
            # Get all predictions and labels from task3_data table
            query = "SELECT id, predictions_pred, label_label3 FROM task3_data"
            results = cur.execute(query).fetchall()

            # Extract predictions and true labels
            y_pred = []
            y_true = []

            for id, pred, label in results:
                # Ensure that prediction value and label are integers
                try:
                    pred_val = int(pred)
                    label_val = int(label)
                    y_pred.append(pred_val)
                    y_true.append(label_val)
                except (ValueError, TypeError):
                    continue  # Skip values that cannot be converted to integers

            # Calculate accuracy
            if y_pred and y_true:
                accuracy = accuracy_score(y_true, y_pred)

                metric = accuracy

        else:
            logging.error("Invalid task type")
            return None, False

        # # $ In the normal stage, need to analyze changes in data distribution:
        # if data_analysis:
        #     # $ Corresponding analysis of data distribution, this function might be derived from rewritten ETL code
        #     # Directly return metrics about data shift rather than data distribution, analyze directly here
        #     data_info = 0
        #     pass

        if e_analysis:

            query = """SELECT COUNT(*) FROM task{}_data 
                    WHERE predictions_pred IS NULL 
                    OR time > {}""".format(
                itype + 1, SLA
            )
            error_count = cur.execute(query).fetchall()[0][0]

            total_query = """SELECT COUNT(*) FROM task{}_data""".format(itype + 1)
            total_count = cur.execute(total_query).fetchall()[0][0]

            # Calculate the current task's error rate
            current_error_rate = error_count / total_count if total_count > 0 else 0

            # Save the calculated error rate to the global array at the corresponding position
            error_rates[itype] = current_error_rate

            error_status = True if current_error_rate > critical_err else False
            # Conditions for judging as error
            # 1. Model inference response time exceeds LSA time
            # 2. pred is none
            # $ Then judge whether error_rate exceeds critical_err value
            # $ If it exceeds, then error_status = True
    finally:
        cur.close()
        conn.close()
        return metric, error_status


def datashift_analysis(database):
    """
    return: word_counts:[] all length of words, label_counts:[]
    """
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    # Input length frequency distribution

    input_query = """select text from task1_data"""
    input_data = cur.execute(input_query).fetchall()

    word_counts = []

    for row in input_data:
        text = row[0]
        if text is not None:
            words = text.split()  # word count instead of char count
            word_count = len(words)
            word_counts.append(word_count)

    # Classification label frequency changes
    label_query = """select pred from task2_data"""
    label_data = cur.execute(label_query).fetchall()

    label_counts = {}

    for row in label_data:
        label = row[0]

        if label is None:
            label = "None"

        # Update count
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    return word_counts, label_counts


@app.route("/init", methods=["POST"])
def init():
    # There are only two types of messages received: candidate / serving
    # 'candidate' means that train triggers deploy, and monitor needs to monitor the database of the corresponding model
    # 'serving' means that the deploy container has completed the canary phase, and monitor needs to monitor the corresponding database
    # Additionally, as mentioned above, it's not possible for three models to be deployed asynchronously, so there is no situation where multiple models need to execute the same instruction at the same time
    # Why is this? Because 1. the train container supports training and offline testing of at most one model at a time, 2. going from shadow to canary to serving requires sample inflow, not just determined by time
    # Even if such a situation truly exists, the deploy container will send 2-3 requests, as init() only executes state modification for one model at a time, and is already protected by thread locks
    # The triggering of init() is unlikely to occur simultaneously with monitor(), but to avoid this small probability situation, thread locks have already been used for protection
    global serving, candidate, stage, s_metrics, c_metrics

    data = request.get_json(force=True)
    message = data.get("type")
    model = data.get("model")  # Here model is the real model name, not an index number
    model_index = data.get("index")

    if not message:
        logging.error("No notification type provided in the request.")
        return "No notification type provided", 400

    if model_index not in [0, 1, 2]:
        logging.error("Invalid model index.")
        return "Invalid model index", 400

    if message == "candidate":
        with status_lock:
            candidate[model_index] = model
            stage[model_index] = "shadow"
        with metric_lock:
            # $ Need to reset serving-related metrics because the shadow phase needs to compare the concurrent performance of serving and candidate:
            s_metrics[model_index] = []
            # Need to reset serving-related metrics, currently there seems to be only one metric
            c_metrics[model_index] = []  # Reset candidate as well
            # Previous serving monitoring records are no longer meaningful because the focus now is on examining the candidate
        # No need to report, deploy will report this deployment to prometheus

    elif message == "serving":
        with status_lock:
            candidate[model_index] = None
            serving[model_index] = model
            stage[model_index] = "normal"
            # $ Migrate the values of various global metrics:
            # For example, overwrite serving with the records from candidate (generated during canary phase) and reset the corresponding position in candidate:
        with metric_lock:
            s_metrics[model_index] = c_metrics[model_index]
            c_metrics[model_index] = []  # Reset
            # No need to additionally select for the "canary" phase because all records will be reset once from shadow to canary
            # $ Also need to initialize the data drift monitoring metric [model_index] to []

    else:
        logging.error(f"Unknown notification type: {message}")
        return "Unknown notification type", 400


@app.route(
    "/monitor", methods=["POST"]
)  # The most complex part of model monitoring, the core mechanism
def monitor():
    global serving, candidate, stage, sample_num, s_metrics, c_metrics, error_, data_shift_metrics
    data = request.get_json(force=True)
    index = data.get("index")
    status = data.get("status")

    # The HTTP message from the deploy container notifying monitor to perform monitoring contains an index key corresponding to the model ID 0~2, which is also the task ID
    # Similarly, one message will not simultaneously request monitoring for two tasks
    # The mechanism of this message: The results produced after each API call in the deploy container will not be immediately written to SQLite, but stored in memory variables
    # When the number of samples in memory reaches, for example, 1000, the previous records will be deleted (assuming the monitor container has already taken them away in the previous round)
    # Then write to deploy data all at once and notify the monitor container via HTTP to monitor this batch of data - read, match, and transfer
    # After obtaining the index, it is also necessary to obtain the status, because the monitor container in this program also needs to maintain in real-time and synchronously the global variables about the model status in the deploy container
    # Only when the status matches the local record can the corresponding monitoring operation be performed. This mechanism is set up to prevent any potential information conflict, although logically, deploy and monitor are already closely interlocked
    # ! Regardless of the status being monitored, make sure the process includes several of the following:
    # 1) Temporary database concatenation (read from deploy data -> cache concatenation -> place in monitor data)
    # 2) Metric calculation
    # 3) Global metric record update
    # 4) Database writing, temporary database deletion
    # 5) Critical condition judgment (error)
    # 6) Container HTTP notification (highest priority) (deploy&frontend&etl)
    # 7) Metric update or reset
    # 8) Global status variable modification / status file modification
    # 9) Report all relevant content to prometheus
    def background_processing():
        if stage[index] != status:  # If mismatched message, report error and ignore
            return "Unmatched model stage status", 400

        if status == "normal":
            # 1) First match and concatenate the corresponding database with the data in the label volume, and place it in monitor data
            db_dir = database_merge("serving")

            # 2) Next, traverse the database for matching, metric calculation, integration, migration;
            metric, err_info = metric_analysis(db_dir, index, True)

            data_shift_metrics["word_counts"], data_shift_metrics["label_counts"] = (
                datashift_analysis(db_dir)
            )

            # 3) Then delete irrelevant entries from the temporary database and add to the corresponding task's database
            # database_output(db_dir, None, index, "normal")
            # 4) Update and store global metric variables:
            # Use thread lock to access metric global variables:
            with thread_lock:
                avg_metric = (
                    np.mean(s_metrics[index]) if len(s_metrics[index]) > 0 else 0
                )
                s_metrics[index].append(metric)

                # $ Access global metric variables about data distribution data_distribution
                # $ Perform similar operations: analyze the average of previous metrics, then add data variable to become part of the record
            data_shift = (
                False  # This is for code completeness, temporarily written this way
            )
            # 5) Then enter critical condition judgment: whether frontend statistical error rate exceeds threshold / whether data drift occurs / whether performance decays
            # Actually, for the efficiency of this part of the process, critical condition judgment has priority, but since all need to be reported to prometheus, all three need to be judged
            # If the critical condition is triggered, trigger ETL according to LOCK.json and alert prometheus
            if data_shift or err_info or (avg_metric - metric < critical_decay[index]):
                with thread_lock:
                    # Read the value of the LOCK key in the LOCK.json file
                    lock = False
                    try:
                        with open(LOCK_file, "r") as f:
                            lock_data = json.load(f)
                            lock = lock_data.get("LOCK", False)
                    except Exception as e:
                        print(f"Error reading lock file: {e}")

                    if lock:
                        code = notify(
                            "etl", None, "trigger"
                        )  # The message content is not important, what matters is triggering
                        if code == 200:  # Indicates ETL was successfully triggered
                            # Modify the LOCK key value in the LOCK.json file to True
                            try:
                                with open(LOCK_file, "r") as f:
                                    lock_data = json.load(f)

                                lock_data["LOCK"] = True

                                with open(LOCK_file, "w") as f:
                                    json.dump(lock_data, f)
                            except Exception as e:
                                print(f"Error modifying lock file: {e}")

        elif status == "shadow":
            fail = False
            # After analyzing the two databases, the data in bind amount needs to be deleted
            # 1) Database concatenation (serving+candidate)
            db_dir_s = database_merge("serving")
            db_dir_c = database_merge("candidate")

            # 2) Matching, metric calculation
            metric_s, _ = metric_analysis(db_dir_s, index, False)
            metric_c, error_status = metric_analysis(db_dir_c, index, True)
            # 3) Add temporary database to the corresponding task's database
            # database_output(db_dir_s, db_dir_c, index, "shadow")
            # 4) Global variable operations
            with metric_lock:
                s_metrics[index].append(metric_s)
                c_metrics[index].append(metric_c)
                sample_num[index] = (
                    sample_num[index] + 1000
                )  # The 1000 here might be changed, depending on the frequency of notification by the deploy container
                count = sample_num[index]
            # 5) Enter critical condition judgment:
            if (
                error_status
            ):  # First judge error rate, if it exceeds the threshold, directly enter deployment rollback process
                notify("deploy", index, "normal")
                fail = True
                # Wait to immediately enter the standard phase of shadow fail
            elif count >= critical_sample[index]:  # Enter critical judgment state
                s_metric_avg = (
                    np.mean(s_metrics[index][:-1]) if len(s_metrics[index]) > 0 else 0
                )
                c_metric_avg = (
                    np.mean(c_metrics[index][:-1]) if len(c_metrics[index]) > 0 else 0
                )
                if (
                    c_metric_avg - s_metric_avg >= t[index]  # t size
                ):  # Pass online evaluation, enter canary phase from shadow
                    notify("deploy", index, "canary")

                    with status_lock:
                        stage[index] = "canary"
                    with metric_lock:
                        s_metrics[index], c_metrics[index] = (
                            [],
                            [],
                        )  # Reset metrics for recalculation in canary phase
                        sample_num[index] = 0
                else:  # Failed the test, enter standard shadow fail phase
                    notify("deploy", index, "normal")

                    fail = True
                    # Wait to immediately enter the standard phase of shadow (online evaluation) fail
            if fail:
                with status_lock:
                    stage[index] = "normal"
                with metric_lock:
                    c_metrics[index] = []  # Reset for next time
                    sample_num[index] = 0
                    # No need to initialize data drift monitoring metric [model_index]

        elif status == "canary":
            # No longer need to calculate metrics of both, as serving and candidate models receive different traffic
            # 1) Database concatenation, delete label data:

            db_dir_s = database_merge("serving")
            db_dir_c = database_merge("candidate")
            # count
            # 2) No need for metric calculation, add temporary database to the corresponding task's database
            # database_output(db_dir_s, db_dir_c, index, "canary")
            # 3) Get global metric variables:
            # First calculate sample num, then judge. If less than critical - batch (batch = 500), just end
            # If greater than or equal, then calculate error rate. If it exceeds threshold, take action

            count = sample_num[index]
            if count < critical_sample[index] - 500:
                return "OK", 200
            else:

                metric_s, error_status = metric_analysis(db_dir_s, index, True)
                metric_c, error_status = metric_analysis(db_dir_c, index, True)

                with metric_lock:
                    s_metrics[index].append(metric_s)
                    c_metrics[index].append(metric_c)
                    sample_num[index] = 0

        else:
            pass

    thread = threading.Thread(target=background_processing)
    thread.daemon = (
        True  # Set as daemon thread, thread will exit when main program exits
    )
    thread.start()
    return "OK", 200


@app.route("/metrics", methods=["GET"])
def metrics():
    global serving, candidate, s_metrics, c_metrics, data_shift_metrics, error_rates
    agent = get_update_agent()

    for i in range((len(serving))):
        agent.update_model_metrics(i, serving[i], s_metrics[i])
        if candidate[i] is not None:
            agent.update_model_metrics(i, candidate[i], c_metrics[i])
            agent.update_error_rate(candidate[i], "candidate", error_rates[i])
        else:
            agent.update_error_rate(serving[i], "serving", error_rates[i])
    agent.update_word_length(data_shift_metrics["word_counts"])
    agent.update_label_frequency(data_shift_metrics["label_counts"])

    return Response(generate_latest(), mimetype="text/plain")


# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
