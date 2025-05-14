import threading, numpy as np

# from threading import Thread
from fastapi import FastAPI, Response
from starlette.types import ASGIApp, Scope, Receive, Send
from contextlib import asynccontextmanager
from prometheus_client import Histogram, generate_latest, CONTENT_TYPE_LATEST, Gauge
from pydantic import BaseModel, constr, field_validator, Field
from db_opt import *  # Contains all the functions needed for SQLite database operations
from func import *  # Contains functions for sending information to external Docker containers & modifying `model_status.json`
from preprocess import *  # Contains functions for filtering and tokenizing information when responding to API
from Triton import *  # Contains Triton inference and Triton weight loading functions


train_data_dir = "/app/models"
deploy_data_dir = "/app/deploy_data"


# ---------------- Logging Initialization ----------------
log_file = os.path.join(train_data_dir, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)

# --------------------------
# | ID | Model  |  Mission |
# --------------------------
# | 0  |  BART  |  Summary |
# | 1  |  XLNet | Identify |
# | 2  |  BERT  | Classify |
# --------------------------

# ---------------- Global Variables ----------------
MAIN_LOOP: asyncio.AbstractEventLoop | None = None
status_lock = threading.Lock()
serving_name: List[str] = ["BART-v1", "XLN-v1", "BERT-v1"]
candidate_name: List[Optional[str]] = [None, None, None]
stage: List[str] = ["normal", "normal", "normal"]

p_list = [0.2, 0.4, 0.6, 0.8, 1.0]
BATCH = 500  # Record model prediction results every BATCH samples
factor = 3
# For example, if factor=3, this means BATCH*factor=1500 API calls are required in the canary phase to move to the next p value.
# Therefore, the canary phase requires a total of 6000 samples to complete.

# ---------------- Results & SQLite ----------------
batch_lock0 = threading.Lock()
batch_lock1 = threading.Lock()
batch_lock2 = threading.Lock()
count_lock0 = threading.Lock()
count_lock1 = threading.Lock()
count_lock2 = threading.Lock()
api_lock = threading.Lock()

api_call_count_0, api_call_count_1, api_call_count_2 = 0, 0, 0
# The calculation of api call count is not incremented immediately but based on the buffer.
result0_buffer, result1_buffer, result2_buffer = [], [], []
# Lists used to store 500 serving results; protected by thread locks `batch_lock`.
_result0_buffer, _result1_buffer, _result2_buffer = [], [], []
# Explanation of the API data recording mechanism:
# For every API call, the model performs inference sequentially and returns the result to the API immediately.
# Then, the results are recorded into the buffer using `record_result`.
# Why not record each result individually? Because API calls need to be efficient; frequent use of `batch_lock` will cause program blocking.
# So, the API returns the result to the front-end user first, and then records the results in a unified manner.


# ---------------- Initialization ----------------
async def init():
    """
    Initialize global variables and databases, and load necessary models.
    """
    # Step 1: Initialize databases
    try:
        init_db()  # Assuming init_db() is a synchronous function
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing the database: {e}")
        return  # If database initialization fails, stop further initialization

    # Step 2: Load models asynchronously
    models_to_load = [
        {"model_name": "BART", "version": 1},
        {"model_name": "XLN", "version": 1},
        {"model_name": "BERT", "version": 1},
    ]

    # for model in models_to_load:
    #     try:
    #         await triton_config(
    #             model_name=model["model_name"], version=model["version"], action="load"
    #         )
    #         logging.info(f"Model {model['model_name']} loaded successfully.")
    #     except Exception as e:
    #         logging.error(f"Error loading model {model['model_name']}: {e}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()
    logging.info(f"MAIN_LOOP initialized: {MAIN_LOOP}")
    global_status_visual(stage, serving_name, candidate_name)

    # 调用 init() 初始化
    try:
        logging.info("Starting application initialization...")
        await init()  # 确保在异步上下文中调用
        logging.info("Initialization completed successfully.")
    except Exception as e:
        logging.error(f"Error during application initialization: {e}")
        raise  # 让 FastAPI 知道初始化失败

    try:
        yield  # Application runtime
    finally:
        logging.info("Application shutdown. Cleaning up resources...")
        # 在这里进行清理操作，比如关闭数据库连接、清理缓存等


app = FastAPI(title="deploy-gateway", version="1.0", lifespan=lifespan)


# ---------------- Timer for TTLB ----------------
# TTLB: time-to-last-byte
# Create ASGI middleware - Asynchronous Server Gateway Interface
# The essence of ASGI is to wrap the APP application
class LatencyMiddleware:
    """
    Record the start time when entering the handler,
    and observe the cost when sending the last body block (more_body == False).
    """

    def __init__(self, application: ASGIApp, observer):
        self.app = application  # Wrapped FastAPI core
        self.observer = observer  # Any callable object, e.g., Histogram.observe

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ):  # Coroutine function
        """
        scope: Metadata of the request (method, path, headers, etc.)
        receive: await receive() to get client data
        send: await send(message) to send response
        """
        if scope["type"] != "http":  # Only count HTTP requests
            return await self.app(scope, receive, send)

        start = time.perf_counter()

        async def send_wrapper(message):
            # Starlette / Uvicorn splits the response into several body blocks
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                cost = time.perf_counter() - start
                self.observer(cost)  # <── Record TTLB to Prometheus
            await send(message)

        return await self.app(scope, receive, send_wrapper)


# ---------------- Prometheus Visualization ----------------
# TTLB: the time length between "just when HTTP received" and "last byte sent" - a precise metric
ttlb_hist = Histogram("predict_ttlb_seconds", "Time-To-Last-Byte of /predict")
# client  →  Middleware(timer on)  →  FastAPI routing  → timer on → business logic → |
# client  ←  Middleware(timer off) ←  FastAPI routing  ← time off ← business logic ← |

# service_hist: the time length between "after HTTP received" and "before returning result" - a rough metric
# service_hist = Histogram("predict_service_time_seconds","handler enter -> before return")
# candidate_hist = Histogram("candidate_latency_seconds","shadow model latency")
infer_hist = Histogram(
    "model_inference_seconds",
    "Latency of model inference",
    labelnames=("model", "role"),
)  # role = serving | candidate
# client  →  Middleware  →  FastAPI routing  → timer on → business logic → |
# client  ←  Middleware  ←  FastAPI routing  ← time off ← business logic ← |

# Register pluggable hooks - Middleware LatencyMiddleware:
app.add_middleware(LatencyMiddleware, observer=ttlb_hist.observe)  # type: ignore

task_stage_gauge = Gauge(
    "task_stage_info", "Current stage of each task", labelnames=("task", "stage")
)
serving_model_gauge = Gauge(
    "serving_model_info",
    "Current serving model of each task",
    labelnames=("task", "model"),
)
candidate_model_gauge = Gauge(
    "candidate_model_info",
    "Current candidate model of each task",
    labelnames=("task", "model"),
)

MAX = 65_535
TASK_NAMES = ("summary", "identification", "classification")
_counter = 0
_counter_lock = threading.Lock()


def _next_counter() -> int:
    global _counter
    with _counter_lock:
        _counter = (_counter + 1) % MAX
        return _counter


def global_status_visual(
    stage_lst: list[str], serving_lst: list[str], candidate_lst: list[str | None]
) -> None:
    """
    Write the current global status to Prometheus.
    Should be called after acquiring `status_lock` and completing global variable updates.
    """
    ts = _next_counter()

    for idx, task in enumerate(TASK_NAMES):
        task_stage_gauge.labels(task=task, stage=stage_lst[idx]).set(ts)
        serving_model_gauge.labels(task=task, model=serving_lst[idx]).set(ts)
        cand_val = candidate_lst[idx] if candidate_lst[idx] is not None else "None"
        candidate_model_gauge.labels(task=task, model=cand_val).set(ts)


# ---------------- Associated Functions ----------------


def record_result(
    serving_load, candidate_load, status
):  # Pure Python logic, set as synchronous `def`, no need for async
    # Unified behavior function instead of being guided by a specific index
    """
    After each API call, three result groups (to be encapsulated into payloads by the main program)
    are produced: load0~load2, formatted as JSON: {"":"", "":""}.
    Use this function to record the results into `pred` and check whether memory release
    and monitor notification are required.

    The `status` here needs to be obtained by the main program using `status_lock` to acquire
    the global variable `stage` before passing in.
    This is because `status` not only determines the database recording behavior for different
    tasks but also needs to be passed to the monitor container via HTTP.

    `batch_locks` are used to protect their respective `result_buffer` global variables and
    the global variables for candidate prediction results (`_result_buffer`).
    The following content cannot be consolidated or simplified into a function form because
    doing so would limit API performance due to shared locks.
    """
    global result0_buffer, result1_buffer, result2_buffer, _result0_buffer, _result1_buffer, _result2_buffer
    global api_call_count_0, api_call_count_1, api_call_count_2
    global stage, serving_name, candidate_name

    # If `status[index] = "normal"`: only record `serving_load`, and the triggered threshold is `BATCH`.
    # If `status[index] = "shadow"`: simultaneously record `serving_load` and the corresponding `candidate_load` content.
    # If `status[index] = "canary"`: simultaneously record `serving_load` and the corresponding `candidate_load`,
    #                                but the triggered threshold is dynamically calculated.

    threshold = False
    with batch_lock0:
        # Step 1: First, record the results into the corresponding intermediate buffer
        if serving_load[0]:
            result0_buffer.append(serving_load[0])
        buffer_length = len(result0_buffer)
        if status[0] in ("shadow", "canary"):
            if candidate_load[0]:
                _result0_buffer.append(candidate_load[0])
            _buffer_length = len(_result0_buffer)
        # Step 2: Check whether the buffer length has reached or exceeded the threshold.
        # If the condition is met, memory needs to be released.
        if (status[0] in ("shadow", "normal")) and buffer_length > BATCH:
            threshold = True  # Mark as needing action
        elif status[0] == "canary":
            length = buffer_length + _buffer_length
            # if length >= BATCH*factor:
            if length >= BATCH:  # Use `BATCH` instead of `BATCH*factor`:
                # Ensures that monitor always processes 500 samples per batch during the canary phase.
                # Note: In the canary phase, the monitor only performs data merging,
                # without any analysis or alarms. It only accepts statistical alarms from the front end.
                with count_lock0:
                    api_call_count_0 += (
                        BATCH  # Update `api_call_count` every 500 samples
                    )
                    if api_call_count_0 >= 6000:
                        if MAIN_LOOP:
                            asyncio.run_coroutine_threadsafe(
                                canary_to_normal(0), MAIN_LOOP
                            )
                        else:
                            logging.error(
                                "MAIN_LOOP not ready, skip canary→normal trigger"
                            )
                threshold = True
        if threshold:
            if status[0] in ("shadow", "canary"):
                _pred = _result0_buffer.copy()
                _result0_buffer = []
                threading.Thread(
                    target=result_to_db,
                    args=(
                        True,
                        _pred,
                        0,
                        None,
                    ),
                ).start()
            pred = result0_buffer.copy()
            result0_buffer = []
            threading.Thread(
                target=result_to_db,
                args=(
                    False,
                    pred,
                    0,
                    status[0],
                ),
            ).start()
    threshold = False

    with batch_lock1:
        if serving_load[1]:
            result1_buffer.append(serving_load[1])
        buffer_length = len(result1_buffer)
        if status[1] in ("shadow", "canary"):
            if candidate_load[1]:
                _result1_buffer.append(candidate_load[1])
            _buffer_length = len(_result1_buffer)
        if (status[1] in ("shadow", "normal")) and buffer_length > BATCH:
            threshold = True
        elif status[1] == "canary":
            length = buffer_length + _buffer_length
            if length >= BATCH:
                with count_lock1:
                    api_call_count_1 += BATCH
                    if api_call_count_1 >= 6000:
                        if MAIN_LOOP:
                            asyncio.run_coroutine_threadsafe(
                                canary_to_normal(1), MAIN_LOOP
                            )
                        else:
                            logging.error(
                                "MAIN_LOOP not ready, skip canary→normal trigger"
                            )
                threshold = True
        if threshold:
            if status[1] in ("shadow", "canary"):
                _pred = _result1_buffer.copy()
                _result1_buffer = []
                threading.Thread(
                    target=result_to_db,
                    args=(
                        True,
                        _pred,
                        1,
                        None,
                    ),
                ).start()
            pred = result1_buffer.copy()
            result1_buffer = []
            threading.Thread(
                target=result_to_db,
                args=(
                    False,
                    pred,
                    1,
                    status[1],
                ),
            ).start()
    threshold = False

    with batch_lock2:
        if serving_load[2]:
            result2_buffer.append(serving_load[2])
        buffer_length = len(result2_buffer)
        if status[2] in ("shadow", "canary"):
            if candidate_load[2]:
                _result2_buffer.append(candidate_load[2])
            _buffer_length = len(_result2_buffer)
        if (status[2] in ("shadow", "normal")) and buffer_length > BATCH:
            threshold = True
        elif status[2] == "canary":
            length = buffer_length + _buffer_length
            if length >= BATCH:
                with count_lock2:
                    api_call_count_2 += BATCH
                    if api_call_count_2 >= 6000:
                        if MAIN_LOOP:
                            asyncio.run_coroutine_threadsafe(
                                canary_to_normal(2), MAIN_LOOP
                            )
                        else:
                            logging.error(
                                "MAIN_LOOP not ready, skip canary→normal trigger"
                            )
                threshold = True
        if threshold:
            if status[2] in ("shadow", "canary"):
                _pred = _result2_buffer.copy()
                _result2_buffer = []
                threading.Thread(
                    target=result_to_db,
                    args=(
                        True,
                        _pred,
                        2,
                        None,
                    ),
                ).start()
            pred = result2_buffer.copy()
            result2_buffer = []
            threading.Thread(
                target=result_to_db,
                args=(
                    False,
                    pred,
                    2,
                    status[2],
                ),
            ).start()


def result_to_db(cand, pred, index, status):
    """
    Write all released memory to the corresponding database, but reset it before writing.
    """
    # 1) Target the object: the target database and table
    # table_name = "task" + str(index)
    if cand:
        db_filepath = os.path.join(deploy_data_dir, "candidate.db")
    else:
        db_filepath = os.path.join(deploy_data_dir, "serving.db")

    # 2) Reset the corresponding table in the `db_filepath` database. Note: reset, not direct clearing
    reset_table(index, db_filepath)  # This function is in `db_opt`

    # 3) Batch write results into the corresponding database
    batch_write(pred, index, db_filepath)

    # 4) Notify the monitor container for monitoring
    try:
        asyncio.run(
            notify_docker("monitor", "monitor", {"index": index, "status": status})
        )
    except Exception as e:
        logging.warning("notify_docker failed: %s", e)


async def canary_to_normal(ind: int):
    """
    Transition a task from the canary stage to the normal stage.
    This must be completed within the deploy container itself.

    Actions when API call count reaches 6000 (triggered in `record_result`):
    0. Reset `api_call_count` to 0 (already implemented in `record_result`).
    1. Immediately use the global lock to modify global variables (this must be done first to prevent further candidate API calls).
    2. Notify the Triton container to unload the old serving model and wait for the result
       (this gives time for `result_to_db` to notify the monitor container for monitoring and analysis of results).
    3. Notify the monitor container about the stage transition.
    4. Finally, use the `status_modify` function to update `model_status.json` and delete the weights for the model two versions ago.

    Note: The parent thread (`record_result`) will automatically clean up the buffer,
    so there is no need to clean it here. Also, don't reset the corresponding database table,
    as `batch_write` will automatically reset the table before writing.
    """
    global stage, serving_name, candidate_name
    with status_lock:
        old_model = serving_name[ind]
        new_serving = candidate_name[ind]
        serving_name[ind] = new_serving
        candidate_name[ind] = None
        stage[ind] = "normal"
        stage_ss = stage.copy()
        serving_ss = serving_name.copy()
        candidate_ss = candidate_name.copy()
    global_status_visual(stage_ss, serving_ss, candidate_ss)
    old_model_name, old_model_version = split_model_tag(old_model)
    # Wait for Triton to unload the results
    await triton_config(old_model_name, old_model_version, "unload")
    await notify_docker(
        "monitor", "init", {"type": "serving", "model": new_serving, "index": ind}
    )
    weight_clean("replace", old_model)


# ---------------- Flask App Routes ----------------
# Route1 -> Status Transfer
class NotifyPayload(BaseModel):
    type: Literal["shadow", "canary", "normal"]
    index: Literal[
        0, 1, 2
    ]  # Passing "0", "1", or "2" will automatically be converted to int
    model: constr(min_length=3)


@app.post("/notify", status_code=200)
async def notify(payload: NotifyPayload):
    """Receive stage transition notifications from train/monitor Docker containers."""
    logging.info("Received notify: %s", payload)
    # Non-blocking for the client: toss the real logic into the event loop.
    asyncio.create_task(stage_transfer(payload.type, payload.index, payload.model))
    return {"msg": "accepted"}


async def stage_transfer(cmd, index, model):
    global serving_name, candidate_name, stage, _result0_buffer, _result1_buffer, _result2_buffer
    assert model is not None

    model_name, version = split_model_tag(model)
    if cmd == "shadow":  # Candidate passed offline evaluation, enters the shadow stage
        # The project requires the train container to prepare all resources before notifying deploy.
        # Similarly, deploy must prepare everything before notifying the monitor.

        ok = await asyncio.to_thread(download_candidate_weights, model)
        if not ok:
            logging.error(
                "[stage_transfer] unable to download candidate weight '%s', abort.",
                model,
            )
            return

        _pass = await asyncio.to_thread(
            load_test, model
        )  # Wait for load test (run in thread pool)
        if _pass:
            await triton_config(model_name=model_name, version=version, action="load")
            with status_lock:
                candidate_name[index] = model
                stage[index] = "shadow"
                stage_ss = stage.copy()
                serving_ss = serving_name.copy()
                candidate_ss = candidate_name.copy()
            global_status_visual(stage_ss, serving_ss, candidate_ss)
            transfer_info = {"type": "candidate", "model": model, "index": index}
            asyncio.create_task(
                notify_docker("monitor", "init", transfer_info)
            )  # No result needed
            logging.info(
                "Candidate model has passed the load test. Operations: status change & notify monitor."
            )
        else:  # Failed load test: unregister from model_status.json, delete weights, and log the operation
            weight_clean("revoke", model)
            logging.info(
                "Candidate model failed in the load test. Operations: revoke on model_status.json."
            )

    elif cmd == "canary":  # Newly trained candidate model passed shadow evaluation
        # Enter the canary stage
        with status_lock:
            stage[index] = "canary"  # No need to notify the monitor container
            stage_ss = stage.copy()
            serving_ss = serving_name.copy()
            candidate_ss = candidate_name.copy()
        global_status_visual(stage_ss, serving_ss, candidate_ss)
        logging.info("Entered canary stage.")
        # Reset memory because the candidate results in memory are from the shadow stage, not traffic-split results.
        # To avoid duplicate sample storage in ETL, clear candidate memory here.
        if index == 0:
            with batch_lock0:
                _result0_buffer = []
        elif index == 1:
            with batch_lock1:
                _result1_buffer = []
        else:  # index == 2
            with batch_lock2:
                _result2_buffer = []

    elif cmd == "normal":  # Candidate failed shadow/canary evaluation
        # 1) Transition stage first to prevent further non-expected data generation
        with status_lock:
            stage[index] = "normal"
            candidate_name[index] = None
            stage_ss = stage.copy()
            serving_ss = serving_name.copy()
            candidate_ss = candidate_name.copy()
        global_status_visual(stage_ss, serving_ss, candidate_ss)
        # 2) Notify Triton to unload the new model weights (no need to wait here)
        asyncio.create_task(
            triton_config(model_name=model_name, version=version, action="unload")
        )
        # 3) Reset candidate prediction results in memory
        if index == 0:
            with batch_lock0:
                _result0_buffer = []
        elif index == 1:
            with batch_lock1:
                _result1_buffer = []
        else:  # index == 2
            with batch_lock2:
                _result2_buffer = []
        # 4) Reset the candidate table in the candidate database (not necessary as `batch_write` already resets)
        # 5) Delete model weights and unregister from model_status.json
        weight_clean("revoke", model)
        logging.info("Candidate model failed online evaluation. Reset to normal stage.")


# Route 2 -> API Service


class PredictIn(BaseModel):  # Lightweight validation for BFF
    text: constr(min_length=10 * 5, max_length=1000 * 12) = Field(
        ...
    )  # 10~1000 words; 50~12000 characters
    prediction_id: int = Field(
        ..., gt=0, description="Prediction ID must be a positive integer."
    )

    # BaseModel is a Pydantic type (used for mapping JSON to Python objects + auto-validation).
    # If the input exceeds 12000 characters, it will automatically return an error code using `constr`.
    # noinspection PyMethodParameters
    @field_validator("text", mode="before")
    def clean(cls, v: str) -> str:
        return data_filter(v)  # `data_filter` source -> preprocess.py


@app.post("/predict")
async def predict(data: PredictIn):
    # FastAPI automatically unpacks JSON information, instantiating it as a PredictIn object assigned to `data`.
    """
    In the model inference section, this route acts as a lightweight BFF (Backend For Frontend)
    and records the results produced by the API.
    The response is only returned after the model prediction results are fully obtained.

    Full logic chain:
    1. Pre-package payload information for Triton.
    2. Acquire global variables under a thread lock, transfer them to intermediate variables, and immediately release the lock.
    3. For each of the three tasks, if the stage is "normal" or "shadow", send a request to Triton immediately while recording time.
    4. Wait for Triton to return the results, validate and package them (e.g., set results to None if Triton is unresponsive).
    5. Stop the timer, package the results into a JSON, and start a second thread for candidate predictions and result recording.
    6. Return the results to the frontend.

    The process ensures the shortest possible response time for essential tasks.
    """

    # .json info has been checked when loaded, passed information is already validated
    # -> .json: {"input":"xxxx", "prediction_id":"xxx"}
    # 1. Payload Packing
    payloads = build_payloads(data.text)

    # 2. Global status snapshot
    global stage, serving_name, candidate_name
    with status_lock:
        stg = stage.copy()
        srv = serving_name.copy()
        cand = candidate_name.copy()
    # models = ["BART", "XLN", "BERT"]
    serving_res: list[Dict[str, Any] | None] = [None] * 3
    candidate_res: list[Dict[str, Any] | None] = [None] * 3
    frontend_res = {}

    # 3. Model inference definition
    async def infer(idx: int):
        # model = models[idx]
        model, m_ver = split_model_tag(srv[idx])  # e.g. ('BART','0')
        ver = int(m_ver)
        # Create an intermediate variable `result_slot` to determine which model performs the final inference
        result_slot = serving_res  # Default to write into `serving`
        role = "serving"  # Prometheus business time observation default: `serving`

        # ---------- Canary: Use probability `p` to route to candidate --------------------
        if stg[idx] == "canary":
            if (
                idx == 0
            ):  # No need for global thread lock when only reading global variables
                n_calls = api_call_count_0
            elif idx == 1:
                n_calls = api_call_count_1
            else:
                n_calls = api_call_count_2
            p = p_list[min(n_calls // (BATCH * factor), len(p_list) - 1)]
            if random.random() < p and cand[idx]:
                ver = int(split_model_tag(cand[idx])[1])
                result_slot = candidate_res
                role = "candidate"
            else:
                logging.info("No candidate model found in the Canary stage.")

        # --------------------- Send Inference ------------------------------
        with infer_hist.labels(model=model, role=role).time():  # <── Timer point
            try:
                t0 = time.perf_counter()
                rsp = await triton_infer(model, ver, payloads[model])
                t_elapsed = time.perf_counter() - t0
            except Exception as e:
                logging.error("%s v%s inference failed: %s", model, ver, e)
                rsp, t_elapsed = None, -1

        # --------------------- Parse Output ------------------------------
        if rsp and model == "BART":
            tokens = np.array(rsp["outputs"][0]["data"])
            pred_text = TOKENS["BART"].decode(tokens, skip_special_tokens=True)
        elif rsp:
            logits = np.array(rsp["outputs"][0]["data"])
            pred_text = int(logits.argmax())
        else:
            pred_text = None

        result_slot[idx] = {
            "id": data.prediction_id,
            "text": data.text,  # Store the original text for all three tasks
            "pred": pred_text,
            "time": round(t_elapsed, 4),
        }
        frontend_res[model.lower()] = (
            pred_text  # Return information to the frontend, note it's in lowercase
        )

    # 4. Model inference
    await asyncio.gather(*(infer(i) for i in range(3)))

    # 5. Task creation
    asyncio.create_task(
        predict_(
            orig_text=data.text,
            pred_id=data.prediction_id,
            stage_snapshot=stg,
            serving_res=serving_res,
            candidate_res=candidate_res,
            candidate_models=cand,
            payloads=payloads,
        )
    )

    # 6. Return to front-end
    return {"prediction_id": data.prediction_id, **frontend_res}


async def predict_(
    orig_text: str,
    pred_id: int,
    stage_snapshot: list[str],
    serving_res: list[dict],
    candidate_res: list[dict],
    candidate_models,
    payloads: dict,
):
    async def shadow_infer(idx: int):
        tag = candidate_models[idx]
        if stage_snapshot[idx] == "shadow" and tag:
            model, ver = split_model_tag(tag)

            with infer_hist.labels(model=model, role="candidate").time():  # Timer
                try:
                    t0 = time.perf_counter()
                    rsp = await triton_infer(model, int(ver), payloads[model])
                    t_elapsed = time.perf_counter() - t0
                except Exception as e:
                    logging.error("shadow %s v%s failed: %s", model, ver, e)
                    rsp, t_elapsed = None, -1

            if rsp and model == "BART":
                tokens = np.array(rsp["outputs"][0]["data"])
                pred = TOKENS["BART"].decode(tokens, skip_special_tokens=True)
            elif rsp:
                pred = int(np.array(rsp["outputs"][0]["data"]).argmax())
            else:
                pred = None

            candidate_res[idx] = {
                "id": pred_id,
                "text": orig_text,
                "pred": pred,
                "time": round(t_elapsed, 4),
            }

    await asyncio.gather(*(shadow_infer(i) for i in range(3)))

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,  # None = Default ThreadPoolExecutor
        record_result,
        serving_res,
        candidate_res,
        stage_snapshot,
    )


# ---------------- Prometheus Monitoring ----------------
@app.get("/metrics")
def metrics():
    """
    Expose Prometheus metrics for monitoring.
    """
    return Response(generate_latest(), mimetype="text/plain")
