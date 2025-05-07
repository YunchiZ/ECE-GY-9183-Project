import threading, numpy as np
from fastapi import FastAPI, Response
from starlette.types import ASGIApp, Scope, Receive, Send
# from pydantic import BaseModel
from contextlib import asynccontextmanager
# from typing import Literal, Dict, Any, List
# import prometheus_client as prom
from prometheus_client import Histogram, generate_latest, CONTENT_TYPE_LATEST, Gauge
from pydantic import BaseModel, constr, field_validator, Field
from db_opt import *      # 包含SQLite数据库操作所需要的全部函数
from func import *        # 包含了向外部docker发送信息&改写model_status.json的函数
from preprocess import *  # 包含了API响应时对信息进行过滤及tokenize的函数
from Triton import *      # 包含了Triton推理以及Triton权重装载的函数


train_data_dir    = '/app/models'
deploy_data_dir   = '/app/deploy_data'


# ---------------- Logging Initialization ----------------
log_file = os.path.join(train_data_dir, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file),logging.StreamHandler()]
)

# --------------------------
# | ID | Model  |  Mission |
# --------------------------
# | 0  |  BART  |  Summary |
# | 1  |  XLNet | Classify |
# | 2  |  BERT  | Identify |
# --------------------------

# ---------------- Global Variables ----------------
MAIN_LOOP: asyncio.AbstractEventLoop | None = None
status_lock = threading.Lock()
serving_name: List[str] = ["BART-v1", "XLN-v1", "BERT-v1"]
candidate_name: List[Optional[str]] = [None, None, None]
stage: List[str] = ["normal", "normal", "normal"]

p_list = [0.2, 0.4, 0.6, 0.8, 1.0]
BATCH = 500  # record model prediction results every BATCH samples
factor = 3
# 例如 factor=3 那么意味着 BATCH*factor=1500 API在canary阶段需要总共经过1500个样本才能从一个p值跳到下一个p值
# 所以canary阶段则需要总共6000的样本来完成

# ---------------- Results & SQlite ----------------
batch_lock0 = threading.Lock()
batch_lock1 = threading.Lock()
batch_lock2 = threading.Lock()
count_lock0 = threading.Lock()
count_lock1 = threading.Lock()
count_lock2 = threading.Lock()
api_lock    = threading.Lock()

api_call_count_0, api_call_count_1, api_call_count_2 = 0, 0, 0
# api call count的计算不是随时增加的 而是根据缓冲buffer进行计算的
result0_buffer, result1_buffer, result2_buffer = [], [], []
# 用于装载500条serving结果的列表  # 已启用线程锁batch lock进行保护
_result0_buffer, _result1_buffer, _result2_buffer = [], [], []
# 项目中API数据录入的机制详细阐述:
# 在每次API调用的时候 首先模型按顺序进行预测 并且立即返回给API
# 然后使用record_result对这些结果进行依次录入
# 为什么不能针对每个结果单独录入?? 因为API的调用需要效率 在高频使用batch_lock的情况 会使得程序阻塞
# 所以先返回API的结果给前端用户 再进行结果的统一录入


# ---------------- Initialization ----------------
def init():  # 所有的初始化内容
    """
        Initialize global variable `serving` for the API
        & establish corresponding database
    """
    # 一次性初始化所有数据库和表单
    try:
        init_db()
        logging.info(f"Database initialized for databases successfully.")
    except Exception as e:
        logging.error(f"Error occurred in initializing databases: {e}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()
    logging.info(f"MAIN_LOOP initialized: {MAIN_LOOP}")
    global_status_visual(stage, serving_name, candidate_name)
    init()   # 应用启动前调用
    try:
        yield  # 应用运行期
    finally:
        pass # 可选收尾：关闭 DB 连接、flush 日志…

app = FastAPI(title="deploy-gateway", version="1.0", lifespan=lifespan)


# ---------------- Timer for TTLB ----------------
# TTLB: time - to - last - byte
# create ASGI middleware - Asynchronous Server Gateway Interface
# ASGI的本质是将APP应用程序包裹起来
class LatencyMiddleware:
    """
    进入 handler 时记 start，
    直到 send() 最后一块 body (more_body == False) 时 observe(cost)。
    """
    def __init__(self, application: ASGIApp, observer):
        self.app = application   # 被包裹的 FastAPI 内核
        self.observer = observer # 任何可调用对象，eg. Histogram.observe

    async def __call__(self, scope: Scope, receive: Receive, send: Send): # 协程函数
        """
        scope: 请求的元数据 (method, path, headers…)
        receive: await receive() 获取客户端数据
        send: await send(message) 发送响应
        """
        if scope["type"] != "http":       # 只统计 HTTP 请求
            return await self.app(scope, receive, send)

        start = time.perf_counter()

        async def send_wrapper(message):
            # Starlette / Uvicorn 会把响应拆成若干块 body
            if (message["type"] == "http.response.body"
                    and not message.get("more_body", False)):
                cost = time.perf_counter() - start
                self.observer(cost)       # <── 记录 TTLB 到 prometheus
            await send(message)

        return await self.app(scope, receive, send_wrapper)

# ---------------- Prometheus Visualization ----------------
# ttlb: time length between "just when http receive" and "last byte been sent" - a precise metric
ttlb_hist = Histogram("predict_ttlb_seconds","Time-To-Last-Byte of /predict")
# client  →  Middleware(timer on)  →  FastAPI 路由  → timer on → 业务代码 → |
# client  ←  Middleware(timer off) ←  FastAPI 路由  ← time off ← 业务代码 ← |

# service_hist: time length between "after http receive" and "before return result" - a rough metric
# service_hist = Histogram("predict_service_time_seconds","handler enter -> before return")
# candidate_hist = Histogram("candidate_latency_seconds","shadow model latency")
infer_hist = Histogram("model_inference_seconds","Latency of model inference",
                        labelnames=("model", "role"))   # role = serving | candidate
# client  →  Middleware  →  FastAPI 路由  → timer on → 业务代码 → |
# client  ←  Middleware  ←  FastAPI 路由  ← time off ← 业务代码 ← |

# 注册可插拔钩子 - 中间件 LatencyMiddleware:
app.add_middleware(LatencyMiddleware, observer=ttlb_hist.observe)  # type: ignore

task_stage_gauge = Gauge("task_stage_info","Current stage of each task",
                         labelnames=("task", "stage"))
serving_model_gauge = Gauge("serving_model_info","Current serving model of each task",
                            labelnames=("task", "model"))
candidate_model_gauge = Gauge("candidate_model_info","Current candidate model of each task",
                              labelnames=("task", "model"))

TASK_NAMES = ("summary", "classification", "identification")
def global_status_visual(stage_lst: list[str],
                           serving_lst: list[str],
                           candidate_lst: list[str | None]) -> None:
    """
    把当前全局状态写到 Prometheus。
    应当在已获取 status_lock、并完成全局变量更新后调用。
    """
    # 先清空旧 label，避免留下陈旧时间序列
    task_stage_gauge.clear()
    serving_model_gauge.clear()
    candidate_model_gauge.clear()

    for idx, task in enumerate(TASK_NAMES):
        task_stage_gauge.labels(task=task, stage=stage_lst[idx]).set(1)
        serving_model_gauge.labels(task=task, model=serving_lst[idx]).set(1)
        cand_val = candidate_lst[idx] if candidate_lst[idx] is not None else "None"
        candidate_model_gauge.labels(task=task, model=cand_val).set(1)


# ---------------- Associated Functions ----------------

def record_result(serving_load, candidate_load, status):  # 纯python逻辑 设置为同步def 不需要为异步
    # 统一行为函数 而非特定index引导下的函数
    """
    每次API调用之后会产生三个结果组(结果组需要主程序封装成payload) load0~load2 其样式为json: {"":"", "":""}
    使用该函数将结果录入pred中 并检测是否需要进行内存释放以及monitor通知

    这里的status需要在传入之前由主程序使用status lock获取全局变量stage
    因为status不仅决定了不同任务的数据库录入行为 还需要通过http传给monitor容器

    batch_locks用于保护各自的result_buffer全局变量以及candidate预测结果的全局变量_result_buffer
    以下内容不能够被整合简化为函数形式 因为那样会因为锁共享而限制API的性能
    """
    global result0_buffer, result1_buffer, result2_buffer, _result0_buffer, _result1_buffer, _result2_buffer
    global api_call_count_0, api_call_count_1, api_call_count_2
    global stage, serving_name, candidate_name

    # 如果status[index] = "normal": 仅录入serving_load 触发的数量阈值为BATCH
    # 如果status[index] = "shadow" 需要同时录入serving_load以及candidate_load中对应内容(使用index访问) 触发的数量阈值为BATCH
    # 如果status[index] = "canary" 需要同时录入serving_load以及candidate_load中对应内容 触发的数量阈值为动态计算


    threshold = False
    with batch_lock0:
        # step1: 首先录入结果到对应中间缓存区
        if serving_load[0]:
            result0_buffer.append(serving_load[0])
        buffer_length = len(result0_buffer)
        if status[0] in ("shadow", "canary"):
            if candidate_load[0]:
                _result0_buffer.append(candidate_load[0])
            _buffer_length = len(_result0_buffer)
        # step2: 判断缓存长度是否达到或超过阈值 若满足条件则需要释放内存
        if (status[0] in ("shadow", "normal")) and buffer_length > BATCH:
            threshold = True  # 先写成标志 这里需要有对应的操作
        elif status[0] == "canary":
            length = buffer_length + _buffer_length
            # if length >= BATCH*factor:  #
            if length >= BATCH: # 这里还是应该用BATCH而不是上面的BATCH*factor 这样的设置:
                # 使得从始至终 monitor每次对单个任务进行数据库融合的样本总数量维持在BATCH=500个
                # 注意 在canary阶段 monitor只进行数据融合 不做任何数据分析和告警判断 仅接受来自前端的统计告警
                with count_lock0:
                    api_call_count_0 += BATCH # 这个api_call_count在后面也会被用到
                    # 原本打算在api调用时即时更新api_count 现在改用每总共500个样本流入 就更新一次
                    # 至于如何更新p值 不是属于这里的任务
                    if api_call_count_0 >= 6000:
                        if MAIN_LOOP:
                            asyncio.run_coroutine_threadsafe(canary_to_normal(0), MAIN_LOOP)
                        else:
                            logging.error("MAIN_LOOP not ready, skip canary→normal trigger")
                threshold = True
        if threshold:
            if status[0] in ("shadow", "canary"):
                _pred = _result0_buffer.copy()
                _result0_buffer = []
                threading.Thread(target=result_to_db, args=(True, _pred, 0, None,)).start()
            pred = result0_buffer.copy()
            result0_buffer = []
            threading.Thread(target=result_to_db, args=(False, pred, 0, status[0],)).start()
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
                            asyncio.run_coroutine_threadsafe(canary_to_normal(1), MAIN_LOOP)
                        else:
                            logging.error("MAIN_LOOP not ready, skip canary→normal trigger")
                threshold = True
        if threshold:
            if status[1] in ("shadow", "canary"):
                _pred = _result1_buffer.copy()
                _result1_buffer = []
                threading.Thread(target=result_to_db, args=(True, _pred, 1, None,)).start()
            pred = result1_buffer.copy()
            result1_buffer = []
            threading.Thread(target=result_to_db, args=(False, pred, 1, status[1],)).start()
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
                            asyncio.run_coroutine_threadsafe(canary_to_normal(2), MAIN_LOOP)
                        else:
                            logging.error("MAIN_LOOP not ready, skip canary→normal trigger")
                threshold = True
        if threshold:
            if status[2] in ("shadow", "canary"):
                _pred = _result2_buffer.copy()
                _result2_buffer = []
                threading.Thread(target=result_to_db, args=(True, _pred, 2, None,)).start()
            pred = result2_buffer.copy()
            result2_buffer = []
            threading.Thread(target=result_to_db, args=(False, pred, 2, status[2],)).start()


def result_to_db(cand, pred, index, status):
    """
    将释放的内存全部存入对应的数据库中 但在存入之前需要对其进行重置
    """
    # 1) 靶定对象 目标数据库及表单
    # table_name = "task" + str(index)
    if cand:
        db_filepath = os.path.join(deploy_data_dir, "candidate.db")
    else:
        db_filepath = os.path.join(deploy_data_dir, "serving.db")

    # 2) 重置该db_filepath数据库中对应的tabel_name名称的表单 注意是重置不是直接清空
    reset_table(index, db_filepath)  # 该函数在db_opt中

    # 3) 将结果批量写入对应数据库中
    batch_write(pred, index, db_filepath)

    # 4) 通知monitor容器进行监控
    try:
        asyncio.run(notify_docker("monitor", "monitor", {"index":index, "status": status}))
    except Exception as e:
        logging.warning("notify_docker failed: %s", e)


async def canary_to_normal(ind: int):
    """
    对应任务的canary到normal的转换 需要在deploy容器内自行完成
    也就是若无monitor容器阻拦 当api调用次数达到6000次时 自动从normal阶段步入canary阶段
    0. 将api_call_count重置为0 (已在record_result中生效)
    1. 立即使用全局锁更改全局变量(必须的首要执行任务) 防止再次产生candidate API调用
    2. 通知Triton容器对旧serving模型进行下线 等待其结果 (此时恰好留出时间让result_to_db通知monitor进行结果的监控分析)
    3. 通知monitor容器关于阶段转换
    4. 最无关紧要的 使用 status_modify函数修改model_status.json及删除两个版本前模型的权重文件
    由于父线程record_result会自动对buffer进行清理 所以无需在此处进行清理
    也不需要对其对应的数据库表单进行重置 因为每次batch write写入之前程序会自动重置表单 况且按照时间逻辑 此时monitor容器正在读取分析该表单
    """
    global stage, serving_name, candidate_name
    with status_lock:
        old_model = serving_name[ind]
        serving_name[ind] = candidate_name[ind]
        candidate_name[ind] = None
        stage[ind] = "normal"
        stage_ss = stage.copy()
        serving_ss = serving_name.copy()
        candidate_ss = candidate_name.copy()
    global_status_visual(stage_ss, serving_ss, candidate_ss)
    old_model_name, old_model_version = split_model_tag(old_model)
    # 等待Triton对结果进行下线
    await triton_config(old_model_name, old_model_version, "unload")
    await notify_docker("monitor", "init", {"type":"serving", "model":old_model, "index":ind})
    status_modify("replace", old_model, ind)


# ---------------- Flask App Routes ----------------
# Route1 -> Status Transfer
class NotifyPayload(BaseModel):
    type : Literal["shadow", "canary", "normal"]
    index: Literal[0, 1, 2]  # 传 "0"/"1"/"2" 也会被自动转成 int
    model: constr(min_length=3)

@app.post("/notify", status_code=200)
async def notify(payload: NotifyPayload):
    """接收 train / monitor docker 的阶段切换通知"""
    logging.info("Received notify: %s", payload)
    # 不阻塞客户端：把真正逻辑扔给事件循环
    asyncio.create_task(stage_transfer(payload.type,
                                       payload.index,
                                       payload.model))
    return {"msg": "accepted"}

async def stage_transfer(cmd, index, model):
    global serving_name, candidate_name, stage, _result0_buffer, _result1_buffer, _result2_buffer
    assert model is not None

    model_name, version = split_model_tag(model)
    if cmd == "shadow": # Candidate passed offline evaluation, enter shadow stage
        # 所以项目要求train容器整备好所有资源 再通过deploy 同样的deploy也是整备好 再通知monitor
        # _pass = False
        # with api_lock:   # 由于需要进行负载测试 如果全部占用GPU 则意味着API将无法进行工作
        _pass = await asyncio.to_thread(load_test, model)  # 该部分函数需要完善 暂时仅传入模型名称
        # 等待triton-test容器进行负载测试 使用 to_thread丢进线程池中运行 - 必须阻塞拿结果
        if _pass:
            await triton_config(model_name=model_name, version=version, action="load")
            with status_lock:
                candidate_name[index] = model
                stage[index] = "shadow"
                stage_ss = stage.copy()
                serving_ss = serving_name.copy()
                candidate_ss = candidate_name.copy()
            global_status_visual(stage_ss, serving_ss, candidate_ss)
            # asyncio.create_task(triton_config(model_name=model_name, version=version, action="load"))
            # init_db(model) 不需要reset表单 在canary -> normal的时候需要
            transfer_info = {"type":"candidate", "model":model, "index":index}
            # await notify_docker("monitor", "init", transfer_info)
            asyncio.create_task(notify_docker("monitor", "init", transfer_info)) # 无需该结果
            logging.info("Candidate model has passed the load test. Operations: status change & notify monitor")

        else:  # 未通过负载测试 则对model_status.json进行操作 删除对于该模型的注册 并且删除对应的权重文件 并且logging消息记录此次操作
            status_modify("revoke", model, index)   # 删除对应的模型权重文件 并且使用logging进行消息记录
            logging.info("Candidate model failed in the load test. Operations： revoke on model_status.json")

    elif cmd == "canary": # newly trained candidate model has passed through the shadow evaluation
        # enter into canary stage
        with status_lock:
            stage[index] = "canary" # 不需要通知monitor容器
            stage_ss = stage.copy()
            serving_ss = serving_name.copy()
            candidate_ss = candidate_name.copy()
        global_status_visual(stage_ss, serving_ss, candidate_ss)
        logging.info("Entered canary stage.")
        # 需要reset内存 尽管此时内存中的candidate结果即将作为canary阶段测试的一部分
        # 但是monitor容器的机制会导致对candidate和serving的结果都进行记录匹配 而这部分的结果是之前的shadow结果而非流量分流的结果
        # 所以这样会导致monitor容器将样本结果进行两次记录到ETL中   这里选择清空candidate内存 保证避免样本重复
        # 但不用清空serving内容 因为需要保证样本不被遗漏地进入ETL  另外 此时的serving & candidate 内存数量不相等本身就是canary的特征
        if index == 0:
            with batch_lock0:
                _result0_buffer = []
        elif index == 1:
            with batch_lock1:
                _result1_buffer = []
        else: # index == 2
            with batch_lock2:
                _result2_buffer = []

    elif cmd == "normal": # newly trained candidate model has failed in shadow/canary evaluation
        # 1） # 先进行阶段的转换 避免API调用产生更多非期望数据
        with status_lock:
            stage[index] = "normal"
            candidate_name[index] = None
            stage_ss = stage.copy()
            serving_ss = serving_name.copy()
            candidate_ss = candidate_name.copy()
        global_status_visual(stage_ss, serving_ss, candidate_ss)
        # 2） 通知Triton对新模型权重进行unload 此处不需要await
        asyncio.create_task(triton_config(model_name=model_name, version=version, action="unload"))
        # 3） 重置内存中的candidate预测结果
        if index == 0:
            with batch_lock0:
                _result0_buffer = []
        elif index == 1:
            with batch_lock1:
                _result1_buffer = []
        else: # index == 2
            with batch_lock2:
                _result2_buffer = []
        # 4） 重初始化对应的candidate数据库中对应的表单(但其实这一步已经不需要了 因为batch write写入之前会自动重置表单中的内容)
        # database_reset = os.path.join(deploy_data_dir, "candidate.db")
        # reset_table(index, database_reset)
        # 5) 删除对应的模型权重 并且在model_status中进行注册
        status_modify("revoke", model, index)
        logging.info("Candidate model failed online evaluation. Reset to normal stage.")
        # 因为指令是来自于monitor容器的 所以不需要再通知monitor


# Route 2 -> API Service

class PredictIn(BaseModel): # BFF轻量化校验
    text: constr(min_length=10*5, max_length=1000*12) = Field(...) # 10~1000 单词; 50~12000 字符
    prediction_id: int = Field(..., gt=0, description="Prediction ID must be a positive integer.")
    # BaseModel是pydantic(用于把json映射成python对象+自动校验的工具)的一个数据类型
    # 如果超过12000个字符串 就会自动返回错误码 通过constr实现
    # noinspection PyMethodParameters
    @field_validator("text", mode="before")
    def clean(cls, v: str) -> str:
        return data_filter(v)  # data_filter source -> preprocess.py

@app.post("/predict")
async def predict(data: PredictIn):
    # fastAPI自动完成json信息的解包 也就是传入的json会被实例化为PredictIn的对象 并赋值给data进入该路由内
    """
    在模型推理部分 该路由不仅需要作为前端的轻量化BFF 还需要对API产生的结果进行记录
    需要在完整获取模型推理结果之后才能应答 完整逻辑链:
    0.1 在端口路由收到信息之后 中间件LatencyMiddleware自动开始工作
    0.2 尽管前端已有一个文本过滤 此处依然需要进行json内容校验和多项过滤 (已在该路由端口自动pydantic实现)
    . 此条路由被触发后 计时器开始打点
    1. 预先封装Triton需要的payload信息组
    2. 使用状态线程锁保护下获取状态全局变量 转移至中间变量 并立即释放线程锁
    3. for循环分别获取三个任务的阶段类型 如果是normal或者shadow 则立即向Triton发送请求 同时进行计时
    4. 等待Triton返回结果 进行校验封装 例如Triton无响应 则需要将结果设置为None  (存在顺序和时间打点逻辑的问题)
    5. 停止计时器 将结果封装至json信息中 马上启用第二线程(candidate推理以及全部结果的记录)
    6. 向前端返回结果从而结束该此触发的逻辑链
    整个流程已经保证在尽可能短的时间内完成必须完成的事情
    第二线程: (命名为predict_) 传入的变量为serving模型的预测结果组、candidate模型的预测结果组、status、candidate名称租
    1. for循环遍历三个任务的阶段类型 如果是shadow阶段 则开始计时 向Triton再次请求对于candidate模型的推理
    2. 收到结果后停止计时 并将结果保存至candidate结果组中 重复1、2直至candidate均测试完毕(极小概率情况下存在3个candidate模型的shadow情况)
    3. 使用record_result函数一次性录入全部结果
    """

    # .json info has been checked when loaded in 传入信息已经经过自动校验
    # -> .json: {"input":"xxxx", "prediction_id":"xxx"}
    # 1. Payload Packing
    # tokenize -> pack
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

    # 3. model inference definition
    async def infer(idx: int):
        # model = models[idx]
        model, m_ver = split_model_tag(srv[idx])  # e.g. ('BART','0')
        ver = int(m_ver)
        # 由于暂时不清楚是哪个模型做最后的推理 生成一个中间变量result_slot
        result_slot = serving_res  # 默认写到 serving
        role = "serving"  # prometheus业务时间观测默认 serving

        # ---------- canary：按 p 概率走 candidate --------------------
        if stg[idx] == "canary":
            if idx == 0:  # 仅读取全局变量的情况下 不需要使用全局线程锁
                n_calls = api_call_count_0
            elif idx == 1:
                n_calls = api_call_count_1
            else:
                n_calls = api_call_count_2
            p = p_list[min(n_calls // (BATCH*factor), len(p_list) - 1)]
            if random.random() < p and cand[idx]:
                ver = int(split_model_tag(cand[idx])[1])
                result_slot = candidate_res
                role = "candidate"
            else:
                logging.info("No candidate model founded in Canary stage.")

        # --------------------- 发送推理 ------------------------------
        with infer_hist.labels(model=model, role=role).time():  # ← 计时点
            try:
                t0 = time.perf_counter()
                rsp = await triton_infer(model, ver, payloads[model])
                t_elapsed = time.perf_counter() - t0
            except Exception as e:
                logging.error("%s v%s infer failed: %s", model, ver, e)
                rsp, t_elapsed = None, -1

        # --------------------- 解析输出 ------------------------------
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
            "text": data.text,  # 3 个任务都存原文
            "pred": pred_text,
            "time": round(t_elapsed, 4)
        }
        frontend_res[model.lower()] = pred_text  # 返回给前端的json信息 注意这里是小写

    # 4. model inference
    await asyncio.gather(*(infer(i) for i in range(3)))

    # 5. task creation
    asyncio.create_task(predict_(orig_text=data.text, pred_id=data.prediction_id, stage_snapshot=stg,
                                 serving_res=serving_res, candidate_res=candidate_res,
                                 candidate_models=cand, payloads=payloads))

    # 6. return to front-end
    return {"prediction_id": data.prediction_id, **frontend_res}


async def predict_(orig_text: str, pred_id: int, stage_snapshot: list[str],
                   serving_res: list[dict], candidate_res: list[dict],
                   candidate_models, payloads: dict):

    async def shadow_infer(idx: int):
        tag = candidate_models[idx]
        if stage_snapshot[idx] == "shadow" and tag:
            model, ver = split_model_tag(tag)

            with infer_hist.labels(model=model, role="candidate").time():  # 计时
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
                "time": round(t_elapsed, 4)
            }

    await asyncio.gather(*(shadow_infer(i) for i in range(3)))

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,  # None = 默认 ThreadPoolExecutor
        record_result,
        serving_res,
        candidate_res,
        stage_snapshot
    )


# prometheus 监控部分 ...
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)




