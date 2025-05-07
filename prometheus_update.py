import logging
import threading
import time
import queue
from prometheus_client import start_http_server
import prometheus_client as prom
from collections import defaultdict


# dashboard

# - 数据偏移
# 输入长度频率分布
# 分类标签频率变化
# 数据质量

# - 模型状态，阶段状态 文本 -----
#
# - 指标
# metrics 折线图-----
# 总响应时间ttlb  折线图----
# service time 折线图-----
# 模型推理时间  折线图 -----
# 错误率-


class UpdateAgent:

    def __init__(self, port=9091, batch_interval=5):
        self.logger = logging.getLogger(__name__)
        self.port = port
        self.batch_interval = batch_interval

        # 创建消息队列
        self.update_queue = queue.Queue()

        self.batch_data = {
            "word_length": [],
            "label_frequency": defaultdict(int),
            "model_status": {},
            "model_metrics": {},
            "response_time": defaultdict(list),
            "service_time": defaultdict(list),
            "inference_time": defaultdict(list),
        }

        # 批处理锁
        self.batch_lock = threading.RLock()

        start_http_server(self.port)
        self.logger.info(f"Started Prometheus metrics server on port {self.port}")

        # 初始化Prometheus指标
        self._init_prometheus_metrics()

        # 启动处理线程
        self._start_worker_threads()

        # 启动批处理线程
        self._start_batch_thread()

        self.logger.info(
            f"UpdateAgent initialized with batch interval of {batch_interval}s"
        )

    def _init_prometheus_metrics(self):
        """初始化所有Prometheus指标"""
        self.word_length_counter = prom.Histogram(
            "data_word_length",
            "Length of input text data",
            buckets=(50, 100, 250, 500, 1000),
        )
        self.label_counter = prom.Gauge(
            "data_label_frequency",
            "Frequency of category labels",
            ["label"],
        )

        self.model_status = prom.Enum(
            "model_deployment_status",
            "Current deployment status of the model",
            ["model_name"],
            states=["canary", "shadow", "serving"],
        )

        self.model_metrics = prom.Gauge(
            "model_metrics",
            "Performance metrics for different models",
            ["model_name", "metric_type"],
        )

        self.response_time = prom.Histogram(
            "model_response_time",
            "Response time of the model",
            ["model_name"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
        )
        self.service_time = prom.Histogram(
            "model_service_time",
            "Service time of the model",
            ["model_name"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
        )
        self.inference_time = prom.Histogram(
            "model_inference_time",
            "inference time of the model",
            ["model_name"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
        )

        # 监控指标
        self.queue_size = prom.Gauge(
            "update_queue_size", "Current size of the update queue"
        )

        self.batch_size = prom.Gauge(
            "batch_size", "Size of current batch by update type", ["update_type"]
        )

        self.batch_processing_time = prom.Histogram(
            "batch_processing_time",
            "Time taken to process a complete batch",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        )

    def _start_worker_threads(self, num_workers=2):
        """启动工作线程来处理队列中的更新请求"""
        self.workers = []
        self.should_stop = threading.Event()

        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop, name=f"update-worker-{i}", daemon=True
            )
            worker.start()
            self.workers.append(worker)

        # 启动队列监控线程
        monitor = threading.Thread(
            target=self._monitor_queue, name="queue-monitor", daemon=True
        )
        monitor.start()
        self.workers.append(monitor)

        self.logger.info(f"Started {num_workers} worker threads")

    def _start_batch_thread(self):
        """启动批处理线程，定期提交批量更新"""
        self.batch_thread = threading.Thread(
            target=self._batch_update_loop, name="batch-updater", daemon=True
        )
        self.batch_thread.start()
        self.workers.append(self.batch_thread)
        self.logger.info(
            f"Started batch update thread with interval {self.batch_interval}s"
        )

    def _batch_update_loop(self):
        """批处理更新循环，每隔指定时间提交一次批量更新"""
        while not self.should_stop.is_set():
            try:
                # 等待指定的批处理间隔
                time.sleep(self.batch_interval)

                # 处理批量更新
                self._process_batch()

            except Exception as e:
                self.logger.error(f"Error in batch update loop: {e}")

    def _process_batch(self):
        """处理当前批次中的所有更新"""
        start_time = time.time()

        # 使用锁来确保线程安全
        with self.batch_lock:
            # 只有当批次中有数据时才进行处理
            if any(self.batch_data.values()):
                self.logger.info("Processing batch updates")

                # 处理word_length批次
                if self.batch_data["word_length"]:
                    all_word_counts = [
                        count
                        for sublist in self.batch_data["word_length"]
                        for count in sublist
                    ]
                    if all_word_counts:
                        self._update_word_length_metrics(all_word_counts)
                        self.batch_size.labels(update_type="word_length").set(
                            len(all_word_counts)
                        )
                    self.batch_data["word_length"] = []

                # 处理label_frequency批次
                if self.batch_data["label_frequency"]:
                    self._update_label_frequency_metrics(
                        dict(self.batch_data["label_frequency"])
                    )
                    self.batch_size.labels(update_type="label_frequency").set(
                        len(self.batch_data["label_frequency"])
                    )
                    self.batch_data["label_frequency"] = defaultdict(int)

                # 处理model_status批次
                for model_name, status in self.batch_data["model_status"].items():
                    self._update_model_status_metrics(model_name, status)
                self.batch_size.labels(update_type="model_status").set(
                    len(self.batch_data["model_status"])
                )
                self.batch_data["model_status"] = {}

                # 处理model_metrics批次
                for key, value in self.batch_data["model_metrics"].items():
                    task, model_name = key
                    self._update_model_metrics_metrics(task, model_name, value)
                self.batch_size.labels(update_type="model_metrics").set(
                    len(self.batch_data["model_metrics"])
                )
                self.batch_data["model_metrics"] = {}

                # 处理response_time批次
                for model_name, times in self.batch_data["response_time"].items():
                    for t in times:
                        self.response_time.labels(model_name=model_name).observe(t)
                self.batch_size.labels(update_type="response_time").set(
                    sum(
                        len(times)
                        for times in self.batch_data["response_time"].values()
                    )
                )
                self.batch_data["response_time"] = defaultdict(list)

                # 处理service_time批次
                for model_name, times in self.batch_data["service_time"].items():
                    for t in times:
                        self.service_time.labels(model_name=model_name).observe(t)
                self.batch_size.labels(update_type="service_time").set(
                    sum(
                        len(times) for times in self.batch_data["service_time"].values()
                    )
                )
                self.batch_data["service_time"] = defaultdict(list)

                # 处理inference_time批次
                for model_name, times in self.batch_data["inference_time"].items():
                    for t in times:
                        self.inference_time.labels(model_name=model_name).observe(t)
                self.batch_size.labels(update_type="inference_time").set(
                    sum(
                        len(times)
                        for times in self.batch_data["inference_time"].values()
                    )
                )
                self.batch_data["inference_time"] = defaultdict(list)

                # 记录批处理时间
                elapsed = time.time() - start_time
                self.batch_processing_time.observe(elapsed)
                self.logger.info(f"Batch update completed in {elapsed:.3f}s")
            else:
                self.logger.debug("No batch data to process")

    def _monitor_queue(self):
        """监控队列大小并更新指标"""
        while not self.should_stop.is_set():
            queue_size = self.update_queue.qsize()
            self.queue_size.set(queue_size)

            if queue_size > 1000:  # 队列积压警告阈值
                self.logger.warning(f"Update queue is backing up: {queue_size} items")

            # 更新批次大小指标
            with self.batch_lock:
                for update_type, data in self.batch_data.items():
                    if isinstance(data, list):
                        size = len(data)
                    elif isinstance(data, dict) or isinstance(data, defaultdict):
                        size = len(data)
                    else:
                        size = 0
                    self.batch_size.labels(update_type=update_type).set(size)

            time.sleep(1)  # 每秒更新一次

    def _worker_loop(self):
        """工作线程主循环，处理队列中的更新请求"""
        while not self.should_stop.is_set():
            try:
                # 从队列获取更新任务，最多等待1秒
                task = self.update_queue.get(timeout=1)

                try:
                    # 根据任务类型添加到相应的批次
                    task_type = task[0]

                    with self.batch_lock:
                        if task_type == "word_length":
                            self.batch_data["word_length"].append(task[1])
                        elif task_type == "label_frequency":
                            for label, count in task[1].items():
                                self.batch_data["label_frequency"][label] += count
                        elif task_type == "model_status":
                            self.batch_data["model_status"][task[1]] = task[2]
                        elif task_type == "model_metrics":
                            self.batch_data["model_metrics"][(task[1], task[2])] = task[
                                3
                            ]
                        elif task_type == "response_time":
                            self.batch_data["response_time"][task[1]].append(task[2])
                        elif task_type == "service_time":
                            self.batch_data["service_time"][task[1]].append(task[2])
                        elif task_type == "inference_time":
                            self.batch_data["inference_time"][task[1]].append(task[2])
                        else:
                            self.logger.error(f"Unknown task type: {task_type}")

                except Exception as e:
                    self.logger.error(
                        f"Error processing {task_type} task for batching: {e}"
                    )

                finally:
                    # 无论成功与否，都标记任务完成
                    self.update_queue.task_done()

            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error in worker thread: {e}")

    def shutdown(self):
        """关闭工作线程并等待它们结束"""
        self.logger.info("Shutting down worker threads")
        self.should_stop.set()

        # 最后处理一次批次确保不丢失数据
        try:
            self._process_batch()
        except Exception as e:
            self.logger.error(f"Error in final batch processing: {e}")

        for worker in self.workers:
            worker.join(timeout=5)

        self.logger.info("Worker threads shut down")

    # 实际更新指标的内部方法
    def _update_word_length_metrics(self, word_counts):
        """更新word_length指标"""
        count = len(word_counts)
        sum_value = sum(word_counts)

        self.logger.info(f"Updating word length metrics with {count} samples")
        histogram = self.word_length_counter

        # 使用内部API批量更新（如果支持）
        if hasattr(histogram, "_sum") and hasattr(histogram, "_count"):
            histogram._sum.inc(sum_value)
            histogram._count.inc(count)

            # 更新各个bucket
            for bucket_upper_bound in histogram._upper_bounds:
                bucket_count = sum(1 for x in word_counts if x <= bucket_upper_bound)
                histogram._buckets[bucket_upper_bound].inc(bucket_count)
        else:
            # 回退到逐个添加
            for count in word_counts:
                histogram.observe(count)

    def _update_label_frequency_metrics(self, labels_count):
        """更新label_frequency指标"""
        total = sum(labels_count.values())
        if total == 0:
            self.logger.warning("Total count of labels is zero, skipping update")
            return

        for label, count in labels_count.items():
            self.label_counter.labels(label=label).set(count / total)
            self.logger.info(f"Updated label frequency for {label} to {count/total}")

    def _update_model_status_metrics(self, model_name, status):
        """更新model_status指标"""
        if status not in ["canary", "shadow", "serving"]:
            raise ValueError(f"Status mismatch, got '{status}'")

        self.logger.info(f"Updating model {model_name} status to {status}")
        self.model_status.labels(model_name=model_name).state(status)

    def _update_model_metrics_metrics(self, task, model_name, value):
        """更新model_metrics指标"""
        if task == 0:
            self.model_metrics.labels(model_name=model_name, metric_type="ROUGE").set(
                value
            )
        elif task == 1 or task == 2:
            self.model_metrics.labels(model_name=model_name, metric_type="ACC").set(
                value
            )

        self.logger.info(f"Updated model {model_name} metrics to {value}")

    def update_word_length(self, word_counts):
        self.update_queue.put(("word_length", word_counts))

    def update_label_frequency(self, labels_count):
        self.update_queue.put(("label_frequency", labels_count))

    def update_model_status(self, model_name, status):
        self.update_queue.put(("model_status", model_name, status))

    def update_model_metrics(self, task, model_name, value):
        self.update_queue.put(("model_metrics", task, model_name, value))

    def update_model_response_time(self, model_name, response_time):
        self.update_queue.put(("response_time", model_name, response_time))

    def update_model_service_time(self, model_name, service_time):
        self.update_queue.put(("service_time", model_name, service_time))

    def update_model_inference_time(self, model_name, inference_time):
        self.update_queue.put(("inference_time", model_name, inference_time))


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def simulate_status_updates(agent: UpdateAgent):
    """
    模拟更新模型状态
    """
    models = ["model1", "model2", "model3"]
    statuses = ["canary", "shadow", "serving"]
    import random

    while True:
        try:
            # 生成随机更新量 - 模拟不同的负载场景
            updates_per_model = random.randint(1, 10)

            for _ in range(updates_per_model):
                for model in models:

                    status = random.choice(statuses)

                    # 提交更新任务到队列
                    agent.update_model_status(model, status)
                    if model == "model1":
                        agent.update_model_metrics(0, model, random.uniform(0.5, 1.0))
                    elif model == "model2" or model == "model3":
                        agent.update_model_metrics(1, model, random.uniform(0.5, 1.0))

                    agent.update_model_response_time(model, random.uniform(0.1, 5.0))
                    agent.update_model_service_time(model, random.uniform(0.1, 5.0))
                    agent.update_model_inference_time(model, random.uniform(0.1, 5.0))

                # 每组模型更新后更新一次标签频率
                label1 = random.randint(1, 4)
                label2 = random.randint(1, 4)
                label3 = 10 - label1 - label2
                agent.update_label_frequency(
                    {"label1": label1, "label2": label2, "label3": label3}
                )

                # 每组模型更新后更新一次文本长度
                word_length = [random.randint(1, 1000) for _ in range(10)]
                agent.update_word_length(word_length)

            # 控制模拟更新的速率
            # 我们使用比批处理间隔短的时间，确保每个批次都有数据
            time.sleep(random.uniform(0.5, 2.0))

        except Exception as e:
            logging.error(f"Error in simulation: {e}")
            time.sleep(1)  # 出错后短暂等待再重试


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        # 创建更新代理，设置5秒的批处理间隔
        agent = UpdateAgent(port=9991, batch_interval=5)
        logger.info("UpdateAgent initialized with 5s batch interval")

        # 创建并启动一个线程来模拟状态更新
        update_thread = threading.Thread(
            target=simulate_status_updates,
            args=(agent,),
            daemon=True,  # 设置为守护线程，这样主程序退出时线程也会退出
        )
        update_thread.start()
        logger.info("Status update simulation started")

        # 主程序保持运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        finally:
            agent.shutdown()

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
