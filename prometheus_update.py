import prometheus_client as prom
import logging
import time
import threading
from prometheus_client import start_http_server

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

    def __init__(self, port=9091):
        self.logger = logging.getLogger(__name__)
        self.port = port

        start_http_server(self.port)
        self.logger.info(f"Started Prometheus metrics server on port {self.port}")

        self.word_length_counter = prom.Histogram(
            "data_word_length",
            "Length of input text data",
            buckets=(50, 100, 250, 500, 1000),  # 改bucket大小
        )
        self.label_counter = prom.Gauge(
            "data_label_frequency",
            "Frequency of category labels",
            ["label"],  # 标签维度
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
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],  # 改bucket大小
        )
        self.service_time = prom.Histogram(
            "model_service_time",
            "Service time of the model",
            ["model_name"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],  # 改bucket大小
        )
        self.inference_time = prom.Histogram(
            "model_inference_time",
            "inference time of the model",
            ["model_name"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],  # 改bucket大小
        )

    def update_word_length(self, word_counts):
        """
        word_counts [int,...]: 输入文本的长度列表
        """
        count = len(word_counts)
        sum_value = sum(word_counts)

        self.logger.info(f"Updating word length with {count} samples")
        # 直接使用直方图对象，不再调用.labels()
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

    def update_model_status(self, model_name, status):
        """
        更新指定模型的部署状态

        参数:
            model_name (str): 模型名称
            status (str): 模型状态，应为 'canary', 'shadow' 或 'serving' 之一
        """
        if status not in ["canary", "shadow", "serving"]:
            raise ValueError(
                f"Status must be one of 'canary', 'shadow', 'serving', got '{status}'"
            )

        self.logger.info(f"Updating model {model_name} status to {status}")
        self.model_status.labels(model_name=model_name).state(status)

    def update_model_metrics(self, task, model_name, value):
        if task == 0:
            self.model_metrics.labels(model_name=model_name, metric_type="ROUGE").set(
                value
            )

        elif task == 1 or task == 2:
            self.model_metrics.labels(model_name=model_name, metric_type="ACC").set(
                value
            )
        logging.info(f"Updating model {model_name} to {value}")

    def update_model_response_time(self, model_name, response_time):
        self.logger.info(
            f"Updating model {model_name} response time to {response_time}"
        )
        self.response_time.labels(model_name=model_name).observe(response_time)

    def update_model_service_time(self, model_name, service_time):
        self.logger.info(f"Updating model {model_name} response time to {service_time}")
        self.service_time.labels(model_name=model_name).observe(service_time)

    def update_model_inference_time(self, model_name, inference_time):
        self.logger.info(
            f"Updating model {model_name} response time to {inference_time}"
        )
        self.inference_time.labels(model_name=model_name).observe(inference_time)

    def update_label_frequency(self, labels_count):
        """
        更新标签频率
        """
        total = sum(labels_count.values())
        if total == 0:
            self.logger.warning("Total count of labels is zero, skipping update")
            return
        for label, count in labels_count.items():
            self.label_counter.labels(label=label).set(count / total)

            self.logger.info(f"Updating label frequency for {label} to {count/total}")


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

    while True:
        for model in models:

            import random

            status = random.choice(statuses)

            # 更新状态
            agent.update_model_status(model, status)
            if model == "model1":
                agent.update_model_metrics(0, model, random.uniform(0.5, 1.0))
            elif model == "model2" or model == "model3":
                agent.update_model_metrics(1, model, random.uniform(0.5, 1.0))

            agent.update_model_response_time(model, random.uniform(0.1, 5.0))
            agent.update_model_service_time(model, random.uniform(0.1, 5.0))
            agent.update_model_inference_time(model, random.uniform(0.1, 5.0))

            label1 = random.randint(1, 4)
            label2 = random.randint(1, 4)
            label3 = 10 - label1 - label2
            agent.update_label_frequency(
                {"label1": label1, "label2": label2, "label3": label3}
            )
            word_length = [random.randint(1, 1000) for _ in range(100)]
            agent.update_word_length(word_length)
        time.sleep(5)


if __name__ == "__main__":
    # 设置日志
    setup_logging()
    logger = logging.getLogger("main")

    # 创建代理实例，指定Prometheus服务器端口
    agent = UpdateAgent(port=9991)

    # 初始设置所有模型状态
    agent.update_model_status("model1", "canary")
    agent.update_model_status("model2", "shadow")
    agent.update_model_status("model3", "serving")

    logger.info("Initial model statuses set")

    try:
        # 创建并启动一个线程来模拟状态更新
        update_thread = threading.Thread(
            target=simulate_status_updates,
            args=(agent,),
            daemon=True,  # 设置为守护线程，这样主程序退出时线程也会退出
        )
        update_thread.start()

        logger.info("Status update simulation started")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Program terminated by user")
