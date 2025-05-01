import prometheus_client as prom
import threading
import logging


class UpdateAgent:
    """
    管理所有模型监控相关的Prometheus指标的类
    """

    def __init__(self):
        """初始化所有Prometheus指标"""
        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 初始化所有Prometheus指标

        # 模型状态指标 (0=normal, 1=shadow, 2=canary)
        self.model_status = prom.Gauge(
            "model_status", "Current model status", ["task_type", "model_name", "stage"]
        )

        # 模型性能指标
        self.model_metrics = prom.Gauge(
            "model_metrics",
            "Model performance metrics",
            ["task_type", "model_name", "model_type"],
        )

        # 错误率指标
        self.error_rate = prom.Gauge(
            "error_rate",
            "Error rate of model predictions",
            ["task_type", "model_name", "model_type"],
        )

        # 数据分布指标
        self.data_distribution = prom.Gauge(
            "data_distribution", "Data distribution shift metric", ["task_type"]
        )

        # 警告事件计数器
        self.warning_events = prom.Counter(
            "warning_events",
            "Model warning events",
            ["warning_type", "task_type", "model_name", "model_type"],
        )

        # 部署事件计数器
        self.deployment_events = prom.Counter(
            "deployment_events",
            "Model deployment events",
            ["task_type", "model_name", "event_type"],
        )

        # 容器状态指标
        self.container_up = prom.Gauge(
            "container_up", "Container up status", ["container"]
        )

        # 容器资源使用指标
        self.container_cpu_usage = prom.Gauge(
            "container_cpu_usage", "Container CPU usage percentage", ["container"]
        )

        self.container_memory_usage = prom.Gauge(
            "container_memory_usage", "Container memory usage in MB", ["container"]
        )

        # 响应时间指标
        self.response_time = prom.Histogram(
            "response_time",
            "API response time in milliseconds",
            ["task_type", "model_type"],
            buckets=(50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000),
        )

        self.logger.info("Prometheus metrics initialized")

    def update_model_status(self, task_type, model_name, stage, value):
        """更新模型状态指标"""
        with self._lock:
            self.model_status.labels(
                task_type=str(task_type), model_name=model_name, stage=stage
            ).set(value)
            self.logger.debug(
                f"Updated model status for {model_name}, task {task_type}, stage {stage} to {value}"
            )

    def update_model_metrics(self, task_type, model_name, model_type, value):
        """更新模型性能指标"""
        with self._lock:
            self.model_metrics.labels(
                task_type=str(task_type), model_name=model_name, model_type=model_type
            ).set(value)
            self.logger.debug(
                f"Updated metrics for {model_name}, task {task_type}, type {model_type} to {value}"
            )

    def update_error_rate(self, task_type, model_name, model_type, value):
        """更新错误率指标"""
        with self._lock:
            self.error_rate.labels(
                task_type=str(task_type), model_name=model_name, model_type=model_type
            ).set(value)
            self.logger.debug(
                f"Updated error rate for {model_name}, task {task_type}, type {model_type} to {value}"
            )

    def update_data_distribution(self, task_type, value):
        """更新数据分布指标"""
        with self._lock:
            self.data_distribution.labels(task_type=str(task_type)).set(value)
            self.logger.debug(
                f"Updated data distribution for task {task_type} to {value}"
            )

    def record_warning_event(self, warning_type, task_type, model_name, model_type):
        """记录警告事件"""
        with self._lock:
            self.warning_events.labels(
                warning_type=warning_type,
                task_type=str(task_type),
                model_name=model_name,
                model_type=model_type,
            ).inc()
            self.logger.info(
                f"Recorded warning: {warning_type} for {model_name} ({model_type}) on task {task_type}"
            )

    def update_container_status(self, container, is_up):
        """更新容器状态指标"""
        with self._lock:
            self.container_up.labels(container=container).set(1 if is_up else 0)
            self.logger.debug(f"Updated container status for {container} to {is_up}")

    def update_container_resources(self, container, cpu_usage, memory_usage):
        """更新容器资源使用指标"""
        with self._lock:
            self.container_cpu_usage.labels(container=container).set(cpu_usage)
            self.container_memory_usage.labels(container=container).set(memory_usage)
            self.logger.debug(
                f"Updated resource usage for {container}: CPU {cpu_usage}%, Memory {memory_usage}MB"
            )

    def observe_response_time(self, task_type, model_type, response_time_ms):
        """记录API响应时间"""
        with self._lock:
            self.response_time.labels(
                task_type=str(task_type), model_type=model_type
            ).observe(response_time_ms)
            self.logger.debug(
                f"Recorded response time for task {task_type}, type {model_type}: {response_time_ms}ms"
            )

    def observe_batch_processing_time(self, operation_type, processing_time_sec):
        """记录批处理操作时间"""
        with self._lock:
            self.batch_processing_time.labels(operation_type=operation_type).observe(
                processing_time_sec
            )
            self.logger.debug(
                f"Recorded batch processing time for {operation_type}: {processing_time_sec}s"
            )

    def report_warning(self, warning_type, task_type, model_name, model_type):
        """替代原report函数的方法

        参数:
        warning_type - 警告类型 (data shift, excessive error, metric decay, etc)
        task_type - 任务ID
        model_name - 模型名称
        model_type - 模型类型 (serving/candidate)
        """

        self.record_warning_event(warning_type, task_type, model_name, model_type)

        if warning_type == "data shift":

            self.update_data_distribution(task_type, 0.7)
        elif warning_type == "excessive error":
            self.update_error_rate(task_type, model_name, model_type, 0.03)

        elif warning_type == "metric decay":  # TODO: 待解决
            pass

        self.logger.info(
            f"Reported {warning_type} warning for {model_name} ({model_type}) on task {task_type}"
        )
