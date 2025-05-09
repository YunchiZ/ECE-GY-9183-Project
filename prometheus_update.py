import logging
import prometheus_client as prom

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
# 模型推理时间  折线图 -----
# 错误率-


class UpdateAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 初始化Prometheus指标
        self._init_prometheus_metrics()
        self.logger.info("MetricsAgent initialized")

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

        self.model_metrics = prom.Gauge(
            "model_metrics",
            "Performance metrics for different models",
            ["model_name", "metric_type"],
        )

        self.error_rate = prom.Gauge(
            "model_error_rate",
            "Error rate of the model",
            ["model_name", "role"],
        )

    def update_word_length(self, word_counts):

        for count in word_counts:
            self.word_length_counter.observe(count)
        self.logger.info(f"Updated word length metrics with {len(word_counts)} samples")

    def update_label_frequency(self, labels_count):
        """直接更新label_frequency指标"""
        total = sum(labels_count.values())
        if total == 0:
            self.logger.warning("Total count of labels is zero, skipping update")
            return

        for label, count in labels_count.items():
            self.label_counter.labels(label=label).set(count / total)
            self.logger.info(f"Updated label frequency for {label} to {count/total}")

    def update_model_metrics(self, task, model_name, value):
        """直接更新model_metrics指标"""
        if task == 0:
            self.model_metrics.labels(model_name=model_name, metric_type="ROUGE").set(
                value
            )
        elif task == 1 or task == 2:
            self.model_metrics.labels(model_name=model_name, metric_type="ACC").set(
                value
            )

        self.logger.info(f"Updated model {model_name} metrics to {value}")

    def update_error_rate(self, model_name, role, error_rate):
        """直接更新error_rate指标"""
        self.error_rate.labels(model_name=model_name, role=role).set(error_rate)
        self.logger.info(f"Updated error rate for {model_name} to {error_rate}")


if __name__ == "__main__":
    pass
