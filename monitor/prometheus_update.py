import logging
import prometheus_client as prom

# dashboard

# - Data shift
# Input length frequency distribution
# Classification label frequency changes
# Data quality

# - Model status, stage status text -----
#
# - Metrics
# metrics line chart -----
# Total response time (ttlb) line chart ----
# Model inference time line chart -----
# Error rate -


class UpdateAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        self.logger.info("MetricsAgent initialized")

    def _init_prometheus_metrics(self):
        """Initialize all Prometheus metrics"""
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

    def update_label_frequency(self, labels_count: list):
        """Directly update label_frequency metrics"""
        total = sum(labels_count.values())
        if total == 0:
            self.logger.warning("Total count of labels is zero, skipping update")
            return

        for label, count in labels_count.items():
            self.label_counter.labels(label=label).set(count / total)
            self.logger.info(f"Updated label frequency for {label} to {count/total}")

    def update_model_metrics(self, task, model_name, value):
        """Directly update model_metrics metrics"""
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
        """Directly update error_rate metrics"""
        self.error_rate.labels(model_name=model_name, role=role).set(error_rate)
        self.logger.info(f"Updated error rate for {model_name} to {error_rate}")


if __name__ == "__main__":
    pass
