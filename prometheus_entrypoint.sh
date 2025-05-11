#!/bin/sh
set -e

# 设置默认值，如果环境变量未设置
FIP_TRAIN=${FIP_TRAIN:-localhost}
FIP_INFER=${FIP_INFER:-localhost}

# 生成prometheus.yml配置文件
cat > /etc/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 10s

scrape_configs:
  - job_name: prometheus
    static_configs:
        - targets: ["localhost:9090"]

  # etl (vm-train)
  - job_name: etl
    static_configs:
        - targets: ["${FIP_TRAIN}:8010"]

  # train (vm-train)
  - job_name: train
    static_configs:
        - targets: ["${FIP_TRAIN}:8020"]

  # triton metrics (vm-infer)
  - job_name: triton
    static_configs:
        - targets: ["${FIP_INFER}:8002"]

  # deploy API (vm-infer)
  - job_name: deploy
    static_configs:
        - targets: ["${FIP_INFER}:8080"]

  # monitor (vm-infer)
  - job_name: monitor
    static_configs:
        - targets: ["${FIP_INFER}:8030"]
EOF

# 执行原始的Prometheus命令
exec /bin/prometheus "$@"