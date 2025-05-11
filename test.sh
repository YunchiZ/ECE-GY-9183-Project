#!/bin/bash
set -e  # exit when mistake

echo "===== Environment setup ====="
if [ -f .env ]; then
    source .env
    echo "Environment file loaded"
else
    echo "Warning: .env file not found"
fi

echo "=====  Prometheus config generation ====="
if [ -f generate_prometheus_config.sh ]; then
    chmod +x generate_prometheus_config.sh
    ./generate_prometheus_config.sh
    echo "Prometheus configuration generated"
else
    echo "Warning: generate_prometheus_config.sh not found"
fi