#!/bin/bash
set -e

echo "Starting Monitor container..."

mkdir -p /app/logs

python app.py