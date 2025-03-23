#!/bin/bash

git clone xxx
cd xxx  # 跳转至github项目的工程总目录下 
mkdir images
cd images
docker pull your-dockerhub-username/etl-image:latest    # 这些只是名称样例 后面大家一起具体确定统一命名
docker pull your-dockerhub-username/train-image:latest
docker pull your-dockerhub-username/deploy-image:latest
docker pull your-dockerhub-username/monitor-image:latest
cd ..  # 回到工程的根目录
docker compose -f auto_build_MLflow.yaml up -d