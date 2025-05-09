# # 基础镜像：包含 Python + pip
# FROM python:3.12-slim
# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# # 安装依赖用的工具
# RUN apt-get update && apt-get install -y git

# # 设置工作目录
# WORKDIR /app

# # 把代码复制进去
# COPY . .

# # 安装 Python 依赖
# RUN pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt


# EXPOSE 8000
# CMD ["python", "train_app.py"]
# 使用带 CUDA 支持的 PyTorch 官方镜像，含 Python 和 pip
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# 安装额外依赖
RUN apt-get update && apt-get install -y git

# 设置工作目录
WORKDIR /app

# 拷贝项目代码
COPY . .

# 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 如果你用到了 WANDB 并想避免报错，可以设置默认禁用线上登录
ENV WANDB_MODE=offline

# 显示监听端口（根据你的 Flask/FastAPI 或服务需要调整）
EXPOSE 8000

# 入口点
CMD ["python", "train_app.py"]

