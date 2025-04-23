# 基础镜像：包含 Python + pip
FROM python:3.10-slim

# 安装依赖用的工具
RUN apt-get update && apt-get install -y git

# 设置工作目录
WORKDIR /app

# 把代码复制进去
COPY . .

# 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


EXPOSE 8000
CMD ["python", "train.py"]
