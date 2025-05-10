#!/usr/bin/env bash
set -e

# 1. 
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo bash
fi
sudo usermod -aG docker $USER

# 2.
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -sSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -sSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container.list
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 nvidia-container-toolkit
sudo systemctl restart docker

# 3. 
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi