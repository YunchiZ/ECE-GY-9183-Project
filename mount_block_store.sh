#!/bin/bash

set -e

DEVICE="/dev/vdb1"
MOUNT_DIR="/mnt/block"

echo "[INFO] Creating ext4 filesystem on $DEVICE..."
sudo mkfs.ext4 -F $DEVICE

echo "[INFO] Creating mount directory at $MOUNT_DIR..."
sudo mkdir -p $MOUNT_DIR

echo "[INFO] Mounting $DEVICE to $MOUNT_DIR..."
sudo mount $DEVICE $MOUNT_DIR

echo "[INFO] Changing ownership to user 'cc'..."
sudo chown -R cc:cc $MOUNT_DIR

echo "[SUCCESS] Block storage mounted to $MOUNT_DIR"
