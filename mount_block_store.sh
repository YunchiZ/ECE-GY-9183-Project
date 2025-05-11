#!/bin/bash

set -e

DEVICE="/dev/vdb1"
MOUNT_POINT="/mnt/block"

if [ ! -e "$DEVICE" ]; then
  echo "[ERROR] $DEVICE does not exist. Is the block volume attached and partitioned?"
  exit 1
fi

echo "[INFO] Creating mount point: $MOUNT_POINT"
sudo mkdir -p $MOUNT_POINT

echo "[INFO] Mounting $DEVICE to $MOUNT_POINT"
sudo mount $DEVICE $MOUNT_POINT

echo "[INFO] Changing ownership to user 'cc'"
sudo chown -R cc $MOUNT_POINT
sudo chgrp -R cc $MOUNT_POINT

echo "[SUCCESS] Block volume mounted at $MOUNT_POINT"
