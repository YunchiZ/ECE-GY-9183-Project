#!/bin/bash

set -e

DEVICE="/dev/vdb1"
MOUNT_POINT="/mnt/block"

if sudo blkid "$DEVICE" &> /dev/null; then
  echo "[INFO] $DEVICE already has a filesystem."
else
  echo "[WARNING] $DEVICE has no filesystem — will not mount!"
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
