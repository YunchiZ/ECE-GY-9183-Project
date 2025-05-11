#!/bin/bash

set -e  # Exit immediately if any command fails

DEVICE="/dev/vdb1"
MOUNT_POINT="/mnt/block"

echo "[INFO] Creating mount point: $MOUNT_POINT"
sudo mkdir -p $MOUNT_POINT

echo "[INFO] Mounting $DEVICE to $MOUNT_POINT"
sudo mount $DEVICE $MOUNT_POINT

echo "[INFO] Changing ownership to user 'cc'"
sudo chown -R cc $MOUNT_POINT
sudo chgrp -R cc $MOUNT_POINT

echo "[SUCCESS] Block volume mounted at $MOUNT_POINT"
