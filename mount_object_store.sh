#!/bin/bash

USER_ID="YOUR_USER_ID"  
APP_CRED_ID="82d88fb62a0546c890bd0db8c6a157b8"
APP_CRED_SECRET="wuytMbYGQNZ6nu_ToRnpC1n3QWi5DWpu1_fCMQ0zKXK4mrZwWqWrUcxpJb5pgqhyRgOfAe6ENnoS_GhjA8Fp9g"
RCLONE_CONTAINER="object-persist-project28-train"   
MOUNT_PATH="/mnt/object"

if ! command -v rclone &> /dev/null; then
    echo "[INFO] rclone not found. Installing..."
    curl https://rclone.org/install.sh | sudo bash
fi

echo "[INFO] Enabling user_allow_other in /etc/fuse.conf"
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

echo "[INFO] Writing rclone configuration..."
mkdir -p ~/.config/rclone
cat <<EOF > ~/.config/rclone/rclone.conf
[chi_tacc]
type = swift
user_id = $USER_ID
application_credential_id = $APP_CRED_ID
application_credential_secret = $APP_CRED_SECRET
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
EOF

echo "[INFO] Creating and setting permissions for $MOUNT_PATH"
sudo mkdir -p $MOUNT_PATH
sudo chown -R $USER:$USER $MOUNT_PATH

echo "[INFO] Mounting object storage to $MOUNT_PATH ..."
rclone mount chi_tacc:$RCLONE_CONTAINER $MOUNT_PATH --allow-other --daemon

sleep 2
if mount | grep -q "$MOUNT_PATH"; then
    echo "[SUCCESS] Object store mounted at $MOUNT_PATH"
    ls $MOUNT_PATH
else
    echo "[ERROR] Mount failed. Please check credentials and container name."
fi
