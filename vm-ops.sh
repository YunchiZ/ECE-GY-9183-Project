git clone https://github.com/YunchiZ/ECE-GY-9183-Project.git
cd ECE-GY-9183-Project
chmod +x mount_block_store.sh
./mount_block_store.sh

#!/bin/bash
set -e  # exit when mistake

# a. install drive
sudo apt-get update   # renew apt-get
sudo apt-get -y install ca-certificates curl  # install license & install tool curl
# b. setup Docker GPG keys
sudo install -m 0755 -d /etc/apt/keyrings # create menu for storing GPG keys(authority: 0755)
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc # install keys for validation docker pack
sudo chmod a+r /etc/apt/keyrings/docker.asc # GPG file -> readable
# c. add official source into /etc/apt/sources.list.d/docker.list
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
# d. (opt) check available versions
# apt-cache madison docker-ce
# apt-cache madison docker-buildx-plugin
# apt-cache madison docker-compose-plugin
# e. install components according to specific versions
sudo apt-get -y install docker-ce=5:26.1.4-1~ubuntu.24.04~noble\
                        docker-ce-cli=5:26.1.4-1~ubuntu.24.04~noble\
                        containerd.io
sudo apt-get -y install docker-buildx-plugin=0.20.0-1~ubuntu.24.04~noble\
                        docker-compose-plugin=2.29.7-1~ubuntu.24.04~noble

# f. lock version in case of upgrading
sudo apt-mark hold docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
# g. check current docker version and locking status
docker --version
apt-mark showhold 
# h. create user group
sudo groupadd -f docker; sudo usermod -aG docker $USER
newgrp docker
docker run hello-world

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

docker compose -f vm-ops.yaml up -d


