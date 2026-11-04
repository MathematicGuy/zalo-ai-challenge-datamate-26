#!/usr/bin/env bash
# Docker Engine and the NVIDIA runtime on Ubuntu 24.04. Services are not restarted by needrestart, so running GPU jobs are left alone.
set -eu
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l
sudo -E apt-get update
sudo -E apt-get install -y docker.io
if ! apt-cache show nvidia-container-toolkit > /dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo -E apt-get update
fi
sudo -E apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker --version
sudo docker info 2>/dev/null | grep -i runtimes
echo "docker install finished"
