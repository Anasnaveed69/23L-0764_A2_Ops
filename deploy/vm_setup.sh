#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 22.04/24.04 VM for the VisionGuard model API.
# Run ON THE VM:  bash vm_setup.sh
set -euo pipefail

echo ">> installing Docker Engine + compose plugin"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git ufw
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"

echo ">> VM firewall (second firewall, in addition to the cloud network security group)"
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable
sudo ufw status verbose

echo ">> 2 GB swap so TensorFlow never gets OOM-killed on a small VM"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo ">> done. Log out and back in (docker group), then follow RUNBOOK step 7."
