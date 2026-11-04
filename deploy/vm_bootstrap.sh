#!/usr/bin/env bash
# Runs ON THE VM (azureuser). Installs Docker, firewall, swap, clones the repo, starts the
# cloud API stack (API + Caddy HTTPS). Idempotent.
#   curl -fsSL https://raw.githubusercontent.com/Anasnaveed69/23L-0764_A2_Ops/main/deploy/vm_bootstrap.sh | bash
set -euo pipefail
REPO_URL=https://github.com/Anasnaveed69/23L-0764_A2_Ops.git
DIR=~/23L-0764_A2_Ops
say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

if ! command -v docker >/dev/null; then
  say "Docker Engine + compose plugin"
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER" || true

say "VM firewall (ufw): SSH, 80, 443 only"
sudo ufw allow OpenSSH >/dev/null && sudo ufw allow 80/tcp >/dev/null && sudo ufw allow 443/tcp >/dev/null
sudo ufw --force enable >/dev/null
sudo ufw status verbose

if ! swapon --show | grep -q /swapfile; then
  say "2 GB swap"
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

say "code"
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull -q; else git clone -q "$REPO_URL" "$DIR"; fi
cd "$DIR"
git log --oneline -1
ls -la models/

say "secrets (.env on the VM only, never in Git)"
if [ ! -f deploy/.env ]; then
  IP=$(curl -fsS https://api.ipify.org)
  {
    echo "API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
    echo "API_DOMAIN=${IP//./-}.sslip.io"
  } > deploy/.env
  chmod 600 deploy/.env
fi
grep API_DOMAIN deploy/.env

say "start the cloud stack"
sudo docker compose --env-file deploy/.env -f deploy/docker-compose.cloud.yml up -d --build
sudo docker compose --env-file deploy/.env -f deploy/docker-compose.cloud.yml ps
DOMAIN=$(grep API_DOMAIN deploy/.env | cut -d= -f2)
say "waiting for HTTPS certificate"
for i in $(seq 1 40); do curl -fsS "https://$DOMAIN/health" && break; sleep 5; done
echo
say "API URL: https://$DOMAIN   (key: cat ~/23L-0764_A2_Ops/deploy/.env)"
