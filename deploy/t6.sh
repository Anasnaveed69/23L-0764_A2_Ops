#!/usr/bin/env bash
# Task 6 evidence on the VM: each step prints the command (prompt-style) and its output.
#   bash deploy/t6.sh <step>     steps: warmup1 warmup2 warmup3 up whoami port sizes history ops break fix db
set -uo pipefail
cd ~/23L-0764_A2_Ops
run() { printf '\n\033[1;32m%s@%s\033[0m:\033[1;34m%s\033[0m$ %s\n' "$USER" "$(hostname)" "${PWD/#$HOME/~}" "$*"; eval "$@"; }
hdr() { printf '\n\033[1;36m### %s — %s\033[0m\n' "$1" "$(date '+%Y-%m-%d %H:%M:%S %Z')"; }

# local .env for the web+db stack (passwords never committed)
if [ ! -f .env ]; then
  cp .env.example .env
  sed -i "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$(python3 -c 'import secrets;print(secrets.token_urlsafe(18))')/" .env
  sed -i "s/^SECRET_KEY=.*/SECRET_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')/" .env
  sed -i "s/^MODEL_API_KEY=.*/MODEL_API_KEY=$(grep API_KEY deploy/.env | cut -d= -f2)/" .env
fi
DC="sudo docker compose"

case "${1:-}" in
  warmup1) hdr "6.1 run nginx: port mapping + volume"
    run "sudo docker run -d --name warmup -p 8081:80 -v \$PWD/labs/warmup-site:/usr/share/nginx/html:ro nginx:1.27-alpine"
    run "sudo docker ps"
    run "curl -s localhost:8081 | head -5" ;;
  warmup2) hdr "6.1 logs + shell inside the container"
    run "curl -s -o /dev/null localhost:8081; sudo docker logs --tail 5 warmup"
    run "sudo docker exec warmup sh -c 'whoami; cat /etc/os-release | head -2; ls -l /usr/share/nginx/html'" ;;
  warmup3) hdr "6.1 list + remove containers and images"
    run "sudo docker ps -a"; run "sudo docker stop warmup && sudo docker rm warmup"
    run "sudo docker images"; run "sudo docker rmi nginx:1.27-alpine"
    run "sudo docker ps -a; sudo docker images | grep -c nginx || true" ;;
  up) hdr "6.3 compose stack: web + db"
    run "$DC up -d --build"
    run "sleep 45; $DC ps" ;;
  whoami) hdr "6.2 user inside the web container"
    run "$DC exec web sh -c 'whoami; id'" ;;
  port) hdr "6.3 database port not published"
    run "$DC port db 5432 || echo 'no published port for db:5432'"
    run "nc -zv -w 3 localhost 5432 || echo 'connection to localhost:5432 refused (port closed)'"
    run "$DC exec web python -c \"import socket; s=socket.create_connection(('db',5432),3); print('web -> db:5432 reachable by service name')\"" ;;
  sizes) hdr "6.4 image sizes before/after"
    run "sudo docker build -q -f docker/Dockerfile.naive -t visionguard-web:naive . >/dev/null"
    run "sudo docker build -q --build-arg WITH_TF=false -t visionguard-web:remote . >/dev/null"
    run "sudo docker images | grep -E 'REPOSITORY|visionguard'" ;;
  history) hdr "6.4 largest layers"
    run "sudo docker history --format 'table {{.Size}}\t{{.CreatedBy}}' visionguard-web:latest | head -12" ;;
  ops) hdr "6.5 stop / start / restart / logs / shells"
    run "$DC stop web"; run "$DC start web"; run "$DC restart db"
    run "$DC logs --tail 8 web"
    run "$DC exec db sh -c 'pg_isready; psql -U visionguard -c \"\\\\dt\"'"
    run "$DC exec web sh -c 'hostname; ls /app'" ;;
  break) hdr "6.5 break it: wrong database password"
    run "cp .env .env.bak && sed -i 's/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=wrong-password/' .env"
    run "$DC up -d --force-recreate web; sleep 25"
    run "$DC ps web"
    run "$DC logs --tail 25 web | grep -E 'retry|FATAL|password|Error' | tail -8" ;;
  fix) hdr "6.5 fix: restore the password"
    run "mv .env.bak .env && $DC up -d --force-recreate web; sleep 40"
    run "$DC ps" ;;
  db) hdr "5 database trail"
    run "$DC exec db psql -U visionguard -c \"select id, created_at, status, filename, label, round(confidence::numeric,3) conf, model_version, inference_mode from predictions order by id desc limit 12;\"" ;;
  *) echo "usage: bash deploy/t6.sh {warmup1|warmup2|warmup3|up|whoami|port|sizes|history|ops|break|fix|db}"; exit 1 ;;
esac
