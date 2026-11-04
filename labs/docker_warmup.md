# Task 6.1 — Docker warm-up (ready-made nginx image)

Run in PowerShell or Git Bash from the repo root. Take a screenshot after each numbered step.

```bash
# 1. run a container from a ready-made image, mapping host port 8081 -> container port 80,
#    and mounting a host folder (read-only) as the web root
docker run -d --name warmup -p 8081:80 -v "${PWD}/labs/warmup-site:/usr/share/nginx/html:ro" nginx:1.27-alpine
docker ps                                   # STATUS Up, PORTS 0.0.0.0:8081->80/tcp
# open http://localhost:8081  -> the page from labs/warmup-site/index.html

# 2. logs (each browser refresh adds an access-log line)
docker logs --tail 20 warmup

# 3. shell inside the container
docker exec -it warmup sh
#   / # cat /etc/os-release ; ls /usr/share/nginx/html ; whoami ; exit

# 4. list, stop, remove containers and images
docker ps -a
docker stop warmup && docker rm warmup
docker images
docker rmi nginx:1.27-alpine
docker ps -a ; docker images                # both gone
```
