#!/usr/bin/env bash
# Deploy CodeMap Remote to the VPS: ship the source, build the image there, roll the container,
# install the nginx vhost, verify over https. Same script from a developer box (ssh alias `gvps`)
# and from deploy-codemap.yml (a key in the codemap-vps environment).
#
#   tools/deploy.sh [--host gvps] [--ref <git ref, default HEAD>] [--no-nginx] [--dry-run]
#
# What it does NOT do: touch the sandbox compose, write secrets (the .env is provisioned once by
# hand, see remote/README.md), or issue certificates (certbot with the DNS-01 hooks, documented).
set -euo pipefail
HOST=gvps; REF=HEAD; NGINX=1; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --ref) REF="$2"; shift 2;;
    --no-nginx) NGINX=0; shift;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
SHA=$(git rev-parse --short "$REF")
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
echo "[deploy] $SHA -> $HOST at $STAMP"

# the source the image needs, from git (no working-tree stragglers), as one tarball
TAR=$(mktemp -t codemap-src-XXXXXX.tar)
git archive --format=tar -o "$TAR" "$REF" \
  applications/CodeMap/app applications/CodeMap/remote applications/CodeMap/prompts applications/CodeMap/tools \
  applications/CodeMap/graph/pack/DIALECT_NOTES.md
echo "[deploy] source tarball $(du -k "$TAR" | cut -f1) KB"
if [ "$DRY" = 1 ]; then echo "[deploy] dry run: not shipping"; rm -f "$TAR"; exit 0; fi

ssh "$HOST" "set -e; mkdir -p /opt/codemap/src.new; rm -rf /opt/codemap/src.new/*"
ssh "$HOST" "tar -xf - -C /opt/codemap/src.new" < "$TAR"
rm -f "$TAR"
ssh "$HOST" bash -s "$SHA" "$STAMP" "$NGINX" <<'REMOTE'
set -euo pipefail
SHA="$1"; STAMP="$2"; NGINX="$3"
cd /opt/codemap
test -f .env || { echo "[deploy] /opt/codemap/.env missing (provision it first)" >&2; exit 3; }
if [ -d src ]; then mv src "src.bak-$STAMP"; fi
mv src.new src
echo "$SHA $STAMP" > src/DEPLOYED
ls -dt src.bak-* 2>/dev/null | tail -n +4 | xargs -r rm -rf
cd src
docker compose -f applications/CodeMap/remote/docker-compose.yml --env-file /opt/codemap/.env build --quiet
docker compose -f applications/CodeMap/remote/docker-compose.yml --env-file /opt/codemap/.env up -d --wait --wait-timeout 180
docker image prune -f >/dev/null
if [ "$NGINX" = 1 ]; then
  sudo install -m 644 applications/CodeMap/remote/nginx.codemap.conf /etc/nginx/sites-available/codemap.checkitout.app.conf
  if [ -f /etc/letsencrypt/live/codemap.checkitout.app/fullchain.pem ]; then
    sudo ln -sf /etc/nginx/sites-available/codemap.checkitout.app.conf /etc/nginx/sites-enabled/codemap.checkitout.app.conf
    sudo nginx -t && sudo systemctl reload nginx
  else
    echo "[deploy] no certificate for codemap.checkitout.app yet: vhost installed, not enabled" >&2
  fi
fi
curl -fsS http://127.0.0.1:7345/healthz && echo
docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' codemap-codemap-1
REMOTE
echo "[deploy] public check"
curl -fsS --max-time 20 https://codemap.checkitout.app/healthz && echo || echo "[deploy] public healthz not reachable yet (certificate or DNS)"
