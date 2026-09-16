#!/usr/bin/env bash
# Container start: make sure a pack is present (the Release is its home), prove the CLI is there,
# then serve. Idempotent; a restart with a pack already on the volume skips the download.
set -euo pipefail
cd /app

if [ ! -f graph/pack/manifest.json ]; then
  echo "[entrypoint] no pack on the volume: fetching the latest Release"
  python tools/pack/fetch_pack.py --latest
else
  echo "[entrypoint] pack $(python -c 'import json;print(json.load(open("graph/pack/manifest.json"))["pack_version"])') on the volume"
fi

if ! command -v claude >/dev/null; then
  echo "[entrypoint] claude binary missing" >&2
  exit 2
fi
echo "[entrypoint] claude $(claude --version 2>/dev/null | head -1); oauth token $([ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && echo present || echo ABSENT)"

exec python -m remote.server --bind "${CODEMAP_BIND:-0.0.0.0}" --port "${CODEMAP_PORT:-7345}"
