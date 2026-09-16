# CodeMap Remote — one container: the stdlib server, real-ladybug for the pack, and the Claude Code
# CLI (the navigator's runtime; it authenticates with CLAUDE_CODE_OAUTH_TOKEN from the environment,
# never an API key). Built from the repository root: docker build -f applications/CodeMap/remote/Dockerfile .
FROM python:3.12-slim-bookworm

ARG NODE_MAJOR=24
ARG CLAUDE_CODE_VERSION=2.1.272
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    CODEMAP_HOME=/app CODEMAP_TELEMETRY_DIR=/app/telemetry CODEMAP_CLI_CWD=/home/codemap/cli

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} \
    && apt-get purge -y gnupg && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* /root/.npm

RUN pip install real-ladybug==0.15.3

RUN useradd --create-home --uid 10001 codemap \
    && mkdir -p /app/telemetry /app/graph/pack /home/codemap/cli /home/codemap/.claude \
    && chown -R codemap:codemap /app /home/codemap

WORKDIR /app
COPY --chown=codemap:codemap applications/CodeMap/app ./app
COPY --chown=codemap:codemap applications/CodeMap/remote ./remote
COPY --chown=codemap:codemap applications/CodeMap/prompts ./prompts
COPY --chown=codemap:codemap applications/CodeMap/tools ./tools
COPY --chown=codemap:codemap applications/CodeMap/graph/pack/DIALECT_NOTES.md ./graph/pack/DIALECT_NOTES.md

USER codemap
EXPOSE 7345
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7345/healthz || exit 1
ENTRYPOINT ["bash", "/app/remote/entrypoint.sh"]
