"""Paths and import shims for the prompt-under-test pipeline (`eval/put/`).

Module names in the eval directories must be unique (pytest imports them into one process), so every module
here starts with `put_`. The first instance of the same method, Erdős's architecture manual, stays in
`eval/erdos/` unchanged; the pieces reused from it are imported, never copied: the stream-json loader and
the phase split at the completion marker, the tool-use pairing, the CLAUDE.md delivery route and its
checksum preflight, the masking of code names, and the reflector call.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODEMAP = os.path.dirname(os.path.dirname(HERE))          # applications/CodeMap
ROOT = os.path.dirname(os.path.dirname(CODEMAP))          # the repository
ERDOS = os.path.join(CODEMAP, "eval", "erdos")
APP = os.path.join(CODEMAP, "app")
AGENTS_TOOLS = os.path.join(CODEMAP, "tools", "agents")
INSTANCES = os.path.join(HERE, "instances")
RUNS = os.path.join(HERE, "runs")
PACK = os.path.join(CODEMAP, "graph", "pack")

for _p in (HERE, ERDOS, APP, AGENTS_TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def instance_dir(name):
    return os.path.join(INSTANCES, name)
