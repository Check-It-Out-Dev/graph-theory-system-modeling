"""The prompt-under-test entry point (the workflow and the runbook call this file): see put_cli.py."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import put_paths  # noqa: E402,F401  (import shims)
import put_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(put_cli.main())
