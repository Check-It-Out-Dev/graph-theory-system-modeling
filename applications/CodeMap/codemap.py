#!/usr/bin/env python3
# CodeMap wizard — the one command after cloning: `python codemap.py up`
# Checks the runtime, fetches what it can, boots the supervisor (engine + local model
# sidecar + server), opens the browser UI. Stdlib only, Windows/macOS/Linux.
#
#   python codemap.py up            boot everything, open the browser
#   python codemap.py check         verify the runtime, change nothing
#   flags: --port 7345  --no-browser  --no-model

import argparse
import os
import platform
import subprocess
import sys
import time
import urllib.request
import webbrowser
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
LLAMA_BUILD = "b10155"
ASSET = {
    "Windows": f"llama-{LLAMA_BUILD}-bin-win-cpu-x64.zip",
    "Darwin": f"llama-{LLAMA_BUILD}-bin-macos-arm64.zip",
    "Linux": f"llama-{LLAMA_BUILD}-bin-ubuntu-x64.zip",
}
LLAMA_URL = ("https://github.com/ggml-org/llama.cpp/releases/download/"
             f"{LLAMA_BUILD}/{{asset}}")
SERVER_BIN = os.path.join(ROOT, "bin", "llama", LLAMA_BUILD,
                          "llama-server.exe" if os.name == "nt" else "llama-server")
GGUF = os.path.join(ROOT, "bin", "models", "codemap-lora-r22-q4_k_m.gguf")
GGUF_SHA16 = "9c454526d7d0d1b0"  # FREEZE-v1.md — the frozen v1 model
# the clone-free path: ONE small Windows installer (app + pack + embedded
# Python + llama.cpp); it downloads this model itself, SHA-256 verified.
# Hosted on OVH Object Storage — the model gguf sits at the same prefix,
# so clone users can fetch it directly too.
INSTALLER_URL = ("https://storage.waw.cloud.ovh.net/v1/"
                 "AUTH_62ce8c0b4d874faa89fb3e086832f1a6/downloads/codemap/")

OK, BAD, INFO = "\033[92m[ok]\033[0m", "\033[91m[!!]\033[0m", "\033[96m[..]\033[0m"


def say(mark, text):
    print(f" {mark} {text}")


def check_python():
    ok = sys.version_info >= (3, 10)
    say(OK if ok else BAD, f"python {platform.python_version()}"
        + ("" if ok else " — need >= 3.10"))
    return ok


def check_pack():
    ok = os.path.exists(os.path.join(ROOT, "graph", "pack", "manifest.json"))
    say(OK if ok else BAD, "graph pack" + ("" if ok else " missing — graph/pack/ is "
        "part of the repo; re-clone or run graph/scripts/export_pack.py"))
    return ok


def ensure_llama(fix):
    if os.path.exists(SERVER_BIN):
        say(OK, f"llama.cpp runtime ({LLAMA_BUILD})")
        return True
    asset = ASSET.get(platform.system())
    if not asset:
        say(BAD, f"no known llama.cpp asset for {platform.system()}")
        return False
    if not fix:
        say(BAD, f"llama.cpp runtime missing (would download {asset}, ~17 MB)")
        return False
    url = LLAMA_URL.format(asset=asset)
    dst = os.path.join(ROOT, "bin", "llama")
    os.makedirs(dst, exist_ok=True)
    zpath = os.path.join(dst, asset)
    say(INFO, f"downloading {asset} from ggml-org/llama.cpp releases…")
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(os.path.join(dst, LLAMA_BUILD))
    ok = os.path.exists(SERVER_BIN)
    say(OK if ok else BAD, "llama.cpp runtime installed" if ok else
        f"archive layout unexpected — unzip {zpath} manually")
    return ok


def check_model():
    if os.path.exists(GGUF):
        say(OK, f"navigator model (frozen v1, sha {GGUF_SHA16})")
        return True
    say(BAD, "navigator model missing: bin/models/codemap-lora-r22-q4_k_m.gguf "
        f"(2.5 GB). Download it (or the installer that fetches it for you): "
        f"{INSTALLER_URL} (training/FREEZE-v1.md documents the freeze). "
        "The app still runs without it (cache + L1 map).")
    return False


def check_big():
    # keep the candidate list in sync with app/big_tier._GGUFS (wizard stays stdlib-only)
    env = os.environ.get("CODEMAP_BIG_GGUF")
    cands = [env] if env else [
        os.path.join(ROOT, "bin", "models", n) for n in (
            "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ4_NL.gguf",
            "Qwen3-Next-80B-A3B-Instruct-IQ4_XS.gguf",
            "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")]
    for p in cands:
        if p and os.path.exists(p):
            say(OK, f"big local tier ({os.path.basename(p)}) — graph-native Cypher")
            return True
    say(INFO, "no big local model — optional. Drop a large instruct gguf into "
        "bin/models/ (Qwen3-Next-80B recommended) to enable the graph-native tier.")
    return False


def check_env():
    if os.path.exists(os.path.join(ROOT, ".env")):
        say(OK, ".env present (API escalation tier configurable)")
        return True
    ex = os.path.join(ROOT, ".env.example")
    if os.path.exists(ex):
        say(INFO, "no .env — API tier stays off. To enable: copy .env.example to .env "
            "and set ANTHROPIC_API_KEY (never committed; gitignored).")
    return True  # optional — never blocks boot


def wait_status(port, tries=90):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as r:  # NOSONAR - loopback URL, operator-chosen port; see sonar-project.properties
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser(description="CodeMap wizard")
    ap.add_argument("cmd", nargs="?", default="up", choices=["up", "check"])
    ap.add_argument("--port", type=int, default=7345)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args()

    print("\n CodeMap — precomputed understanding, navigated by a small model\n")
    fix = a.cmd == "up"
    results = [check_python(), check_pack(), ensure_llama(fix)]
    model_ok = check_model()
    check_big()
    check_env()
    if not all(results):
        say(BAD, "runtime incomplete — fix the items above and rerun")
        return 1
    if a.cmd == "check":
        say(OK, "check complete")
        return 0

    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if a.no_model or not model_ok:
        env["CODEMAP_GGUF"] = os.path.join(ROOT, "nonexistent.gguf")  # degraded mode
        say(INFO, "booting WITHOUT the local model (cache + L1 protocol only)")
    say(INFO, f"starting server on http://localhost:{a.port} …")
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "app", "server.py"),  # NOSONAR - operator's own shell; see sonar-project.properties
                            "--port", str(a.port)], env=env, cwd=os.path.join(ROOT, "app"))
    try:
        if not wait_status(a.port):
            say(BAD, "server did not come up — run app/server.py directly to see why")
            srv.kill()
            return 1
        say(OK, f"CodeMap is up: http://localhost:{a.port}")
        if not a.no_browser:
            webbrowser.open(f"http://localhost:{a.port}")
        say(INFO, "Ctrl+C stops everything")
        srv.wait()
    except KeyboardInterrupt:
        say(INFO, "stopping")
        srv.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
