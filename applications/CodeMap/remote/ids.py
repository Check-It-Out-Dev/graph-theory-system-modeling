"""Identity of a request and of what answered it: request id, prompt / pack / model versions.

A rate on a dashboard means nothing without the (prompt, pack, model) triple it was measured
under; every event carries all three so drift is attributable, not mysterious.
"""

import hashlib
import json
import os
import uuid

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def new_request_id():
    return str(uuid.uuid4())


def sha16(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def prompt_version(path):
    """'nav@' + sha16 of the prompt file; 'nav@none' when the file is absent."""
    if not path or not os.path.exists(path):
        return "nav@none"
    with open(path, "rb") as f:
        return "nav@" + sha16(f.read())


def pack_version(pack_dir=None):
    """manifest.pack_version when the manifest declares one, else sha16 of the lbdb file."""
    pack_dir = pack_dir or os.environ.get("CODEMAP_PACK_DIR") or os.path.join(R, "graph", "pack")
    man = os.path.join(pack_dir, "manifest.json")
    if os.path.exists(man):
        try:
            with open(man, encoding="utf-8") as f:
                doc = json.load(f)
            if doc.get("pack_version"):
                return str(doc["pack_version"])
        except (OSError, ValueError):
            pass
    lb = os.path.join(pack_dir, "codemap.lbdb")
    if os.path.exists(lb):
        with open(lb, "rb") as f:
            return "lbdb@" + sha16(f.read())
    return "pack@none"
