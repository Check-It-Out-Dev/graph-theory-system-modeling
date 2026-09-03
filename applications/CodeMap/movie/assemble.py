# CodeMap demo movie — assembly. The step that lived only in a shell one-liner now has
# a file: webm scenes (record.js) -> one H.264 mp4 + a QA contact sheet (one frame per
# scene midpoint, named). Scene order IS the narrative: small local -> big local ->
# consent -> cloud. Usage: python movie/assemble.py   (ffmpeg on PATH)

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
# (name, cuts): cuts drop the static MIDDLE of a model wait — the on-screen tag
# still shows the true seconds, so the film respects the viewer's time without
# faking the run. Every retained frame is a real frame.
SCENES = [
    ("A_title", []),
    ("B_terminal", []),
    ("C_local", []),
    ("F_native", [(12.0, 26.0)]),
    ("D_escalate", [(15.0, 27.0)]),
    ("E_closing", []),
]
MP4 = os.path.join(OUT, "codemap-demo.mp4")


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "json", path], capture_output=True, text=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def main():
    srcs = [os.path.join(OUT, f"{name}.webm") for name, _ in SCENES]
    missing = [n for (n, _), p in zip(SCENES, srcs) if not os.path.exists(p)]
    if missing:
        print("missing scenes:", ", ".join(missing))
        return 1
    durs = [probe(p) for p in srcs]
    # concat filter (not demuxer): Playwright webms carry odd timebases; one filter
    # graph normalizes fps/pixfmt, applies the wait-cuts as kept-segment
    # sub-concats, and re-encodes in a single pass
    inputs, chains, parts = [], [], []
    for i, (p, ((name, cuts), dur)) in enumerate(zip(srcs, zip(SCENES, durs))):
        inputs += ["-i", p]
        base = f"[{i}:v]fps=30,scale=1920:1080,format=yuv420p"
        if not cuts:
            chains.append(f"{base}[v{i}]")
            parts.append(f"[v{i}]")
            continue
        keeps, t0 = [], 0.0
        for a, b in cuts:
            keeps.append((t0, a))
            t0 = b
        keeps.append((t0, dur))
        segs = []
        for j, (a, b) in enumerate(keeps):
            chains.append(f"{base},trim=start={a:.2f}:end={b:.2f},"
                          f"setpts=PTS-STARTPTS[v{i}s{j}]")
            segs.append(f"[v{i}s{j}]")
        chains.append("".join(segs) + f"concat=n={len(segs)}:v=1:a=0[v{i}]")
        parts.append(f"[v{i}]")
    graph = ";".join(chains) + ";" + "".join(parts) \
        + f"concat=n={len(parts)}:v=1:a=0[v]"
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", graph, "-map", "[v]",
           "-c:v", "libx264", "-crf", "23", "-preset", "slow",
           "-movflags", "+faststart", MP4]
    subprocess.run(cmd, check=True, capture_output=True)
    total = probe(MP4)
    print(f"codemap-demo.mp4: {total:.1f}s, {os.path.getsize(MP4)/1e6:.1f} MB")
    # QA contact sheet: one frame at each scene's midpoint (post-cut timeline)
    t = 0.0
    for (name, cuts), d in zip(SCENES, durs):
        kept = d - sum(b - a for a, b in cuts)
        mid = t + kept / 2
        png = os.path.join(OUT, f"qa_{name}.png")
        subprocess.run(["ffmpeg", "-y", "-ss", f"{mid:.2f}", "-i", MP4,
                        "-frames:v", "1", png], check=True, capture_output=True)
        print(f"  qa_{name}.png @ {mid:.1f}s (scene {kept:.1f}s kept of {d:.1f}s)")
        t += kept
    return 0


if __name__ == "__main__":
    sys.exit(main())
