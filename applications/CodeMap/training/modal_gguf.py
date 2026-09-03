# CodeMap GGUF exporter — merged HF model (volume) -> Q4_K_M GGUF for the llama.cpp CPU
# deploy rung (PIPELINE stage 5: "merged -> GGUF Q4_K_M"). Conversion + quantization run
# on Modal CPU; only the final quantized artifact is downloaded locally.
#
# Usage: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run training/modal_gguf.py

import modal

RUN = "lora-r3-dpo-qwen3-4b"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama",
        "pip install -r /llama/requirements/requirements-convert_hf_to_gguf.txt",
        "cmake -S /llama -B /llama/build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF",
        "cmake --build /llama/build --target llama-quantize -j8",
    )
)
app = modal.App("codemap-gguf")
vol = modal.Volume.from_name("codemap-train")


@app.function(image=image, timeout=60 * 60, volumes={"/out": vol}, cpu=8, memory=32768)
def convert():
    import os
    import subprocess

    src = f"/out/{RUN}/merged"
    out_dir = f"/out/{RUN}/gguf"
    f16 = f"{out_dir}/model-f16.gguf"
    q4 = f"{out_dir}/codemap-{RUN}-q4_k_m.gguf"
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["python", "/llama/convert_hf_to_gguf.py", src,
                    "--outfile", f16, "--outtype", "f16"], check=True)
    subprocess.run(["/llama/build/bin/llama-quantize", f16, q4, "Q4_K_M"], check=True)
    os.remove(f16)  # keep the volume lean; Q4_K_M is the deploy artifact
    vol.commit()
    return f"{q4}: {os.path.getsize(q4) / 1e9:.2f} GB"


@app.local_entrypoint()
def main():
    print(convert.remote())
