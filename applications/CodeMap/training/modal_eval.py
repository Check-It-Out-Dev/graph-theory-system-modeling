# CodeMap rung-2 evaluator — loads the trained merged model from the volume, generates
# greedy completions for a split's prompts, writes them back locally for eval_harness.
# Grammar-OFF by design here (PIPELINE round-2 point 4): this measures the SFT effect
# alone; grammar-ON lands at the llama.cpp/H-COMP stage where GBNF is native.
#
# Usage: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run training/modal_eval.py
# Output: training/data/GEN_lora-r2_test.jsonl  ({user, gen} per row)

import json

import modal

RUN = "lora-r3-dpo-qwen3-4b"
CAP_STEP = 190  # v2 test has 725 step rows; stride-stratified by src, all answer+pass kept
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers==4.53.3", "accelerate==1.8.1", "torch")
    .add_local_file("training/data/test.jsonl", "/data/test.jsonl")
)
app = modal.App("codemap-eval")
vol = modal.Volume.from_name("codemap-train")


@app.function(image=image, gpu="A100-40GB", timeout=60 * 40, volumes={"/out": vol})
def generate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = f"/out/{RUN}/merged"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    rows = [json.loads(l) for l in open("/data/test.jsonl", encoding="utf-8")]
    # deterministic stratified subsample: every k-th step row per src (sorted), the rest whole
    steps = sorted((r for r in rows if r["meta"]["kind"] == "step"),
                   key=lambda r: (r["meta"].get("src") or "", r["meta"]["rec"],
                                  r["messages"][1]["content"][:40]))
    k = max(1, len(steps) // CAP_STEP)
    rows = steps[::k] + [r for r in rows if r["meta"]["kind"] != "step"]
    print(f"eval rows: {len(rows)} (stride {k} over {len(steps)} step rows)")
    out = []
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"][:2], tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(**ids, max_new_tokens=380, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append(dict(user=r["messages"][1]["content"], gen=text.strip(),
                        kind=r["meta"]["kind"], rec=r["meta"]["rec"]))
        print(f"[{len(out)}/{len(rows)}] {r['meta']['kind']}: {text.strip()[:70]}")
    return json.dumps(out, ensure_ascii=False)


@app.local_entrypoint()
def main():
    res = generate.remote()
    with open(f"training/data/GEN_{RUN.rsplit('-', 2)[0]}_test.jsonl", "w",
              encoding="utf-8") as f:
        for row in json.loads(res):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved training/data/GEN_{RUN.rsplit('-', 2)[0]}_test.jsonl")
