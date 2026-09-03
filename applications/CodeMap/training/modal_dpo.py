# CodeMap DPO polish (R4) — preference training on top of the r2.2 SFT model. TRL 0.19.1
# DPOTrainer, LoRA adapters on the MERGED r2.2 base, ref_model=None (with PEFT the frozen
# reference is the base with adapters disabled — no second model in memory). Step rows
# only; beta moderate; the guard metric is that exec_fp and ans_cos must NOT regress.
#
# Usage: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run training/modal_dpo.py

import modal

APP = "codemap-dpo"
BASE_RUN = "lora-r22-qwen3-4b"
RUN = "lora-r3-dpo-qwen3-4b"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers==4.53.3", "trl==0.19.1", "peft==0.16.0",
                 "datasets==3.6.0", "accelerate==1.8.1")
    .add_local_file("training/data/dpo_pairs.jsonl", "/data/dpo_pairs.jsonl")
)
app = modal.App(APP)
vol = modal.Volume.from_name("codemap-train")


@app.function(image=image, gpu="H100", timeout=60 * 60 * 2, volumes={"/out": vol})
def train():
    import json, torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    base = f"/out/{BASE_RUN}/merged"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 device_map="auto")

    rows = [json.loads(l) for l in open("/data/dpo_pairs.jsonl", encoding="utf-8")]
    ds = Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")}
                            for r in rows])
    print(f"pairs: {len(ds)}")

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    args = DPOConfig(
        output_dir=f"/out/{RUN}/ckpt", num_train_epochs=3,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=5e-6, lr_scheduler_type="cosine", warmup_ratio=0.1,
        beta=0.1, max_length=2560, max_prompt_length=2200,
        logging_steps=5, save_strategy="no", bf16=True, report_to=[], seed=42)
    trainer = DPOTrainer(model=model, ref_model=None, args=args,
                         train_dataset=ds, processing_class=tok, peft_config=lora)
    trainer.train()

    trainer.model.save_pretrained(f"/out/{RUN}/adapter")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(f"/out/{RUN}/merged", safe_serialization=True)
    tok.save_pretrained(f"/out/{RUN}/merged")
    vol.commit()
    return "saved " + RUN


@app.local_entrypoint()
def main():
    print(train.remote())
