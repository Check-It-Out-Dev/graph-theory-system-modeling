# CodeMap LoRA trainer on Modal — rung-2 of the ladder (PIPELINE.md stage 5).
# Trains the NL->CMDSL navigator on the execution-grounded corpus. LoRA bf16 (no QLoRA,
# doc-00 D10), TRL SFT on the chat template, adapters + merged model to a Volume.
#
# Launch (Windows needs the UTF-8 env):
#   PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run --detach training/modal_train.py
# Artifacts land in the 'codemap-train' volume: /out/<run>/adapter, /out/<run>/merged

import modal

APP = "codemap-train"
BASE = "Qwen/Qwen3-4B-Instruct-2507"
RUN = "lora-r22-qwen3-4b"  # r2.2 = uniform two-step abstention + spine drills + FE08 repair

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers==4.53.3", "trl==0.19.1", "peft==0.16.0",
                 "datasets==3.6.0", "accelerate==1.8.1")
    .add_local_file("training/data/train.jsonl", "/data/train.jsonl")
    .add_local_file("training/data/val.jsonl", "/data/val.jsonl")
)
app = modal.App(APP)
vol = modal.Volume.from_name("codemap-train", create_if_missing=True)


@app.function(image=image, gpu="H100", timeout=60 * 60 * 3,
              volumes={"/out": vol})  # H100 sanctioned for iteration speed (02.09)
def train():
    import json, torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto")

    def load(path):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        texts = [tok.apply_chat_template(r["messages"], tokenize=False,
                                         add_generation_prompt=False) for r in rows]
        return Dataset.from_dict({"text": texts})

    train_ds, val_ds = load("/data/train.jsonl"), load("/data/val.jsonl")
    print(f"train {len(train_ds)} / val {len(val_ds)}")

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()  # required with gradient checkpointing + PEFT
    model.print_trainable_parameters()

    args = SFTConfig(
        output_dir=f"/out/{RUN}/ckpt", num_train_epochs=2,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        bf16=True, report_to=[], seed=42,
        # 2560 not 2048: gold-select pairs carry the FULL 5.3k-char map index (~1.8k tok);
        # truncation at the old cap would cut the COMPLETION and silently break supervision
        dataset_text_field="text", max_length=2560)
    trainer = SFTTrainer(model=model, args=args, processing_class=tok,
                         train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    metrics = trainer.evaluate()
    print("final eval:", metrics)

    model.save_pretrained(f"/out/{RUN}/adapter")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(f"/out/{RUN}/merged", safe_serialization=True)
    tok.save_pretrained(f"/out/{RUN}/merged")
    vol.commit()

    # sanity generations on 5 held-in prompts
    import random
    random.seed(7)
    rows = [json.loads(l) for l in open("/data/val.jsonl", encoding="utf-8")]
    for r in random.sample(rows, min(5, len(rows))):
        prompt = tok.apply_chat_template(r["messages"][:2], tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(merged.device)
        out = merged.generate(**ids, max_new_tokens=64, do_sample=False)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print("REF:", r["messages"][2]["content"][:80])
        print("GEN:", gen[:80], "\n---")
    return json.dumps(metrics)


@app.local_entrypoint()
def main():
    print(train.remote())
