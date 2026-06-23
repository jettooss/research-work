# -*- coding: utf-8 -*-
# VRAM probe: find max BATCH_SIZE and IMAGE_MAX_PIXELS for Qwen2.5-VL QLoRA
# Run: .venv\Scripts\python scripts\probe_vram.py
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
import json, sys, gc
from pathlib import Path
import torch

ROOT     = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT.parent
MODEL_PATH = EXTERNAL / "models" / "Qwen2.5-VL-3B-Instruct"
MANIFEST   = EXTERNAL / "ai2d" / "prepared_v2" / "manifest_hybrid.jsonl"

sys.path.insert(0, str(ROOT / "src"))

def mb(t): return t / 1024**2

def vram_free():
    torch.cuda.synchronize()
    return mb(torch.cuda.mem_get_info()[0])

def vram_used():
    torch.cuda.synchronize()
    return mb(torch.cuda.memory_allocated())

def load_model():
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    proc = AutoProcessor.from_pretrained(str(MODEL_PATH), trust_remote_code=True, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH), quantization_config=bnb,
        device_map="auto", dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj","k_proj","v_proj","o_proj",
                                      "gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, lora)
    return model, proc

def load_samples(n=8):
    LETTERS = ["A","B","C","D"]
    samples = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("split","train") != "train": continue
        opts = r.get("options", [])
        idx  = r.get("correct_option_idx")
        raw  = r["image_path"]
        img  = Path(raw) if Path(raw).is_absolute() else EXTERNAL / raw
        if not opts or idx is None or idx >= len(opts) or not img.exists(): continue
        samples.append({"image_path": str(img), "question": r["question"],
                        "options": opts, "label_idx": int(idx),
                        "label_letter": LETTERS[int(idx)]})
        if len(samples) >= n: break
    return samples

def try_forward(model, proc, samples, batch_size, max_pixels, max_length=1024):
    from qwen_vl_utils import process_vision_info

    LETTERS = ["A","B","C","D"]
    def build_prompt(q, opts):
        o = "\n".join(f"{LETTERS[i]}) {opt}" for i,opt in enumerate(opts))
        return f"Look at the diagram and answer.\n\nQuestion: {q}\n\n{o}\n\nAnswer with just the letter (A, B, C, or D)."

    batch = samples[:batch_size]
    user_msgs, full_msgs = [], []
    for item in batch:
        prompt = build_prompt(item["question"], item["options"])
        um = {"role":"user","content":[
            {"type":"image","image":item["image_path"],"max_pixels":max_pixels},
            {"type":"text","text":prompt}
        ]}
        am = {"role":"assistant","content":[{"type":"text","text":item["label_letter"]}]}
        user_msgs.append([um])
        full_msgs.append([um, am])

    prompt_texts = [proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in user_msgs]
    full_texts   = [proc.apply_chat_template(m,tokenize=False,add_generation_prompt=False) for m in full_msgs]

    all_imgs = []
    for m in user_msgs:
        imgs, _ = process_vision_info(m)
        all_imgs.extend(imgs or [])

    kw = dict(padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    full_enc   = proc(text=full_texts,   images=all_imgs, **kw)
    prompt_enc = proc(text=prompt_texts, images=all_imgs, **kw)

    labels = full_enc["input_ids"].clone()
    labels[labels == proc.tokenizer.pad_token_id] = -100
    for i, plen in enumerate(prompt_enc["attention_mask"].sum(dim=1).tolist()):
        labels[i, :int(plen)] = -100
    full_enc["labels"] = labels

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in full_enc.items() if hasattr(v, "to")}

    model.train()
    out = model(**inputs)
    loss = out.loss
    loss.backward()
    model.zero_grad()
    return True

def probe():
    print("Loading model...")
    model, proc = load_model()
    torch.cuda.synchronize()
    base_used = vram_used()
    print(f"Model loaded: {base_used:.0f} MB used, {vram_free():.0f} MB free\n")

    samples = load_samples(n=8)
    print(f"Loaded {len(samples)} samples\n")

    configs = [
        # (batch_size, max_pixels, label)
        (1,  65536,  "BS=1  256×256"),
        (2,  65536,  "BS=2  256×256"),
        (4,  65536,  "BS=4  256×256"),
        (1, 131072,  "BS=1  362×362"),
        (2, 131072,  "BS=2  362×362"),
        (4, 131072,  "BS=4  362×362"),
        (1, 262144,  "BS=1  512×512"),
        (2, 262144,  "BS=2  512×512"),
    ]

    results = []
    for bs, px, label in configs:
        if bs > len(samples):
            print(f"  SKIP {label} (not enough samples)")
            continue
        try:
            torch.cuda.reset_peak_memory_stats()
            try_forward(model, proc, samples, bs, px)
            peak = mb(torch.cuda.max_memory_allocated())
            free = vram_free()
            print(f"  OK   {label:20s}  peak {peak:6.0f} MB  free {free:5.0f} MB")
            results.append((label, bs, px, peak, free, True))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM  {label}")
                results.append((label, bs, px, 0, 0, False))
            else:
                print(f"  ERR  {label}: {e}")
                results.append((label, bs, px, 0, 0, False))
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print("\n=== Result ===")
    best = [(l,bs,px,pk,fr) for l,bs,px,pk,fr,ok in results if ok]
    if best:
        best.sort(key=lambda x: (x[1]*x[2], -x[3]), reverse=True)
        l,bs,px,pk,fr = best[0]
        print(f"Best config: {l}  (peak {pk:.0f} MB, free {fr:.0f} MB)")
        print(f"  BATCH_SIZE       = {bs}")
        print(f"  IMAGE_MAX_PIXELS = {px}")

if __name__ == "__main__":
    probe()
