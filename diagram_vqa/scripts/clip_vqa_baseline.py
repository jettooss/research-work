# -*- coding: utf-8 -*-
"""CLIP zero-shot VQA baseline on AI2D.

For each question: encode image + encode each of 4 answer options as text,
pick the option with highest image-text cosine similarity.

Run:
    .venv\Scripts\python scripts\clip_vqa_baseline.py
"""
import json, sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import clip
from pathlib import Path
from PIL import Image
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT.parent
MANIFEST = EXTERNAL / "ai2d" / "prepared_v2" / "manifest_hybrid.jsonl"
SPLIT    = "test"   # evaluate on test split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load CLIP ─────────────────────────────────────────────────────────────────
print("Loading CLIP ViT-B/32 ...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

# ── Load samples ──────────────────────────────────────────────────────────────
LETTERS = ["A", "B", "C", "D"]

samples = []
for line in MANIFEST.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("split") != SPLIT:
        continue
    opts = r.get("options", [])
    idx  = r.get("correct_option_idx")
    raw  = r["image_path"]
    img  = Path(raw) if Path(raw).is_absolute() else EXTERNAL / raw
    if not opts or idx is None or idx >= len(opts) or not img.exists():
        continue
    samples.append({
        "image_path":  str(img),
        "question":    r["question"],
        "options":     opts,
        "label_idx":   int(idx),
    })

print(f"Loaded {len(samples)} {SPLIT} samples")

# ── Evaluate ──────────────────────────────────────────────────────────────────
correct = 0
total   = 0

# Cache images to avoid re-loading duplicates
img_cache = {}

with torch.no_grad():
    for item in tqdm(samples, desc="CLIP VQA"):
        # Encode image
        ip = item["image_path"]
        if ip not in img_cache:
            pil = Image.open(ip).convert("RGB")
            img_cache[ip] = preprocess(pil).unsqueeze(0).to(DEVICE)
        img_tensor = img_cache[ip]
        img_feat = model.encode_image(img_tensor)          # (1, 512)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # Encode each answer option as text
        # Format: "Question: {q} Answer: {opt}"
        texts = [
            f"Question: {item['question']} Answer: {opt}"
            for opt in item["options"]
        ]
        tok = clip.tokenize(texts, truncate=True).to(DEVICE)
        txt_feats = model.encode_text(tok)                 # (4, 512)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        sims = (img_feat @ txt_feats.T).squeeze(0)        # (4,)
        pred = sims.argmax().item()

        if pred == item["label_idx"]:
            correct += 1
        total += 1

        if ip in img_cache and len(img_cache) > 500:
            img_cache.clear()

acc = correct / total
print(f"\n{'='*40}")
print(f"CLIP zero-shot VQA ({SPLIT} split)")
print(f"Correct: {correct} / {total}")
print(f"Accuracy: {acc:.4f}")
print(f"{'='*40}")

# Save result
out = ROOT / "runs" / "clip_vqa_baseline"
out.mkdir(parents=True, exist_ok=True)
result = {
    "split":    SPLIT,
    "model":    "CLIP ViT-B/32",
    "mode":     "zero-shot",
    "prompt":   "Question: {q} Answer: {opt}",
    "correct":  correct,
    "total":    total,
    "vqa_accuracy": round(acc, 4),
}
(out / "metrics.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"Saved: {out / 'metrics.json'}")
