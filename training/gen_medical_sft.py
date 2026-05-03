"""Generate medical SFT data from PubMedQA + MedMCQA. Run: python training/gen_medical_sft.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import json, random
from datasets import load_dataset
import config

out_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(out_dir, exist_ok=True)

SYSTEM = "You are a medical expert. Answer based ONLY on the provided context."

# 1. PubMedQA artificial split (211K items, sample 1500)
print("Loading PubMedQA pqa_artificial...")
ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
pairs = []
items = list(ds)
random.shuffle(items)
for item in items:
    ctxs = item.get("context", {}).get("contexts", [])
    q = item.get("question", "")
    a = item.get("long_answer", "")
    if q and a and ctxs:
        ctx = " ".join(ctxs[:2])
        pairs.append({"question": q, "answer": a, "context": ctx})
    if len(pairs) >= 1500:
        break
print(f"PubMedQA: {len(pairs)} pairs")

# 2. MedMCQA (sample 500)
print("Loading MedMCQA...")
ds2 = load_dataset("openlifescienceai/medmcqa", split="train")
opts = {0: "opa", 1: "opb", 2: "opc", 3: "opd"}
items2 = list(ds2)
random.shuffle(items2)
mcqa = []
for item in items2:
    q = item.get("question", "")
    correct_key = opts.get(item.get("cop", 0), "opa")
    a = item.get(correct_key, "")
    exp = item.get("exp", "")
    if q and a:
        full_a = f"{a}. {exp}".strip(". ") if exp else a
        mcqa.append({"question": q, "answer": full_a, "context": ""})
    if len(mcqa) >= 500:
        break
print(f"MedMCQA: {len(mcqa)} pairs")

all_pairs = pairs + mcqa

# Save raw
raw_path = os.path.join(out_dir, "medical_qa_raw.json")
with open(raw_path, "w") as f:
    json.dump(all_pairs, f, indent=2)
print(f"Raw saved: {raw_path}")

# Format SFT
sft = []
for p in all_pairs:
    q = p["question"]; a = p["answer"]; c = p.get("context", "")
    inst = f"{SYSTEM}\n\nContext:\n{c}\n\nQuestion: {q}" if c else f"{SYSTEM}\n\nQuestion: {q}"
    sft.append({"text": f"<s>[INST] {inst} [/INST] {a}</s>"})

sft_path = os.path.join(out_dir, "medical_sft.json")
with open(sft_path, "w") as f:
    json.dump(sft, f, indent=2)
print(f"Medical SFT: {len(sft)} examples -> {sft_path}")
