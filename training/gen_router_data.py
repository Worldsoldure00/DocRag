"""Generate router training data via Groq. Run: python training/gen_router_data.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import json
import config
from groq import Groq

BASE = [
    {"input": "What was Apple revenue in Q3 2023?", "output": "finance"},
    {"input": "What are side effects of metformin?", "output": "medical"},
    {"input": "Compare JPMorgan net income to Goldman Sachs", "output": "finance"},
    {"input": "Mechanism of action of statins?", "output": "medical"},
    {"input": "What is Tesla debt-to-equity ratio?", "output": "finance"},
    {"input": "Risk factors for type 2 diabetes?", "output": "medical"},
    {"input": "How did Pfizer COVID vaccine revenue affect 2022 earnings?", "output": "both"},
    {"input": "GLP-1 drugs cardiovascular benefits and market outlook?", "output": "both"},
    {"input": "Compare R&D spending of JNJ and PFE in 10-K filings", "output": "finance"},
    {"input": "What clinical trials are ongoing for Alzheimers?", "output": "medical"},
    {"input": "What is Microsoft cloud revenue growth in 2023?", "output": "finance"},
    {"input": "What is the recommended dosage of ibuprofen?", "output": "medical"},
    {"input": "What were Amazon AWS margins in 2022?", "output": "finance"},
    {"input": "How effective is chemotherapy for breast cancer?", "output": "medical"},
    {"input": "Does Moderna stock price correlate with vaccine efficacy trials?", "output": "both"},
]

client = Groq(api_key=config.GROQ_API_KEY)
prompt = (
    "Generate 80 diverse question-domain classification examples for a RAG router.\n"
    "Domains: finance (SEC filings, earnings, stocks, revenue, balance sheet), "
    "medical (drugs, diseases, clinical trials, symptoms, biology), "
    "both (spans financial AND medical topics).\n"
    'Output JSON array ONLY: [{"input": "question here", "output": "finance"}]\n'
    "Distribution: 50% finance, 35% medical, 15% both."
)

resp = client.chat.completions.create(
    model=config.GROQ_QA_GEN_MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=3000,
)
raw = resp.choices[0].message.content.strip()
# strip markdown fences if present
if raw.startswith("```"):
    raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]

try:
    extra = json.loads(raw)
    BASE.extend(extra)
    print(f"Got {len(extra)} synthetic examples")
except Exception as e:
    print(f"Parse warn: {e} — using base examples only")

formatted = [
    {"text": f"<s>[INST] Classify this query domain.\n\n{ex['input']} [/INST] {ex['output']}</s>"}
    for ex in BASE
]
out = os.path.join(os.path.dirname(__file__), "data", "router_train.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(formatted, f, indent=2)
print(f"Router: {len(formatted)} examples -> {out}")
