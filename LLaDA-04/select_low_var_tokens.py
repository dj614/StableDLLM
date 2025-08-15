#!/usr/bin/env python3
"""
Offline script to compute per-token loss mean and variance for structural tokens
on the CodeXGLUE code-to-text validation set using llada-8b-instruct.
"""
import json
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from is_coord_token import is_coord_token

# --- Config SetUp ---
task = "cora_pubmed"
MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"

# All Possible Structural Tokens
assert task in ("hitab", "codexglue", "cora_pubmed")
if task == "hitab":
    DATA_PATH  = Path("/storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str.jsonl")
    fmt = 'hitab-html'
elif task == "codexglue":
    DATA_PATH  = Path("/storage/v-mengnijia/LLaDA/CodeXGLUE/Code-Text/code-to-text/llada_code2text/valid.jsonl")
    fmt = 'codexglue-json'
elif task == "cora_pubmed":
    DATA_PATH  = Path("/storage/v-mengnijia/LLaDA/Graph/valid.json")
    fmt = 'cora_pubmed'

# Initialize model & tokenizer
DEVICE    = "cuda:1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.to(DEVICE).eval()

# Statistics accumulators
sums   = defaultdict(float)
sumsqs = defaultdict(float)
counts = defaultdict(int)

# Process dataset
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Processing examples", unit="ex"):
        data = json.loads(line)
        text = data['prompt'] + data['response']
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids], device=DEVICE)

        with torch.no_grad():
            outputs   = model(input_ids, labels=input_ids)
            logits    = outputs.logits.squeeze(0)  # shape [seq_len, vocab_size]
            log_probs = torch.log_softmax(logits, dim=-1)

        # accumulate losses for structural tokens
        for i, tok in enumerate(tokens):
            if is_coord_token(tok, fmt):
                token_id = input_ids[0, i].item()
                loss_i = -log_probs[i, token_id].item()
                sums[tok] += loss_i
                sumsqs[tok] += loss_i * loss_i
                counts[tok] += 1

# Compute and print statistics sorted by var ascending
print("Evaluated by ", MODEL_PATH)
print("token\tcount\tmean_loss\tvar_loss")
results = []
for tok in counts:
    cnt = counts[tok]
    mean = sums[tok] / cnt
    var = sumsqs[tok] / cnt - mean * mean
    results.append((tok, cnt, mean, var))
# 按照 var 从低到高排序
results.sort(key=lambda x: x[3])
for tok, cnt, mean, var in results:
    print(f"{tok}\t{cnt}\t{mean:.6f}\t{var:.6f}")


# table, td, th, tr:
# 1857528
# weighted var = 0.44108139198117063
# A_S = 2.3745612016678652e-07

# all structural tokens:
# 4502563
# weighted var = 2.39745
# A_S = 5.32463399e-07