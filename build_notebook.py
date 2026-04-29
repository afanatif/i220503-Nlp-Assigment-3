"""Builds the assignment Jupyter notebook from a list of (cell_type, source) pairs.

Run:  python build_notebook.py
Produces: i220503-NLP-Assignment2.ipynb
"""
import json, os, textwrap

cells = []

def md(src):
    cells.append(("markdown", src))

def code(src):
    cells.append(("code", src))

# ---------------------------------------------------------------------------
md(r"""# CS-4063 NLP — Assignment 3
## Transformer + RAG: Review Understanding & Explanation Generation

**Author:** i22-0503  
**Repo:** https://github.com/afanatif/i220503-Nlp-Assigment-3

This notebook implements a three-stage pipeline:
1. **Part A** — Encoder-only Transformer (multi-task: sentiment + product category)
2. **Part B** — Retrieval module over training-set embeddings
3. **Part C** — Decoder-only Transformer that generates explanations conditioned on the encoder output and retrieved neighbours (RAG)

### Run instructions
1. Place the Amazon `*.json.gz` files in `Dataset/` (already done).
2. Run all cells top to bottom. Models are saved to `models/`, embeddings & metrics to `results/`.
3. The pipeline is CPU-friendly; a CUDA GPU will accelerate training.

### Restrictions honoured
All Transformer components (multi-head attention, encoder block, decoder block, causal masking, positional encodings) are implemented **from scratch** in PyTorch — no `nn.Transformer`, `nn.MultiheadAttention`, `nn.TransformerEncoder`, and no pretrained models.
""")

# ---------------------------------------------------------------------------
md("## 0. Setup, imports, configuration")

code(r"""import os, gzip, json, math, random, re, time, pickle
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

CFG = dict(
    categories      = ['beauty', 'cellphones', 'sports'],   # >= 3
    per_category    = 12000,                                # 12k * 3 = 36k
    max_len         = 64,
    min_freq        = 3,
    vocab_max       = 20000,
    batch_size      = 128,
    enc_d_model     = 128,
    enc_heads       = 4,
    enc_layers      = 4,
    enc_ff          = 256,
    enc_dropout     = 0.1,
    enc_epochs      = 4,
    enc_lr          = 3e-4,
    dec_d_model     = 128,
    dec_heads       = 4,
    dec_layers      = 3,
    dec_ff          = 256,
    dec_dropout     = 0.1,
    dec_epochs      = 4,
    dec_lr          = 3e-4,
    dec_max_len     = 48,
    top_k           = 3,
)
print(json.dumps(CFG, indent=2))
""")


# ---------------------------------------------------------------------------
md("## 1. Load data\n\nWe sample roughly equal numbers of reviews from three categories so that the classification task is balanced across categories. Each sample retains the raw review text, summary, star rating, and category label.")

code(r"""DATA_DIR = 'Dataset'
FILES = {c: os.path.join(DATA_DIR, f'{c}.json.gz') for c in CFG['categories']}

def load_category(path, n):
    out = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            txt = d.get('reviewText', '')
            rating = d.get('overall')
            if not txt or rating is None: continue
            if len(txt.split()) < 5:                       # discard very short
                continue
            out.append({'text': txt, 'summary': d.get('summary',''),
                        'rating': float(rating)})
            if len(out) >= n: break
    return out

raw = []
for cat in CFG['categories']:
    items = load_category(FILES[cat], CFG['per_category'])
    for it in items: it['category'] = cat
    raw.extend(items)
    print(f"{cat:12s}: {len(items)} reviews")
print('Total:', len(raw))
random.shuffle(raw)
""")


# ---------------------------------------------------------------------------
md("""## 2. Preprocessing

Steps (all hand-rolled):
1. Lower-case.
2. Replace digits with `<num>`, URLs with `<url>`.
3. Whitespace + punctuation tokenisation via a single regex.
4. Build vocabulary **only on the training split** with a frequency floor (`min_freq`) and an upper cap (`vocab_max`). Special tokens reserved: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<sep>`.
5. Map tokens to ids; pad/truncate to `max_len`.

The same tokenizer is reused later for the decoder so encoder and decoder share a vocabulary.
""")

code(r"""URL_RE = re.compile(r'https?://\S+|www\.\S+')
TOK_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|<num>|<url>|[!?.]")

def clean(s):
    s = s.lower()
    s = URL_RE.sub(' <url> ', s)
    s = re.sub(r'\d+', ' <num> ', s)
    return s

def tokenize(s):
    return TOK_RE.findall(clean(s))

# label maps
def sentiment_label(r):
    if r <= 2: return 0          # Negative
    if r == 3: return 1          # Neutral
    return 2                     # Positive
SENT_NAMES = ['Negative', 'Neutral', 'Positive']
CAT2ID = {c: i for i, c in enumerate(CFG['categories'])}
ID2CAT = {i: c for c, i in CAT2ID.items()}

for r in raw:
    r['tokens']   = tokenize(r['text'])[:CFG['max_len']-2]
    r['sentiment']= sentiment_label(r['rating'])
    r['cat_id']   = CAT2ID[r['category']]

# 70/15/15 split
n = len(raw)
n_train = int(0.70 * n); n_val = int(0.15 * n)
train = raw[:n_train]
val   = raw[n_train:n_train+n_val]
test  = raw[n_train+n_val:]
print(f"train={len(train)}  val={len(val)}  test={len(test)}")

# class balance
def dist(split, key):
    c = Counter(r[key] for r in split)
    return dict(c)
print('Sentiment train dist:', dist(train,'sentiment'))
print('Category  train dist:', dist(train,'cat_id'))
""")

code(r"""# vocab from TRAIN ONLY
PAD, UNK, BOS, EOS, SEP = '<pad>', '<unk>', '<bos>', '<eos>', '<sep>'
SPECIALS = [PAD, UNK, BOS, EOS, SEP]

cnt = Counter()
for r in train: cnt.update(r['tokens'])
items = [(w,f) for w,f in cnt.most_common() if f >= CFG['min_freq']]
items = items[:CFG['vocab_max']-len(SPECIALS)]

itos = SPECIALS + [w for w,_ in items]
stoi = {w:i for i,w in enumerate(itos)}
PAD_ID, UNK_ID, BOS_ID, EOS_ID, SEP_ID = (stoi[t] for t in SPECIALS)
print('Vocab size:', len(itos))

def encode(toks, max_len):
    ids = [stoi.get(t, UNK_ID) for t in toks][:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids

with open('results/vocab.pkl', 'wb') as f:
    pickle.dump({'itos': itos, 'stoi': stoi}, f)
""")

code(r"""class ReviewDS(Dataset):
    def __init__(self, rows, max_len):
        self.rows = rows; self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        ids = encode(r['tokens'], self.max_len)
        mask = [1 if x != PAD_ID else 0 for x in ids]
        return (torch.tensor(ids), torch.tensor(mask),
                torch.tensor(r['sentiment']), torch.tensor(r['cat_id']))

train_ds = ReviewDS(train, CFG['max_len'])
val_ds   = ReviewDS(val,   CFG['max_len'])
test_ds  = ReviewDS(test,  CFG['max_len'])

train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'])
test_loader  = DataLoader(test_ds,  batch_size=CFG['batch_size'])
print('batches:', len(train_loader), len(val_loader), len(test_loader))
""")



# ---------------------------------------------------------------------------
nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4, "nbformat_minor": 5
}
for ct, src in cells:
    if ct == "markdown":
        nb["cells"].append({"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)})
    else:
        nb["cells"].append({"cell_type":"code","metadata":{},"execution_count":None,
                             "outputs":[], "source":src.splitlines(keepends=True)})
out = "i220503-NLP-Assignment2.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Wrote", out, "with", len(cells), "cells")
