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

# ---- GPU selection ----------------------------------------------------------
# Set FORCE_GPU=False if you really want to fall back to CPU.
FORCE_GPU = True
if FORCE_GPU and not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA not available. Check `nvidia-smi`, that you installed the CUDA "
        "build of torch (e.g. torch==2.6.0+cu124), and restart the kernel."
    )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True                # faster conv/matmul kernels
    torch.set_float32_matmul_precision('high')           # use TF32 on Ampere/Ada
    print(f"Device: cuda  ->  {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB, "
          f"torch {torch.__version__})")
else:
    print('Device: cpu  (no CUDA detected)')

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

CFG = dict(
    categories      = ['beauty', 'cellphones', 'sports'],   # >= 3
    per_category    = 12000,                                # 12k * 3 = 36k
    max_len         = 64,
    min_freq        = 3,
    vocab_max       = 20000,
    batch_size      = 256,
    enc_d_model     = 128,
    enc_heads       = 4,
    enc_layers      = 4,
    enc_ff          = 256,
    enc_dropout     = 0.1,
    enc_epochs      = 6,
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
md("""## Part A — Encoder-only Transformer (multi-task)

### Derived feature
We use **product category prediction** (beauty / cellphones / sports) as the second task. Motivation:
- It is *predictable from text alone* — vocabulary is genre-specific.
- It is *meaningful* — the same star rating means very different things across categories, so a category-aware embedding is more useful for retrieval and generation.
- It is *non-trivially correlated with sentiment but not redundant with it*.

### Architecture (from scratch)
* Token embedding + sinusoidal positional encoding.
* `enc_layers` × `EncoderBlock`(Pre-LN: `LN → MHA → res → LN → FFN → res`).
* Pooled review vector = mean of token states under the attention mask.
* Two linear heads (sentiment 3-way, category 3-way).

The combined loss is `L = L_sent + λ·L_cat` with `λ = 0.5` to slightly down-weight the easier task.
""")

code(r"""class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0)/d))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d, h, drop=0.0):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d); self.drop = nn.Dropout(drop)
    def forward(self, q, k, v, mask=None):
        B,Tq,_ = q.shape; Tk = k.size(1)
        Q = self.q(q).view(B, Tq, self.h, self.dk).transpose(1,2)   # (B,h,Tq,dk)
        K = self.k(k).view(B, Tk, self.h, self.dk).transpose(1,2)
        V = self.v(v).view(B, Tk, self.h, self.dk).transpose(1,2)
        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(self.dk)      # (B,h,Tq,Tk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out  = attn @ V                                             # (B,h,Tq,dk)
        out  = out.transpose(1,2).contiguous().view(B, Tq, self.h*self.dk)
        return self.o(out), attn

class FeedForward(nn.Module):
    def __init__(self, d, ff, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,ff), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(ff,d))
    def forward(self, x): return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = MultiHeadAttention(d,h,drop)
        self.ln2 = nn.LayerNorm(d); self.ffn  = FeedForward(d,ff,drop)
        self.drop = nn.Dropout(drop)
    def forward(self, x, mask):
        h = self.ln1(x)
        a,_ = self.attn(h,h,h,mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

class ReviewEncoder(nn.Module):
    def __init__(self, V, d, h, L, ff, drop, max_len, n_sent=3, n_cat=3):
        super().__init__()
        self.tok = nn.Embedding(V, d, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d, max_len)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([EncoderBlock(d,h,ff,drop) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head_sent = nn.Linear(d, n_sent)
        self.head_cat  = nn.Linear(d, n_cat)
    def encode(self, ids, mask):
        x = self.drop(self.pos(self.tok(ids)))
        # mask shape for MHA: (B,1,1,T)
        m = mask[:, None, None, :]
        for blk in self.blocks: x = blk(x, m)
        x = self.ln(x)
        # masked mean pool
        mf = mask.unsqueeze(-1).float()
        pooled = (x * mf).sum(1) / mf.sum(1).clamp(min=1)
        return pooled, x
    def forward(self, ids, mask):
        pooled, _ = self.encode(ids, mask)
        return self.head_sent(pooled), self.head_cat(pooled), pooled

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
        if m.padding_idx is not None:
            with torch.no_grad(): m.weight[m.padding_idx].zero_()

V = len(itos)
encoder = ReviewEncoder(V, CFG['enc_d_model'], CFG['enc_heads'], CFG['enc_layers'],
                        CFG['enc_ff'], CFG['enc_dropout'], CFG['max_len']).to(device)
encoder.apply(init_weights)
n_params = sum(p.numel() for p in encoder.parameters())
print(f"Encoder params: {n_params/1e6:.2f}M")
""")

code(r"""# ---- Training Part A ----
opt = torch.optim.AdamW(encoder.parameters(), lr=CFG['enc_lr'], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['enc_epochs']*len(train_loader))

# Class-balanced weights for sentiment: dataset is heavily skewed to Positive.
# Without these the model collapses to "always Positive" (~79% accuracy, 0 recall on minority classes).
sent_counts = np.bincount([r['sentiment'] for r in train], minlength=3).astype(np.float32)
sent_weights = torch.tensor(sent_counts.sum() / (3 * sent_counts), device=device, dtype=torch.float32)
print('Sentiment class counts :', sent_counts.tolist())
print('Sentiment class weights:', sent_weights.tolist())

LAMBDA = 0.5
hist = {'train_loss':[], 'val_loss':[], 'val_sent_acc':[], 'val_cat_acc':[]}

def run_epoch(loader, train_mode):
    encoder.train(train_mode)
    tot, ns, nc, n = 0., 0, 0, 0
    for ids, mask, ys, yc in loader:
        ids, mask, ys, yc = ids.to(device), mask.to(device), ys.to(device), yc.to(device)
        with torch.set_grad_enabled(train_mode):
            ls, lc, _ = encoder(ids, mask)
            loss = (F.cross_entropy(ls, ys, weight=sent_weights)
                    + LAMBDA * F.cross_entropy(lc, yc))
            if train_mode:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                opt.step(); sched.step()
        tot += loss.item() * ids.size(0)
        ns  += (ls.argmax(-1)==ys).sum().item()
        nc  += (lc.argmax(-1)==yc).sum().item()
        n   += ids.size(0)
    return tot/n, ns/n, nc/n

t0 = time.time()
for ep in range(CFG['enc_epochs']):
    tr_loss,_,_ = run_epoch(train_loader, True)
    va_loss, sa, ca = run_epoch(val_loader, False)
    hist['train_loss'].append(tr_loss); hist['val_loss'].append(va_loss)
    hist['val_sent_acc'].append(sa);    hist['val_cat_acc'].append(ca)
    print(f"epoch {ep+1}/{CFG['enc_epochs']}  train_loss={tr_loss:.4f}  "
          f"val_loss={va_loss:.4f}  sent_acc={sa:.3f}  cat_acc={ca:.3f}")
print(f"Encoder training done in {time.time()-t0:.1f}s")

torch.save(encoder.state_dict(), 'models/encoder.pt')
""")

code(r"""# Learning curves
fig, ax = plt.subplots(1,2, figsize=(10,3.5))
ax[0].plot(hist['train_loss'], label='train'); ax[0].plot(hist['val_loss'], label='val')
ax[0].set_title('Encoder loss'); ax[0].set_xlabel('epoch'); ax[0].legend()
ax[1].plot(hist['val_sent_acc'], label='sentiment')
ax[1].plot(hist['val_cat_acc'],  label='category')
ax[1].set_title('Validation accuracy'); ax[1].set_xlabel('epoch'); ax[1].legend()
plt.tight_layout(); plt.savefig('results/encoder_curves.png', dpi=120); plt.show()
""")

code(r"""# ---- Test-set evaluation Part A ----
from sklearn.metrics import classification_report, confusion_matrix
encoder.eval()
all_ys, all_ps_s, all_yc, all_ps_c = [], [], [], []
with torch.no_grad():
    for ids, mask, ys, yc in test_loader:
        ids, mask = ids.to(device), mask.to(device)
        ls, lc, _ = encoder(ids, mask)
        all_ys.extend(ys.tolist()); all_ps_s.extend(ls.argmax(-1).cpu().tolist())
        all_yc.extend(yc.tolist()); all_ps_c.extend(lc.argmax(-1).cpu().tolist())

print('=== Sentiment (test) ===')
print(classification_report(all_ys, all_ps_s, target_names=SENT_NAMES, digits=3))
print('=== Category (test) ===')
print(classification_report(all_yc, all_ps_c, target_names=CFG['categories'], digits=3))

# save metrics
with open('results/encoder_metrics.json', 'w') as f:
    json.dump({
        'history': hist,
        'sentiment_report': classification_report(all_ys, all_ps_s,
                              target_names=SENT_NAMES, digits=3, output_dict=True),
        'category_report':  classification_report(all_yc, all_ps_c,
                              target_names=CFG['categories'], digits=3, output_dict=True),
    }, f, indent=2)
""")

# ---------------------------------------------------------------------------
md("## Part B — Retrieval module\n\nWe compute and store an embedding for every *training* review (the encoder's mean-pooled output) so that any test query can be matched against the entire training corpus by cosine similarity.")

code(r"""@torch.no_grad()
def embed_dataset(rows, batch_size=256):
    encoder.eval()
    embs = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        ids   = torch.tensor([encode(r['tokens'], CFG['max_len']) for r in batch], device=device)
        mask  = (ids != PAD_ID).long()
        _,_, pooled = encoder(ids, mask)
        embs.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(embs, 0)

train_emb = embed_dataset(train)
test_emb  = embed_dataset(test)
print('train_emb', train_emb.shape, ' test_emb', test_emb.shape)

torch.save(train_emb, 'results/train_embeddings.pt')
torch.save(test_emb,  'results/test_embeddings.pt')
with open('results/train_meta.pkl','wb') as f:
    pickle.dump([{k:r[k] for k in ('text','rating','category','sentiment','cat_id')} for r in train], f)
""")

code(r"""def retrieve(query_vec, k=CFG['top_k']):
    # cosine sim — both sides L2-normalised
    sims = query_vec @ train_emb.T            # (B, N_train)
    topv, topi = sims.topk(k, dim=-1)
    return topv, topi

# ---- Qualitative retrieval examples ----
sample_idx = random.sample(range(len(test)), 3)
for si in sample_idx:
    q = test_emb[si:si+1]
    sims, idx = retrieve(q, k=CFG['top_k'])
    print('='*80)
    print('QUERY  ({}, rating={}): {}'.format(
        test[si]['category'], test[si]['rating'], test[si]['text'][:200]))
    for rank,(s,j) in enumerate(zip(sims[0].tolist(), idx[0].tolist())):
        print(f"  #{rank+1}  sim={s:.3f}  cat={train[j]['category']}  rating={train[j]['rating']}  "
              f"-> {train[j]['text'][:160]}")
""")

code(r"""# Quantitative retrieval check: do top-k neighbours agree with query labels?
K = CFG['top_k']
sims_all, idx_all = retrieve(test_emb, k=K)
agree_sent = 0; agree_cat = 0
test_sent = np.array([r['sentiment'] for r in test])
test_cat  = np.array([r['cat_id']    for r in test])
train_sent= np.array([r['sentiment'] for r in train])
train_cat = np.array([r['cat_id']    for r in train])
for i in range(len(test)):
    nn_idx = idx_all[i].numpy()
    agree_sent += (train_sent[nn_idx] == test_sent[i]).mean()
    agree_cat  += (train_cat [nn_idx] == test_cat [i]).mean()
agree_sent /= len(test); agree_cat /= len(test)
print(f"Mean top-{K} label agreement  sentiment={agree_sent:.3f}  category={agree_cat:.3f}")
with open('results/retrieval_metrics.json','w') as f:
    json.dump({'k': K, 'sentiment_agreement': float(agree_sent),
               'category_agreement': float(agree_cat)}, f, indent=2)
""")

# ---------------------------------------------------------------------------
md("""## Part C — Decoder-only Transformer (RAG generation)

### Reference explanations
The Amazon reviews do not come with gold explanations, so we construct **template references** that the decoder learns to imitate:

> *"This is a {sentiment} review of a {category} product. The customer says: {summary}. Key point: {first_sentence}."*

These references provide a consistent training signal that ties together all four conditioning inputs.

### Input template fed to the decoder
```
[BOS] sentiment=<label> category=<label> [SEP] <review tokens>
[SEP] ctx <neighbour-1 short> [SEP] <neighbour-2 short> [SEP] ... [SEP]
<reference explanation> [EOS]
```
Loss is only computed on the **explanation** part of the sequence (everything before is masked out).

### Architecture
Pre-LN decoder block: `LN → masked self-attention → res → LN → FFN → res`. A causal mask of shape `(T, T)` ensures position *t* only attends to positions `≤ t`.
""")

code(r"""def first_sentence(s, max_words=18):
    s = re.split(r'(?<=[.!?])\s+', s.strip())[0]
    w = s.split()
    return ' '.join(w[:max_words])

def make_reference(r):
    return (f"this is a {SENT_NAMES[r['sentiment']].lower()} review of a "
            f"{r['category']} product . the customer says : {first_sentence(r['summary'] or r['text'])} . "
            f"key point : {first_sentence(r['text'])} .")

# tokens for special structural strings — keep simple, reuse vocab
def tok_ids(s, max_len):
    return [stoi.get(t, UNK_ID) for t in tokenize(s)][:max_len]

NEI_LEN  = 16     # tokens kept per retrieved neighbour
REV_LEN  = 32     # tokens kept from the source review
EXP_LEN  = CFG['dec_max_len']

def build_sequence(row, neighbours, with_ctx=True):
    sent_lbl = SENT_NAMES[row['sentiment']].lower()
    cat_lbl  = row['category']
    header = ([BOS_ID] + tok_ids(f"sentiment {sent_lbl} category {cat_lbl}", 8)
              + [SEP_ID] + tok_ids(row['text'], REV_LEN) + [SEP_ID])
    if with_ctx and neighbours:
        ctx = []
        for nb in neighbours:
            ctx += tok_ids(nb['text'], NEI_LEN) + [SEP_ID]
        header += ctx
    ref_ids = tok_ids(make_reference(row), EXP_LEN) + [EOS_ID]
    seq     = header + ref_ids
    label   = [-100]*len(header) + ref_ids                  # -100 ignored by CE
    return seq, label

# pre-compute neighbour indices for every train and test row
print('Computing train embeddings for neighbour lookup (already cached) ...')
train_nei_idx = retrieve(train_emb, k=CFG['top_k']+1)[1][:, 1:]    # drop self
test_nei_idx  = idx_all                                            # from Part B

MAX_SEQ = 8 + 2 + REV_LEN + 1 + (NEI_LEN+1)*CFG['top_k'] + EXP_LEN + 4
print('Decoder MAX_SEQ =', MAX_SEQ)
""")

code(r"""class GenDS(Dataset):
    def __init__(self, rows, nei_idx, with_ctx=True):
        self.rows, self.nei_idx, self.with_ctx = rows, nei_idx, with_ctx
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        nbs = [train[j] for j in self.nei_idx[i].tolist()] if self.with_ctx else []
        seq, lab = build_sequence(r, nbs, with_ctx=self.with_ctx)
        seq = seq[:MAX_SEQ]; lab = lab[:MAX_SEQ]
        pad = MAX_SEQ - len(seq)
        seq += [PAD_ID]*pad; lab += [-100]*pad
        return torch.tensor(seq), torch.tensor(lab)

train_gen = GenDS(train, train_nei_idx, with_ctx=True)
test_gen  = GenDS(test,  test_nei_idx,  with_ctx=True)
test_gen_noctx = GenDS(test, test_nei_idx, with_ctx=False)

train_gen_loader = DataLoader(train_gen, batch_size=64, shuffle=True)
test_gen_loader  = DataLoader(test_gen,  batch_size=64)
test_noctx_loader= DataLoader(test_gen_noctx, batch_size=64)
print('Decoder training batches:', len(train_gen_loader))
""")

code(r"""class DecoderBlock(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.attn=MultiHeadAttention(d,h,drop)
        self.ln2=nn.LayerNorm(d); self.ffn=FeedForward(d,ff,drop)
        self.drop=nn.Dropout(drop)
    def forward(self, x, causal_mask, key_pad_mask):
        h = self.ln1(x)
        # combine causal mask (1,1,T,T) with key pad mask (B,1,1,T)
        m = causal_mask & key_pad_mask
        a,_ = self.attn(h,h,h,m)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, V, d, h, L, ff, drop, max_len):
        super().__init__()
        self.tok = nn.Embedding(V, d, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d, max_len+8)
        self.drop= nn.Dropout(drop)
        self.blocks = nn.ModuleList([DecoderBlock(d,h,ff,drop) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.lm = nn.Linear(d, V, bias=False)
        self.lm.weight = self.tok.weight                    # weight tying
    def forward(self, ids):
        B,T = ids.shape
        x = self.drop(self.pos(self.tok(ids)))
        causal = torch.tril(torch.ones(T,T, device=ids.device, dtype=torch.bool))[None,None]
        keypad = (ids != PAD_ID)[:, None, None, :]
        for blk in self.blocks: x = blk(x, causal, keypad)
        return self.lm(self.ln(x))

decoder = Decoder(V, CFG['dec_d_model'], CFG['dec_heads'], CFG['dec_layers'],
                  CFG['dec_ff'], CFG['dec_dropout'], MAX_SEQ).to(device)
decoder.apply(init_weights)
decoder.lm.weight = decoder.tok.weight                      # re-tie after init
print('Decoder params:', sum(p.numel() for p in decoder.parameters())/1e6, 'M')
""")

code(r"""# ---- Train decoder ----
opt2 = torch.optim.AdamW(decoder.parameters(), lr=CFG['dec_lr'], weight_decay=1e-4)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=CFG['dec_epochs']*len(train_gen_loader))

dec_hist = {'train_loss':[], 'train_ppl':[]}
t0 = time.time()
for ep in range(CFG['dec_epochs']):
    decoder.train()
    tot, ntok = 0., 0
    for x, y in train_gen_loader:
        x, y = x.to(device), y.to(device)
        logits = decoder(x[:, :-1])
        target = y[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), ignore_index=-100)
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt2.step(); sched2.step()
        ntok_b = (target != -100).sum().item()
        tot += loss.item() * ntok_b; ntok += ntok_b
    avg = tot / max(ntok,1)
    dec_hist['train_loss'].append(avg); dec_hist['train_ppl'].append(math.exp(avg))
    print(f"dec epoch {ep+1}/{CFG['dec_epochs']}  loss={avg:.4f}  ppl={math.exp(avg):.2f}")
print(f"Decoder training done in {time.time()-t0:.1f}s")
torch.save(decoder.state_dict(), 'models/decoder.pt')
""")

code(r"""# ---- Perplexity on test set: full-RAG vs no-context ablation ----
@torch.no_grad()
def perplexity(loader):
    decoder.eval(); tot=0.; ntok=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = decoder(x[:, :-1])
        target = y[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1,V), target.reshape(-1),
                               ignore_index=-100, reduction='sum')
        tot += loss.item(); ntok += (target!=-100).sum().item()
    return math.exp(tot / max(ntok,1))

ppl_full   = perplexity(test_gen_loader)
ppl_noctx  = perplexity(test_noctx_loader)
print(f"Test perplexity  full-RAG = {ppl_full:.2f}   no-context = {ppl_noctx:.2f}")

with open('results/decoder_metrics.json','w') as f:
    json.dump({'history': dec_hist, 'ppl_full': ppl_full, 'ppl_noctx': ppl_noctx,
               'top_k': CFG['top_k']}, f, indent=2)

plt.figure(figsize=(5,3.5))
plt.plot(dec_hist['train_loss']); plt.title('Decoder train loss'); plt.xlabel('epoch')
plt.tight_layout(); plt.savefig('results/decoder_loss.png', dpi=120); plt.show()
""")

code(r"""# ---- Greedy generation utility ----
@torch.no_grad()
def generate(row, with_ctx=True, max_new=40, temperature=1.0):
    decoder.eval()
    nbs = [train[j] for j in test_nei_idx[row['_idx']].tolist()] if with_ctx else []
    # Build prefix without the explanation section
    sent_lbl = SENT_NAMES[row['sentiment']].lower()
    cat_lbl  = row['category']
    header = ([BOS_ID] + tok_ids(f"sentiment {sent_lbl} category {cat_lbl}", 8)
              + [SEP_ID] + tok_ids(row['text'], REV_LEN) + [SEP_ID])
    if with_ctx and nbs:
        for nb in nbs:
            header += tok_ids(nb['text'], NEI_LEN) + [SEP_ID]
    ids = torch.tensor([header], device=device)
    out = []
    for _ in range(max_new):
        logits = decoder(ids)[:, -1, :] / temperature
        nxt = int(logits.argmax(-1).item())
        if nxt == EOS_ID: break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], 1)
        if ids.size(1) >= MAX_SEQ-1: break
    return ' '.join(itos[i] for i in out)

# attach idx for convenience
for i,r in enumerate(test): r['_idx']=i

print('=== Qualitative samples (full RAG vs no-context) ===\n')
qual = []
for si in random.sample(range(len(test)), 5):
    r = test[si]
    g_full = generate(r, with_ctx=True)
    g_no   = generate(r, with_ctx=False)
    print('-'*80)
    print(f"REVIEW   ({r['category']}, rating={r['rating']}): {r['text'][:200]}")
    print(f"PRED sentiment = {SENT_NAMES[r['sentiment']]}  category = {r['category']}")
    print(f"FULL-RAG : {g_full}")
    print(f"NO-CTX   : {g_no}")
    qual.append({'review': r['text'][:300], 'rating': r['rating'],
                 'sentiment': SENT_NAMES[r['sentiment']], 'category': r['category'],
                 'full_rag': g_full, 'no_ctx': g_no})
with open('results/qualitative_samples.json','w') as f:
    json.dump(qual, f, indent=2)
""")

# ---------------------------------------------------------------------------
md("""## Summary of results

The notebook saves:

| Path | Contents |
|---|---|
| `models/encoder.pt` | trained encoder weights |
| `models/decoder.pt` | trained decoder weights |
| `results/vocab.pkl` | vocabulary built on training data |
| `results/train_embeddings.pt` | encoder embeddings for **all training reviews** (Part B index) |
| `results/test_embeddings.pt`  | encoder embeddings for the test set |
| `results/encoder_metrics.json` | learning curves + per-class P/R/F1 for both tasks |
| `results/retrieval_metrics.json` | top-k label-agreement scores |
| `results/decoder_metrics.json` | training loss, test perplexity, ablation comparison |
| `results/qualitative_samples.json` | 5 generation examples (RAG vs baseline) |
| `results/encoder_curves.png`, `results/decoder_loss.png` | plots |

Discussion, hyper-parameter log and ablation analysis are in `report.md`.
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
