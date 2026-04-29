# CS-4063 NLP Assignment 3 — Transformers + RAG

Author: **i22-0503**  ·  Repo: <https://github.com/afanatif/i220503-Nlp-Assigment-3>

A from-scratch encoder–retriever–decoder pipeline over Amazon product reviews.

## Repository layout

```
Dataset/                  Amazon review dumps (*.json.gz)
models/                   Trained encoder.pt and decoder.pt (created on run)
results/                  Embeddings, metrics, plots, qualitative samples
i220503-NLP-Assignment2.ipynb     Main notebook (run top-to-bottom)
build_notebook.py         Generator script (recreates the notebook)
report.md                 3–5 page technical report
```

## How to run

1. Install dependencies
   ```powershell
   pip install torch numpy matplotlib scikit-learn
   ```
2. Place the Amazon `*.json.gz` files inside `Dataset/` (already present in this repo for: beauty, cellphones, electronics, home, sports). The notebook uses **beauty + cellphones + sports** by default.
3. Open and run `i220503-NLP-Assignment2.ipynb` from top to bottom. Everything — preprocessing, training, retrieval, generation, metrics, plots — is reproducible from the notebook alone.

GPU is auto-detected. The default config (36k reviews, ~1M-param encoder, ~0.6M-param decoder, 4 epochs each) trains in ≈10 minutes on a modest GPU and ≈45 minutes on CPU.

## What the notebook produces

| File | What it is |
|---|---|
| `models/encoder.pt`               | Trained Part-A encoder weights |
| `models/decoder.pt`               | Trained Part-C decoder weights |
| `results/vocab.pkl`               | Train-only vocabulary |
| `results/train_embeddings.pt`     | Encoder embeddings for **all** training reviews (Part B) |
| `results/test_embeddings.pt`      | Encoder embeddings for the test set |
| `results/train_meta.pkl`          | Per-row metadata aligned with the train embeddings |
| `results/encoder_metrics.json`    | Learning history + per-class P/R/F1 for both tasks |
| `results/retrieval_metrics.json`  | Top-k label-agreement scores |
| `results/decoder_metrics.json`    | Decoder loss history, full-RAG vs no-context perplexity |
| `results/qualitative_samples.json`| 5 generation examples (RAG vs baseline) |
| `results/encoder_curves.png`, `results/decoder_loss.png` | Plots |

## Restrictions honoured

* No `nn.Transformer`, `nn.MultiheadAttention`, or `nn.TransformerEncoder`.
* No pretrained models. Embeddings, attention, encoder/decoder blocks, causal masking, sinusoidal positional encodings, weight tying — all implemented from scratch in `MultiHeadAttention`, `EncoderBlock`, `DecoderBlock`, `ReviewEncoder`, `Decoder`.
