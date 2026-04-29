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
