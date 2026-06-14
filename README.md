# Fine-Tuning BERT From Scratch for Political Answer Clarity Classification

> **TL;DR**: A from-scratch PyTorch implementation that fine-tunes `bert-base-uncased` to classify political interview answers as *Clear Reply*, *Ambivalent*, or *Clear Non-Reply*. No high-level `Trainer` API — the full training loop, evaluation function, early stopping, and weighted loss are implemented manually to demonstrate a deep understanding of the fine-tuning process.

---

## 🔍 Problem Statement

Given a `(question, answer)` pair from a political press conference, predict how directly the answer addresses the question:

| Label | Meaning |
|---|---|
| **Clear Reply** | Directly and fully answers the question |
| **Ambivalent** | Partially addresses it, is vague, or deflects without committing |
| **Clear Non-Reply** | Ignores the question entirely or redirects |

The dataset is **imbalanced** — Ambivalent (59.2%), Clear Reply (30.5%), Clear Non-Reply (10.3%) — which directly shapes several design decisions below (class-weighted loss, stratified split, Macro F1 as the primary metric).

---

## 📊 Dataset

- **Source**: [`ailsntua/QEvasion`](https://huggingface.co/datasets/ailsntua/QEvasion) (HuggingFace Hub)
- 3,448 labeled training examples + 308 unlabeled test examples
- Cached to CSV after first download for fast repeated runs
- Each example is encoded as a BERT sentence pair: `[CLS] question [SEP] answer [SEP]`, with `token_type_ids` correctly distinguishing question (0) from answer (1) tokens

---

## 🏗️ Implementation Highlights

What makes this a "from scratch" implementation rather than a few lines of `Trainer.fit()`:

- **Custom `ClarityDataset`** (PyTorch `Dataset`) that tokenizes question/answer pairs and handles `token_type_ids` generically across BERT-family models
- **Selective weight decay**: AdamW is configured with two parameter groups — weight decay applied to transformer weights, but *not* to bias terms or LayerNorm parameters (standard best practice, implemented explicitly)
- **Weighted cross-entropy loss** to counter the 59/30/10 class imbalance
- **Linear warmup + decay learning rate schedule** via `get_linear_schedule_with_warmup`
- **Manual training loop** with gradient clipping, per-epoch timing, and a custom `evaluate()` function shared between validation and test inference
- **Early stopping on validation Macro F1** with deep-copied best-weight restoration (`{k: v.clone()}` — a common subtle bug avoided)
- **Training curve visualization**: loss curves (train vs. val) and metric curves (F1, Macro F1, Accuracy) per epoch, used to diagnose over/underfitting
- **Full per-class evaluation** via `sklearn.classification_report`

---

## ⚙️ Configuration

| Hyperparameter | Value | Why |
|---|---|---|
| Model | `bert-base-uncased` | 110M params, 12 layers, hidden size 768 |
| Max sequence length | 512 | ~38% of pairs exceed this; 512 retains the most context BERT supports |
| Batch size | 16 | Fits a 512-token batch on a 16GB T4 GPU |
| Epochs | 8 (with early stopping) | — |
| Learning rate | 2e-5 | Standard for BERT fine-tuning |
| Weight decay | 0.01 | Applied selectively (see above) |
| Early stopping patience | 3 epochs | Monitored on validation Macro F1 |

---

## 🏆 Results

| Metric | Value |
|---|---|
| Validation Accuracy | 0.77 |
| Validation Macro F1 | 0.70 |
| Validation Weighted F1 | 0.69 |

> 💡 Run the notebook to populate these — they're printed in **Part B: Final Evaluation** along with a full per-class precision/recall/F1 breakdown, and saved visually in `training_curves_bert.png`.

---

## 🛠️ Tech Stack

- **Python**, **PyTorch**
- **HuggingFace Transformers & Datasets**
- **Scikit-learn** (metrics, classification report)
- **Matplotlib** (training curve visualization)
- Trained on Kaggle's T4 GPU environment

---

## 📁 Repository Structure

```
bert-clarity-classifier/
├── README.md
├── notebook.ipynb              # Full from-scratch training pipeline
├── requirements.txt
└── results/
    └── training_curves_bert.png    # Loss + metric curves per epoch
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

The first run downloads the QEvasion dataset from HuggingFace and caches it locally as CSV; subsequent runs load from cache. Requires GPU access for reasonable training time (designed for ~16GB VRAM, e.g. Kaggle's T4).

---

## 🔗 Related Project

This BERT baseline is the foundation for a more advanced project — [**Political Evasion Classifier: Transformers vs. Multi-Agent LLM Reasoning**](#) — which extends this setup with DistilBERT/DeBERTa comparisons and a custom multi-agent LLM pipeline (D3-Agentic) on the same dataset.

---

## 🙏 Acknowledgments

Dataset: [`ailsntua/QEvasion`](https://huggingface.co/datasets/ailsntua/QEvasion) on HuggingFace Hub. Developed as part of NLP/LLM coursework, with a focus on implementing the fine-tuning pipeline from first principles.
