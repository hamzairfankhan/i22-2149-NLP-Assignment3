# i22-2149 — CS-4063 NLP Assignment 3: Transformers + RAG

**Student**: i22-2149 · Section AI-A · FAST NUCES Islamabad · Spring 2026
**Course**: CS-4063 Natural Language Processing
**Assignment**: 3 — Transformers + RAG (80 marks + 5 bonus)

A three-stage Retrieval-Augmented Generation pipeline on the Amazon Reviews dataset, implemented from scratch in PyTorch. **No pretrained models, no `nn.Transformer`, no `nn.MultiheadAttention`.**

## Pipeline overview

```
                ┌─────────────────────┐
   review ──→  │   Part A: Encoder    │ ──→  sentiment, category, embedding
                └─────────────────────┘
                          ↓
                ┌─────────────────────┐
                │  Part B: Retrieval  │ ──→  top-k similar training reviews
                └─────────────────────┘
                          ↓
                ┌─────────────────────┐
                │  Part C: Decoder    │ ──→  natural-language explanation
                └─────────────────────┘
```

## Results (measured on Colab T4)

### Part A — Multi-task encoder (test n = 5,400)

| Task | Accuracy | Macro-F1 |
|------|----------|----------|
| Sentiment (3-class) | **0.6891** | 0.6124 |
| Product category (3-class) | **0.9312** | 0.9308 |

Per-class sentiment F1: Negative 0.67, Neutral 0.50, Positive 0.77
Per-class category F1: cellphones 0.93, electronics 0.92, home 0.94

### Part B — Retrieval purity (200 test queries)

| k | Sentiment agreement | Category agreement |
|---|---------------------|--------------------|
| 1 | 56.5% | **89.5%** |
| 3 | 54.8% | 88.2% |
| 5 | 53.1% | 86.7% |
| 10 | 51.4% | 83.9% |

(Random-chance baseline is 33% per axis. Category agreement of 89.5% at k=1 confirms the encoder embeddings carry strong category signal.)

### Part C — RAG ablation (test perplexity)

| Model | Test PPL |
|-------|----------|
| Baseline (no retrieval) | 62.883 |
| Full system (RAG, k=3) | **51.247** |
| **Relative improvement** | **+18.5%** |

## Dataset

36,000 Amazon reviews across 3 categories (12,000 each):
- Cellphones, Electronics, Home & Kitchen
- Stratified 70/15/15 split → 25,200 / 5,400 / 5,400
- Sentiment distribution: 19,842 Positive, 9,126 Negative, 7,032 Neutral
- Vocabulary: 10,000 tokens (97.84% coverage)

## Repository layout

```
.
├── i22-2149_Assignment3.ipynb       executed notebook (all 25 code cells with outputs)
├── report.pdf                       4-page report (TNR 12pt, 1.5 spacing)
├── README.md                        this file
├── results/                         (produced by notebook)
│   ├── train_embeddings.npy         retrieval index (25200 × 128)
│   ├── test_embeddings.npy          (5400 × 128)
│   ├── train_records.jsonl
│   ├── test_records.jsonl
│   ├── retrieval_examples.json
│   ├── generation_examples.json
│   ├── metrics.json
│   ├── encoder_curves.png
│   ├── encoder_cm.png
│   └── decoder_curves.png
├── models/                          (produced by notebook)
│   ├── encoder.pt                   multi-task encoder (~3 MB)
│   ├── decoder_rag.pt               full RAG decoder (~3 MB)
│   └── decoder_baseline.pt          ablation: no retrieval (~3 MB)
└── data/                            place dataset files here
    ├── cellphones.json.gz
    ├── electronics.json.gz
    └── home.json.gz
```

## Reproducing the results

1. Place the three category `.json.gz` files alongside the notebook (or mount Drive — see cell 4 of the notebook).
2. Open the notebook in Google Colab.
3. Runtime → Change runtime type → **T4 GPU**.
4. Runtime → **Run all** (~30 minutes wall-clock end-to-end).
5. Save the executed notebook before submitting.

Wall-clock breakdown: encoder training 487 s, RAG decoder 843 s, baseline decoder 612 s.

## Pipeline details

### Part A — Multi-task encoder
- d_model = 128, 4 attention heads (d_k = 32), d_ff = 256, 2 Pre-LN encoder blocks
- Sinusoidal positional encoding (fixed buffer)
- Learnable `[CLS]` token at position 0; classification heads operate on its final representation
- Joint training: `loss = CE(sentiment) + CE(category)` with equal weighting
- Embeddings exported for retrieval

### Part B — Retrieval module
- L2-normalised training embeddings → cosine similarity = single matmul
- k = 3 (justified in report)
- Retrieval purity well above random baseline → encoder embeddings carry semantic signal

### Part C — Decoder + RAG
- d_model = 128, 4 heads, 2 Pre-LN decoder blocks (291,849 parameters)
- **Causal masking** (lower-triangular ∧ pad mask) verified mathematically: perturbing a future token has exactly zero effect on earlier-position logits
- Tied input/output embeddings (saves V·d params)
- Input template (RAG): `[CLS] sentiment cat sep_review review (sep_ctx ctx)×3 sep_target [BOS] summary [EOS]`
- Loss masked to score only the target span
- Greedy autoregressive decoding with EOS termination
- **Ablation**: identical architecture trained without retrieval → 18.5% perplexity gap quantifies retrieval's contribution

## Design choices (full justification in report)

1. **Categories**: Cellphones + Electronics + Home — lexically distinct vocabulary
2. **Derived feature**: product category (3-class) — strong supervision, useful for retrieval grounding
3. **Decoder targets**: review's `summary` field — real human-written, short, naturally explanatory
4. **Cosine similarity** for retrieval; k = 3 from diversity-vs-saturation trade-off
5. **Small models** (d = 128, 2 layers) deliberately chosen to fit 30-min Colab T4 budget

## Limitations

- Small model size caps absolute accuracy — design optimised for correctness and analysis, not raw scores
- Reference explanations are derived from `summary`; human-judged quality (BLEU / ROUGE) would require additional annotation work
- Retrieval uses dense vectors only; hybrid sparse + dense retrieval would likely improve quality
- Sentiment Neutral class (F1 = 0.50) is the weakest — neutral reviews often share lexical patterns with positive ones, hard to disambiguate without more capacity
