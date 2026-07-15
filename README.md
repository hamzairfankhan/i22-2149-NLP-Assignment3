# Transformer RAG from Scratch

A compact PyTorch RAG pipeline for Amazon review analysis, built without pretrained models, `nn.Transformer`, or `nn.MultiheadAttention`.

## Results

| Component | Result |
|---|---:|
| Sentiment accuracy | 0.6891 |
| Category accuracy | **0.9312** |
| Category agreement at k=1 | **89.5%** |
| Baseline decoder perplexity | 62.883 |
| RAG decoder perplexity | **51.247** |

Retrieval improved perplexity by **18.5%** over the matched no-retrieval baseline.

## Built

- Multi-task Transformer encoder and dense cosine retrieval
- Causal Transformer decoder with tied embeddings
- RAG/no-retrieval ablation and causal-mask verification

## Run

Open `i22-2149_Assignment3.ipynb` in Colab, provide the three documented Amazon review files, select a T4 GPU, and run all cells.

This is a small, compute-conscious implementation study; human evaluation would strengthen the generation analysis.

