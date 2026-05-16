# GPT-2 from Scratch

A PyTorch implementation of a GPT-2-style decoder-only Transformer, written for study rather than deployment.

The repository keeps the main parts of the model visible: tokenization, causal self-attention, multi-head attention, transformer blocks, next-token pretraining, text generation, classification fine-tuning, and instruction fine-tuning. The goal is to understand the model from the inside, not to hide it behind a high-level API.

## Why this project exists

I built this project as a way to read the original language-model papers with code open next to them. GPT-like models are easy to use today, but much harder to reason about if the architecture stays abstract.

Here, each piece is small enough to inspect: how embeddings are built, how the causal mask blocks future tokens, how residual transformer blocks are assembled, how the training loop learns next-token prediction, and how sampling changes the behavior of the model.

This also connects to my research interests in representation learning and predictive world models. Before using large models as components in larger systems, I want to be able to rebuild the core machinery myself.

## Literature-first implementation

The project started from research papers and local reading notes kept in `litterature/`. The local PDFs and diagrams stay out of git, but the public references are:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer blocks, scaled dot-product attention, multi-head attention, positional encodings, residual connections, and feed-forward layers.
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - The attention idea before Transformers: learn soft alignments instead of forcing a sequence into one fixed vector.
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - GPT-1, generative pretraining, and supervised fine-tuning.
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2, decoder-only language modeling, WebText-style pretraining, and zero-shot transfer through next-token prediction.
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3 scaling, in-context learning, and few-shot prompting.
- [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) - GELU activations in transformer feed-forward layers.
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) - Instruction-data generation and instruction-following fine-tuning.
- [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) - MMLU and broad multitask evaluation.
- [Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394) - Objective design for instruction fine-tuning.

The workflow is paper-to-code: take one idea from the literature, implement it in PyTorch, test it in a notebook or a small script, then connect it to the rest of the model.

## Repository structure

```text
src/
  attention.py                 causal and multi-head self-attention
  transformer.py               layer norm, feed-forward blocks, transformer blocks
  gpt_model.py                 GPT-style model definition
  gpt_config.py                model configuration presets
  tokenizer.py                 tokenization utilities
  dataloaders.py               dataset and dataloader helpers
  train.py                     training and evaluation loops
  pretraining.py               pretraining utilities
  text_generator.py            autoregressive generation helpers
  classification_finetuning.py spam/classification fine-tuning workflow
  instruction_finetuning.py    instruction tuning workflow
  metrics.py                   evaluation helpers
  plots.py                     plotting utilities

Notebooks/
  attention.ipynb
  GPT_model.ipynb
  tokenizer.ipynb
  pretraining.ipynb
  finetunning.ipynb
  Toy_exemple_token_embedding.ipynb

reports/figures/
  Small training and evaluation figures for GitHub.
```

## What stays local

Some files are useful on my machine but should not be in a public repository:

- `models/` for checkpoints and converted weights
- `data/` for local datasets
- `litterature/` for PDFs, diagrams, and reading notes
- `.venv/` and `.vscode/` for local environment state

Keeping them out of git keeps the repository light and avoids redistributing papers, datasets, or model weights.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a small model sanity check:

```bash
python src/gpt_model.py
```

Most experiments are run from the notebooks or from `src/text_generator.py`.

## Topics covered

- Token and positional embeddings
- Scaled dot-product attention
- Causal masking
- Multi-head attention
- Transformer blocks
- GPT-style decoder-only modeling
- Next-token prediction
- Temperature and top-k sampling
- Pretraining loops
- Classification fine-tuning
- Instruction fine-tuning

## Status

The core implementation is present. I am still cleaning the notebooks and scripts so the public version reads like a study project someone can follow, not just a folder of experiments.
