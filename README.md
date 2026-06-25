# PROBIX

## Overview

 This repository implements PROBIX, a probing-guided adversarial suffix attack designed to uncover vulnerabilities and assess the robustness of large language models (LLMs) across various NLP tasks.


## Features

- **Probe-guided optimization**: Uses task-specific linear probes as surrogate predictors during adversarial suffix optimization.
- **Optimization Strategy**: Employs a composite objective to balance adversarial effectiveness, semantic similarity, and linguistic fluency during adversarial text generation. It further incorporates a part-of-speech (POS)-aware perturbation strategy and an LLM-based paraphrasing refinement module to improve exploration and mitigate artifacts.
- **Evaluation metrics**: Tracks attack success rate (ASR), semantic similarity, perplexity, and runtime statistics.

## Requirements

- Python 3.11
- PyTorch 2.6.0
- Transformers 4.50.1
- NLTK
- SentenceTransformers

## Environment setup

- `conda create --name probix_env`
- `conda activate probix_env`
- `conda install --file conda-requirements.txt`

## Supported datasets

- `sst2`
- `AG-News`
- `StrategyQA`

## Main interface

`run_optim.py` performs the generation of adversarial examples and saves them to a JSONL file.

Example:

```bash
python run_optim.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset_name sst2 \
  --n_epochs 50 \
  --similarity_threshold 0.5 \
  --suffix_len 5 \
  --output_file adv_examples.jsonl
```

`run_predict.py` evaluates the attack success rate and quality of the saved examples.

White-box example via openrouter:

```bash
python run_predict.py \
  --input_file adv_examples.jsonl \
  --output_file prediction_results.jsonl \
  --transfer_mode off
```

Transfer example:

```bash
python run_predict.py \
  --input_file adv_examples.jsonl \
  --output_file prediction_results.jsonl \
  --transfer_mode on \
  --target_model openai:gpt-4o-mini
```

#### Note

Since the API of `mistralai/Mistral-7B-Instruct-v0.3` is currently unavailable, you can instead download the [model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) locally and provide its path via `--hf_local_path`.

Example:

```bash
python run_predict.py \
  --input_file adv_examples.jsonl \
  --output_file prediction_results.jsonl \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --transfer_mode off \
  --hf_local_path /path/to/Mistral-7B-Instruct-v0.3
```


## Train a probe

```bash
python train_probe.py
```
