# README

## Overview

 This repository implements PROBIX, a probing-guided adversarial suffix attack designed to uncover vulnerabilities and assess the robustness of large language models (LLMs) across various NLP tasks.

## Features

- **Probe-guided Adversarial Attack**: Leverages task-specific linear probes
as surrogate predictors of model behavior, significantly reducing repeated queries to the target model.
- **Optimization Strategy**: Employs a composite objective to balance adversarial effectiveness, semantic similarity, and linguistic fluency during adversarial text generation. It further incorporates a part-of-speech (POS)-aware perturbation strategy and an LLM-based paraphrasing refinement module to improve exploration and mitigate artifacts.
- **Evaluation Metrics**: Computes Attack Success Rate (ASR), semantic similarity, and perplexity to assess both adversarial effectiveness and the quality of the generated adversarial text. It also reports the computational cost of adversarial example generation.

## Requirements

- Python 3.11
- PyTorch 2.6.0
- Transformers 4.50.1
- NLTK
- SentenceTransformers

## Set environments
- `conda create --name new_env`
- `conda env list`
- `conda activate new_env`

## Install dependencies:
- `conda install --file conda-requirements.txt`

## Main interface
- Supported `dataset_name` values: `sst2`, `AG-News`, `StrategyQA`
- White-box setting: use `--transfer_mode off`

```bash
python run_optim.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset_name sst2 \
  --num_examples 5 \
  --n_epochs 50 \
  --suffix_len 5 \
  --transfer_mode off
```

- Transfer setting: use `--transfer_mode on` and also provide `--target_model`

```bash
python run_optim.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset_name sst2 \
  --num_examples 5 \
  --n_epochs 50 \
  --suffix_len 5 \
  --transfer_mode on \
  --target_model gpt-4
```

- For custom target routes, `--target_model` also supports `openrouter:<model_id>`, `openai:<model_id>`, or `ollama:<local_tag>`.

## Train probe for new task or new model
- `python train_probe.py`
