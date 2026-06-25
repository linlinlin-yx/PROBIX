import argparse
import json
import os
import time

import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    PreTrainedTokenizerFast,
)

from run_paraphrase import get_paraphrase_text
from src.dataset_utils import load_dataset_custom
from src.generate_and_save_category_ids import (
    load_category_ids,
    load_emotional_words,
    select_candidate_ids,
    select_candidate_ids_for_agnews,
)
from src.prompt_optim import compute_perplexity, compute_use_similarity, optimize_adversarial_suffix

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Run PROBIX adversarial suffix optimization (stage 1).")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset_name", default="sst2")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--similarity_threshold", type=float, default=0.5)
    parser.add_argument("--suffix_len", type=int, default=5)
    parser.add_argument("--ppl_threshold", type=float, default=5000.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--output_file", default="stage1_candidates.jsonl")
    return parser.parse_args()


def build_stage1_record(sample_id, prompt_text, optimized_text, paraphrased_text, ground_truth_label, target_label, similarity, perplexity, ppl_before, epoch, dataset_name, model_name):
    return {
        "sample": sample_id,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "original_prompt": prompt_text,
        "optimized_prompt": optimized_text,
        "paraphrased_prompt": paraphrased_text,
        "orig_pred": ground_truth_label,
        "target_label": target_label,
        "epoch": epoch,
        "similarity": similarity,
        "perplexity": perplexity,
        "perplexity_opt": ppl_before,
    }


def save_stage1_results(records, output_file, execution_time, execution_time_data, args):
    with open(output_file, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "num_candidates": len(records),
        "execution_time": execution_time,
        "execution_time_for_adv_examples": execution_time_data,
        "similarity_threshold": args.similarity_threshold,
        "ppl_threshold": args.ppl_threshold,
        "suffix_length": args.suffix_len,
        "num_iterations": args.n_epochs,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
    }
    summary_path = os.path.splitext(output_file)[0] + "_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved {len(records)} stage-1 candidates to {output_file}")
    print(f"Saved stage-1 summary to {summary_path}")


args = parse_args()
print("Initialize experimental configuration")

use_model = SentenceTransformer("all-MiniLM-L6-v2")
model_name = args.model_name
dataset_name = args.dataset_name
num_examples = args.num_examples
n_epochs = args.n_epochs
similarity_threshold = args.similarity_threshold
suffix_len = args.suffix_len

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token = "[PAD]"

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

if args.dataset_name == "AG-News":
    classifier_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news", num_labels=4)
    classifier_tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
else:
    classifier_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2", num_labels=2)
    classifier_tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
bert_model.to(device)
gpt2_model.to(device)
classifier_model.to(device)
use_model.to(device)

classifier_probe_path = f"./src/probe/classifier_probe_{args.model_name.replace('/', '_')}_{args.dataset_name}.pt"
num_labels = 4 if args.dataset_name == "AG-News" else 2
classifier_probe = torch.nn.Linear(model.config.hidden_size, num_labels).to(device)
if os.path.exists(classifier_probe_path):
    classifier_probe.load_state_dict(torch.load(classifier_probe_path))
    print(f"Loaded classifier probe from {classifier_probe_path}")
else:
    raise FileNotFoundError(f"Classifier probe not found at {classifier_probe_path}")
classifier_probe.eval()

dataset_class = load_dataset_custom(args)

stage1_records = []
print("Starting stage-1 adversarial optimization...")
start_time = time.time()
category_ids, positive_ids, negative_ids = load_emotional_words(bert_tokenizer, dataset_name)

if args.dataset_name == "AG-News":
    category_ids = load_category_ids()

execution_time_data = 0.0
for i, (prompt_text, ground_truth_label) in enumerate(dataset_class):
    start_time_data = time.time()
    print(f"Processing sample {i+1}/{len(dataset_class)}")

    if args.dataset_name == "AG-News":
        label_priority = {
            0: [1, 2, 3],
            1: [0, 2, 3],
            2: [0, 1, 3],
            3: [2, 0, 1],
        }
        target_label = label_priority[ground_truth_label][0]
        candidate_ids = select_candidate_ids_for_agnews(bert_tokenizer, target_label, category_ids)
    else:
        target_label = 0 if ground_truth_label == 1 else 1
        candidate_ids = select_candidate_ids(dataset_name, target_label, positive_ids, negative_ids)
        category_ids = None

    optimized_text = optimize_adversarial_suffix(
        model=model,
        tokenizer=tokenizer,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
        gpt2_model=gpt2_model,
        gpt2_tokenizer=gpt2_tokenizer,
        classifier_model=classifier_model,
        classifier_tokenizer=classifier_tokenizer,
        orig_prompt=prompt_text,
        orig_pred=ground_truth_label,
        target_label=target_label,
        dataset_name=args.dataset_name,
        suffix_len=suffix_len,
        similarity_threshold=args.similarity_threshold,
        ppl_threshold=args.ppl_threshold,
        n_steps=n_epochs,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        use_model=use_model,
        candidate_ids=candidate_ids,
        classifier_probe=classifier_probe,
    )

    paraphrased_text = get_paraphrase_text(args.dataset_name, optimized_text)
    if paraphrased_text is None:
        print("Paraphrasing failed: return None.")
        continue

    end_time_data = time.time()
    execution_time_data += end_time_data - start_time_data

    similarity = compute_use_similarity(prompt_text, paraphrased_text, use_model)
    perplexity = compute_perplexity(gpt2_model, gpt2_tokenizer, paraphrased_text, device)
    ppl_before = compute_perplexity(gpt2_model, gpt2_tokenizer, optimized_text, device)

    stage1_records.append(
        build_stage1_record(
            sample_id=i + 1,
            prompt_text=prompt_text,
            optimized_text=optimized_text,
            paraphrased_text=paraphrased_text,
            ground_truth_label=ground_truth_label,
            target_label=target_label,
            similarity=similarity,
            perplexity=perplexity,
            ppl_before=ppl_before,
            epoch=n_epochs,
            dataset_name=args.dataset_name,
            model_name=model_name,
        )
    )

end_time = time.time()
execution_time = end_time - start_time
save_stage1_results(stage1_records, args.output_file, execution_time, execution_time_data, args)
